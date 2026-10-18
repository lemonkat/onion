import dill
import typing

import numpy as np
from tqdm import tqdm

from onion.util import softmax, batch
from onion.layer import Layer, Parameter, L2Regularization

class Model:
    """
    Models represent collections of layers and parameters.
    """
    MAX_BATCH_SIZE = 2000
    def __init__(self, layers: list[Layer] | None = None, parameters: list[Parameter] | None = None, mode: str = "train") -> None:
        self.layers = layers or []
        self.parameters = parameters or []
        self.parameters.extend(layer for layer in self.layers if isinstance(layer, Parameter))
        self.mode = mode

    def train(self) -> None:
        self.mode = "train"
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        self.mode = "eval"
        for layer in self.layers:
            layer.eval()

    def is_train(self) -> bool:
        return self.mode == "train"
        
    def is_eval(self) -> bool:
        return self.mode == "eval"
    
    @property
    def params(self) -> dict[str, np.ndarray[np.float64]]:
        return {param.name: param.value for param in self.parameters}

    @property
    def gradients(self) -> dict[str, np.ndarray[np.float64]]:
        return {param.name: param.output_grad for param in self.parameters}
    
    def _predict(self, X: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        raise NotImplementedError
    
    def predict(self, X: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        if len(X) > self.MAX_BATCH_SIZE:
            if self.is_eval():
                return np.concatenate([self._predict(Xb) for Xb in tqdm(batch(X, batch_size=self.MAX_BATCH_SIZE, shuffle=False)[0])], axis=0)
            return np.concatenate([self._predict(Xb) for Xb in batch(X, batch_size=self.MAX_BATCH_SIZE, shuffle=False)[0]], axis=0)
        return self._predict(X)

    def probs(self, X: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        raise NotImplementedError
    
    def _grad(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> None:
        raise NotImplementedError
    
    def classify(self, X: np.ndarray[np.float64]) -> np.ndarray[np.int32]:
        return np.argmax(self.predict(X), axis=-1)
    
    def accuracy(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> float:
        return np.mean(self.classify(X) == np.argmax(y, axis=-1))
    
    def _loss(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> float:
        raise NotImplementedError
    
    def loss(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> float:
        if len(X) > self.MAX_BATCH_SIZE:
            if self.is_eval():
                return sum(self._loss(Xb, yb) * len(Xb) / len(X) for Xb, yb in tqdm(list(zip(*batch(X, y, batch_size=self.MAX_BATCH_SIZE, shuffle=False)))))
            return sum(self._loss(Xb, yb) * len(Xb) / len(X) for Xb, yb in zip(*batch(X, y, batch_size=self.MAX_BATCH_SIZE, shuffle=False)))
        return self._loss(X, y)
    
    def grad(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> None:
        self._grad(X, y)
        return self.gradients
    
    def grad_check(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64], eps: float = 1e-5) -> None:
        grad = self.grad(X, y)
        for param in self.parameters:
            print(f"Checking {param.name}:")
            print(f"Analytical gradient {grad[param.name]}")
            numgrad = np.zeros(param.shape, dtype=np.float64)
            for i, val in tqdm(np.ndenumerate(param.value), total=len(param.value.flatten())):
                param.value[i] = val + eps
                out1 = self.loss(X, y)
                param.value[i] = val - eps
                out2 = self.loss(X, y)
                param.value[i] = val
                numgrad[i] = np.sum(out1 - out2) / (2 * eps)
            print(f"Numerical gradient {numgrad}")
            print(f"Difference {abs(numgrad - grad[param.name])}")
            print()

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, path: str) -> typing.Self:
        with open(path, "rb") as file:
            return dill.load(file)
        
class DAGNode:
    def __init__(self, layer: Layer,  **inputs: typing.Self) -> None:

        self.layer = layer
        self.inputs = inputs

        assert set(self.layer.input_names) == set(self.inputs)

        self.name = self.layer.name

        for name, node in self.inputs.items():
            setattr(self, name, node)

    def __hash__(self) -> int:
        return hash(self.layer.name)
    
def node(
    layer: Layer, 
    loss_list: list[typing.Self] | None = None, 
    name_counter: dict[str, int] | None = None,
    activation: Layer | None = None, 
    **inputs: DAGNode | Layer | tuple[np.ndarray[np.float64] | tuple[list[int], float, float], float | Layer]
) -> DAGNode:
    """
    Utility DAGNode constructor.
    Handles parameter creation and initialization, duplicate name removal, activation and regularization integration, and more.
    """
    while layer.name in name_counter:
        name_counter[layer.name] += 1
        layer.name = layer.name + "_" + str(name_counter[layer.name])
    name_counter[layer.name] = 0
    inp_nodes = {}
    for name, obj in inputs.items():
        if isinstance(obj, DAGNode):
            inp_nodes[name] = obj
        elif isinstance(obj, Layer):
            inp_nodes[name] = DAGNode(obj)
        else:
            val, reg = obj
            if isinstance(val, tuple):
                shape, shift, scale = val
                val = np.random.normal(shift, scale, shape)
            
            param_node = inp_nodes[name] = node(Parameter(layer.name + "_" + name, val), name_counter=name_counter)
            if isinstance(reg, float) and reg > 0.0:
                reg = L2Regularization(param_node.name + "_reg", reg)
            if isinstance(reg, Layer):
                reg_node = node(reg, name_counter=name_counter, **{list(reg.input_names)[0]: param_node})
                loss_list.append(reg_node)

    nd = DAGNode(layer, **inp_nodes)

    if activation is not None:
        nd = node(activation, name_counter=name_counter, **{list(activation.input_names)[0]: nd})

    return nd
        
    
class DAGModel(Model):
    """
    In a DAGModel, the computation is represented as a directed acyclic graph of DAGNodes.
    Each node wraps a layer and ensures computations happen in order.
    4 nodes must be specified: 
    the model's input, the model's output, the ground truth for loss computation, and the model's loss(es).
    """
    def __init__(self, input: DAGNode | Parameter, output: DAGNode, truth: DAGNode | Parameter, loss: DAGNode | list[DAGNode], mode: str = "train") -> None:
        if isinstance(input, Parameter):
            input = DAGNode(input)

        if isinstance(truth, Parameter):
            truth = DAGNode(truth)

        assert isinstance(input.layer, Parameter), "Input node must be a Parameter layer!"
        assert isinstance(truth.layer, Parameter), "Truth node must be a Parameter layer!"

        if isinstance(loss, DAGNode):
            loss = [loss]
        
        self.input_node, self.output_node, self.truth_node, self.loss_nodes = input, output, truth, loss
        self.nodes = self.get_nodes()
        
        super().__init__([nd.layer for nd in self.nodes], mode=mode)

        # print("NODES: ", *[nd.name + ((" on " + ", ".join(inp + "=" + pnd.name for inp, pnd in nd.inputs.items())) if nd.inputs else "") for nd in self.nodes], sep="\n")

    # find all nodes, in order of dependency
    def get_nodes(self) -> list[DAGNode]:
        out = []
        seen = {self.output_node}
        def dfs(nd: DAGNode) -> None:
            for prev in nd.inputs.values():
                if prev not in seen:
                    seen.add(prev)
                    dfs(prev)
            if nd != self.input_node and nd != self.truth_node:
                out.append(nd)
        
        dfs(self.output_node)

        for nd in self.loss_nodes:
            if nd not in seen:
                seen.add(nd)
                dfs(nd)

        return out
    
    def forward(self, values: dict[DAGNode, np.ndarray[np.float64]], no_loss: bool = False) -> dict[str, np.ndarray[np.float64]]:
        for nd in self.nodes:
            values[nd] = nd.layer.forward(**{name: values[prev] for name, prev in nd.inputs.items()})
            # print(nd.name, "=", np.array(values[nd]).shape)
            if no_loss and nd == self.output_node:
                break
        return values

    def _predict(self, X: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return self.forward({self.input_node: X}, True)[self.output_node]
        
    def probs(self, X: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return softmax(self.predict(X))
    
    def _loss(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> float:
        values = self.forward({self.input_node: X, self.truth_node: y}, False)
        return sum(values[nd] for nd in self.loss_nodes)
    
    def _grad(self, X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> None:
        self.forward({self.input_node: X, self.truth_node: y}, False)

        grads = {nd: np.array(1.0, dtype=np.float64) for nd in self.loss_nodes}

        for nd in reversed(self.nodes):
            inp_grads = nd.layer.backward(grads[nd])
            for name, prev in nd.inputs.items():
                grads[prev] = inp_grads[name] + grads.get(prev, 0.0)
                # print(nd.name, name, inp_grads[name])

    # trying a custom saving and loading system
    # basically store all data as plain-text
    # by saving and loading model structure + params + layer settings
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            print("INPUT", self.input_node.name, file=f)
            print("OUTPUT", self.output_node.name, file=f)
            print("TRUTH", self.truth_node.name, file=f)
            print("LOSS", *[nd.name for nd in self.loss_nodes], file=f)
            nodes = [self.input_node, self.truth_node] + self.nodes
            print(len(nodes), file=f)
            for nd in nodes:
                print(str(nd.layer.__class__)[8:-2], nd.name, "ON", *[f"{name}={inp.name}" for name, inp in nd.inputs.items()], file=f)
                cfg = nd.layer.to_config()
                print(len(cfg), file=f)
                for key, val in cfg.items():
                    match val:
                        case str():
                            print("STR", key, file=f)
                            print("VALUE", val, file=f)
                        case int():
                            print("INT", key, file=f)
                            print("VALUE", val, file=f)
                        case float():
                            print("FLOAT", key, file=f)
                            print("VALUE", val, file=f)
                        case np.ndarray():
                            print("ARR", key, file=f)
                            print("DTYPE", val.dtype, file=f)
                            print("SHAPE", *val.shape, file=f)
                            print("VALUE", *val.flatten(), file=f)
                        case list() | tuple(): # mostly shapes
                            print("TUPLE", key, file=f)
                            print("VALUE", *val, file=f)


    @classmethod
    def load(cls, path: str) -> typing.Self:
        layer_classes = Layer.all_subclasses()
        # print(layer_classes)
        with open(path, "r") as f:
            def next_line() -> str:
                return f.readline().strip().split()
            _, inp_name = next_line()
            _, out_name = next_line()
            _, truth_name = next_line()
            _, *loss_names = next_line()
            nodes = {}
            for _ in range(int(next_line()[0])):
                layer_type, name, _, *args = next_line()
                inputs = {key: nodes[val] for key, val in (entry.split("=") for entry in args)}
                cfg = {}
                for _ in range(int(next_line()[0])):
                    value_type, key = next_line()
                    match value_type:
                        case "STR":
                            value = next_line()[1]
                        case "INT":
                            value = int(next_line()[1])
                        case "FLOAT":
                            value = float(next_line()[1])
                        case "ARR":
                            dtype = next_line()[1]
                            shape = next_line()[1:]
                            value = np.array(next_line()[1:], dtype=dtype).reshape([int(d) for d in shape])
                        case "TUPLE":
                            value = tuple(int(n) for n in next_line()[1:])
                    cfg[key] = value
                layer = layer_classes[layer_type].from_config(name, cfg)
                nodes[name] = DAGNode(layer, **inputs)
            return DAGModel(
                nodes[inp_name], 
                nodes[out_name], 
                nodes[truth_name], 
                [nodes[name] for name in loss_names],
            )
            





            

    
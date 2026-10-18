import typing

import numpy as np
import torch as th

from onion.layer import Layer

# WIP, never tested nor used
class TorchWrapper(Layer):
    def __init__(
            self, 
            name: str = "TorchLayer", 
            layer: th.nn.Module | None = None,
            inputs: set[str] | None = None,
            mode: str = "train",
            **kwargs: typing.Any,
        ) -> None:
        self.layer = layer.double()
        self.layer_inputs = set(inputs)
        self.layer_params = {p for p, _ in self.layer.named_parameters()}
        super().__init__(name, self.layer_inputs | self.layer_params, mode)
        self.kwargs = kwargs
        self.cache = None, None

    def _forward(self, **inputs: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        self.layer.train(self.is_train())
        self.layer.zero_grad()
        inputs = {key: th.tensor(val, dtype=th.float64, requires_grad=True) for key, val in inputs.items()}
        layer_input = {}
        layer_state = self.layer.state_dict()
        for key, val in inputs.items():
            if key in self.layer_params:
                layer_state[key][()] = val
            else:
                layer_input[key] = val
        layer_output = self.layer.forward(**layer_input, **self.kwargs)
        self.cache = layer_input, layer_output
        return layer_output.numpy(force=True)
    
    def _backward(
        self, grad: np.ndarray[np.float64]
    ) -> dict[str, np.ndarray[np.float64]]:
        layer_input, layer_output = self.cache
        layer_output.backward(th.tensor(grad, dtype=th.float64))
        layer_input_grad = {key: layer_input[key].grad.numpy(force=True) for key in self.layer_inputs}
        layer_param_grad = {name: param.grad.numpy(force=True) for name, param in self.layer.named_parameters()}
        return layer_input_grad | layer_param_grad
    
    def to_config(self) -> dict[str, object]:
        for p in self.layer.parameters():
            p[()] = 0
        

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, object]) -> typing.Self:
        pass
        # return MaxPool(name=name, shape=cfg["shape"], stride=cfg.get("stride", None))
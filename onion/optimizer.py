
import numpy as np

from onion.model import Model

class Optimizer:
    def __init__(self, model: Model):
        self.model = model

    def step(self, grad: dict[str, np.ndarray[np.float64]]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, model: Model, learning_rate: float = 1e-3) -> None:
        super().__init__(model)
        self.learning_rate = learning_rate

    def step(self, grad: dict[str, np.ndarray[np.float64]]) -> None:
        for param in self.model.params:
            self.model.params[param] -= self.learning_rate * grad[param]

class ADAM(Optimizer):
    def __init__(
        self, model: Model, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999
    ) -> None:
        super().__init__(model)
        self.lr = lr
        self.b1, self.b2 = b1, b2
        self.m1 = {param: 0.0 for param in model.params}
        self.m2 = {param: 0.0 for param in model.params}

    def step(self, grad: dict[str, np.ndarray[np.float64]]) -> None:
        for param in self.model.params:
            self.m1[param] = self.m1[param] * self.b1 + grad[param] * (1 - self.b1)
            self.m2[param] = self.m2[param] * self.b2 + grad[param] ** 2 * (1 - self.b2)
            self.model.params[param] -= (
                self.lr * self.m1[param] / np.sqrt(self.m2[param] + 1e-7)
            )


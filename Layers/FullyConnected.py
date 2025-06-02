from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)

        self._optimizer = None
        self._gradient_weights = None
        self.last_input = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor: np.ndarray):
        batch_size = len(input_tensor)
        self.last_input = np.append(input_tensor, np.ones((batch_size, 1)), axis=1)
        return self.last_input @ self.weights

    def backward(self, error_tensor: np.ndarray):
        self.gradient_weights = self.last_input.T @ error_tensor
        gradient = error_tensor @ self.weights[:-1].T
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient

#a = np.array([[1, 2], [2, 3], [1, 3]])
#b = np.array([[1, 0], [0, 1]])
#print(np.append(a, np.ones((3, 1)), axis=1))
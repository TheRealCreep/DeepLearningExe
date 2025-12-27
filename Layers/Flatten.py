from Layers.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_dim = None

    def forward(self, input_tensor: np.ndarray):
        self.input_dim = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor: np.ndarray):
        return error_tensor.reshape(self.input_dim)
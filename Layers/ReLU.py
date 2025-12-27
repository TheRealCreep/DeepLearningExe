from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.last_activation = None

    def forward(self, input_tensor: np.ndarray):
        self.last_activation = np.where(input_tensor <= 0, 0, 1)
        return input_tensor * self.last_activation

    def backward(self, error_tensor: np.ndarray):
        return error_tensor * self.last_activation

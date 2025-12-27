from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.last_output = None

    def forward(self, input_tensor: np.ndarray):
        exponential = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.last_output = exponential / np.sum(exponential, axis=1, keepdims=True)
        return self.last_output

    def backward(self, error_tensor: np.ndarray):
        return self.last_output * (error_tensor - np.sum(error_tensor * self.last_output, axis=1, keepdims=True))

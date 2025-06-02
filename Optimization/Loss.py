import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction = None

    def forward(self, prediction_tensor: np.ndarray, label_tensor: np.ndarray):
        self.prediction = prediction_tensor
        dtype = prediction_tensor.dtype
        epsilon = np.finfo(dtype).eps
        ce = np.mean(np.sum(-label_tensor * np.log((self.prediction + epsilon))))
        return ce

    def backward(self, label_tensor: np.ndarray):
        dtype = label_tensor.dtype
        epsilon = np.finfo(dtype).eps
        gradient = - label_tensor / (self.prediction + epsilon)
        return gradient

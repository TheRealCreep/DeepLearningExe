import numpy as np


class Sgd():
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.last_update = None

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        if self.last_update is not None:
            update = self.momentum_rate * self.last_update - self.learning_rate * gradient_tensor
        else:
            update = - self.learning_rate * gradient_tensor
        self.last_update = update
        return weight_tensor + update


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 0
        self.v = None
        self.r = None

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray):
        gradient_tensor = np.asarray(gradient_tensor)
        weight_tensor = np.asarray(weight_tensor)

        if self.v is None:
            self.v = np.zeros_like(gradient_tensor)
        if self.r is None:
            self.r = np.zeros_like(gradient_tensor)

        self.k += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)

        v_cor = self.v / (1 - self.mu ** self.k)
        r_cor = self.r / (1 - self.rho ** self.k)

        dtype = gradient_tensor.dtype
        epsilon = np.finfo(dtype).eps

        update = -self.learning_rate * v_cor / (np.sqrt(r_cor) + epsilon)
        return weight_tensor + update

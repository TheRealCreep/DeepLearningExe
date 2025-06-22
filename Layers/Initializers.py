import numpy as np


class Constant:
    def __init__(self, weight=0.1):
        self.weight = weight

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.weight)


class UniformRandom:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.rand(weights_shape[0], weights_shape[1])


class Xavier:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)


class He:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)

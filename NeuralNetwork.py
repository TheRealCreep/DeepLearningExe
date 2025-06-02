import numpy as np
import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.labels = None

    def forward(self):
        data_in, labels = self.data_layer.next()
        self.labels = labels
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        loss = self.loss_layer.forward(out, labels)
        return loss

    def backward(self):
        gradient = self.loss_layer.backward(self.labels)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        out = input_tensor
        for layer in self.layers:
            out = layer.forward(out)
        return out

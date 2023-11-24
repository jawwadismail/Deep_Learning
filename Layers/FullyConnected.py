# FullyConnected.py

import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.optimizer = None
        self.input_tensor = None
        self.gradient_weights = None

    def forward(self, input_tensor):
        self.input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))  # Add bias
        return np.dot(self.input_tensor, self.weights)
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer



    def backward(self, error_tensor):
        gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        error_tensor_prev = np.dot(error_tensor, self.weights[:-1, :].T)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_weights)
        self.gradient_weights = gradient_weights
        return error_tensor_prev
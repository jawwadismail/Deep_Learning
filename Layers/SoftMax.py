from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        max_arr = np.max(input_tensor, axis=1, keepdims=True)

        sum = np.sum(np.exp(input_tensor - max_arr), axis=1, keepdims=True)
        prediction = np.divide(np.exp(input_tensor - max_arr), sum)
        self.prediction = prediction

        return prediction

    def backward(self, error_tensor):
        multiplication = np.multiply(error_tensor, self.prediction)
        summation = np.sum(multiplication, axis=1, keepdims=True)
        error_difference = error_tensor - summation
        error_back = np.multiply(error_difference, self.prediction)

        return error_back
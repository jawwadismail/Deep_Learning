import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        loss = -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))
        self.prediction_tensor = prediction_tensor
        return loss

    def backward(self,label_tensor):
        gradient = -(np.divide(label_tensor,self.prediction_tensor + np.finfo(float).eps))
        return gradient
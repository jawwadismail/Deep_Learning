
import copy
class NeuralNetwork:

    def __init__(self ,optimizer) -> None:
        self.optimizer = optimizer
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.loss = []
    def append_layer(self ,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def forward(self):
        input_tensor ,label_tensor =self.data_layer.next()
        self.label_tensor = label_tensor
        for i ,layer in enumerate(self.layers):
            if i== 0:
                output_tensor = layer.forward(input_tensor)
            else:
                output_tensor = layer.forward(output_tensor)
        self.prediction = output_tensor
        loss = self.loss_layer.forward(output_tensor, label_tensor)
        return loss

    def backward(self):
        # input_tensor, label_tensor = self.data_layer.backward()

        gradient = self.loss_layer.backward(self.label_tensor)

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                output_gradient = layer.backward(gradient)
            else:
                output_gradient = layer.backward(output_gradient)

    def train(self, iterations):
        for iter in range(0, iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):

        for i, layer in enumerate(self.layers):
            if i == 0:
                output_tensor = layer.forward(input_tensor)
            else:
                output_tensor = layer.forward(output_tensor)
        return output_tensor
# import copy
# import numpy as np
#
# class NeuralNetwork:
#     def __init__(self, optimizer):
#         self.optimizer = optimizer
#         self.loss = []
#         self.layers = []
#         self.data_layer = None
#         self.loss_layer = None
#
#     def forward(self):
#         input_tensor, label_tensor = self.data_layer.next()
#         output_tensor = input_tensor
#
#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)
#
#         return self.loss_layer.forward(output_tensor, label_tensor)
#
#     def append_layer(self, layer):
#
#         if layer.trainable:
#             optimizer_copy = copy.deepcopy(self.optimizer)
#             layer.optimizer = optimizer_copy
#         self.layers.append(layer)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     def forward(self):
#         input_tensor, label_tensor = self.data_layer.next()  # Get input and label from the data layer
#         output_tensor = input_tensor
#
#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)
#
#         return self.loss_layer.forward(output_tensor, label_tensor)
#
#     def backward(self):
#         error_tensor = self.loss_layer.backward()  # Start backward pass from the loss layer
#
#         for layer in reversed(self.layers):
#             error_tensor = layer.backward(error_tensor)
#
#     def append_layer(self, layer):
#
#         if layer.trainable:
#             optimizer_copy = copy.deepcopy(self.optimizer)
#             layer.optimizer = optimizer_copy
#         self.layers.append(layer)
#
#     def train(self, iterations):
#         for _ in range(iterations):
#             loss_value = self.forward()  # Forward pass
#             self.loss.append(loss_value)
#             self.backward()  # Backward pass
#
#     def test(self, input_tensor):
#         output_tensor = input_tensor
#
#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)
#
#         return output_tensor
#
#
#
#









# # NeuralNetwork.py

# import copy
# import numpy as np

# class NeuralNetwork:
#     def __init__(self, optimizer):
#         self.optimizer = optimizer
#         self.loss = []  # List to store loss values for each iteration
#         self.layers = []  # List to hold the architecture of the network
#         self.data_layer = None  # Placeholder for the data layer
#         self.loss_layer = None  # Placeholder for the loss layer

#     def forward(self):
#         input_tensor, label_tensor = self.data_layer.next()  # Get input and label from the data layer
#         output_tensor = input_tensor

#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)

#         return self.loss_layer.forward(output_tensor, label_tensor)

#     def backward(self):
#         error_tensor = self.loss_layer.backward()  # Start backward pass from the loss layer

#         for layer in reversed(self.layers):
#             error_tensor = layer.backward(error_tensor)

#     def append_layer(self, layer):
#         self.layers.append(layer)
#         if layer.trainable:
#             layer.optimizer = copy.deepcopy(self.optimizer)

#     def train(self, iterations):
#         for _ in range(iterations):
#             loss_value = self.forward()  # Forward pass
#             self.loss.append(loss_value)
#             self.backward()  # Backward pass

#     def test(self, input_tensor):
#         output_tensor = input_tensor

#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)

#         return output_tensor

# NeuralNetwork.py
class BaseLayer:
    def __init__(self):
        self.trainable = False
        # self.optimizer = None

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass

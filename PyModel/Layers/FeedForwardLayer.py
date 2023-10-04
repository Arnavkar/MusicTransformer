import sys
# appending Layer path
sys.path.append('./PyModel')

from tensorflow.keras.layers import Layer, Dense, ReLU
from Layers.utils import check_shape
    
class FeedForward(Layer):
    def __init__(self, d_in, d_out, **kwargs):
        super(FeedForward,self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_in)  # First fully connected layer takes in input dimensions
        self.fully_connected2 = Dense(d_out)  # Second fully connected layer, based on model output dimensions
        self.activation = ReLU()

    def call(self, x):
        output = self.fully_connected1(x)
        output = self.activation(output)
        output = self.fully_connected2(output)
        return output
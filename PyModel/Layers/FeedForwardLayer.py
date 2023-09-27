import sys
# appending Layer path
sys.path.append('Layers')

from tensorflow.keras.layers import Layer, Dense
from Layers.utils import check_shape
    
class FeedForward(Layer):
    def __init__(self, d_in, d_out, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_in, activation = "relu")  # First fully connected layer takes in input dimensions
        self.fully_connected2 = Dense(d_out)  # Second fully connected layer, based on model output dimensions

    def call(self, x):
        return self.fully_connected2(self.fully_connected1(x))
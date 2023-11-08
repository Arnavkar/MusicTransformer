from tensorflow.keras.layers import Layer, Dense, ReLU
import tensorflow as tf
from .utils import check_shape
    
class FeedForward(Layer):
    def __init__(self, d_in, d_out, **kwargs):
        super(FeedForward,self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_in, kernel_initializer="glorot_uniform", bias_initializer=tf.keras.initializers.Zeros())  # First fully connected layer takes in input dimensions
        self.fully_connected2 = Dense(d_out, kernel_initializer="glorot_uniform", bias_initializer=tf.keras.initializers.Zeros())  # Second fully connected layer, based on model output dimensions
        self.activation = ReLU()

    def call(self, x):
        output = self.fully_connected1(x)
        output = self.activation(output)
        output = self.fully_connected2(output)
        return output
    

if __name__ == "__main__":
    test_tensor = tf.constant([[1,3,4,2,0]])
    layer = FeedForward(5,3)
    output = layer(test_tensor)
    print(f"feedforward output: {output}")
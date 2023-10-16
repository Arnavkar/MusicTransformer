import sys
# appending Layer path
sys.path.append('./PyModel')

from tensorflow.keras.layers import Layer, LayerNormalization
import tensorflow as tf
from Transformer.utils import check_shape

class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
    
    def call(self, x, sublayer_x):
        add = x + sublayer_x     
        return self.layer_norm(add)
    
if __name__ == "__main__":
    test_tensor = tf.constant([[1,3,4,2,0]])
    addnorm_layer = AddNormalization()
    addnorm_output = addnorm_layer(test_tensor, test_tensor)
    print(addnorm_output)

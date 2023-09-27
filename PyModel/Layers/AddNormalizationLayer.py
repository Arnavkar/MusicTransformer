from tensorflow.keras.layers import Layer, LayerNormalization
import tensorflow as tf
from utils import check_shape

class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
    
    def call(self, x, sublayer_x):       
        return self.layer_norm(x + sublayer_x)
    
if __name__ == "__main__":
    test_tensor = tf.constant([[1,3,4,2,0]])
    addnorm_layer = AddNormalization()
    addnorm_output = addnorm_layer(test_tensor, test_tensor)
    print(addnorm_output)

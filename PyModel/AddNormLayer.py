from tensorflow.keras.layers import LayerNormalization

class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
    
def call(self, x, sublayer_x):
    add = x + sublayer_x
    return self.layer_norm(add)
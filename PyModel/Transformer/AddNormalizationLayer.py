from tensorflow.keras.layers import Layer, LayerNormalization
import tensorflow as tf
from .utils import check_shape
import json

@tf.keras.saving.register_keras_serializable()
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
    
    def call(self, x, sublayer_x):
        add = x + sublayer_x     
        return self.layer_norm(add)
    
    #Config to serialize Custom Layer
    #Explicit Desrialization not required because we are are not passing layers or models to __init__()
    def get_config(self):
        base_config = super(AddNormalization, self).get_config()
        config = {
            "layer_norm": tf.keras.saving.serialize_keras_object(self.layer_norm),
        }
        return {**base_config, **config}
    
if __name__ == "__main__":
    test_tensor = tf.constant([[1,3,4,2,0]])
    addnorm_layer = AddNormalization()
    addnorm_output = addnorm_layer(test_tensor, test_tensor)
    print(f"Addnorm output: {addnorm_output}")

    layer_config = addnorm_layer.get_config()
    print(f"Layer config: {json.dumps(layer_config,indent=4)}")

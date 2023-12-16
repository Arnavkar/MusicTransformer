from tensorflow.keras.layers import Layer, LayerNormalization, Add
import tensorflow as tf
from .utils import check_shape
import json

@tf.keras.saving.register_keras_serializable()
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
         # Layer normalization layer
        self.layer_norm = LayerNormalization() 
        # Add layer - the add layers ensure that keras masks are properly propagated
        self.add = Add()  
    
    def call(self, x, sublayer_x):
        #Skip connection
        output = self.add([x + sublayer_x])     
        return self.layer_norm(output)
    
    def get_config(self):
        base_config = super(AddNormalization, self).get_config()
        config = {
            "layer_norm": tf.keras.saving.serialize_keras_object(self.layer_norm),
        }
        return {**base_config, **config}
    
if __name__ == "__main__":
    test = [1,3,4,2,0]
    test_tensor = tf.constant([test])
    addnorm_layer = AddNormalization()
    addnorm_output = addnorm_layer(test_tensor, test_tensor)

    divide_value = sum(test)*2
    check = [x*2/divide_value for x in test]
    print(check)
    print(f"Addnorm output: {addnorm_output}")

    layer_config = addnorm_layer.get_config()
    print(f"Layer config: {json.dumps(layer_config,indent=4)}")

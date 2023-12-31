from tensorflow.keras.layers import Layer, Dense, ReLU
import tensorflow as tf
from .utils import check_shape
import json

@tf.keras.saving.register_keras_serializable()
class FeedForward(Layer):
    def __init__(self, d_in, d_out, **kwargs):
        super(FeedForward,self).__init__(**kwargs)
        self.d_in = d_in
        self.d_out = d_out

        # First fully connected layer takes in input dimensions
        self.dense1 = Dense(self.d_in) 
        # Second fully connected layer, based on model output dimensions 
        self.dense2 = Dense(self.d_out)  
        self.activation = ReLU()
    
    def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({
            'd_in': self.d_in,
            'd_out': self.d_out,
            'dense1': tf.keras.saving.serialize_keras_object(self.dense1),
            'dense2': tf.keras.saving.serialize_keras_object(self.dense2),
        })
        return config

    def call(self, x):
        #Send inputs through the first fully connected layer
        output = self.dense1(x)
        #Apply ReLU activation
        output = self.activation(output)
        #Send inputs through the second fully connected layer
        output = self.dense2(output)
        return output
    

if __name__ == "__main__":
    test_tensor = tf.constant([[1,3,4,2,0]])
    layer = FeedForward(5,3)
    output = layer(test_tensor)
    print(f"feedforward output: {output}")

    layer_config = layer.get_config()
    print(f"Layer config: {json.dumps(layer_config,indent=4)}")
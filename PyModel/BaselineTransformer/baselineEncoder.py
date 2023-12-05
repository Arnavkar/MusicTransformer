import keras
from Transformer.params import Params
from tensorflow.keras import layers
import tensorflow as tf

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = p.model_dim
        self.dense_dim = p.feed_forward_dim
        self.num_heads = p.num_heads

        self.attention = layers.MultiHeadAttention(
            num_heads=p.num_heads, key_dim=p.model_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(p.feed_forward_dim, activation="relu"),
                layers.Dense(p.model_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(p.dropout_rate)
        self.dropout2 = layers.Dropout(p.dropout_rate)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        attention_output = self.dropout1(attention_output)
        proj_input = self.layernorm_1(inputs + attention_output)

        proj_output = self.dense_proj(proj_input)
        proj_output = self.dropout2(proj_output)

        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.num_encoder_layers = p.num_encoder_layers
        self.enc_layers = [TransformerEncoderLayer(p) for _ in range(self.num_encoder_layers)]

    def call(self, inputs, mask=None):
        for i in range(self.num_encoder_layers):
            inputs = self.enc_layers[i](inputs, mask=mask)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_encoder_layers": self.num_encoder_layers,
            }
        )
        return config
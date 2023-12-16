import keras
from CustomTransformer.params import Params
from tensorflow.keras import layers
import tensorflow as tf

class TransformerDecoder(layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = p.model_dim
        self.feed_forward_dim = p.feed_forward_dim
        self.num_heads = p.num_heads

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.model_dim
        )

        self.attention_2 = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.model_dim
        )

        self.dense_proj = keras.Sequential(
            [
                layers.Dense(self.feed_forward_dim, activation="relu"),
                layers.Dense(self.model_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(p.dropout_rate)
        self.dropout2 = layers.Dropout(p.dropout_rate)
        self.dropout3 = layers.Dropout(p.dropout_rate)

        self.add = layers.Add()  # instead of `+` to preserve mask
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        attention_output_1 = self.attention_1(
            query=inputs, 
            value=inputs, 
            key=inputs, use_causal_mask=True
        )
        attention_output_1 = self.dropout1(attention_output_1)
        out_1 = self.layernorm_1(self.add([inputs, attention_output_1]))

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
        )
        attention_output_2 = self.dropout2(attention_output_2)
        out_2 = self.layernorm_2(self.add([out_1, attention_output_2]))

        proj_output = self.dense_proj(out_2)
        proj_output = self.dropout3(proj_output)

        return self.layernorm_3(self.add([out_2, proj_output]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "feed_forward_dim": self.feed_forward_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
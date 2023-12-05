import keras
from Transformer.params import Params
from tensorflow.keras import layers
import tensorflow as tf

class TransformerDecoderLayer(layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = p.model_dim
        self.dense_dim = p.feed_forward_dim
        self.num_heads = p.num_heads

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=p.num_heads, key_dim=p.model_dim
        )

        self.attention_2 = layers.MultiHeadAttention(
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
        self.layernorm_3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(p.dropout_rate)
        self.dropout2 = layers.Dropout(p.dropout_rate)
        self.dropout3 = layers.Dropout(p.dropout_rate)

        self.add = layers.Add()  # instead of `+` to preserve mask
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, use_causal_mask=True
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
                "embed_dim": self.model_dim,
                "latent_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
    
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.num_decoder_layers = p.num_decoder_layers
        self.dec_layers = [TransformerDecoderLayer(p) for _ in range(self.num_decoder_layers)]

    def call(self, decoder_inputs, encoded_seq_inputs, mask=None):
        for i in range(self.num_decoder_layers):
            decoder_inputs = self.dec_layers[i](decoder_inputs, encoded_seq_inputs, mask=mask)
        return decoder_inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_decoder_layes": self.num_decoder_layers
            }
        )
        return config
import keras_nlp
import keras
from Transformer.params import Params
from Transformer.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from tensorflow.keras import layers
import tensorflow as tf

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = p.model_dim
        self.dense_dim = p.feed_forward_dim
        self.num_heads = p.num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=p.num_heads, key_dim=p.model_dim
        )
        # self.attention = layers.MultiHeadAttentionLayer(p, isRelative=False)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(p.feed_forward_dim, activation="relu"),
                layers.Dense(p.model_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        # attention_output = self.attention([inputs, inputs, inputs])
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
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


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=sequence_length,
            embedding_dim=embed_dim,
            mask_zero=True,
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        return self.embedding_layer(inputs)

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, p:Params, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = p.model_dim
        self.dense_dim = p.feed_forward_dim
        self.num_heads = p.num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=p.num_heads, key_dim=p.model_dim
        )
        # self.attention_1 = layers.MultiHeadAttentionLayer(p, isRelative=False)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=p.num_heads, key_dim=p.model_dim
        )
        # self.attention_2 = layers.MultiHeadAttentionLayer(p, isRelative=False)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(p.feed_forward_dim, activation="relu"),
                layers.Dense(p.model_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.add = layers.Add()  # instead of `+` to preserve mask
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, use_causal_mask=True
        )
        out_1 = self.layernorm_1(self.add([inputs, attention_output_1]))

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
        )
        out_2 = self.layernorm_2(self.add([out_1, attention_output_2]))

        proj_output = self.dense_proj(out_2)
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
    
def createBaselineTransformer(p:Params):
    encoder_inputs = keras.Input(shape=(None,), dtype="uint16", name="encoder_inputs")
    print("encoder inputs shape: ", encoder_inputs.shape)
    x = PositionalEmbedding(p.encoder_seq_len, p.encoder_vocab_size, p.model_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(p)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="uint16", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, p.model_dim), name="decoder_state_inputs")

    x = PositionalEmbedding(p.decoder_seq_len, p.decoder_vocab_size, p.model_dim)(decoder_inputs)
    x = TransformerDecoder(p)(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(p.decoder_vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        inputs = (encoder_inputs, decoder_inputs), outputs = decoder_outputs, name="transformer"
    )
    
    return transformer
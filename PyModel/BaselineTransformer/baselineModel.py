import keras_nlp
import keras
from Transformer.params import Params, midi_test_params_v2
from .baselineEncoder import TransformerEncoder
from .baselineDecoder import TransformerDecoder
from tensorflow.keras import layers
import tensorflow as tf

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
    
def createBaselineTransformer(p:Params):
    encoder_inputs = keras.Input(shape=(None,), dtype="uint16", name="encoder_inputs")
    x = PositionalEmbedding(p.encoder_seq_len, p.encoder_vocab_size, p.model_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(p)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="uint16", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, p.model_dim), name="decoder_state_inputs")

    x = PositionalEmbedding(p.decoder_seq_len, p.decoder_vocab_size, p.model_dim)(decoder_inputs)
    x = TransformerDecoder(p)(x, encoded_seq_inputs)
    decoder_outputs = layers.Dense(p.decoder_vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        inputs = (encoder_inputs, decoder_inputs), outputs = decoder_outputs, name="transformer"
    )
    
    return transformer

if __name__ == "__main__":
    p = Params(midi_test_params_v2)
    model = createBaselineTransformer(p)
    model.summary()


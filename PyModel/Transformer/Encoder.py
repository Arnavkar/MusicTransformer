import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Input
from tensorflow.keras.models import Model
from .MultiHeadAttentionLayer import MultiHeadAttentionLayer
from .FeedForwardLayer import FeedForward
from .PositionalEncodingLayer import PositionEmbeddingFixedWeights
from .AddNormalizationLayer import AddNormalization
from .utils import check_shape
from .params import baseline_test_params, Params

class EncoderLayer(Layer):
    def __init__(self, p:Params, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.seq_len = p.encoder_seq_len
        self.model_dim = p.model_dim
        self.dropout_rate = p.dropout_rate
        self.feed_forward_dim = p.feed_forward_dim

        self.mha_layer = MultiHeadAttentionLayer(p, isRelative=False)
        self.add_norm1 = AddNormalization()
        self.dropout1 = Dropout(p.dropout_rate)
        self.add_norm2 = AddNormalization()
        self.dropout2 = Dropout(p.dropout_rate)
        self.feed_forward = FeedForward(p.feed_forward_dim, p.model_dim)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'feed_forward_dim': self.feed_forward_dim,
            'mha_layer': self.mha_layer.get_config(),
            'feed_forward': self.feed_forward.get_config(),
        })
        return config

    def call(self, x, padding_mask, training):
        attention_output = self.mha_layer([x, x, x], padding_mask)
        # check_shape("attention_output", attention_output, (x.shape[0], self.seq_len, self.model_dim))

        attention_output = self.dropout1(attention_output, training=training)
        #the input itself + the scaled attention values
        addnorm_output = self.add_norm1(x, attention_output)

        ff_output = self.feed_forward(addnorm_output)
        ff_output = self.dropout2(ff_output, training=training)
        #the previous addnorm output + values from the feedforward network
        final = self.add_norm2(addnorm_output, ff_output)
        return final
    
class Encoder(Layer):
    def __init__(self, p:Params, **kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.encoder_seq_len = p.encoder_seq_len
        self.model_dim = p.model_dim
        self.dropout_rate = p.dropout_rate
        self.encoder_vocab_size = p.encoder_vocab_size
        self.num_encoder_layers = p.num_encoder_layers

        self.positional_encoding = PositionEmbeddingFixedWeights(self.encoder_seq_len, self.encoder_vocab_size, self.model_dim)
        self.dropout = Dropout(self.dropout_rate)
        self.encoder_layers = [EncoderLayer(p) for _ in range(self.num_encoder_layers)]

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'encoder_seq_len': self.encoder_seq_len,
            'encoder_vocab_size': self.encoder_vocab_size,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'num_encoder_layers': self.num_encoder_layers,
            # 'positional_encoding' is also a layer and needs to handle its config
            'positional_encoding': self.positional_encoding.get_config(),
            'encoder_layers': [layer.get_config() for layer in self.encoder_layers.layers]
        })
    
    def call(self, x, padding_mask, training):
        positional_encoding_output = self.positional_encoding(x)
        positional_encoding_output = self.dropout(positional_encoding_output, training=training)

        for layer in self.encoder_layers:
            output = layer(
                positional_encoding_output,
                padding_mask=padding_mask,
                training = training 
            )
        return output

if __name__ == "__main__":
    p = Params(baseline_test_params)

    #encoder_layer = EncoderLayer(input_seq_len, num_heads, embedding_dim, model_dim, feed_forward_dim, dropout_rate)
    #encoder_layer.build_graph().summary()

    input_seq = tf.random.uniform((p.batch_size, p.seq_len))
    encoder_layer_input_seq = tf.random.uniform((p.batch_size, p.seq_len,p.model_dim))

    encoderLayer = EncoderLayer(p)
    output = encoderLayer(encoder_layer_input_seq, None, True)
    print(f'Encoder Layer output: {output}')
    #encoderLayer.summary()

    encoder = Encoder(p)
    output = encoder(input_seq, None, True)
    print(f'Encoder output shape: {output}')
    # encoder.summary()

    
    #None is the mask, True is the flag for training which only applies droput when flag value is set to true
    
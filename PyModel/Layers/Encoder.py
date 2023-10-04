import sys
sys.path.append('./PyModel')

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Input
from tensorflow.keras.models import Model
from Layers.MultiHeadAttentionLayer import MultiHeadAttention
from Layers.FeedForwardLayer import FeedForward
from Layers.PositionalEncodingLayer import PositionEmbeddingFixedWeights
from Layers.AddNormalizationLayer import AddNormalization
from Layers.utils import check_shape
from params import baseline_test_params, Params

class EncoderLayer(Model):
    def __init__(self, p:Params, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        #for building graph
        #self.build(input_shape=[None, p.seq_len, p.model_dim])
        self.seq_len = p.encoder_seq_len

        self.model_dim = p.model_dim
        self.mha_layer = MultiHeadAttention(p, isRelative=False)

        self.dropout1 = Dropout(p.dropout_rate)
        #Not sure about this class yet
        self.add_norm1 = AddNormalization()

        #Not sure about these dimensions yet
        self.feed_forward = FeedForward(p.feed_forward_dim, p.model_dim)

        self.dropout2 = Dropout(p.dropout_rate)
        self.add_norm2 = AddNormalization()

    # def build_graph(self):
    #     input_layer = Input(shape=(self.seq_len, self.model_dim))
    #     return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

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
    
class Encoder(Model):
    def __init__(self, p:Params, **kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.positional_encoding = PositionEmbeddingFixedWeights(p.encoder_seq_len, p.encoder_vocab_size, p.model_dim)
        self.dropout = Dropout(p.dropout_rate)
        self.encoder_layers = [EncoderLayer(p) for _ in range(p.num_encoder_layers)]
    
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

    encoder = Encoder(p)
    encoder(input_seq, None, True)
    encoder.summary()

    encoderLayer = EncoderLayer(p)
    encoderLayer(encoder_layer_input_seq, None, True)
    encoderLayer.summary()
    #None is the mask, True is the flag for training which only applies droput when flag value is set to true
    
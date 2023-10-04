import sys
# appending Layer path
sys.path.append('./PyModel')

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Input
from tensorflow.keras.models import Model
from Layers.MultiHeadAttentionLayer import MultiHeadAttention
from Layers.FeedForwardLayer import FeedForward
from Layers.AddNormalizationLayer import AddNormalization
from Layers.PositionalEncodingLayer import PositionEmbeddingFixedWeights
from Layers.utils import check_shape
from params import baseline_test_params, Params

class DecoderLayer(Layer):
    def __init__(self, p:Params, **kwargs):
        super(DecoderLayer,self).__init__(**kwargs)
        #to build graph
        self.build(input_shape=[None, p.seq_len, p.model_dim])
        self.seq_len = p.seq_len
        self.model_dim = p.model_dim

        self.masked_mha_layer = MultiHeadAttention(p,isRelative=False)
        self.dropout1 = Dropout(p.dropout_rate)
        self.add_norm1 = AddNormalization()

        self.mha_layer = MultiHeadAttention(p,isRelative=False)
        self.dropout2 = Dropout(p.dropout_rate)
        self.add_norm2 = AddNormalization()
        
       #Not sure about these dimensions yet
        self.feed_forward = FeedForward(p.feed_forward_dim, p.model_dim)

        self.dropout3 = Dropout(p.dropout_rate)
        self.add_norm3 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.seq_len, self.model_dim))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, None, None, True))

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        attention_output1 = self.masked_mha_layer([x, x, x], lookahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        addnorm_output1 = self.add_norm1(x, attention_output1)

        attention_output2 = self.mha_layer([addnorm_output1,encoder_output,encoder_output],padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        addnorm_output2 = self.add_norm2(addnorm_output1, attention_output2)

        ff_output = self.feed_forward(addnorm_output2)
        ff_output = self.dropout3(ff_output, training=training)
        final = self.add_norm3(addnorm_output2, ff_output)
        return final
    
class Decoder(Layer):
    def __init__(self,p:Params, **kwargs):
        super().__init__(**kwargs)
        self.positional_encoding = PositionEmbeddingFixedWeights(p.seq_len, p.decoder_vocab_size, p.model_dim)
        self.dropout = Dropout(p.dropout_rate)
        self.decoder_layers = [
            DecoderLayer(p) for _ in range(p.num_decoder_layers)]

    def call(self, input, encoder_output, lookahead_mask, padding_mask, training):
        positional_encoding_output = self.positional_encoding(input)
        positional_encoding_output = self.dropout(positional_encoding_output, training=training)

        for layer in self.decoder_layers:
            output = layer(positional_encoding_output, encoder_output, lookahead_mask, padding_mask, training)
        
        return output

if __name__ == "__main__":
    p = Params(baseline_test_params)
    input_seq = tf.random.uniform((p.batch_size, p.seq_len))
    enc_output = tf.random.uniform((p.batch_size, p.seq_len,p.model_dim))

    decoder = Decoder(p)
    decoderLayer = DecoderLayer(p)
    decoderLayer.build_graph().summary()
    print(decoder(input_seq, enc_output, None, True))
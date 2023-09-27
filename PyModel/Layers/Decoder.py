import sys
# appending Layer path
sys.path.append('Layers')

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Input
from tensorflow.keras.models import Model
from Layers.MultiHeadAttentionLayer import MultiHeadAttention
from Layers.FeedForwardLayer import FeedForward
from Layers.AddNormalizationLayer import AddNormalization
from Layers.PositionalEncodingLayer import PositionEmbeddingFixedWeights
from Layers.utils import check_shape

class DecoderLayer(Layer):
    def __init__(self, seq_len, num_heads, embedding_dim, model_dim, feed_forward_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        #to build graph
        self.build(input_shape=[None, seq_len, model_dim])

        self.model_dim = model_dim
        self.masked_mha_layer = MultiHeadAttention(num_heads,embedding_dim,  model_dim)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = AddNormalization()

        self.mha_layer = MultiHeadAttention(num_heads,embedding_dim, model_dim)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddNormalization()
        
       #Not sure about these dimensions yet
        self.feed_forward = FeedForward(feed_forward_dim, model_dim)

        self.dropout3 = Dropout(dropout_rate)
        self.add_norm3 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training, **kwargs):
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
    def __init__(self, vocab_size, seq_len, num_heads, embedding_dim, model_dim, feed_forward_dim, dropout_rate, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.positional_encoding = PositionEmbeddingFixedWeights(seq_len, vocab_size, model_dim)
        self.dropout = Dropout(dropout_rate)
        self.decoder_layers = [
            DecoderLayer(
                seq_len,
                num_heads,
                embedding_dim, 
                model_dim, 
                feed_forward_dim, 
                dropout_rate) for _ in range(num_layers)]

    def call(self, input, encoder_output, lookahead_mask, padding_mask, training):
        positional_encoding_output = self.positional_encoding(input)
        positional_encoding_output = self.dropout(positional_encoding_output, training=training)

        for layer in self.decoder_layers:
            output = layer(positional_encoding_output, encoder_output, lookahead_mask, padding_mask, training)
        
        return output

if __name__ == "__main__":
    num_heads = 8
    embedding_dim = 64 #number of dimensions for each token in the sequence 
    model_dim = 512
    feed_forward_dim = 2048
    num_decoder_layers = 6

    batch_size = 64 #number of sequences in a batch
    dropout_rate = 0.1

    dec_vocab_size = 20 
    input_seq_len = 5

    input_seq = tf.random.uniform((batch_size, input_seq_len))
    enc_output = tf.random.uniform((batch_size, input_seq_len,model_dim))

    decoder = Decoder(
        vocab_size = dec_vocab_size,
        seq_len = input_seq_len,
        num_heads = num_heads,
        embedding_dim = embedding_dim,
        model_dim = model_dim,
        feed_forward_dim = feed_forward_dim,
        num_layers = num_decoder_layers,
        dropout_rate = dropout_rate
    )
    print(decoder(input_seq, enc_output, None, True))
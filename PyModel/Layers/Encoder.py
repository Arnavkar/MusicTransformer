import sys
# appending Layer path
sys.path.append('Layers')

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Input
from tensorflow.keras.models import Model
from Layers.MultiHeadAttentionLayer import MultiHeadAttention
from Layers.FeedForwardLayer import FeedForward
from Layers.PositionalEncodingLayer import PositionEmbeddingFixedWeights
from Layers.AddNormalizationLayer import AddNormalization
from Layers.utils import check_shape

class EncoderLayer(Layer):
    def __init__(self, seq_len, num_heads, embedding_dim, model_dim, feed_forward_dim, dropout_rate,  **kwargs):
        super().__init__(**kwargs)
        #for building graph
        self.build(input_shape=[None, seq_len, model_dim])

        self.model_dim = model_dim
        self.mha_layer = MultiHeadAttention(num_heads, embedding_dim, model_dim, isRelative=False)

        self.dropout1 = Dropout(dropout_rate)
        #Not sure about this class yet
        self.add_norm1 = AddNormalization()
        
        #Not sure about these dimensions yet
        self.feed_forward = FeedForward(feed_forward_dim, model_dim)

        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, training, mask=None, **kwargs):
        attention_output = self.mha_layer([x, x, x], mask)
        attention_output = self.dropout1(attention_output, training=training)
        #the input itself + the scaled attention values
        addnorm_output = self.add_norm1(x, attention_output)

        ff_output = self.feed_forward(addnorm_output)
        ff_output = self.dropout2(ff_output, training=training)
        #the previous addnorm output + values from the feedforward network
        final = self.add_norm2(addnorm_output, ff_output)
        return final
    
class Encoder(Layer):
    def __init__(self, vocab_size, seq_len, num_heads, embedding_dim, model_dim, feed_forward_dim, dropout_rate, num_layers,   **kwargs):
        super().__init__(**kwargs)
        self.positional_encoding = PositionEmbeddingFixedWeights(seq_len, vocab_size, model_dim)
        self.dropout = Dropout(dropout_rate)
        self.encoder_layers = [
            EncoderLayer(
                seq_len,
                num_heads,
                embedding_dim,
                model_dim,
                feed_forward_dim,
                dropout_rate) for _ in range(num_layers)]
    
    def call(self, input, padding_mask, training):
        positional_encoding_output = self.positional_encoding(input)
        # Expected output shape = (batch_size, sequence_length, d_model)

        positional_encoding_output = self.dropout(positional_encoding_output, training=training)

        for layer in self.encoder_layers:
            output = layer(positional_encoding_output, training, padding_mask)
        
        return output

if __name__ == "__main__":
    num_heads = 8
    embedding_dim = 64 #number of dimensions for each token in the sequence 
    model_dim = 512
    feed_forward_dim = 2048
    num_encoder_layers = 6

    batch_size = 64 #number of sequences in a batch
    dropout_rate = 0.1

    enc_vocab_size = 20 
    input_seq_len = 5

    test_tensor = tf.random.uniform((batch_size, input_seq_len))

    encoder = Encoder(
        vocab_size = enc_vocab_size,
        seq_len = input_seq_len,
        num_heads = num_heads,
        embedding_dim = embedding_dim,
        model_dim = model_dim,
        feed_forward_dim = feed_forward_dim,
        num_encoder_layers= num_encoder_layers,
        dropout_rate= dropout_rate
    )
    #None is the mask, True is the flag for training which only applies droput when flag value is set to true
    print(encoder(test_tensor, None, True))
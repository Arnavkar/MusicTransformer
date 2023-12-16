from tensorflow.keras.layers import Layer, Embedding
import numpy as np
import tensorflow as tf
from .utils import check_shape
from matplotlib import pyplot as plt

class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_len, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        self.model_dim = output_dim
        self.voacb_size = vocab_size
        self.seq_len = seq_len
        
        # Initialize the positional encoding matrices
        self.position_embedding_matrix = self.get_positional_encoding(seq_len, output_dim)

        # Input embedding layer
        self.input_embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
        )
    
    def get_config(self):
        config = super(PositionEmbeddingFixedWeights, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'vocab_size': self.vocab_size,
            'output_dim': self.output_dim,
            'input_embedding_layer': tf.keras.saving.serialize_keras_object(self.input_embedding_layer),
            'position_embedding_layer': tf.keras.saving.serialize_keras_object(self.position_embedding_layer),
        })
        return config

    #Based on "attention is all you need"
    #Given a input sequence of size L and model dimension, initialize the positional encoding matrix

    # P(k,2i) = sin(k/n^(2i/d)) , P(k,2i+1) = cos(k/n^(2i/d)) 
    # k = position, where k < L/2 (since we alternate between sin and cos)
    # d = dimension of the output embedding space
    # n user defined scalar, 10000 in the paper
    # i = dimension index, where i < d/2
    @classmethod
    def get_positional_encoding(self, size, output_dim, n=10000):
        P = np.zeros((size, output_dim))
        for pos in range(size):
            for i in range(output_dim // 2):
                denominator = np.power(n, (2 * i) / output_dim)
                P[pos, 2 * i] = np.sin(pos / denominator)
                P[pos, 2 * i + 1] = np.cos(pos / denominator)

        # Ensure the data type matches TensorFlow expectations
        P = P.astype(np.float32)  
        return P

    def call(self, inputs):
        # Generate a sequence of position indices
        position_indices = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        # Embed the input tokens and the positions
        embedded_input = self.input_embedding_layer(inputs)
        #Scale the input embedding by sqrt(model_dim)
        embedded_input *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        # Sum the token embeddings and position embeddings
        return embedded_input + self.position_embedding_matrix
    
if __name__ == "__main__":
    #Plot the positional encoding
    pos_encoding = PositionEmbeddingFixedWeights.get_positional_encoding(2048, 512)
    plt.pcolormesh(pos_encoding.T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
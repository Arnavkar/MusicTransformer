from tensorflow.keras.layers import Layer, Embedding
import numpy as np
import tensorflow as tf
from .utils import check_shape

#Positional encoding as specified in the paper "Attention is all you need"
#Embedding layer is used to convert integer values into vectors
#In this case we use pre-defined weights
class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_len, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        input_embedding_matrix = self.get_positional_encoding(vocab_size, output_dim)   
        position_embedding_matrix = self.get_positional_encoding(seq_len, output_dim)    

        #trainiable must be false since we are using pre-defined weights
        self.input_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[input_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_len, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    #TODO: How did the embedding layer for the output get created? 
    
    #Based on "attention is all you need"
    #Given a input sequence of size L
    # P(k,2i) = sin(k/n^(2i/d)) , P(k,2i+1) = cos(k/n^(2i/d)) 
    # k = position, where k < L/2 (since we alternate between sin and cos)
    # d = dimension of the output embedding space
    # n user defined scalar, 10000 in the paper
    # i = dimension index, where i < d/2
    #from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    def get_positional_encoding(self, seq_len, d, n=10000):
        #create the positional matrix
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_input = self.input_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_input + embedded_indices
    
if __name__ == "__main__":
    #test from tutorial
    output_sequence_length = 5
    vocab_size = 10 
    output_length = 6
    test_tensor = tf.constant([[5,6,7,2,0],[3,4,2,0,0]])
    embedding_layer = PositionEmbeddingFixedWeights(output_sequence_length,
                                            vocab_size, output_length)
    positional_encoding_output = embedding_layer(test_tensor)
    #In this case, 2,5,6 -> 2 sequences, 5 integers each, where each integer is now spread into a vector of 6 elements
    check_shape("positional encoding",
                positional_encoding_output,
                (test_tensor.shape[0],test_tensor.shape[1],output_length))
    print(f"positional encoding output: {positional_encoding_output}")
    
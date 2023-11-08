from tensorflow.keras.layers import Layer, Embedding
import numpy as np
import tensorflow as tf
from .utils import check_shape

# #Positional encoding as specified in the paper "Attention is all you need"
# #Embedding layer is used to convert integer values into vectors
# #In this case we use pre-defined weights
# class PositionEmbeddingFixedWeights(Layer):
#     def __init__(self, seq_len, vocab_size, output_dim, **kwargs):
#         super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
#         input_embedding_matrix = self.get_positional_encoding(vocab_size, output_dim)  
#         #print(f"input embedding matrix shape: {input_embedding_matrix.shape}") 
#         position_embedding_matrix = self.get_positional_encoding(seq_len, output_dim)    
#         #print(f"input embedding matrix shape: {input_embedding_matrix.shape}") 

#         #trainiable must be false since we are using pre-defined weights
#         self.input_embedding_layer = Embedding(
#             input_dim=vocab_size, output_dim=output_dim,
#             weights=[input_embedding_matrix],
#             trainable=False
#         )
#         self.position_embedding_layer = Embedding(
#             input_dim=seq_len, output_dim=output_dim,
#             weights=[position_embedding_matrix],
#             trainable=False
#         )

#     #TODO: How did the embedding layer for the output get created? 
    
#     
#     def get_positional_encoding(self, seq_len, d, n=10000):
#         #create the positional matrix
#         P = np.zeros((seq_len, d))
#         for k in range(seq_len):
#             for i in np.arange(int(d/2)):
#                 denominator = np.power(n, 2*i/d)
#                 P[k, 2*i] = np.sin(k/denominator)
#                 P[k, 2*i+1] = np.cos(k/denominator)
#         return P

#     def call(self, inputs):        
#         position_indices = tf.range(tf.shape(inputs)[-1])
#         #print(f"position indices shape: {position_indices.shape}")
#         embedded_input = self.input_embedding_layer(inputs)
#         #print(f"embedded input shape: {embedded_input.shape}")
#         embedded_indices = self.position_embedding_layer(position_indices)
#         #print(f"embedded indices shape: {embedded_indices.shape}")
#         return embedded_input + embedded_indices
    
# if __name__ == "__main__":
#     #test from tutorial
#     output_sequence_length = 5
#     vocab_size = 10 
#     output_length = 6
#     test_tensor = tf.constant([[5,6,7,2,0],[3,4,2,0,0]])
#     embedding_layer = PositionEmbeddingFixedWeights(output_sequence_length,
#                                             vocab_size, output_length)
#     positional_encoding_output = embedding_layer(test_tensor)
#     #In this case, 2,5,6 -> 2 sequences, 5 integers each, where each integer is now spread into a vector of 6 elements
#     check_shape("positional encoding",
#                 positional_encoding_output,
#                 (test_tensor.shape[0],test_tensor.shape[1],output_length))
#     print(f"positional encoding output: {positional_encoding_output}")
#     print(f"positional encoding shape: {positional_encoding_output.shape}")

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Layer, Embedding

class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_len, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        # Initialize the positional encoding matrices
        self.input_embedding_matrix = self.get_positional_encoding(vocab_size, output_dim)
        self.position_embedding_matrix = self.get_positional_encoding(seq_len, output_dim)

        # Non-trainable embeddings
        self.input_embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
            weights=[self.input_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_len,
            output_dim=output_dim,
            weights=[self.position_embedding_matrix],
            trainable=False
        )

    #Based on "attention is all you need"
#     #Given a input sequence of size L
#     # P(k,2i) = sin(k/n^(2i/d)) , P(k,2i+1) = cos(k/n^(2i/d)) 
#     # k = position, where k < L/2 (since we alternate between sin and cos)
#     # d = dimension of the output embedding space
#     # n user defined scalar, 10000 in the paper
#     # i = dimension index, where i < d/2
#     #from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    def get_positional_encoding(self, size, output_dim, n=10000):
        # Initialize the positional encoding matrix
        P = np.zeros((size, output_dim))
        for pos in range(size):
            for i in range(output_dim // 2):
                denominator = np.power(n, (2 * i) / output_dim)
                P[pos, 2 * i] = np.sin(pos / denominator)
                P[pos, 2 * i + 1] = np.cos(pos / denominator)

        P = P.astype(np.float32)  # Ensure the data type matches TensorFlow expectations
        return P

    def call(self, inputs):
        # Generate a sequence of position indices
        position_indices = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        # Embed the input tokens and the positions
        embedded_input = self.input_embedding_layer(inputs)
        embedded_positions = self.position_embedding_layer(position_indices)
        
        # Sum the token embeddings and position embeddings
        return embedded_input + embedded_positions
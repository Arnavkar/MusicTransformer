import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

#test params
num_heads = 8
embedding_dim = 16 #number of dimensions for each token in the sequence
model_dim = 512 
batch_size = 16 #number of sequences in a batch
seq_len = 5 #number of tokens in an input sequence

class MultiHeadAttention(Layer):
    def __init__(self, num_heads, embedding_dim, model_dim, isRelative=False, **kwargs):
        super(MultiHeadAttention).__init__(**kwargs)
        if isRelative: raise NotImplementedError("Relative attention not yet implemented")
        self.num_heads = num_heads
        self.d_embed = embedding_dim
        self.d_model = model_dim
        #Data input must be divisible by the number of heads
        assert model_dim % self.num_heads == 0

        self.W_query = Dense(self.d_embed)
        self.W_key = Dense(self.d_embed)
        self.W_value = Dense(self.d_embed)
        self.W_out = Dense(self.d_model)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        #Q, K, V all have shape [batch_size, num_heads, seq_len, dim_per_head]
        
        #First multiple queries by keys to get similarity scores and normalize
        #TODO: ASK SVEN , HOW TO DETERMINE SHAPE OF MULTI DIMENSIONAL MATRIX MULTIPLICATION
        attention_weights = tf.matmul(q, v, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_embed, tf.float32))

        #Mask if required (Eg. decoder layer), prevent attention from future outputs
        #Essentially multiply by an extremely small negative number to remove future values from softmax calculation
        if mask is not None: attention_weights += -1e9 * mask

        #Use softmax to get attention weights in terms of probability distribution
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        #Multiply by values to get context vector
        context_vector = tf.matmul(attention_weights, v)
            
        return context_vector
    
    def reshape_tensor(self,tensor):
        #in the shape batch_size, num_heads, seq_len, –1, which essentially flattens the last dimension
        '''
        Eg. for a single input query of size 5(seq len),16(query dim) > 80 elements
        Therefore if 8 heads, each head will have 8/8 = 10 elements
        10 elements >  2 sequences of 5 elements, per batch
        '''
        tensor = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], self.num_heads, -1))
        #self.check_shape("reshaped_tensor",tensor,(key_query_dim,seq_len,num_heads,int(batch_size/num_heads)))
        tensor = tf.transpose(tensor, perm=[0,2,1,3])
        #self.check_shape("transposed_tensor",tensor,(key_query_dim,num_heads,seq_len,int(batch_size/num_heads)))
        
        return tensor
    
    def concat_heads(self,tensor):
        tensor = tf.transpose(tensor, perm=[0,2,1,3])
        tensor = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], -1))
        return tensor
    
    def call(self, inputs, mask=None, **kwargs):
        '''
        input: a list of tensors, representing [queries, keys, values]
        mask: for masked multi head attention in decoder

        '''
        q,k,v = inputs[0], inputs[1], inputs[2]

        self.check_shape("base_query",q,(batch_size,seq_len,embedding_dim))
        self.check_shape("base_key",k,(batch_size,seq_len,embedding_dim))
        self.check_shape("base_value",v,(batch_size,seq_len,embedding_dim))
        #First pass through linear layers
        q,k,v = self.W_query(q), self.W_key(k), self.W_value(v)

        #Reshape to [batch_size, num_heads, seq_len, dim_per_head] for dot product attention
        q,k,v = self.reshape_tensor(q), self.reshape_tensor(k), self.reshape_tensor(v)
        self.check_shape("reshaped_query",q,(embedding_dim,num_heads,seq_len,int(batch_size/num_heads)))
        self.check_shape("reshaped_query",k,(embedding_dim,num_heads,seq_len,int(batch_size/num_heads)))
        self.check_shape("reshaped_query",v,(embedding_dim,num_heads,seq_len,int(batch_size/num_heads)))

        #computer scaled dot product attention
        attention = self.scaled_dot_product_attention(q, k, v, mask)
        concat_attention = self.concat_heads(attention)
        self.check_shape("attention",concat_attention,(batch_size,seq_len,embedding_dim))

        #pass through final linear layer
        output = self.W_out(concat_attention)
        self.check_shape("output",output,(batch_size,seq_len,model_dim))
        return output

    @staticmethod
    def check_shape(name,tensor,expectedshape):
        assert tensor.shape == expectedshape, f" expected shape {expectedshape}, {name} shape: {tensor.shape}"

#test the layer
queries = tf.random.uniform((batch_size, seq_len, embedding_dim))
keys = tf.random.uniform((batch_size, seq_len, embedding_dim))
values = tf.random.uniform((batch_size, seq_len, embedding_dim))

attention_layer = MultiHeadAttention(num_heads, embedding_dim, model_dim)
attention_layer([queries,keys,values])
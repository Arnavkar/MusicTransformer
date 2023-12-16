import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from .utils import check_shape
from .params import Params
import json

@tf.keras.saving.register_keras_serializable()
class MultiHeadAttentionLayer(Layer):
    def __init__(self, p:Params, isRelative=False, **kwargs):
        super(MultiHeadAttentionLayer,self).__init__(**kwargs)

        if isRelative: 
            raise NotImplementedError("Relative attention not yet implemented")
        
        self.num_heads = p.num_heads
        self.key_dim = p.key_dim
        self.value_dim = p.value_dim
        self.model_dim = p.model_dim

        #Model dimensionality must be divisible by the number of heads
        assert self.model_dim % self.num_heads == 0

        #Initialize linear layers for projecting queries, keys, values, and output
        self.W_query = Dense(self.key_dim)
        self.W_key = Dense(self.key_dim)
        self.W_value = Dense(self.value_dim)
        self.W_out = Dense(self.model_dim)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        #Q, K, V all have shape [batch_size, num_heads, seq_len, dim_per_head]
        
        #First multiply queries by keys to get similarity scores and normalize
        attention_weights = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        #Mask if required (Eg. decoder layer), prevent attention from future outputs
        #Essentially multiply by an extremely small negative number to remove future values from softmax calculation
        if mask is not None: attention_weights += -1e9 * mask

        #Use softmax to get attention weights in terms of probability distribution
        attention_weights = tf.nn.softmax(attention_weights)

        #Multiply by values to get context vector
        context_vector = tf.matmul(attention_weights, v)
            
        return context_vector
    
    def reshape_tensor(self,tensor):
        '''
        Eg. for a single input query of size 5(seq len),16(query dim) > 80 elements
        Therefore if 8 heads, each head will have 8/8 = 10 elements
        10 elements >  2 sequences of 5 elements, per batch
        '''
        tensor = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], self.num_heads, -1))
        check_shape("reshaped_tensor",tensor,(p.embedding_dim,p.seq_len,p.num_heads,int(p.batch_size/p.num_heads)))

        tensor = tf.transpose(tensor, perm=[0,2,1,3])
        check_shape("transposed_tensor",tensor,(p.embedding_dim,p.num_heads,p.seq_len,int(p.batch_size/p.num_heads)))
        
        return tensor
    
    def concat_heads(self,tensor):
        tensor = tf.transpose(tensor, perm=[0,2,1,3])
        tensor = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], self.key_dim))
        return tensor
    
    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'model_dim': self.model_dim,
            'W_query': tf.keras.saving.serialize_keras_object(self.W_query),
            'W_key': tf.keras.saving.serialize_keras_object(self.W_key),
            'W_value': tf.keras.saving.serialize_keras_object(self.W_value),
            'W_out': tf.keras.saving.serialize_keras_object(self.W_out),
        })
        return config
    
    def call(self, inputs, mask=None, **kwargs):
        '''
        input: a list of tensors, representing [queries, keys, values]
        mask: for masked multi head attention in decoder

        '''
        q,k,v = inputs[0], inputs[1], inputs[2]
        # for tensor in [q,k,v]: 
        #     check_shape("input_tensor",tensor,(p.batch_size,p.seq_len,p.embedding_dim))

        #First project through linear layers
        q,k,v = self.W_query(q), self.W_key(k), self.W_value(v)

        #Reshape to [batch_size, num_heads, seq_len, dim_per_head] for dot product attention
        q,k,v = self.reshape_tensor(q), self.reshape_tensor(k), self.reshape_tensor(v)
        # for tensor in [q,k,v]: 
        #     check_shape("reshaped_query", tensor,(p.embedding_dim,p.num_heads,p.seq_len,int(p.batch_size/p.num_heads)))

        #compute scaled dot product attention for each head
        attention = self.scaled_dot_product_attention(q, k, v, mask)

        #concat attention representations across each head
        concat_attention = self.concat_heads(attention)
        # check_shape("attention",concat_attention,(p.batch_size,p.seq_len,p.embedding_dim))

        #pass through final linear layer
        output = self.W_out(concat_attention)
        # check_shape("output",output,(p.batch_size,p.seq_len,p.model_dim))

        return output

if __name__ == "__main__":
    #test the layer
    p = Params(baseline_test_params)
    queries = tf.random.uniform((p.batch_size, p.encoder_seq_len, p.model_dim))
    keys = tf.random.uniform((p.batch_size, p.encoder_seq_len, p.model_dim))
    values = tf.random.uniform((p.batch_size, p.encoder_seq_len, p.model_dim))

    attention_layer = MultiHeadAttentionLayer(p)
    #Expect shape to be 64, 5, 512
    output = attention_layer([queries,keys,values])
    check_shape("test",output,(p.batch_size,p.encoder_seq_len,p.model_dim))
    print(f'Attention Layer output: {output}')

    layer_config = attention_layer.get_config()
    print(f"Layer config: {json.dumps(layer_config,indent=4)}")
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense,Dropout
from MultiHeadAttention import MultiHeadAttention
from AddNormLayer import AddNormalization

class DecoderLayer(Layer):
    def __init__(self, model_dim, dropout_rate, num_heads, additional, max_seq, **kwargs):
        super(DecoderLayer).__init__(**kwargs)

        self.model_dim = model_dim
        self.masked_mha_layer = MultiHeadAttention(num_heads, model_dim,isRelative=False)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = AddNormalization()

        self.mha_layer = MultiHeadAttention(num_heads, model_dim,isRelative=False)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddNormalization()
        
        #Not sure about these dimensions
        self.feed_forward1 = Dense(self.model_dim, activation='relu')
        self.feed_forward2 = Dense(self.model_dim)
        self.dropout3 = Dropout(dropout_rate)
        self.add_norm3 = AddNormalization()

    def call(self, x, training, encoder_output, lookahead_mask=None,padding=None **kwargs):
        output = self.masked_mha_layer([x, x, x], lookahead_mask)
        output = self.dropout1(output, training=training)
        output = self.add_norm1(x, output)

        output = self.mha_layer = MultiHeadAttention([output,encoder_output,encoder_output],isRelative=False)
        output = self.dropout2(output, training=training)
        output = self.add_norm2(x, output)

        output = self.feed_forward1(output)
        output = self.feed_forward2(output)
        output = self.dropout3(output, training=training)
        final = self.add_norm3(x, output)
        
        return final
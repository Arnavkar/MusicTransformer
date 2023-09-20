import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from MultiHeadAttention import MultiHeadAttention
from AddNormLayer import AddNormalization

class EncoderLayer(Layer):
    def __init__(self, model_dim, dropout_rate, num_heads, additional, max_seq, **kwargs):
        super(EncoderLayer).__init__(**kwargs)

        self.model_dim = model_dim
        self.mha_layer = MultiHeadAttention(num_heads, model_dim,isRelative=False)
        
        #Not sure about these dimensions
        self.feed_forward1 = Dense(self.model_dim, activation='relu')
        self.feed_forward2 = Dense(self.model_dim)

        #Not sure about this class either
        self.add_norm1 = AddNormalization()
        self.add_norm2 = AddNormalization()

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask=None, **kwargs):
        output = self.mha_layer([x, x, x], mask)
        output = self.dropout1(output, training=training)
        output = self.add_norm1(x, output)
        output = self.feed_forward1(output)
        output = self.feed_forward2(output)
        output = self.dropout2(output, training=training)
        final = self.add_norm2(x, output)
        return final




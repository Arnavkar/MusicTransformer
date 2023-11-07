import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from .MultiHeadAttentionLayer import MultiHeadAttentionLayer
from .FeedForwardLayer import FeedForward
from .AddNormalizationLayer import AddNormalization
from .PositionalEncodingLayer import PositionEmbeddingFixedWeights
from .utils import check_shape
from .params import baseline_test_params, Params

class DecoderLayer(Layer):
    def __init__(self, p:Params, **kwargs):
        super(DecoderLayer,self).__init__(**kwargs)
        #to build graph
        # self.build(input_shape=[None, p.seq_len, p.model_dim])
        self.seq_len = p.decoder_seq_len
        self.model_dim = p.model_dim

        self.masked_mha_layer = MultiHeadAttentionLayer(p,isRelative=False)
        self.dropout1 = Dropout(p.dropout_rate)
        self.add_norm1 = AddNormalization()

        self.mha_layer = MultiHeadAttentionLayer(p,isRelative=False)
        self.dropout2 = Dropout(p.dropout_rate)
        self.add_norm2 = AddNormalization()
        
       #Not sure about these dimensions yet
        self.feed_forward = FeedForward(p.feed_forward_dim, p.model_dim)

        self.dropout3 = Dropout(p.dropout_rate)
        self.add_norm3 = AddNormalization()

    # def build_graph(self):
    #     input_layer = Input(shape=(self.seq_len, self.model_dim))
    #     return Model(inputs=[input_layer], outputs=self.call(input_layer, None, None, None, True))

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
        self.positional_encoding = PositionEmbeddingFixedWeights(p.decoder_seq_len, p.decoder_vocab_size, p.model_dim)
        self.dropout = Dropout(p.dropout_rate)
        self.decoder_layers = [
            DecoderLayer(p) for _ in range(p.num_decoder_layers)]

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        positional_encoding_output = self.positional_encoding(x)
        positional_encoding_output = self.dropout(positional_encoding_output, training=training)

        for layer in self.decoder_layers:
            output = layer(         
                positional_encoding_output,
                encoder_output=encoder_output,
                lookahead_mask=lookahead_mask,
                padding_mask=padding_mask,
                training = training 
            )        
        return output

if __name__ == "__main__":
    p = Params(baseline_test_params)
    input_seq = tf.random.uniform((p.batch_size, p.seq_len))
    decoder_layer_input_seq = tf.random.uniform((p.batch_size, p.seq_len,p.model_dim))
    enc_output = tf.random.uniform((p.batch_size, p.seq_len,p.model_dim))

    decoder = Decoder(p)
    output = decoder(input_seq, enc_output, None, None, True)
    print(f'Decoder output: {output}')
    #decoder.summary()

    decoder_layer = DecoderLayer(p)
    output = decoder_layer(decoder_layer_input_seq, enc_output, None, None, True)
    print(f'Decoder Layer output: {output}')
    # decoder_layer.summary()


#One shot, Two shot , 3 Shot training 
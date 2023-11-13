import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from .MultiHeadAttentionLayer import MultiHeadAttentionLayer
from .FeedForwardLayer import FeedForward
from .AddNormalizationLayer import AddNormalization
from .PositionalEncodingLayer import PositionEmbeddingFixedWeights
from .utils import check_shape
from .params import baseline_test_params, Params

@tf.keras.saving.register_keras_serializable()
class DecoderLayer(Layer):
    def __init__(self, p:Params, **kwargs):
        super(DecoderLayer,self).__init__(**kwargs)
        #to build graph
        # self.build(input_shape=[None, p.seq_len, p.model_dim])
        self.seq_len = p.decoder_seq_len
        self.model_dim = p.model_dim
        self.dropout_rate = p.dropout_rate
        self.feed_forward_dim = p.feed_forward_dim

        self.masked_mha_layer = MultiHeadAttentionLayer(p,isRelative=False)
        self.dropout1 = Dropout(self.dropout_rate)
        self.add_norm1 = AddNormalization()

        self.mha_layer = MultiHeadAttentionLayer(p,isRelative=False)
        self.dropout2 = Dropout(self.dropout_rate)
        self.add_norm2 = AddNormalization()
        
        self.feed_forward = FeedForward(self.feed_forward_dim, self.model_dim)

        self.dropout3 = Dropout(self.dropout_rate)
        self.add_norm3 = AddNormalization()

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'feed_forward_dim': self.feed_forward_dim,
            'masked_mha_layer': self.masked_mha_layer.get_config(),
            'mha_layer':self.mha_layer.get_config(),
            'feed_forward': self.feed_forward.get_config(),
        })
        return config

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
        self.decoder_seq_len = p.decoder_seq_len
        self.model_dim = p.model_dim
        self.dropout_rate = p.dropout_rate
        self.decoder_vocab_size = p.decoder_vocab_size
        self.num_decoder_layers = p.num_decoder_layers

        self.positional_encoding = PositionEmbeddingFixedWeights(self.decoder_seq_len, self.decoder_vocab_size, self.model_dim)
        self.dropout = Dropout(self.dropout_rate)
        self.decoder_layers = [
            DecoderLayer(p) for _ in range(self.num_decoder_layers)]

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'decoder_seq_len': self.encoder_seq_len,
            'decoder_vocab_size': self.encoder_vocab_size,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'num_encoder_layers': self.num_encoder_layers,
            # 'positional_encoding' is also a layer and needs to handle its config
            'positional_encoding': self.positional_encoding.get_config(),
            'encoder_layers': [layer.get_config() for layer in self.encoder_layers.layers]
        })
    
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
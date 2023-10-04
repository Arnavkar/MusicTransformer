import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from Layers.Encoder import Encoder
from Layers.Decoder import Decoder
from Layers.utils import padding_mask, lookahead_mask
from params import baseline_test_params, Params

class TransformerModel(Model):
    def __init__(self,p:Params,**kwargs):
        super().__init__(**kwargs)
        self.model_dim = p.model_dim
        self.seq_len = p.seq_len
        self.encoder = Encoder(p)
        self.decoder = Decoder(p)
        
        self.dense = Dense(p.decoder_vocab_size)

    def build_graph(self):
        input_layer = Input(shape=(self.seq_len, self.model_dim))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))
    
    def call(self, encoder_input, decoder_input, training):
        padding = padding_mask(encoder_input)
        lookahead = tf.maximum(padding_mask(decoder_input),lookahead_mask(decoder_input.shape[1]))

        encoder_output = self.encoder(encoder_input, padding, training)
        decoder_output = self.decoder(decoder_input, encoder_output, lookahead, padding, training)
        output = self.dense(decoder_output)
        return output
    


if __name__ == "__main__":
    p = Params(baseline_test_params)

    model = TransformerModel(p)
    model.build_graph().summary()


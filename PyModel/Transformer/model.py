import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from .Encoder import Encoder
from .Decoder import Decoder
from .utils import padding_mask, lookahead_mask
from .params import baseline_test_params, Params

class TransformerModel(Model):
    def __init__(self,p:Params,**kwargs):
        super(TransformerModel,self).__init__(**kwargs)
        self.debug = p.debug
        self.model_dim = p.model_dim
        self.encoder = Encoder(p)
        self.decoder = Decoder(p)
        self.dense = Dense(p.decoder_vocab_size)

    
    def call(self, encoder_input, decoder_input, training):
        padding = padding_mask(encoder_input)
        lookahead = tf.maximum(padding_mask(decoder_input),lookahead_mask(decoder_input.shape[1]))

        encoder_output = self.encoder(encoder_input, padding, training)
        decoder_output = self.decoder(decoder_input, encoder_output, lookahead, padding, training)
        return self.dense(decoder_output)
    

if __name__ == "__main__":
    p = Params(baseline_test_params)
    test_tensor = tf.random.uniform((p.batch_size, p.seq_len))
    model = TransformerModel(p)
    output = model(test_tensor, test_tensor, True)
    print(f'Transformer Model output: {output}')
    model.summary()


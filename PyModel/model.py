import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from Layers.Encoder import Encoder
from Layers.Decoder import Decoder
from Layers.utils import padding_mask, lookahead_mask

class TransformerModel(Model):
    def __init__(self,enc_vocab_size,dec_vocab_size,enc_seq_len,dec_seq_len,num_heads,embedding_dim,model_dim,feed_forward_dim,dropout_rate,num_layers,**kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(
            enc_vocab_size,
            enc_seq_len,
            num_heads,
            embedding_dim,
            model_dim,
            feed_forward_dim,
            dropout_rate,
            num_layers,
        )

        self.decoder = Decoder(
            dec_vocab_size,
            dec_seq_len,
            num_heads,
            embedding_dim,
            model_dim,
            feed_forward_dim,
            dropout_rate,
            num_layers,
        )
        
        self.dense = Dense(dec_vocab_size)
    
    def call(self, encoder_input, decoder_input, training):
        padding = padding_mask(encoder_input)
        lookahead = tf.maximum(padding_mask(decoder_input),lookahead_mask(decoder_input.shape[1]))

        encoder_output = self.encoder(encoder_input, padding, training)
        decoder_output = self.decoder(decoder_input, encoder_output, lookahead, padding, training)
        output = self.dense(decoder_output)
        return output

if __name__ == "__main__":
    enc_vocab_size = 20 # Vocabulary size for the encoder
    dec_vocab_size = 20 # Vocabulary size for the decoder
    
    enc_seq_length = 5  # Maximum length of the input sequence
    dec_seq_length = 5  # Maximum length of the target sequence

    num_heads = 8
    embedding_dim = 64 #number of dimensions for each token in the sequence
    model_dim = 512
    feed_forward_dim = 2048
    num_layers = 6
    num_layers = 6
    dropout_rate = 0.1

    model = TransformerModel(
        enc_vocab_size = enc_vocab_size,
        dec_vocab_size = dec_vocab_size,
        enc_seq_len = enc_seq_length,
        dec_seq_len = dec_seq_length,
        num_heads = num_heads,
        embedding_dim = embedding_dim,
        model_dim = model_dim,
        feed_forward_dim = feed_forward_dim,
        dropout_rate = dropout_rate,
        num_layers = num_layers
    )


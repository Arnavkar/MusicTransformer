import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder
from Transformer.utils import padding_mask, lookahead_mask
from Transformer.params import baseline_test_params, Params
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow import math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64

class TransformerModel(Model):
    def __init__(self,p:Params,**kwargs):
        super(TransformerModel,self).__init__(**kwargs)
        self.debug = p.debug
        self.model_dim = p.model_dim
        self.encoder = Encoder(p)
        self.decoder = Decoder(p)
        self.dense = Dense(p.decoder_vocab_size)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
    
    def call(self, input_data, training):
        encoder_input, decoder_input = input_data
        padding = padding_mask(encoder_input)
        #tf.maximum returns maximum element wise - lookahead mask is a upper triangular matrix of ones
        lookahead = tf.maximum(padding_mask(decoder_input),lookahead_mask(decoder_input.shape[1]))

        encoder_output = self.encoder(encoder_input, padding, training)
        decoder_output = self.decoder(decoder_input, encoder_output, lookahead, padding, training)
        return self.dense(decoder_output)
    
    #@tf.function
    def train_step(self, data):
        train_batchX, train_batchY = data
        encoder_input = train_batchX
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
        with tf.GradientTape() as tape:
            #generate prediction
            prediction = self((encoder_input, decoder_input), training = True)

            #compute loss
            loss = self.compute_loss(decoder_output, prediction)

             # Compute the training accuracy
            accuracy = self.compute_accuracy(decoder_output, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(accuracy)
        return {"train_loss": self.train_loss.result(), "train_accuracy": self.train_accuracy.result()}
    
    #@tf.function
    def test_step(self,val_data):
        val_batchX, val_batchY = val_data
        encoder_input = val_batchX
        decoder_input = val_batchY[:, :-1]
        decoder_output = val_batchY[:, 1:]
        prediction = self((encoder_input, decoder_input), training = False)

        loss = self.compute_loss(decoder_output, prediction)
        self.val_loss.update_state(loss)
        return {"val_loss": self.val_loss.result()}
    
    def compile(self, optimizer,loss_fn,accuracy_fn):
        super().compile(optimizer = optimizer)
        self.optimizer = optimizer
        self.compute_loss = loss_fn
        self.compute_accuracy = accuracy_fn

    @property
    def metrics(self):
        return [self.train_loss, self.train_accuracy, self.val_loss]
    

class DecoderModel(Model):
    def __init__(self,p:Params,**kwargs):
        super(DecoderModel,self).__init__(**kwargs)
        self.debug = p.debug
        self.model_dim = p.model_dim
        self.decoder = Decoder(p)
        self.dense = Dense(p.decoder_vocab_size)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.logger = None
    
    def compile(self, optimizer,loss_fn,accuracy_fn,logger=None):
        super().compile()
        self.optimizer = optimizer
        self.compute_loss = loss_fn
        self.compute_accuracy = accuracy_fn
        if logger:
            self.logger = logger

    def call(self, input_data, training):
        pass

if __name__ == "__main__":
    p = Params(baseline_test_params)
    p.batch_size = 1
    p.seq_len = 10
    test_tensor = tf.constant([[1,234,243,18,37,24,18,57,89,2]])
    model = TransformerModel(p)
    output = model((test_tensor, test_tensor),True)
    print(f'Transformer Model output: {output}')
    model.summary()




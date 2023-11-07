import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from .Encoder import Encoder
from .Decoder import Decoder
from .utils import padding_mask, lookahead_mask
from .params import baseline_test_params, Params
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
    
    # Defining the loss function
    @tf.function
    def compute_loss(self,target,prediction):
        # Create mask so that the zero padding values are not included in the computation of loss
        padding_mask = math.logical_not(equal(target, 0))
        padding_mask = cast(padding_mask, float32)
    
        # Compute a sparse categorical cross-entropy loss on the unmasked values - logits = True because we do not have the softmax in our model
        loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
    
        # Compute the mean loss over the unmasked values
        return reduce_sum(loss) / reduce_sum(padding_mask)
    
    # Defining the accuracy function
    @tf.function
    def compute_accuracy(self,target,prediction):
        # Create mask so that the zero padding values are not included in the computation of accuracy
        padding_mask = math.logical_not(equal(target, 0))
    
        # Find equal prediction and target values, and apply the padding mask
        accuracy = equal(target, argmax(prediction, axis=2))
        accuracy = math.logical_and(padding_mask, accuracy)
    
        # Cast the True/False values to 32-bit-precision floating-point numbers
        padding_mask = cast(padding_mask, float32)
        accuracy = cast(accuracy, float32)
    
        # Compute the mean accuracy over the unmasked values
        return reduce_sum(accuracy) / reduce_sum(padding_mask)
    
    @tf.function
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
    
    @tf.function
    def test_step(self,val_data):
        val_batchX, val_batchY = val_data
        encoder_input = val_batchX
        decoder_input = val_batchY[:, :-1]
        decoder_output = val_batchY[:, 1:]
        prediction = self((encoder_input, decoder_input), training = False)

        loss = self.compute_loss(decoder_output, prediction)
        self.val_loss.update_state(loss)

        return {"val_loss": self.val_loss.result()}

    # @tf.function
    # def predict(self,input_data):
    #     encoder_input, decoder_input = input_data
    #     prediction = self(encoder_input, decoder_input, training = False)
    #     return prediction

    @property
    def metrics(self):
        return [self.train_loss, self.train_accuracy, self.val_loss]

    

if __name__ == "__main__":
    p = Params(baseline_test_params)
    test_tensor = tf.random.uniform((p.batch_size, p.seq_len))
    model = TransformerModel(p)
    output = model(test_tensor, test_tensor, True)
    print(f'Transformer Model output: {output}')
    model.summary()


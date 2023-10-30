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
        self.optimizer = tf.keras.optimizers.Adam(p.l_r, p.beta_1, p.beta_2, p.epsilon)
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    
    def call(self, encoder_input, decoder_input, training):
        padding = padding_mask(encoder_input)
        #tf.maximum returns maximum element wise - lookahead mask is a upper triangular matrix of ones
        lookahead = tf.maximum(padding_mask(decoder_input),lookahead_mask(decoder_input.shape[1]))

        encoder_output = self.encoder(encoder_input, padding, training)
        decoder_output = self.decoder(decoder_input, encoder_output, lookahead, padding, training)
        return self.dense(decoder_output)
    
    # Defining the loss function
    @tf.function
    def compute_loss(self,target, prediction):
        # Create mask so that the zero padding values are not included in the computation of loss
        padding_mask = math.logical_not(equal(target, 0))
        padding_mask = cast(padding_mask, float32)
    
        # Compute a sparse categorical cross-entropy loss on the unmasked values - logits = True because we do not have the softmax in our model
        loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
    
        # Compute the mean loss over the unmasked values
        return reduce_sum(loss) / reduce_sum(padding_mask)
    
    # Defining the accuracy function
    @tf.function
    def compute_accuracy(self,target, prediction):
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
    def train_step(self, data, val_data):
        encoder_input, decoder_input, decoder_output = data
        encoder_input_val, decoder_input_val, decoder_output_val = val_data
        with tf.GradientTape() as tape:
            #generate prediction
            prediction = self(encoder_input, decoder_input, training = True)

            #compute loss
            loss = self.compute_loss(decoder_output, prediction)

             # Compute the training accuracy
            accuracy = self.compute(decoder_output, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(accuracy)
        
        #Prediction on Validation set
        prediction = self(encoder_input_val, decoder_input_val, training = False)
        val_loss = self.compute_loss(decoder_output_val, prediction)

        self.val_loss.update_state(val_loss)

        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self,data):
        encoder_input, decoder_input, decoder_output = data
        prediction = self(encoder_input, decoder_input, training = False)

        loss = self.compute_loss(decoder_output, prediction)
        self.test_loss.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict(self,input_data):
        encoder_input, decoder_input = input_data
        prediction = self(encoder_input, decoder_input, training = False)
        return prediction

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


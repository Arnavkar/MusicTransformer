from keras.optimizers.legacy import Adam
from Transformer.LRSchedule import LRScheduler
from data.neural_translation.PrepareDataset_neural_translation import PrepareDataset
from keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from Transformer.model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import baseline_test_params, Params
from pickle import dump

p = Params(baseline_test_params)
 
# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)
 
# Prepare the training and test splits of the dataset
dataset = PrepareDataset()
trainX, trainY, valX, valY, train_orig, val_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('./data/neural_translation/english-german-both.pkl')


# Prepare the training dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(p.batch_size)

# Prepare the validation dataset batches
val_dataset = data.Dataset.from_tensor_slices((valX, valY))
val_dataset = val_dataset.batch(p.batch_size)

p.encoder_vocab_size = enc_vocab_size
p.decoder_vocab_size = dec_vocab_size
p.encoder_seq_len = enc_seq_length
p.decoder_seq_len = dec_seq_length
p.print_params()

transformer = TransformerModel(p)

# Defining the loss function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)
 
    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)
 
 
# Defining the accuracy function
def accuracy_fcn(target, prediction):
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

transformer.compile(optimizer=optimizer, 
                    loss=loss_fcn,
                    metrics=[accuracy_fcn])
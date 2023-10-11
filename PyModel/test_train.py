from tensorflow.keras.optimizers.legacy import Adam
from LRSchedule import LRScheduler
from data.PrepareDataset_machineTranslation import PrepareDataset
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from time import time
import tensorflow as tf
from params import baseline_test_params, Params

p = Params(baseline_test_params)
 
# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)
 
# Prepare the training and test splits of the dataset
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('PyModel/data/english-german-both.pkl')
 
# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(p.batch_size)

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
 
 
# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
 
# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=transformer, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)
 
# Speeding up the training process
@tf.function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
 
        # Run the forward pass of the model to generate a prediction
        prediction = transformer(encoder_input, decoder_input, training=False)
 
        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)
 
        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)
 
    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, transformer.trainable_weights)
 
    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, transformer.trainable_weights))
 
    train_loss(loss)
    train_accuracy(accuracy)
 
for epoch in range(p.epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
 
    print("\nStart of epoch %d" % (epoch + 1))
 
    start_time = time()
 
    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
 
        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
 
        train_step(encoder_input, decoder_input, decoder_output)
 
        if step % 50 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
            # print("Samples so far: %s" % ((step + 1) * batch_size))
 
    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
 
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
 
print("Training Complete! Total time taken: %.2fs" % (time() - start_time))


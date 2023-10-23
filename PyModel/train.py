from keras.optimizers.legacy import Adam
from Transformer.LRSchedule import LRScheduler
from CustomDataset import CustomDataset
from keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from Transformer.model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import baseline_test_params, midi_test_params_v1, Params
from pickle import dump
import datetime

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

p = Params(midi_test_params_v1)
print(p)
transformer = TransformerModel(p)
 
# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)
 
# Prepare the training and test splits of the dataset
dataset = CustomDataset(p)
print(dataset)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

# # Prepare the training dataset batches
# train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
# train_dataset = train_dataset.batch(p.batch_size)

# # Prepare the validation dataset batches
# val_dataset = data.Dataset.from_tensor_slices((valX, valY))
# val_dataset = val_dataset.batch(p.batch_size)

# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
val_loss = Mean(name='val_loss')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=transformer, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=None)

train_loss_dict = {}
val_loss_dict = {}
 
# Speeding up the training process
@tf.function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
 
        # Run the forward pass of the model to generate a prediction
        prediction = transformer(encoder_input, decoder_input, training=True)
        print(f"encoder input: {encoder_input}, decoder input: {decoder_input}, prediction: {prediction}, decoder output: {decoder_output}")
 
        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)
        print(f"loss: {loss}")
 
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
    val_loss.reset_states()
 
    print("\nStart of epoch %d" % (epoch + 1))
 
    start_time = time()
    for b in range(len(dataset.fileDict) // p.batch_size):
        train_batchX,train_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 1, 'train')
        val_batchX,val_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 1, 'validation')

        encoder_input_train = train_batchX
        decoder_input_train = train_batchY[:, :-1]
        decoder_output_train = train_batchY[:, 1:]

        encoder_input_val = val_batchX
        decoder_input_val = val_batchY[:, :-1]
        decoder_output_val = val_batchY[:, 1:]

        train_step(encoder_input_train, decoder_input_train, decoder_output_train)

        
    prediction = transformer(encoder_input_val, decoder_input_val, training = False)
    loss = loss_fcn(decoder_output_val, prediction)
    val_loss(loss)

    # Print epoch number and loss value at the end of every epoch
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f, Validation Loss %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result(), val_loss.result()))
 
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 1 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))

        transformer.save_weights('./weights/wghts' + str(epoch + 1) +'.ckpt')

        train_loss_dict[epoch] = train_loss.result()
        val_loss_dict[epoch] = val_loss.result()
 
# Save the training loss values
with open('./train_loss.pkl', 'wb') as file:
    dump(train_loss_dict, file)
 
# Save the validation loss values
with open('./val_loss.pkl', 'wb') as file:
    dump(val_loss_dict, file)
 
print("Training Complete! Total time taken: %.2fs" % (time() - start_time))   
print(tf.train.list_variables(ckpt_manager.latest_checkpoint) )


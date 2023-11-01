from CustomDataset import CustomDataset
from tensorflow.keras.metrics import Mean
from Transformer.model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import midi_test_params_v1, Params
from pickle import dump
from datetime import datetime
import argparse
import json
import os
from tqdm.notebook import tqdm, trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n','--name', type=str,required= True)
    parser.add_argument('-o','--overwrite', type=bool, required=False)
    parser.add_argument('-e','--epochs', type=int, required=False)
    parser.add_argument('-b','--batch_size', type=int, required=False)
    parser.add_argument('-s','--max_seq_len',type=int, required=False)
    parser.add_argument('-l','--num_layers', type=int, required=False)
    args = parser.parse_args()

    if not args.overwrite and os.path.exists('./models/' + args.name + '/'):
        print('Model already exists - please choose a different name')
        exit()
    
    base_path = './models/' + args.name +  '/'

    p = Params(midi_test_params_v1)
  
    if args.epochs:
        p.epochs = args.epochs
    
    if args.max_seq_len:
        p.encoder_seq_len = args.max_seq_len
        p.decoder_seq_len = args.max_seq_len

    if args.num_layers:
        p.num_encoder_layers = args.num_layers
        p.num_decoder_layers = args.num_layers

    if args.batch_size:
        p.batch_size = args.batch_size

    #Create Transformer
    model = TransformerModel(p)

    #set up relevant callbacks to use in the custom training loop
    #Early-Stopping
    dataset = CustomDataset(p)
    
    _callbacks = []

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=1,
        mode = "min",
        restore_best_weights=True,
    )

    #necessary to fix callback bug?

    _callbacks.append(early_stopping)

    callbacks = tf.keras.callbacks.CallbackList(
        _callbacks,
        add_history=True,
        model = model
    )

    logs = {}
    # Create a checkpoint object and manager to manage multiple checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        base_path + "checkpoints",
        max_to_keep=None
    )

    train_loss_dict = {}
    val_loss_dict = {}
 
    start_time = time()
    callbacks.on_train_begin(logs=logs)
    for epoch in range(p.epochs):
        callbacks.on_epoch_begin(epoch)
        print("\nStart of epoch %d" % (epoch + 1))
    
        for step in range(len(dataset.fileDict) // p.batch_size):
            train_batchX,train_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 1, 'train')
            val_batchX,val_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 1, 'validation')

            callbacks.on_train_batch_begin((train_batchX,train_batchY),logs=logs)

            encoder_input_train = train_batchX
            decoder_input_train = train_batchY[:, :-1]
            decoder_output_train = train_batchY[:, 1:]

            encoder_input_val = val_batchX
            decoder_input_val = val_batchY[:, :-1]
            decoder_output_val = val_batchY[:, 1:]

            logs = model.train_step((encoder_input_train, decoder_input_train, decoder_output_train),
                             (encoder_input_val, decoder_input_val, decoder_output_val))
            
            callbacks.on_train_batch_end((train_batchX,train_batchY),logs=logs)

        # Print epoch number and loss value at the end of every epoch
        print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f, Validation Loss %.4f" % (epoch + 1, model.train_loss.result(), model.train_accuracy.result(), model.val_loss.result()))
    
        # Save a checkpoint after every five epochs
        if (epoch + 1) % 10 == 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint at epoch %d" % (epoch + 1))

            model.save_weights(base_path+'weights/wghts' + str(epoch + 1) +'.ckpt')

            train_loss_dict[epoch] = float(model.train_loss.result().numpy())
            val_loss_dict[epoch] = float(model.val_loss.result().numpy())
        
        callbacks.on_epoch_end(epoch, logs=logs)
    
    callbacks.on_train_end(logs=logs)

    with open(base_path+'params.json', 'w') as file:
        param_dict = p.get_params()
        param_dict['name'] = args.name
        param_dict['training_date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        json.dump(param_dict, file, indent=4)

    # Save the training loss values
    with open(base_path+'train_loss.json', 'w') as file:
        json.dump(train_loss_dict,file)
    
    # Save the validation loss values
    with open(base_path+'val_loss.json', 'w') as file:
        json.dump(val_loss_dict, file)

    print("Training Complete! Total time taken: %.2fs" % (time() - start_time))   


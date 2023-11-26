from data.CustomDataset import CustomDataset
from tensorflow.keras.metrics import Mean
from model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import midi_test_params_v2, Params
from Transformer.LRSchedule import LRScheduler
from datetime import datetime
import argparse
import json
import os
import logging
from Transformer.utils import custom_loss, custom_accuracy
from train_utils import setup_experiment

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    parser = argparse.ArgumentParser()

    parser.add_argument('-n','--name', type=str,required= True)
    parser.add_argument('-e','--epochs', type=int, required=False)
    parser.add_argument('-b','--batch_size', type=int, required=False)
    parser.add_argument('-s','--max_seq_len',type=int, required=False)
    parser.add_argument('-l','--num_layers', type=int, required=False)
    parser.add_argument('-S','--steps_per_epoch', type=int, required=False)
    parser.add_argument('-f','--save_freq',type=int ,required=False)
    args = parser.parse_args()

    #Set up experiment
    base_path, p, logger = setup_experiment(args)

    # Create a MirroredStrategy to run training across multiple GPUs with single machine
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    #Instantiate Adam optimizer (Either with LRScheduler or without)
    #optimizer = tf.keras.optimizers.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)
    optimizer = tf.keras.optimizers.Adam(p.l_r, p.beta_1, p.beta_2, p.epsilon)
    
    #Create model
    model = TransformerModel(p)

    #Compile step - pass optimizer, loss func and accuracy func to model
    with strategy.scope():
        model.compile(
            optimizer = optimizer,
            loss_fn = custom_loss,
            accuracy_fn = custom_accuracy
        )

    #when using slide_seq2seq_batch - meant for decoder only architecture!
    dataset = CustomDataset(p)

    #When Using tf data set
    data = tf.data.Dataset.load("./data/tf_midi_data_train")
    val_data = tf.data.Dataset.load("./data/tf_midi_data_validation")

    data = data.shuffle(len(data)+1).repeat()
    val_data = val_data.shuffle(len(val_data)+1)

    data = data.batch(p.batch_size, drop_remainder=True)
    val_data = val_data.batch(p.batch_size, drop_remainder=True)

    # Create a checkpoint object and manager to manage multiple checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        base_path + "checkpoints",
        max_to_keep=None
    )
    
    train_loss_dict = {}
    val_loss_dict = {}

    #manual implementation of early stopping
    patience = 5
    wait = 0
    best = float('inf')

    start_time = time()
    for epoch in range(p.epochs):    
        #=======TRAIN WITH SLIDE_SEQ2SEQ_BATCH=======
        for step in range(len(dataset.fileDict) // p.batch_size):
            _,train_batchX,train_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 'train')
            _,val_batchX,val_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 'validation')

            model.train_step((train_batchX,train_batchY))
            model.test_step((val_batchX,val_batchY))

            if step % 50 == 0:
                print("Step %d - Training Loss: %.4f, Training Accuracy: %.4f, Validation Loss: %.4f" % (step, model.train_loss.result(), model.train_accuracy.result(), model.val_loss.result()))

        #=======TRAIN WITH TF.DATA=======
        # for step,(train_batchX,train_batchY) in enumerate(data):
        #     model.train_step((train_batchX,train_batchY))
        #     if step % 50 == 0:
        #         logger.info(f"Step {step} - Training Loss: {model.train_loss.result()}, Training Accuracy: {model.train_accuracy.result()}")
        #     if step == p.steps_per_epoch:
        #         break

        # for _,(val_batchX,val_batchY) in enumerate(val_data):
        #     model.test_step((val_batchX,val_batchY))
        # Print epoch number and loss value at the end of every epoch
        logger.info("Epoch %d: Training Loss %.4f, Training Accuracy %.4f, Validation Loss %.4f" % (epoch + 1, model.train_loss.result(), model.train_accuracy.result(), model.val_loss.result()))
    
        # Save a checkpoint after every five epochs
        if (epoch + 1) % p.save_freq == 0:
            save_path = ckpt_manager.save()
            logger.info("Saved checkpoint at epoch %d" % (epoch + 1))
            
        train_loss = float(model.train_loss.result().numpy())
        val_loss = float(model.val_loss.result().numpy())
        train_loss_dict[epoch] = train_loss
        val_loss_dict[epoch] = val_loss

        wait += 1
        if val_loss < best:
            best = val_loss
            wait = 0
        if wait >= patience:
            logger.info("Early stopping triggered at epoch %d" % (epoch + 1))
            break
        
    logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))

    # Save the training loss values
    try:
        logger.info("Saving Train loss...")
        with open(base_path+'train_loss.json', 'w') as file:
            json.dump(train_loss_dict,file, indent=4)
    except Exception as e:
        logger.error(e)
    
    # Save the validation loss values
    try:
        logger.info("Saving Validation loss...")
        with open(base_path+'val_loss.json', 'w') as file:
            json.dump(val_loss_dict, file, indent=4)
    except Exception as e:
        logger.error(e)


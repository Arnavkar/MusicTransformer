from CustomDataset import CustomDataset
from tensorflow.keras.metrics import Mean
from Transformer.model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import midi_test_params_v2, Params
from Transformer.LRSchedule import LRScheduler
from datetime import datetime
import argparse
import json
import os
import logging
from Transformer.utils import custom_loss


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()

    parser.add_argument('-n','--name', type=str,required= True)
    parser.add_argument('-o','--overwrite', type=bool, required=False)
    parser.add_argument('-e','--epochs', type=int, required=False)
    parser.add_argument('-b','--batch_size', type=int, required=False)
    parser.add_argument('-s','--max_seq_len',type=int, required=False)
    parser.add_argument('-l','--num_layers', type=int, required=False)
    args = parser.parse_args()

    if not args.overwrite and os.path.exists('./models/' + args.name + '/'):
        print('Model already exists - do you want to continue? Y/N')
        char = input().lower()
        while char not in ['y','n']:
            print('Invalid input - do you want to continue? Y/N')
            char = input().lower()
        if char == 'n':
            exit()
    
    base_path = './models/' + args.name +  '/'
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    p = Params(midi_test_params_v2)
  
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

    #set up logger
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.DEBUG)

    #file logger
    fh = logging.FileHandler(base_path + 'output.log', mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

   #Instantiate and Adam optimizer
    #optimizer = tf.keras.optimizers.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)
    optimizer = tf.keras.optimizers.Adam(p.l_r*10, p.beta_1, p.beta_2, p.epsilon)
    
    model = TransformerModel(p)
     # Create a MirroredStrategy to run training across 2 GPUs
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model.compile(
            optimizer = optimizer,
            loss_fn = custom_loss)

    #set up relevant callbacks to use in the custom training loop
    #Early-Stopping
    dataset = CustomDataset(p)
    logs = {}
    # Create a checkpoint object and manager to manage multiple checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        base_path + "checkpoints",
        max_to_keep=None
    )

    try:
        logger.info("Saving Params...")
        with open(base_path+'params.json', 'w') as file:
            param_dict = p.get_params()
            param_dict['name'] = args.name
            param_dict['training_date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            json.dump(param_dict, file, indent=4)
        logger.info("Params Saved!")
    except Exception as e:
        logger.error(e)

    train_loss_dict = {}
    val_loss_dict = {}

    #manual implementation of early stopping
    patience = 5
    wait = 0
    best = float('inf')

    start_time = time()
    for epoch in range(p.epochs):    
        for step in range(len(dataset.fileDict) // p.batch_size):
            _,train_batchX,train_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 'train')
            _,val_batchX,val_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 'validation')

            model.train_step((train_batchX,train_batchY))
            model.test_step((val_batchX,val_batchY))

        # Print epoch number and loss value at the end of every epoch
        logger.info("Epoch %d: Training Loss %.4f, Training Accuracy %.4f, Validation Loss %.4f" % (epoch + 1, model.train_loss.result(), model.train_accuracy.result(), model.val_loss.result()))
    
        # Save a checkpoint after every five epochs
        if (epoch + 1) % 10 == 0:
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


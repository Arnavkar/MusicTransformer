from CustomDataset import CustomDataset
from tensorflow.keras.metrics import Mean
from Transformer.model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import midi_test_params_v2, Params
from Transformer.utils import custom_loss
from pickle import dump
from datetime import datetime
import argparse
import json
import os
from tqdm.notebook import tqdm, trange
import logging
from Transformer.LRSchedule import LRScheduler


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    #handle all command line arguments and parsing
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
    #adjust params as required
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
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Create a MirroredStrategy to run training across 2 GPUs
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    #set up datasets, including shuffling and batching
    data = tf.data.Dataset.load("./data/tf_midi_data_train")
    val_data = tf.data.Dataset.load("./data/tf_midi_data_validation")

    data = data.shuffle(len(data))
    val_data = val_data.shuffle(len(data))

    data = data.batch(p.batch_size, drop_remainder=True)
    val_data = val_data.batch(p.batch_size, drop_remainder=True)

    #Instantiate and Adam optimizer
    optimizer = tf.keras.optimizers.Adam(0.001, p.beta_1, p.beta_2, p.epsilon)

    
    model = TransformerModel(p)

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

    with strategy.scope():
        model.compile(optimizer = optimizer,
                      loss_fn = custom_loss)

    start_time = time()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(base_path + "checkpoints", save_freq='epoch', verbose = 1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_val_loss', patience=3, restore_best_weights=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=base_path+'logs',
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq='epoch',
    )
    #Train model and save history
    try:
        logger.info("Training model...")
        history = model.fit(
            data.repeat(), 
            epochs = p.epochs,
            validation_data = val_data,
            callbacks = [model_checkpoint, early_stopping, tensorboard],
            steps_per_epoch = 500,
        )
        logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))  
        logger.info("Saving History...")
        with open(base_path+'history.json', 'w') as file:
            json.dump(history.history,file)
        logger.info("History Saved!") 
    except Exception as e:
        logger.error(e)
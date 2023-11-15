from CustomDataset import CustomDataset
from tensorflow.keras.metrics import Mean
from model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.params import midi_test_params_v2, Params
from Transformer.utils import custom_loss, custom_accuracy
from pickle import dump
from datetime import datetime
import argparse
import json
import os
from tqdm.notebook import tqdm, trange
import logging
from Transformer.LRSchedule import LRScheduler
from baselineModel import createBaselineTransformer


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #handle all command line arguments and parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', type=str,required= True)
    parser.add_argument('-o','--overwrite', type=bool, required=False)
    parser.add_argument('-e','--epochs', type=int, required=False)
    parser.add_argument('-b','--batch_size', type=int, required=False)
    parser.add_argument('-s','--max_seq_len',type=int, required=False)
    parser.add_argument('-l','--num_layers', type=int, required=False)
    parser.add_argument('--with-baseline', type=bool, required=False)
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
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Create a MirroredStrategy to run training across 2 GPUs
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    #set up datasets, including shuffling and batching
    train_data_path = "./data/tf_midi_train_512_1"
    val_data_path = "./data/tf_midi_validation_512_1"

    if args.with_baseline:
        data=tf.data.Dataset.load(train_data_path+"_baseline")
        val_data=tf.data.Dataset.load(val_data_path+"_baseline")
    else:
        data = tf.data.Dataset.load(train_data_path)
        val_data = tf.data.Dataset.load(val_data_path)

    data = data.shuffle(len(data))
    val_data = val_data.shuffle(len(data))

    # data = data.batch(p.batch_size, drop_remainder=True)
    # val_data = val_data.batch(p.batch_size, drop_remainder=True)

    #Instantiate and Adam optimizer
    optimizer = tf.keras.optimizers.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)
    #optimizer = tf.keras.optimizers.Adam(0.0001, p.beta_1, p.beta_2, p.epsilon)

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

    # with strategy.scope():
    if args.with_baseline:
        print("Creating baseline transformer...")
        model = createBaselineTransformer(p)
        model.compile(
            optimizer = optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
    else:
        model = TransformerModel(p)
        model.compile(optimizer = optimizer,
                    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    accuracy_fn = custom_accuracy,
                    logger = logger)

    start_time = time()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        base_path + "checkpoints",
        save_freq='epoch',
        verbose = 1,
        save_weights_only = True,
    )
    
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
            data, 
            epochs = p.epochs,
            validation_data = val_data,
            callbacks = [model_checkpoint, early_stopping, tensorboard],
        )
        logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))  
        logger.info("Saving History...")
        with open(base_path+'history.json', 'w') as file:
            json.dump(history.history,file)
        logger.info("History Saved!") 
    except Exception as e:
        logger.error(e)
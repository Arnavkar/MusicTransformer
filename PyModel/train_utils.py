import tensorflow as tf
import os
import logging
from Transformer.params import Params, midi_test_params_v2
import json
from datetime import datetime

def setup_path(args):
    #Check if model already exists - ask about override
    if os.path.exists('./models/' + args.name + '/'):
        print('Model already exists - do you want to continue? Y/N')
        char = input().lower()
        while char not in ['y','n']:
            print('Invalid input - do you want to continue? Y/N')
            char = input().lower()
        if char == 'n':
            exit()
    
    #Set up base path for model under models directory
    base_path = './models/' + args.name +  '/'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    return base_path

def setup_params(args,base_params=midi_test_params_v2):
    #Initilaize and adjust params based on cmd line arguments
    p = Params(base_params)
  
    if args.epochs:
        p.epochs = args.epochs
    
    if args.encoder_seq_len:
        p.encoder_seq_len = args.encoder_seq_len
    
    if args.decoder_seq_len:
        p.decoder_seq_len = args.decoder_seq_len

    if args.num_layers:
        p.num_encoder_layers = args.num_layers
        p.num_decoder_layers = args.num_layers

    if args.batch_size:
        p.batch_size = args.batch_size

    # if args.save_freq:
    #     p.save_freq = args.save_freq

    return p

def setup_logger(base_path):
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
    return logger

def save_params(p,base_path,logger,args):
    #Try to save parameters that will be used later to save the model
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

def get_dataset(train_path, validation_path, batch_size):
    #Load data based on given paths
    train_data = tf.data.Dataset.load(train_path)
    val_data = tf.data.Dataset.load(validation_path)

    #Shuffle data
    train_data = train_data.shuffle(len(train_data))
    val_data = val_data.shuffle(len(val_data))

    #Batch data
    train_data = train_data.batch(batch_size, drop_remainder=True)
    val_data = val_data.batch(batch_size, drop_remainder=True)
    
    return train_data, val_data

def setup_experiment(args):
    #Set up base path for model under models directory
    base_path = setup_path(args)

    #Get final params object from params set up
    p = setup_params(args)

    #Setup logger for custom training loop
    logger = setup_logger(base_path)

    #Save Params for inference later
    save_params(p,base_path, logger,args)

    return base_path, p, logger
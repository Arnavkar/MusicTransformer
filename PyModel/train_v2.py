from data.CustomDataset import CustomDataset
from tensorflow.keras.metrics import Mean
from model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.utils import custom_loss, custom_accuracy
from pickle import dump
from datetime import datetime
import argparse
import json
from Transformer.LRSchedule import LRScheduler
from BaselineTransformer.baselineModel import createBaselineTransformer
from train_utils import setup_experiment
import Dataset.testDataSet as test #For training purposes

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #handle all command line arguments and parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', type=str,required= True)
    parser.add_argument('-o','--overwrite', type=bool, required=False)
    parser.add_argument('-e','--epochs', type=int, required=False)
    parser.add_argument('-b','--batch_size', type=int, required=False)
    parser.add_argument('-es','--encoder_seq_len',type=int, required=False)
    parser.add_argument('-ds','--decoder_seq_len',type=int, required=False)
    parser.add_argument('-l','--num_layers', type=int, required=False)
    parser.add_argument('--with-baseline', type=bool, required=False)
    args = parser.parse_args()

    print(args)
    #Set up experiment
    base_path, p, logger = setup_experiment(args)

    #Set up Dataset - different iterations

    #====================================================================
    # simple scales as a sequence of numbers (no encoding) etc.
    #====================================================================
    # train,val,_ = test.mockTfDataset(test.MAJOR_SCALE, 12)

    #====================================================================
    # A window of {encoder_seq_len} taken throughout the entire midi file
    #====================================================================
    # train,val,_ = test.mockTfDataset_from_encoded_midi('./data/processed/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle', p.encoder_seq_len)

    # #Shuffle and Batch data - map for the baseline transformer model
    # train = train.shuffle(len(train))
    # train = train.batch(p.batch_size, drop_remainder=True)
    # train = train.map(test.format_dataset)

    # val = val.shuffle(len(val))
    # val = val.batch(p.batch_size, drop_remainder=True)
    # val = val.map(test.format_dataset)

    #====================================================================
    # Custom dataset class instantiation for train and val
    #====================================================================
    train = CustomDataset(p, 'train', min_event_length=p.encoder_seq_len*2,num_files_to_use=1,logger=logger)
    if train.num_files_to_use != None:
        logger.info(f"Using only {train.num_files_to_use} files for training")
        for file in train.data:
            logger.info(f"Using file:{file.path}")
    
    #====================================================================
    #Model / Optimizer Set up
    #====================================================================

    #Instantiate Adam optimizer (NOTE: using the legacy optimizer for running on MacOS CPu)
    optimizer = tf.keras.optimizers.legacy.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)

    #Choose between custom transformer vs baseline transformer
    if args.with_baseline:
        logger.info("Creating baseline transformer...")
        model = createBaselineTransformer(p)
        model.compile(optimizer = optimizer, loss=custom_loss, metrics=[custom_accuracy])
    else:
        model = TransformerModel(p)
        model.compile(optimizer = optimizer, loss_fn = custom_loss, accuracy_fn=custom_accuracy)

    #====================================================================
    #Callbacks
    #====================================================================
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        base_path + "checkpoints/checkpoints_{epoch:02d}",
        save_freq='epoch',
        verbose = 1,
        save_weights_only = True,
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=base_path+'logs',
        update_freq='epoch',
        histogram_freq = 1,
        profile_batch='5, 10'
    )
    #====================================================================
    #Train model and save history
    #====================================================================

    start_time = time()
    try:
        logger.info("Training model...")
        history = model.fit(
            train, 
            epochs = p.epochs,
            callbacks = [model_checkpoint, early_stopping, tensorboard],
        )
        logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))  
        logger.info("Saving History...")
        with open(base_path+'history.json', 'w') as file:
            json.dump(history.history,file)
        logger.info("History Saved!") 
    except Exception as e:
        logger.error(e)

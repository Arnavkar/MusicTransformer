from Dataset.SequenceDataset import SequenceDataset
from Dataset.testDataSet import TestDataset
from tensorflow.keras.metrics import Mean
from Transformer.model import TransformerModel
from time import time
import tensorflow as tf
from Transformer.utils import custom_loss, custom_accuracy
import argparse
import json
from Transformer.LRSchedule import LRScheduler
from BaselineTransformer.baselineModel import createBaselineTransformer
from train_utils import setup_experiment
import os
import traceback

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
    # train,val,_ = mockTfDataset(MAJOR_SCALE, 12, 2)

    # ====================================================================
    # Using in-memory dataset with TestDataset
    # ====================================================================
    # dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2)

    # train, val, _ = dataset.mockTfDataset_from_encoded_midi(3)
    # # #Shuffle and Batch data - map for the baseline transformer model

    # train = dataset.mockTfDataset_from_encoded_midi_path('./data/processed_numpy/piano_train.mid.npy', 1)
    # val = dataset.mockTfDataset_from_encoded_midi_path('./data/processed_numpy/piano_test.mid.npy', 1)

    # train = train.shuffle(len(train))
    # train = train.batch(p.batch_size, drop_remainder=True)
    # train = train.map(dataset.format_dataset)

    # val = val.shuffle(len(val))
    # val = val.batch(p.batch_size, drop_remainder=True)
    # val = val.map(dataset.format_dataset)

    

    #====================================================================
    # Custom dataset class instantiation for train and val
    #====================================================================
    train = SequenceDataset(p, 'train', min_event_length=p.encoder_seq_len*2,logger=logger)
    # # val = SequenceDataset(p, 'val', min_event_length=p.encoder_seq_len*2,logger=logger,num_files_to_use=1)

    if train.num_files_to_use != None:
        logger.info(f"Using only {train.num_files_to_use} files for training")
        for file in train.data:
            logger.info(f"Using file:{file.path}")
    
    #====================================================================
    #Callbacks and Optimizer Set up
    #====================================================================
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        base_path + "checkpoints/checkpoints_{epoch:02d}",
        save_freq='epoch',
        verbose = 1,
        save_weights_only = True,
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=base_path+'logs',
        update_freq='epoch',
        histogram_freq = 1,
    )

    #Instantiate Adam optimizer (NOTE: using the legacy optimizer for running on MacOS CPu)
    #====================================================================
    #Model Set up
    #====================================================================

    optimizer = tf.keras.optimizers.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)

    #Choose between custom transformer vs baseline transformer
    if args.with_baseline:
        logger.info("Creating baseline transformer...")
        model = createBaselineTransformer(p)
        model.compile(optimizer = optimizer, loss=custom_loss, metrics=[custom_accuracy])
    else:
        model = TransformerModel(p)
        model.compile(optimizer = optimizer, loss_fn = custom_loss, accuracy_fn=custom_accuracy)

    #====================================================================
    #Train model and save history
    #====================================================================

    start_time = time()
    try:
        logger.info("Training model...")
        history = model.fit(
            train, 
            epochs = p.epochs,
            callbacks = [model_checkpoint, tensorboard]
        )
        logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))  
        logger.info("Saving History...")
        with open(base_path+'history.json', 'w') as file:
            json.dump(history.history,file)
        logger.info("History Saved!") 
    except KeyboardInterrupt as ex:
        logger.error(f"KeyBoard Interrupt Occurred")
        tb = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        logger.error(tb)
    except Exception as ex:
        logger.error(f"Unexpected Exception Occurred")
        tb = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        logger.error(tb)
        

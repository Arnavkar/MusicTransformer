from Dataset.SequenceDataset import SequenceDataset
from Dataset.TestDataset import TestDataset
from Dataset.TestDataset import MAJOR_SCALE
from Dataset.RandomDataset import RandomCropDataset

from CustomTransformer.model import TransformerModel
from KerasTransformer.baselineModel import createBaselineTransformer

from time import time
import tensorflow as tf
from CustomTransformer.utils import custom_loss, custom_accuracy
import argparse
import json
from CustomTransformer.LRSchedule import LRScheduler
from utils.train_utils import setup_experiment
import os
import traceback

if __name__ == "__main__":
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    #handle all command line arguments and parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', type=str,required= True)
    parser.add_argument('-o','--overwrite', type=bool, required=False)
    parser.add_argument('-e','--epochs', type=int, required=False)
    parser.add_argument('-b','--batch_size', type=int, required=False)
    parser.add_argument('-es','--encoder_seq_len',type=int, required=False)
    parser.add_argument('-ds','--decoder_seq_len',type=int, required=False)
    parser.add_argument('-l','--num_layers', type=int, required=False)
    parser.add_argument('-d','--dropout', type=int, required=False)
    parser.add_argument('-hs','--hidden_size', type=int, required=False)
    parser.add_argument('--dataset-type', type=str, required=False)
    parser.add_argument('--choose-gpu', type=str, required=False)
    parser.add_argument('--custom', type=bool, required=False)
    args = parser.parse_args()

    #Set up experiment
    base_path, p, logger = setup_experiment(args)

    #log args
    logger.info(f"params: {p}")

    #set visibile gpu devices eg. "0"
    if args.choose_gpu:
        print(f"Setting visible gpu devices to {args.choose_gpu}")
        #set visible gpu devices
        os.environ["CUDA_VISIBLE_DEVICES"]=args.choose_gpu
    
    #Set up Dataset - different iterations
    if not args.dataset_type: args.dataset_type = 'random'

    # ====================================================================
    # Using TestDataset
    # ====================================================================
    if args.dataset_type in ['mocksequence','inmemory']:
        dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2)
        
        if args.dataset_type == 'mocksequence':
            #====================================================================
            # simple scales as a sequence of numbers (no encoding) etc.
            #====================================================================
            train,val,_ = dataset.mockTfDataset_from_scale(MAJOR_SCALE, 12)
        else:
            #====================================================================
            # In memory dataset of sequences
            #====================================================================
            train, val, _ = dataset.mockTfDataset_from_encoded_midi(3)

        train = train.shuffle(len(train))
        train = train.batch(p.batch_size, drop_remainder=True)
        train = train.map(dataset.format_dataset)

        val = val.shuffle(len(val))
        val = val.batch(p.batch_size, drop_remainder=True)
        val = val.map(dataset.format_dataset)

    #====================================================================
    # Sequence dataset class instantiation for train and val
    #====================================================================
    if args.dataset_type == 'sequence':
        train = SequenceDataset(p, 'train', min_event_length=p.encoder_seq_len*2,logger=logger)
        val = SequenceDataset(p, 'validation', min_event_length=p.encoder_seq_len*2,logger=logger)

        if train.num_files_to_use != None:
            logger.info(f"Using only {train.num_files_to_use} files for training")
            for file in train.data:
                logger.info(f"Using file:{file.path}")

    #====================================================================
    # Random dataset class instantiation for train and val
    #====================================================================
    if args.dataset_type == 'random':
        train_dataset = RandomCropDataset(p, 'train', min_event_length=p.encoder_seq_len*2,logger=logger)
        val_dataset = RandomCropDataset(p, 'validation', min_event_length=p.encoder_seq_len*2,logger=logger)
        test_dataset = RandomCropDataset(p, 'test', min_event_length=p.encoder_seq_len*2,logger=logger)

        if train_dataset.num_files_to_use != None:
            logger.info(f"Using only {train_dataset.num_files_to_use} files for training")
            for file in train_dataset.data:
                logger.info(f"Using file:{file}")

        train = train_dataset.batch_generator(p.encoder_seq_len)
        val = val_dataset.batch_generator(p.encoder_seq_len)

        #convert to tf.data.Dataset
        train = tf.data.Dataset.from_generator(
            train,
            output_signature=(
            {
                'encoder_inputs':tf.TensorSpec(shape=(p.batch_size,p.encoder_seq_len), dtype=tf.int32),
                'decoder_inputs':tf.TensorSpec(shape=(p.batch_size,p.decoder_seq_len-1), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(p.batch_size,p.decoder_seq_len-1), dtype=tf.int32))
        )            
        val = tf.data.Dataset.from_generator(
            val,
            output_signature=(
            {
                'encoder_inputs':tf.TensorSpec(shape=(p.batch_size,p.encoder_seq_len), dtype=tf.int32),
                'decoder_inputs':tf.TensorSpec(shape=(p.batch_size,p.decoder_seq_len-1), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(p.batch_size,p.decoder_seq_len-1), dtype=tf.int32))
        )
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
        patience=3, 
        restore_best_weights=True
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=base_path+'logs',
        update_freq='epoch',
        histogram_freq = 1,
    )

    #Instantiate Adam optimizer (NOTE: use the legacy optimizer if running locally on MacOS CPu)
    #====================================================================
    #Model Set up
    #====================================================================
    #Parameters for the Adam optimizer taken fom Vaswani et al. 2017
    optimizer = tf.keras.optimizers.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)

    #Choose between custom transformer vs baseline transformer
    if args.custom:
        logger.info("Creating Custom transformer...")
        model = TransformerModel(p)
        model.compile(optimizer = optimizer, loss_fn = custom_loss, accuracy_fn=custom_accuracy)
    else:
        logger.info("Creating Keras transformer...")
        model = createBaselineTransformer(p)
        model.compile(optimizer = optimizer, loss=custom_loss, metrics=[custom_accuracy])

    #model summary 
    logger.info(model.summary())

    test = tf.data.Dataset.load('./data/test_tf_dataset_instance')

    #====================================================================
    #Train model and save history
    #====================================================================
    start_time = time()
    try:
        logger.info("Training model...")
        history = model.fit(
            train, 
            validation_data = val,
            validation_steps = 100,
            epochs = p.epochs,
            callbacks = [model_checkpoint, early_stopping, tensorboard],
            steps_per_epoch=p.steps_per_epoch
        )
        logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))  
        #evaluate 
        logger.info("Evaluating model...")
        loss, accuracy = model.evaluate(test)
        logger.info(f"Test accuracy: {accuracy}")
        logger.info(f"Test loss: {loss}")
        
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
            

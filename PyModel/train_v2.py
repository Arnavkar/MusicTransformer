from CustomDataset import CustomDataset
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
import testData as test #For training purposes
from shared_mem import shared_mem_multiprocessing

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

    #Set up Dataset

    #=================================
    # simple scales as a sequence of numbers (no encoding) etc.
    #=================================
    # train,val,_ = test.mockTfDataset(test.MAJOR_SCALE, 12)

    #=================================
    # A window of {encoder_seq_len} taken throughout the entire midi file
    #=================================
    # train,val,_ = test.mockTfDataset_from_encoded_midi('./data/processed/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle', p.encoder_seq_len)

    # #Shuffle and Batch data - map for the baseline transformer model
    # train = train.shuffle(len(train))
    # train = train.batch(p.batch_size, drop_remainder=True)
    # train = train.map(test.format_dataset)

    # val = val.shuffle(len(val))
    # val = val.batch(p.batch_size, drop_remainder=True)
    # val = val.map(test.format_dataset)

    #Instantiate a version of our custom dataset class
    train = CustomDataset(p, 'train', min_event_length=p.encoder_seq_len*2)
    for file in train.data:
      logger.info(f"Using file:{file.path}")
    train_gen = shared_mem_multiprocessing(train, workers=32)
    val = CustomDataset(p, 'val', min_event_length=p.encoder_seq_len*2)
    
    #Instantiate Adam optimizer (NOTE: using the legacy optimizer for running on MacOS CPu)
    optimizer = tf.keras.optimizers.legacy.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)

    if args.with_baseline:
        print("Creating baseline transformer...")
        model = createBaselineTransformer(p)
        model.compile(optimizer = optimizer, loss=custom_loss, metrics=[custom_accuracy])
    else:
        model = TransformerModel(p)
        model.compile(optimizer = optimizer, loss_fn = custom_loss, accuracy_fn=custom_accuracy)

    start_time = time()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        base_path + "checkpoints/checkpoints_{epoch:02d}",
        save_freq='epoch',
        verbose = 1,
        save_weights_only = True,
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=base_path+'logs',
        update_freq='epoch',
        histogram_freq = 1,
    )

    #Train model and save history
    try:
        logger.info("Training model...")
        history = model.fit(
            train, 
            epochs = p.epochs,
            callbacks = [model_checkpoint, early_stopping, tensorboard],
            use_multiprocessing=True,
            workers = 8
        )
        logger.info("Training Complete! Total time taken: %.2fs" % (time() - start_time))  
        logger.info("Saving History...")
        with open(base_path+'history.json', 'w') as file:
            json.dump(history.history,file)
        logger.info("History Saved!") 
    except Exception as e:
        logger.error(e)

    #Test model against test data set
    #NOTE : such seq2seq models still need to be evaluated by manually examining output
    # try:
    #     logger.info("Testing model...")
    #     test_data_path = "./data/tf_midi_test_512_1"
    #     if args.with_baseline:
    #         test_data=tf.data.Dataset.load(test_data_path+"_baseline")
    #     else:
    #         test_data = tf.data.Dataset.load(test_data_path)
    #     test_data = test_data.shuffle(len(test_data))
    #     test_data = test_data.batch(p.batch_size, drop_remainder=True)
    #     test_loss, test_acc = model.evaluate(test_data)
    #     logger.info("Test Loss: %.4f, Test Accuracy: %.4f" % (test_loss, test_acc))
    # except Exception as e:
    #     logger.error(e)

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

    #Set up experiment
    base_path, p, logger = setup_experiment(args)

    #Set up Dataset
    #================
    # Test data - simple scales etc.
    #================
    data = test.mockTfDataset(test.MAJOR_SCALE, 12)
    data = data.shuffle(len(data))
    data = data.batch(p.batch_size, drop_remainder=True)
    data = data.map(test.format_dataset)
    p.encoder_seq_len = 12
    p.decoder_seq_len = 14

    #Instantiate and Adam optimizer
    # optimizer = tf.keras.optimizers.Adam(, p.beta_1, p.beta_2, p.epsilon)
    optimizer = tf.keras.optimizers.legacy.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)

    # with strategy.scope():
    if args.with_baseline:
        print("Creating baseline transformer...")
        model = createBaselineTransformer(p)
        model.compile(optimizer = optimizer, loss=custom_loss, metrics=[custom_accuracy])
    else:
        model = TransformerModel(p)
        model.compile(optimizer = optimizer,
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics = ["accuracy"])

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
    )



    #Train model and save history
    try:
        logger.info("Training model...")
        history = model.fit(
            data, 
            epochs = p.epochs,
            # validation_data = val_data,
            callbacks = [model_checkpoint, tensorboard],
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

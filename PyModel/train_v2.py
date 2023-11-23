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
from baselineModel import createBaselineTransformer
from train_utils import setup_experiment


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

    

    # with strategy.scope():
    if args.with_baseline:
        print("Creating baseline transformer...")
        model = createBaselineTransformer(p)
        model.compile(
            optimizer = optimizer, loss=custom_loss, metrics=["accuracy"]
        )
    else:
        model = TransformerModel(p)
        model.compile(optimizer = optimizer,
                    loss_fn = custom_loss,
                    accuracy_fn = custom_accuracy,
                    logger = logger)

    start_time = time()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        base_path + "checkpoints",
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

    #Test model against test data set
    #NOTE : such seq2seq models still need to be evaluated by manually examining output
    try:
        logger.info("Testing model...")
        test_data_path = "./data/tf_midi_test_512_1"
        if args.with_baseline:
            test_data=tf.data.Dataset.load(test_data_path+"_baseline")
        else:
            test_data = tf.data.Dataset.load(test_data_path)
        test_data = test_data.shuffle(len(test_data))
        test_data = test_data.batch(p.batch_size, drop_remainder=True)
        test_loss, test_acc = model.evaluate(test_data)
        logger.info("Test Loss: %.4f, Test Accuracy: %.4f" % (test_loss, test_acc))
    except Exception as e:
        logger.error(e)

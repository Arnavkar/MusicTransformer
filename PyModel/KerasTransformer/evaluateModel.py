import tensorflow as tf
from pickle import load
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import argparse
import json
from CustomTransformer.params import Params
import os
from CustomTransformer.LRSchedule import LRScheduler
from Dataset.TestDataset import TestDataset
from .baselineModel import createBaselineTransformer
from CustomTransformer.utils import custom_loss
from Analysis.analysis import Analysis

class ModelEvaluator:
    def __init__(self, model, p:Params, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.params = p
        self.y_pred = []
        self.y_true = []
        self.nlls = []

    def generate_all_predictions(self):
        for inputs, targets in self.test_dataset:
            encoder_inputs = inputs["encoder_inputs"].numpy()
            decoder_outputs = targets.numpy()

            for i in range(p.batch_size):
                inputs = list(encoder_inputs[i])
                #Strip only end token
                targets = list(decoder_outputs[i][:-1])
                # Strip start and end tokens
                model_output = list(self.generate_prediction([encoder_inputs[i]]))[1:-1]
                print(model_output)

        

    def generate_prediction(self, input_sequence):
        encoder_input = pad_sequences(input_sequence, maxlen=self.params.encoder_seq_len, padding='post')
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)

        start_token = tf.convert_to_tensor([self.params.token_sos],dtype=tf.int64)
        decoder_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0,start_token)
        
        i = 0
        print("Decoding....")
        while True:
            prediction = self.model((encoder_input, tf.transpose(decoder_output.stack())), training=False)
            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = tf.argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][tf.newaxis]

            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)
            # Break if an <EOS> token is predicted
            if predicted_id == self.params.token_eos or i == self.params.decoder_seq_len-1:
                # print("hit eos")
                break
            i+=1
            # print(i,predicted_id)

        output = tf.transpose(decoder_output.stack())[0]
        decoder_output = decoder_output.mark_used()
        output = output.numpy()
        
        return output
            
    def calculate_nll(self):
        total_loss = 0
        num_sequences = 0

        for (input_sequence, true_sequence), predicted_sequence in zip(self.test_dataset, self.predictions):
            # Calculate loss using sparse_categorical_crossentropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(true_sequence, predicted_sequence, from_logits=False)
            total_loss += tf.reduce_sum(loss)
            num_sequences += input_sequence.shape[0]

        # Calculate overall NLL (mean loss per sequence)
        overall_nll = total_loss / num_sequences
        return overall_nll

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    parser = argparse.ArgumentParser()
    parser.add_argument('-n',"--model_name", type=str,required= True)
    parser.add_argument('-c','--checkpoint_type', type=str, required=True)
    parser.add_argument('-p',"--patience", type=int,required= True)
    args = parser.parse_args()

    model_dir = './models_to_analyze/' + args.model_name

    model_params = json.load(open(model_dir + '/params.json', 'rb'))
    p = Params(model_params)

    model = createBaselineTransformer(p)
    optimizer = tf.keras.optimizers.legacy.Adam(LRScheduler(p.model_dim), p.beta_1, p.beta_2, p.epsilon)

    model.compile(
        optimizer = optimizer, loss=custom_loss, metrics=["accuracy"]
    )

    if args.checkpoint_type == 'pb':
        model = tf.keras.models.load_model(model_dir + '/checkpoints')

    elif args.checkpoint_type == 'ckpt':
    #instantiate model
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)    

        if latest_checkpoint == None:
            print("Retrying")
            latest_checkpoint = tf.train.latest_checkpoint(model_dir + "/checkpoints")

        if latest_checkpoint == None:
            raise Exception("No checkpoint found")

        #checkpoint modification to get checkpoint with best val_loss
        latest_checkpoint = latest_checkpoint[:-1] + str(int(latest_checkpoint[-2:]) - args.patience)

        print(f"Latest Checkpoint path: {latest_checkpoint}")
        #Add expect_partial for lazy creation of weights
        model.load_weights(latest_checkpoint).expect_partial()
        model.summary()
        print("Checkpoint restored!")
    
    #load test dataset convert to numpy array
    print("Test Dataset loaded")
    #note, test is batched already
    test = tf.data.Dataset.load('./data/test_tf_dataset_instance')
    print(model.evaluate(test))

    evaluator = ModelEvaluator(model,p,test)
    evaluator.generate_all_predictions()
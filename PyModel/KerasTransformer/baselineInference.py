import tensorflow as tf
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from Dataset.SequenceDataset import SequenceDataset
import numpy as np
import argparse
import json
from CustomTransformer.params import Params
from midi_neural_preprocessor.processor import decode_midi
import os
from CustomTransformer.utils import custom_loss, custom_accuracy
from datetime import datetime
from CustomTransformer.LRSchedule import LRScheduler
from .baselineModel import createBaselineTransformer
import traceback
import sys
from Dataset.TestDataset import TestDataset



class Improvisor(tf.Module):
    def __init__(self,transformer_model,p:Params, **kwargs):
        super(Improvisor, self).__init__(**kwargs)
        self.model = transformer_model
        self.params = p
    
    def __call__(self, input_sequence):
        encoder_input = pad_sequences(input_sequence, maxlen=self.params.encoder_seq_len, padding='post')
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)

        start_token = tf.convert_to_tensor([self.params.token_sos],dtype=tf.int64)
        decoder_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0,start_token)
        
        i = 0
        print("Decoding....")
        while True:
            prediction = self.model((encoder_input, tf.transpose(decoder_output.stack())), training=False)
            print(prediction)
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

if __name__ == '__main__':
    #USE CPU
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
        latest_checkpoint = latest_checkpoint[:-2] + str(int(latest_checkpoint[-2:]) - args.patience)

        print(f"Latest Checkpoint path: {latest_checkpoint}")
        #Add expect_partial for lazy creation of weights
        model.load_weights(latest_checkpoint).expect_partial()
        print("Checkpoint restored!")
    
    improvisor = Improvisor(model,p)

    # =====================================================================
    # Test with very simple sequences of midi notes - not yet midi encoded
    # =====================================================================
    # _ , _ ,  test_data = test.mockTfDataset(test.MAJOR_SCALE, 12)

    # test_data = test_data.shuffle(len(test_data))
    # test_data = test_data.batch(p.batch_size, drop_remainder=True)
    # test_data = test_data.map(test.format_dataset)
    
    # for inputs, targets in test_data.take(1):
    #     encoder_inputs = inputs["encoder_inputs"]
    #     decoder_inputs = inputs["decoder_inputs"]
    #     decoder_outputs = targets

    # encoder_inputs = encoder_inputs.numpy()
    # decoder_inputs = decoder_inputs.numpy()
    # decoder_outputs = decoder_outputs.numpy()

    # for i in range(len(encoder_inputs)):
    #     print('encoderinput :{},\t decoder input:{},\t decoder output:{} \t, improvisor output:{} '.format(encoder_inputs[i],decoder_inputs[i],decoder_outputs[i],improvisor([encoder_inputs[i]])))

    #=====================================================================
    #Test with a single midi_encoded_file
    #=====================================================================
    
    #Grab test data from the same piece
    # _,_,test_data = test.mockTfDataset_from_encoded_midi('./data/processed/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle', p.encoder_seq_len)

    #Grab test data from another piece - same key
    dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2, num_files_by_split={'train':80,'validation':10,'test':10})
    _, _, test_data = dataset.tfDataset_from_encoded_midi(3)

    test_data = test_data.shuffle(len(test_data))
    test_data = test_data.batch(p.batch_size, drop_remainder=True)
    test_data = test_data.map(dataset.format_dataset)
    
    for inputs, targets in test_data.take(1):
        encoder_inputs = inputs["encoder_inputs"]
        decoder_inputs = inputs["decoder_inputs"]
        decoder_outputs = targets

    encoder_inputs = encoder_inputs.numpy()
    decoder_inputs = decoder_inputs.numpy()
    decoder_outputs = decoder_outputs.numpy()

    if not os.path.exists('./samples'):
        os.mkdir('./samples')

    try:
        for i in range(5):
            time_recorded = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            sample_path = './samples' + f'/{args.model_name}_{time_recorded}/'
            os.mkdir(sample_path)

            inputs = list(encoder_inputs[i])
            #Strip only end token
            targets = list(decoder_outputs[i][:-1])
            # Strip start and end tokens
            model_output = list(improvisor([encoder_inputs[i]]))[1:-1]
            
            #Write to files
            decode_midi(inputs,file_path=sample_path + 'input.midi')
            print('input.mid written')
            decode_midi(targets,file_path=sample_path + 'actual.midi')
            print('actual.mid written')
            decode_midi(model_output,file_path=sample_path + 'output.midi')
            print('output.mid written')

    except Exception as ex:
        os.rmdir(sample_path)
        print(ex)
        tb = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        print(traceback)
        
    print("Inference complete")
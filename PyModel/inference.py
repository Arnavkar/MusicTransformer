from Transformer.model import TransformerModel
import tensorflow as tf
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from CustomDataset import CustomDataset
import numpy as np
import argparse
import json
from Transformer.params import Params
from midi_neural_preprocessor.processor import decode_midi
import os

class Improvisor(tf.Module):
    def __init__(self,transformer_model,p:Params, **kwargs):
        super(Improvisor, self).__init__(**kwargs)
        self.model = transformer_model
        self.params = p
    
    def __call__(self, input_sequence):
        input_sequence[0] = [self.params.token_sos] + input_sequence[0] + [self.params.token_eos]

        encoder_input = pad_sequences(input_sequence, maxlen=self.params.encoder_seq_len, padding='post')
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)

        start_token = tf.convert_to_tensor([self.params.token_sos],dtype=tf.int64)
        decoder_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0,start_token)
        
        
        i = 0
        while True:
            prediction = self.model(encoder_input, tf.transpose(decoder_output.stack()), False)
            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = tf.argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][tf.newaxis]

            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)
            # Break if an <EOS> token is predicted
            if predicted_id == self.params.token_eos:
                break
            i+=1

        output = tf.transpose(decoder_output.stack())[0]
        decoder_output = decoder_output.mark_used()
        output = output.numpy()
        
        return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',"--model_name", type=str,required= True)
    args = parser.parse_args()

    model_params = json.load(open('./models/' + args.model_name + '/params.json', 'rb'))
    p = Params(model_params)
    dataset = CustomDataset(p)
    #instantiate model
    model = TransformerModel(p)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    latest_checkpoint = tf.train.latest_checkpoint('./models/' + args.model_name + "/checkpoints")    
    print(f"Latest Checkpoint path: {latest_checkpoint}")
    #Add expect_partial for lazy creation of weights
    checkpoint.restore(latest_checkpoint).expect_partial()

    improvisor = Improvisor(model,p)

    test_batchX,test_batchY = dataset.slide_seq2seq_batch(1, p.encoder_seq_len, 1, 'test')
    #extract a test sequence of the first 20 elements
    test_sequence = list(test_batchX[0][0:300])
    if not os.path.exists('./samples/'):
        os.mkdir('samples')
        
    print(f'test_sequence: {test_sequence}')
    decode_midi(test_sequence,file_path='samples/input_test.mid')

    output_sequence = list(improvisor([test_sequence]))
    print(f'output_sequence: {output_sequence}')
    decode_midi(output_sequence,file_path='samples/output_test.mid')

    print(f'actual_sequence: {test_batchX[0]}')
    decode_midi(test_batchX[0],file_path='samples/actual_test.mid')







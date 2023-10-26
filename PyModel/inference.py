from Transformer.model import TransformerModel
import tensorflow as tf
from Transformer.params import baseline_test_params, midi_test_params_v1, Params
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from CustomDataset import CustomDataset
import numpy as np

class Improvisor(tf.Module):
    def __init__(self,transformer_model,p:Params, **kwargs):
        super(Improvisor, self).__init__(**kwargs)
        self.transformer = transformer_model
        self.params = p
    
    def __call__(self, input_sequence):

        # Append start and end of string tokens to the input sentence
        input_sequence = np.insert(input_sequence, 0, self.params.token_sos, axis = 1)
        input_sequence = np.insert(input_sequence, len(input_sequence[0]), self.params.token_eos, axis = 1)
        print(input_sequence)
        encoder_input = pad_sequences(input_sequence, maxlen=p.encoder_seq_len, padding='post')
        print(encoder_input)
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)
        print(encoder_input)

        # # Prepare the output array of dynamic size
        # decoder_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        # decoder_output = decoder_output.write(0, self.params.token_sos) 

        # for i in range(p.decoder_seq_len):
        #     # Predict an output token
        #     prediction = self.transformer(encoder_input, tf.transpose(decoder_output.stack()), training=False)
 
        #     prediction = prediction[:, -1, :]
 
        #     # Select the prediction with the highest score
        #     predicted_id = tf.argmax(prediction, axis=-1)
        #     predicted_id = predicted_id[0][tf.newaxis]
 
        #     # Write the selected prediction to the output array at the next available index
        #     decoder_output = decoder_output.write(i + 1, predicted_id)
 
        #     # Break if an <EOS> token is predicted
        #     if predicted_id == output_end:
        #         break
 
        # output = tf.transpose(decoder_output.stack())[0]
        # output = output.numpy()
 
        # output_str = []
 
        # # Decode the predicted tokens into an output string
        # for i in range(output.shape[0]):
 
        #     key = output[i]
        #     #print(dec_tokenizer.index_word[key])
        #     output_str.append(dec_tokenizer.index_word[key])
 
        # return output_str

if __name__ == '__main__':
    p = Params(midi_test_params_v1)
    dataset = CustomDataset(p)

    #set dropout to 0 for inference
    p.dropout_rate = 0

    inference = TransformerModel(p)
    #Request sequence size from test dataset

    input_seq_len = 30
    test_batchX,test_batchY = dataset.slide_seq2seq_batch(1, input_seq_len, 1, 'test')
    inference.load_weights('./weights/wghts50.ckpt').expect_partial()
    improv = Improvisor(inference, p)
    improv(test_batchX)

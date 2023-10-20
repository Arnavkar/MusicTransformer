from Transformer.model import TransformerModel
import tensorflow as tf
from Transformer.params import baseline_test_params, Params
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers.legacy import Adam
from Transformer.LRSchedule import LRScheduler


p = Params(baseline_test_params)
p.dropout_rate = 0
p.encoder_seq_len = 7
p.decoder_seq_len = 12
p.encoder_vocab_size = 2225
p.decoder_vocab_size = 3458

#set dropout to 0 for inference
inference = TransformerModel(p)

class Translator(tf.Module):
    def __init__(self,transformer_model, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.transformer = transformer_model
    
    def load_tokenizer(self, name):
        with open(name, 'rb') as handle:
            return load(handle)
    
    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"
 
        # Load encoder and decoder tokenizers
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
 
        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input, maxlen=p.encoder_seq_len, padding='post')
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)
 
        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = tf.convert_to_tensor(output_start[0], dtype=tf.int64)
 
        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = tf.convert_to_tensor(output_end[0], dtype=tf.int64)
 
        # Prepare the output array of dynamic size
        decoder_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start) 
        
        for i in range(p.decoder_seq_len):
 
            # Predict an output token
            prediction = self.transformer(encoder_input, tf.transpose(decoder_output.stack()), training=False)
            print(prediction.shape)
 
            prediction = prediction[:, -1, :]
 
            # Select the prediction with the highest score
            predicted_id = tf.argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][tf.newaxis]
 
            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)
 
            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break
 
        output = tf.transpose(decoder_output.stack())[0]
        output = output.numpy()
 
        output_str = []
 
        # Decode the predicted tokens into an output string
        for i in range(output.shape[0]):
 
            key = output[i]
            #print(dec_tokenizer.index_word[key])
            output_str.append(dec_tokenizer.index_word[key])
 
        return output_str
    
sentence = ['im thirsty']

translator_untrained = Translator(inference)
inference.load_weights('neural_translator/weights/wghts16.ckpt').expect_partial()

translator_trained = Translator(inference)
print(translator_untrained(sentence))
print(translator_trained(sentence))


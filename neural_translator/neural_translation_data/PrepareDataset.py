from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

class PrepareDataset:
    def __init__(self, **kwargs):
        super(PrepareDataset, self).__init__(**kwargs)
        self.n_sentences = 10000  # Number of sentences to include in the dataset
        self.train_split = 0.8  # Ratio of the training data split
        self.val_split = 0.1  # Ratio of the training data split
 
    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)
        return tokenizer
    
    def save_tokenizer(self, tokenizer, name):
        with open( './data/neural_translation/' +name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)
 
    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)
 
    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)
        return len(tokenizer.word_index) + 1
    
    def encode_and_pad(self, dataset, tokenizer, seq_length):
        # Encode and pad the input sequences
        encoded = tokenizer.texts_to_sequences(dataset)
        padded = pad_sequences(encoded, maxlen=seq_length, padding='post')
        return tf.convert_to_tensor(padded, dtype=tf.int64)
 
    def __call__(self, filename, **kwargs):
        # Load a clean dataset
        clean_dataset = load(open(filename, 'rb'))
        
        # Reduce dataset size
        dataset = clean_dataset[:self.n_sentences, :]
        
        # Include start and end of string tokens
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"
        
        # Random shuffle the dataset
        shuffle(dataset)
        
        # Split the dataset
        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split):int(self.n_sentences * (1 - self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]
        
        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(train[:, 0])
        enc_seq_length = self.find_seq_length(train[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer(train[:, 1])
        dec_seq_length = self.find_seq_length(train[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        # Encode and pad the input sequences
        trainX = self.encode_and_pad(train[:, 0], enc_tokenizer, enc_seq_length)
        trainY = self.encode_and_pad(train[:, 1], dec_tokenizer, dec_seq_length)
        
        # Encode and pad the input sequences
        valX = self.encode_and_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_and_pad(val[:, 1], dec_tokenizer, dec_seq_length)

        self.save_tokenizer(enc_tokenizer, 'enc')
        self.save_tokenizer(dec_tokenizer, 'dec')

        # Save the test dataset separately
        savetxt('./data/neural_translation/test_dataset.csv', test, fmt='%s')

        return trainX, trainY, valX, valY, train, val, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size

if __name__ == "__main__":
    dataset = PrepareDataset()
    train_batchX, train_batchY, valX, valY, train_orig, val_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('data/neural_translation/english-german-both.pkl')
    print(f'Train X: {train_batchX[0]}')
    print(f'Train Y: {train_batchY[0]}')
    encoder_input_train = train_batchX
    print(f'encoder input : {encoder_input_train[0]}')
    decoder_input_train = train_batchY[:, :-1]
    print(f'decoder input: {decoder_input_train[0]}')
    decoder_output_train = train_batchY[:, 1:]
    print(f'decoder output: {decoder_output_train[0]}')
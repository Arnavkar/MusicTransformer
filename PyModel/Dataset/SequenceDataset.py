from .BaseDataset import BaseDataset
import tensorflow as tf
from Transformer.params import Params, midi_test_params_v2
import numpy as np
import random
import pickle
import time

class FileData():
    def __init__(self,file_path):
        self.path = file_path
        self.current_note_index = 0

class SequenceDataset(BaseDataset,tf.keras.utils.Sequence):
    def __init__(self, p:Params, mode,min_duration=None,min_event_length=None,num_files_to_use=None,logger=None):
        super().__init__(
            p=p, 
            mode=mode, 
            min_duration=min_duration, 
            min_event_length=min_event_length, 
            num_files_to_use=num_files_to_use, 
            logger=logger)
        self.convert_all_to_FileData()
        
    def __len__(self):
        return self.calculate_num_batches(self.params.encoder_seq_len, 1)
    
    def __getitem__(self, idx):
        x,y = self.seq2seq_batch(self.params.batch_size, self.params.encoder_seq_len)
        return (
            {
                "encoder_inputs": x,
                "decoder_inputs": y[:, :-1],
            },
            y[:, 1:],
        )

    def convert_all_to_FileData(self):
        self.log_or_print("Converting all files to FileData objects")
        for i in range(len(self.data)):
            self.data[i] = FileData(self.data[i])

    def get_batch(self, batch_size, length):
        data = []

        #select k files from the list of files - allows us to grab a batch from the same file multiple times
        for _ in range(batch_size):
            file = (random.choice(self.data))
            data.append(self.extract_sequence_v2(file, length))

        return np.array(data,int)
    
    #extract_sequence, grabs a sequence of size 'length' from file, starting at the start index 0. Then, shifts the sequence down by 1
    def extract_sequence_v2(self, file_data, length):
        #Grab a random sample of length len from a while
        with open(file_data.path, 'rb') as f:
            data = pickle.load(f)

        #pick up from the last recorded start_index
        start_index = file_data.current_note_index

        #if the start index + length is less than the file length, we can grab a sequence of length
        if start_index + length < len(data):
            #extract a sequence of length len from the file
            data = data[start_index:start_index + length] 

            #update the start index for the next sequence
            file_data.current_note_index += 1
        else:
            #if we either 1) perfectly hit the last event in the sequence with a full sequence 2) we hit the end of the file early and need to pad with zeros
            #if there is not enough data left in the file (only possible with stride > 1) then start from start index and take the remaining events in the file, padding the remaining sequence with zeros
            data = data[start_index:]
            while len(data) < length:
                data = np.append(data, self.params.pad_token)
            self.move_to_complete_list(file_data)
        return data
        
    #Move a fileData object from the model_data[mode] list to the complete_files list
    def move_to_complete_list(self,file_data):
        if file_data not in self.complete_files:
            self.complete_files.append(file_data)
        else:
            self.lof_or_print("File{file_data.path} already in complete_files list", isWarning=True)
        
        try:
            self.data.remove(file_data)
        except:
            self.log_or_print(f"File{file_data.path} was already removed from not data list", isWarning=True)

    #if training decoder, num_tokens_to_predict is 0, we teacher force the entire sequence
    #if training regular transformer, length and num_tokens_to_predict are the same

    #first half is passed to encoder as context
    #second half passed to decoder for teacher forcing
    def seq2seq_batch(self, batch_size, length, num_tokens_to_predict=None):
        if num_tokens_to_predict is None:
            num_tokens_to_predict = length
        data = self.get_batch(batch_size, length+num_tokens_to_predict)
        #Accounting for the eos token added in extract_sequence
        x = data[:, 0:length]
        y = data[:, length:length + num_tokens_to_predict]
        
        #attach start and end tokens to each y sequence
        y = np.array([[self.params.token_sos] + list(seq) + [self.params.token_eos] for seq in y])
        return x, y

    def calculate_num_batches(self, seq_len, stride):
        num_examples = 0
        for file in self.data:
            with open(file.path, 'rb') as f:
                data = pickle.load(f)
                #seq_len multiplied by two for the encoder and decoder seq respectively
            num_examples += (len(data) - seq_len*2) // stride
        num_batches = int(num_examples / self.params.batch_size)
        remaining_examples = num_examples % self.params.batch_size
        
        # print(f"Number of batches: {num_batches}, Number of remaining examples: {remaining_examples}")
        assert num_batches*self.params.batch_size + remaining_examples == num_examples

        return num_batches
    
    #Reset all indices
    def on_epoch_end(self):
        self.log_or_print("Epoch ended, resetting fileData objects")
        self.data += self.complete_files
        self.complete_files = []
        for file in self.data:
            file.current_note_index = 0
        random.shuffle(self.data)

if __name__ == "__main__":
    p = Params(midi_test_params_v2)

    data = SequenceDataset(p, 'train', min_event_length=p.encoder_seq_len*2, num_files_to_use=5)
    print(data)

    #============================================================================================
    #Testing out methods required for keras.utils.sequence
    #__getitem__ and __len__ methods
    #__calculate_num_batches method caluculates the number of batches in the dataset, which is needed for __len__
    #__getitem__ returns a batch of data, given an idx
    #============================================================================================
    num_batches = data.calculate_num_batches(p.encoder_seq_len, 1)
    print(num_batches)

    time.sleep(2)
    for i in range(p.epochs):
        print("Epoch:",i)
        for i in range(num_batches):
            batch = data.__getitem__(i)
            if i % 100 == 0:
                print(f'Batch: {i}')
                # print(f'encoder_input shape: {batch[0]["encoder_inputs"].shape}')
                # print(f'decoder_input shape: {batch[0]["decoder_inputs"].shape}')
                # print(f'decoder_output shape: {batch[1].shape}')
                for file in data.data:
                    print(file.path,file.current_note_index)
                print(f"Complete files:{len(data.complete_files)}=========")
                for file in data.complete_files:
                    print(file.path)
        print(len(data.complete_files))
        data.on_epoch_end()
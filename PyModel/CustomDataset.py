import os
import json
import random
import numpy as np
import pickle
from Transformer.params import baseline_test_params, midi_test_params_v1, Params
import tensorflow as tf

'''
Subclasses Kera Sequence object, must implement __getitem__ and __len__ methods

__getitem__ returns a batch of data, where the batch size is the first dimension of the returned array
'''

class CustomDataset():
    def __init__(self,p:Params, path="./data/processed") -> None:
        self.maestroJSON = self.get_maestroJSON()
        self.fileDict = self.get_encoded_files(path)
        self.model_data = {
            "train": [],
            "validation": [],
            "test": []
        }
        #self.partition_by_duration()
        self.partition_by_filecount()
        self.params = p
        self.batch_size = p.batch_size
        #self.num_steps = self.calculate_steps()
    
    def get_maestroJSON(self, path="./data/raw/maestro-v3.0.0.json") -> list:
        with open(path) as f:
            data = json.load(f)
        return data

    def __repr__(self):
        return "<class CustomDataset has {} files, {} for training, {} for validation, {} for testing>".format(str(len(self.fileDict)), str(len(self.model_data["train"])), str(len(self.model_data["validation"])), str(len(self.model_data["test"])))
    
    def get_encoded_files(self,path) -> list:
        if not os.path.exists(path):
            os.mkdir(path)

        #strip the year from the midi filename path, add .pickle to the end and add to base path
        lambda_func = lambda x: os.path.join(path, x.split('/')[-1] +'.pickle')

        #From the json files, get the indexed midi filenames
        midi_filenames_from_json = { key : lambda_func(value) for key, value in self.maestroJSON['midi_filename'].items()}

        #Check for duplicate midi filenames  ensure no duplicates found
        dup_dict = {}
        for key, value in midi_filenames_from_json.items():
            dup_dict.setdefault(value, set()).add(key)
        result = [key for key, values in dup_dict.items() if len(values) > 1]
        assert len(result) == 0, f"Duplicate midi filenames found: {result}"

        return midi_filenames_from_json

    def partition_by_filecount(self):
        total_num_files = len(self.fileDict)

        #gets values as paths, shuffle the file paths
        paths = list(self.fileDict.values())
        random.shuffle(paths)

        train = paths[:int(total_num_files * 0.8)]
        
        validation = paths[int(total_num_files * 0.8):int(total_num_files * 0.9)]
        
        test = paths[int(total_num_files * 0.9):]

        assert len(train) + len(validation) + len(test) == total_num_files, \
            f"Number of files in train, validation and test sets does not match total number of files: {len(train)} + {len(validation)} + {len(test)} != {total_num_files}"
        
        self.model_data["train"] = train
        self.model_data["validation"] = validation
        self.model_data["test"] = test
    
    #TODO: get_batch and slide_seq2seq_batch are from separate code repo, maybe use an alternative batch method??
    #Does this not present the problem of possibly training on the exact same sequence more than one, with no guarantee of using all the training data? 
    def get_batch(self, batch_size, length, mode):
        #select files based on training mode
        files = self.model_data[mode]

        #select k files from the list of files
        files = random.sample(files, batch_size)

        #from each file, extract a random sequence of length
        data = [self.extract_sequence(file, length) for file in files]
        
        for array in data:
            assert len(array) == length+1, f"Length of array {len(array)} is not equal to length {length+2}"
        return np.array(data,int)
    
    def extract_sequence(self, fname, length):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if length <= len(data):
            start = random.randrange(0,len(data) - length)
            data = data[start:start + length]
            data = np.append(data, self.params.token_eos)
        else:
            #concat EOS tokens to the end of the sequence
            data = np.append(data, self.params.token_eos)
            while len(data) < length+1:
                data = np.append(data, self.params.pad_token)
        return data
    
    def slide_seq2seq_batch(self, batch_size, length, num_tokens_predicted=1, mode='train'):
        assert num_tokens_predicted <= length, "Num tokens predicted must be less than length of sequence provided!"
        data = self.get_batch(batch_size, length, mode)
        #Accounting for the eos token added in extract_sequence
        x = data[:, :-num_tokens_predicted]
        y = data[:, num_tokens_predicted:]

        x=np.insert(x,0,self.params.token_sos,axis = 1)

        #no need to add an additonal eos token, already add in extract_sequence
        y=np.insert(y,0,self.params.token_sos,axis=1)
        return x, y
    
    def get_dataset_from_file(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        return tf.keras.utils.timeseries_dataset_from_array(data, 
                                                            None,
                                                            sequence_length = self.params.seq_len, 
                                                            shuffle=True)
        return data

if __name__ == "__main__":
    p = Params(midi_test_params_v1)
    dataset = CustomDataset(p)
    # print(dataset)
    train_batchX,train_batchY = dataset.slide_seq2seq_batch(64, 2048, 1,'train')
    # print(f'Train X: {train_batchX}')
    # print(f'Train Y: {train_batchY}')
    # encoder_input_train = train_batchX
    # print(f'encoder input : {encoder_input_train}')
    # decoder_input_train = train_batchY[:, :-1]
    # print(f'decoder input: {decoder_input_train}')
    # decoder_output_train = train_batchY[:, 1:]
    # print(f'decoder output: {decoder_output_train}')

    # found_dict = {}
    # for _ in range(20):
    #     data = dataset.get_batch(1000, 2048, 'train')

    #     for vec in data:
    #         for value in vec:
    #             if value not in found_dict:
    #                 found_dict[value] = 1
    # print(sorted(list(found_dict.keys())))

    numfiles = len(dataset.fileDict)
    print(f"Number of files: {numfiles}")
    total_events = 0
    maxlen = 0
    minlen = float('inf')
    lengths = []
    for file in dataset.fileDict.values():
        with open(file, 'rb') as f:
            encoded_midi_data = pickle.load(f)
            length = len(encoded_midi_data)
            lengths.append(length)
            if length < minlen: 
                minlen = length
            
            if length > maxlen: 
                maxlen = length

            total_events += length

    avg_event = total_events / numfiles
    print(f"Total number of events: {total_events}")
    print(f"Average number of events per file: {avg_event}")
    print(f"Max length: {maxlen}")
    print(f"Min length: {minlen}")
    #print(f"lengtht_lis: {sorted(lengths)}")

    #Grab a simple file and sample it accordingly to grab more from each file 
    

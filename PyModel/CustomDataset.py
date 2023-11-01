import os
import json
import random
import numpy as np
import pickle
from Transformer.params import baseline_test_params, midi_test_params_v1, Params
import tensorflow as tf
import json
from numba import jit, cuda

'''
Subclasses Kera Sequence object, must implement __getitem__ and __len__ methods

__getitem__ returns a batch of data, where the batch size is the first dimension of the resturned array
__
'''

class CustomDataset():
    def __init__(self,p:Params, path="./data/processed") -> None:
        random.seed(237)
        self.maestroJSON = self.get_maestroJSON()
        self.fileDict = self.get_encoded_files(path)
        self.model_data = {
            "train": [],
            "validation": [],
            "test": []
        }
        #self.partition_by_duration()
        self.params = p
        self.batch_size = p.batch_size
        self.partition_by_filecount()

        if self.params.record_data_stats:
            self.dataset_stats = {
                "train":{},
                "validation":{},
                "test":{}
            }
            self.set_up_stats()

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
        data = [self.extract_sequence(file, length, mode) for file in files]
        
        for array in data:
            assert len(array) == length+1, f"Length of array {len(array)} is not equal to length {length+2}"
        return np.array(data,int)
    
    def extract_sequence(self, fname, length, mode):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if length <= len(data):
            if mode == "test":
                start = 0
            else:
                start = random.randrange(0,len(data) - length)
            data = data[start:start + length]
            data = np.append(data, self.params.token_eos)
        else:
            start = 0
            #concat EOS tokens to the end of the sequence
            data = np.append(data, self.params.token_eos)
            while len(data) < length+1:
                data = np.append(data, self.params.pad_token)
        
        if self.params.record_data_stats:
            self.record_stats(fname, start, length, len(data),mode)

        return data
    
    def record_stats(self, fname, start, seq_len, data_len, mode):
        # stats_object = {
        #     "extracted_sequences":[],
        #     "min_idx":-1,
        #     "max_idx":-1,
        #     "percent_covered":-1
        # }
        #print(f"Recording stats for {fname} in {mode} mode with start {start} and seq_len {seq_len}")
        if fname not in self.dataset_stats[mode]:
            print(f"Stats object for {fname} not found, creating new stats object")
            stats = {
                "extracted_sequences":[],
                "min_idx":-1,
                "max_idx":-1,
                "visit_count":-1
            }
            self.dataset_stats[mode][fname] = stats

        extracted_sequences = self.dataset_stats[mode][fname]["extracted_sequences"]

        if len(extracted_sequences) == 0:
            self.dataset_stats[mode][fname]["min_idx"] = start
            self.dataset_stats[mode][fname]["max_idx"] = start + seq_len
            extracted_sequences.append((start, start + seq_len))
            self.dataset_stats[mode][fname]["visit_count"] = 1

        else:
            self.dataset_stats[mode][fname]["min_idx"] = min(self.dataset_stats[mode][fname]["min_idx"], start)
            self.dataset_stats[mode][fname]["max_idx"] = max(self.dataset_stats[mode][fname]["max_idx"], start + seq_len)
            extracted_sequences.append((start, start + seq_len))
            # for idx_pair in extracted_sequences:
            #     if start >= idx_pair[0] and start + seq_len <= idx_pair[1]:
            #         return
            #     elif start <= idx_pair[0] and start + seq_len >= idx_pair[1]:
            #         idx_pair[0] = start
            #         idx_pair[1] = start + seq_len
            #     elif start <= idx_pair[0] and start + seq_len <= idx_pair[1] and start + seq_len >= idx_pair[0]:
            #         idx_pair[0] = start
            #     elif start >= idx_pair[0] and start <= idx_pair[1] and start + seq_len >= idx_pair[1]:
            #         idx_pair[1] = start + seq_len
            #     elif start > idx_pair[1]:
            #         extracted_sequences.append([start, start + seq_len])
            #     elif start + seq_len < idx_pair[0]:
            #         extracted_sequences.append([start, start + seq_len])
            #     else:
            #         print("ERROR: Should not be here")
            self.dataset_stats[mode][fname]["visit_count"] +=1
    
    def set_up_stats(self):
        stats_object = {
                "extracted_sequences":[],
                "min_idx":float('inf'),
                "max_idx":-1,
                "visit_count":-1
            }
        for path in self.model_data["train"]:
            self.dataset_stats["train"][path] = stats_object.copy()

        for path in self.model_data["validation"]:
            self.dataset_stats["validation"][path] = stats_object.copy()

        for path in self.model_data["test"]:
            self.dataset_stats["test"][path] = stats_object.copy()

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

if __name__ == "__main__":
    '''stats test, using code from training loop'''
    # p = Params(midi_test_params_v1)
    # p.record_data_stats = True
    # p.encoder_seq_len = p.decoder_seq_len = 50
    # dataset = CustomDataset(p)

    

    # for step in range(len(dataset.fileDict) // p.batch_size):
    #     train_batchX,train_batchY = dataset.slide_seq2seq_batch(p.batch_size, p.encoder_seq_len, 1, 'train')

    # with open("./statsv2.json", 'w') as f:
    #     json.dump(dataset.dataset_stats, f, indent=4)

    # print("collecting averages")
    # unique_avg = 0
    # count = 0
    # for file_name, train_file_stats in dataset.dataset_stats["train"].items():
    #     if train_file_stats["visit_count"] > 0:
    #         unique_avg += 1
    #     count += 1
    # unique_avg /= count
    # print(f"Running average of files covered after 1 epoch: {unique_avg}")

    '''Testing out slide_seq2seq - checking average number of events per midi file'''
    # print(dataset)
    # train_batchX,train_batchY = dataset.slide_seq2seq_batch(64, 2048, 1,'train')
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

    # numfiles = len(dataset.fileDict)
    # print(f"Number of files: {numfiles}")
    # total_events = 0
    # maxlen = 0
    # minlen = float('inf')
    # lengths = []
    # for file in dataset.fileDict.values():
    #     with open(file, 'rb') as f:
    #         encoded_midi_data = pickle.load(f)
    #         length = len(encoded_midi_data)
    #         lengths.append(length)
    #         if length < minlen: 
    #             minlen = length
            
    #         if length > maxlen: 
    #             maxlen = length

    #         total_events += length

    # avg_event = total_events / numfiles
    # print(f"Total number of events: {total_events}")
    # print(f"Average number of events per file: {avg_event}")
    # print(f"Max length: {maxlen}")
    # print(f"Min length: {minlen}")
    #print(f"lengtht_lis: {sorted(lengths)}")

    #Grab a simple file and sample it accordingly to grab more from each file 
    

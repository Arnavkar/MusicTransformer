import os
import json
import random
import numpy as np
import pickle
from Transformer.params import midi_test_params_v2, Params
import tensorflow as tf
import json

#==================================================================
#Custom Dataset 
# 
# Subclasses Kera Sequence object, must implement __getitem__ and __len__ methods ?
# __getitem__ returns a batch of data, where the batch size is the first dimension of the resturned array
# __len__ returns the number of batches in the sequence
#==================================================================


class FileData():
    def __init__(self,file_path):
        self.file_path = file_path
        self.current_note_index = 0

class CustomDataset():
    def __init__(self,p:Params, path="./data/processed") -> None:
        self.maestroJSON = self.get_maestroJSON()
        self.fileDict = self.get_encoded_files(path)
        self.model_data = {
            "train": [],
            "validation": [],
            "test": []
        }
        self.params = p
        self.batch_size = p.batch_size
        self.partition_by_filecount()

        # if self.params.record_data_stats:
        #     self.dataset_stats = {
        #         "train":{},
        #         "validation":{},
        #         "test":{}
        #     }
        #     self.set_up_stats()
    
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

        #Convert all into FileData objects
        #paths = [FileData(path) for path in paths]

        #Split paths via list splicing into train, validation and test sets
        train = paths[:int(total_num_files * 0.8)]
        
        validation = paths[int(total_num_files * 0.8):int(total_num_files * 0.9)]
        
        test = paths[int(total_num_files * 0.9):]

        assert len(train) + len(validation) + len(test) == total_num_files, \
            f"Number of files in train, validation and test sets does not match total number of files: {len(train)} + {len(validation)} + {len(test)} != {total_num_files}"
        
        self.model_data["train"] = train
        self.model_data["validation"] = validation
        self.model_data["test"] = test
            
    #NOTE: Taken from Github implementation of Transformer - currently using construct tf_dataset_method
    def get_batch(self, batch_size, length, mode):
        #select files based on training mode
        files = self.model_data[mode]

        #select k files from the list of files
        files = random.sample(files, batch_size)

        #from each file, extract a random sequence of length
        data = [self.extract_sequence(file, length, mode) for file in files]
        
        for array in data:
            assert len(array) == length-1, f"Length of array {len(array)} is not equal to length {length-1}"
        return np.array(data,int)
    
    #NOTE: Taken from Github implementation of Transformer - currently using construct tf_dataset_method
    def extract_sequence(self, fname, length, mode):
        #Grab a random sample of length len from a while
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if length <= len(data):
            #If in test mode, always start from the beginning of the file, else start from a random index
            if mode == "test":
                start = 0
            else:
                start = random.randrange(0,len(data) - length)
            #extract a sequence of length len from the file and append the EOS token to the end
            data = data[start:start + length-2]
            data = np.append(data, self.params.token_eos)
        else:
            #if there is not enough data in the file, start from the beginning of the file
            start = 0
            #concat EOS tokens to the end of the sequence
            data = np.append(data, self.params.token_eos)
            #pad the sequence with pad tokens until the length is equal to the length parameter
            while len(data) < length-1:
                data = np.append(data, self.params.pad_token)

        if self.params.record_data_stats:
            self.record_stats(fname, start, length, len(data),mode)
        return data

    #NOTE: Taken from Github implementation of Transformer
    def slide_seq2seq_batch(self, batch_size, length, mode='train', num_tokens_to_predict = None):
        if num_tokens_to_predict is None:
            num_tokens_to_predict = length
        data = self.get_batch(batch_size, length+num_tokens_to_predict, mode)
        #Accounting for the eos token added in extract_sequence
        x = data[:, :-num_tokens_to_predict]
        y = data[:, num_tokens_to_predict-1:]

        #no need to add an additonal eos token, already add in extract_sequence
        y=np.insert(y,0,self.params.token_sos,axis=1)
        return data, x, y
    
    #Construct a tf dataset directly from the midi files                   
    def construct_tf_dataset(self,mode,seq_len,stride = 1):
        current_file_index = 0
        current_note_index = 0
        paths = self.model_data[mode]
        x, y = [],[]
        #while looping through list of files
        while current_file_index < len(paths):
            fp = paths[current_file_index]
            # print(fp)
            with open(fp, 'rb') as f: 
                f = open(fp, 'rb')
                test_data = pickle.load(f)

            #while looping through each file, grab a sequence of length seq_len 
            #stride sets the distance between each grabbed sequence
            #This while loop ensures we don't grab sequences that are too short, they are discarded
            while current_note_index + seq_len*2 + 1 < len(test_data):

                x_start = current_note_index
                x_end = current_note_index + seq_len
                sequence_x = test_data[x_start:x_end]

                y_start = x_end 
                y_end = x_end + seq_len
                sequence_y = test_data[y_start:y_end]
                sequence_y = np.insert(sequence_y,0,self.params.token_sos)
                sequence_y = np.append(sequence_y,self.params.token_eos)
                x.append(sequence_x)
                y.append(sequence_y)
                current_note_index += stride
            
            print(f'File {current_file_index} at path {fp} complete: {current_note_index} notes processed out of {len(test_data)}')
            #move to next file
            current_file_index += 1
            current_note_index = 0
        
        assert len(x) == len(y), f"num of x samples should equal to y samples, {len(x)} != {len(y)}"
        for i in range(len(x)):
            assert len(x[i]) == len(y[i])-2, f"Length of x elem should be 2 less than y elem: {len(x[i])} != {len(y[i])}"
            # assert x[i][0] == self.params.token_sos, f"First token of x elem at {i} is not sos token: {x[i][0]} != {self.params.token_sos}"
            assert y[i][0] == self.params.token_sos, f"First token of y elem at {i} is not sos token: {y[i][0]} != {self.params.token_sos}"
            #assert x[i][-1] == self.params.token_eos, f"Last token of x elem at {i} is not eos token: {x[i][-1]} != {self.params.token_eos}"
            assert y[i][-1] == self.params.token_eos, f"Last token of y elem at {i} is not eos token: {y[i][-1]} != {self.params.token_eos}"
        #     assert (x[i][2:-1] == y[i][1:-2]).all(), f"Sequence of x elem at {i} is not equal to sequence of y elem at {i}: {x[i][2:-1]} != {y[i][1:-2]}"

        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        path = os.path.join("./data/tf_midi_data_" + mode + "_new")
        tf.data.Dataset.save(dataset, path)
        print(f"Dataset saved at {path}")


if __name__ == "__main__":
    '''Construct and load dataset witf tf.data.Dataset'''
    p = Params(midi_test_params_v2)
    dataset = CustomDataset(p)
    # dataset.construct_tf_dataset('train', p.encoder_seq_len, p.encoder_seq_len)
    # dataset.construct_tf_dataset('validation', p.encoder_seq_len, p.encoder_seq_len)

    #============================================================================================
    # Testing out slide_seq2seq - checking average number of events per midi file
    #============================================================================================
    # for step in range(len(dataset.fileDict) // p.batch_size):
    #     data, train_batchX,train_batchY = dataset.slide_seq2seq_batch(1, 10,'train',1)
    #     for i in range(len(data)):
    #         print(f'Data: {data[i]}')
    #         print(f'Train X: {train_batchX[i]}')
    #         print(f'Train Y: {train_batchY[i]}')
    #         encoder_input_train = train_batchX[i]
    #         print(f'encoder input : {encoder_input_train}')
    #         decoder_input_train = train_batchY[i][:-1]
    #         print(f'decoder input: {decoder_input_train}')
    #         decoder_output_train = train_batchY[i][1:]
    #         print(f'decoder output: {decoder_output_train}')
            
            # test_list = list(encoder_input_train[1:]) + list(decoder_output_train)
            # assert all(x == y for x, y in zip(test_list, list(data[i]))) == True

    #============================================================================================
    #Code to collect dataset statistics:
    # - total/avg num events per file
    # -  min and max length files
    # - memory taken up by all data in memory
    #============================================================================================
    # numfiles = len(dataset.fileDict)
    # print(f"Number of files: {numfiles}")
    # total_events = 0
    # maxlen = 0
    # minlen = float('inf')
    # lengths = []
    # accumulated_events = []
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
    #         accumulated_events += encoded_midi_data

    # avg_event = total_events / numfiles
    # acc1 = accumulated_events[0:int(len(accumulated_events)/2)]
    # acc2 = accumulated_events[int(len(accumulated_events)/2):]
    # print(f"Total number of events: {total_events}")
    # print(f"Average number of events per file: {avg_event}")
    # print(f"Max length: {maxlen}")
    # print(f"Min length: {minlen}")
    # print(f"all_data_in_memory size: {sys.getsizeof(accumulated_events)}")
    # print(f"all_data_in_memory size (split in 2): {sys.getsizeof(acc1) + sys.getsizeof(acc2)}")
    #Grab a simple file and sample it accordingly to grab more from each file 
    
    #============================================================================================
    #Deprecated class methods to record statistics
    #============================================================================================
   
    #NOTE: Used to record statistics about get_batch and extract_sequence, both not in use currently
    # def record_stats(self, fname, start, seq_len, data_len, mode):
    #     if fname not in self.dataset_stats[mode]:
    #         print(f"Stats object for {fname} not found, creating new stats object")
    #         stats = {
    #             "extracted_sequences":[],
    #             "min_idx":-1,
    #             "max_idx":-1,
    #             "visit_count":-1
    #         }
    #         self.dataset_stats[mode][fname] = stats

    #     extracted_sequences = self.dataset_stats[mode][fname]["extracted_sequences"]

    #     if len(extracted_sequences) == 0:
    #         self.dataset_stats[mode][fname]["min_idx"] = start
    #         self.dataset_stats[mode][fname]["max_idx"] = start + seq_len
    #         extracted_sequences.append((start, start + seq_len))
    #         self.dataset_stats[mode][fname]["visit_count"] = 1

    #     else:
    #         self.dataset_stats[mode][fname]["min_idx"] = min(self.dataset_stats[mode][fname]["min_idx"], start)
    #         self.dataset_stats[mode][fname]["max_idx"] = max(self.dataset_stats[mode][fname]["max_idx"], start + seq_len)
    #         extracted_sequences.append((start, start + seq_len))
    #         for idx_pair in extracted_sequences:
    #             if start >= idx_pair[0] and start + seq_len <= idx_pair[1]:
    #                 return
    #             elif start <= idx_pair[0] and start + seq_len >= idx_pair[1]:
    #                 idx_pair[0] = start
    #                 idx_pair[1] = start + seq_len
    #             elif start <= idx_pair[0] and start + seq_len <= idx_pair[1] and start + seq_len >= idx_pair[0]:
    #                 idx_pair[0] = start
    #             elif start >= idx_pair[0] and start <= idx_pair[1] and start + seq_len >= idx_pair[1]:
    #                 idx_pair[1] = start + seq_len
    #             elif start > idx_pair[1]:
    #                 extracted_sequences.append([start, start + seq_len])
    #             elif start + seq_len < idx_pair[0]:
    #                 extracted_sequences.append([start, start + seq_len])
    #             else:
    #                 print("ERROR: Should not be here")
    #         self.dataset_stats[mode][fname]["visit_count"] +=1

    #NOTE: Used to record statistics about get_batch and extract_sequence, both not in use currently
    # def set_up_stats(self):
    #     stats_object = {
    #             "extracted_sequences":[],
    #             "min_idx":float('inf'),
    #             "max_idx":-1,
    #             "visit_count":-1
    #         }
    #     for path in self.model_data["train"]:
    #         self.dataset_stats["train"][path] = stats_object.copy()

    #     for path in self.model_data["validation"]:
    #         self.dataset_stats["validation"][path] = stats_object.copy()

    #     for path in self.model_data["test"]:
    #         self.dataset_stats["test"][path] = stats_object.copy()

    #============================================================================================
    # Code to collect stats from deprecated stats methods
    #============================================================================================
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
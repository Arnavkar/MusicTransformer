import os
import json
import random
import numpy as np
import pickle
from Transformer.params import midi_test_params_v2, Params
import tensorflow as tf
import json
import string
import time

#==================================================================
#Custom Dataset 
# 
# Subclasses Keras Sequence object, must implement __getitem__ and __len__ methods ?
# __getitem__ returns a batch of data, where the batch size is the first dimension of the resturned array
# __len__ returns the number of batches in the sequence
#==================================================================

class FileData():
    def __init__(self,file_path):
        self.path = file_path
        self.current_note_index = 0

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self,p:Params,mode, path="./data/processed", min_duration=None,min_event_length=None) -> None:
        self.maestroJSON = self.get_maestroJSON()
        self.fileDict = self.get_encoded_files(path,min_duration,min_event_length)
        self.mode = mode
        self.data = []
        self.complete_files = []
        self.params = p

        # self.partition_by_filecount()
        self.retrieve_files_by_maestro_split()

        random.seed(self.params.seed)
    
    def get_maestroJSON(self, path="./data/raw/maestro-v3.0.0/maestro-v3.0.0.json") -> list:
        with open(path) as f:
            data = json.load(f)
        return data

    def __repr__(self):
        return "<class CustomDataset_{} has {} files>".format(self.mode, str(len(self.data)))
    
    def get_encoded_files(self,path,min_duration,min_event_length) -> list:
        if not os.path.exists(path):
            os.mkdir(path)

        #strip the year from the midi filename path, add .pickle to the end and add to base path
        lambda_func = lambda x: os.path.join(path, x.split('/')[-1] +'.pickle')

        #From the json files, get the indexed midi filenames
        midi_filenames_from_json = { int(key) : lambda_func(value) for key, value in self.maestroJSON['midi_filename'].items()}

        #Check for duplicate midi filenames  ensure no duplicates found
        dup_dict = {}
        for key, value in midi_filenames_from_json.items():
            dup_dict.setdefault(value, set()).add(key)
        result = [key for key, values in dup_dict.items() if len(values) > 1]
        assert len(result) == 0, f"Duplicate midi filenames found: {result}"

        #Remove midi files that are too short in length by duration
        if min_duration is not None:
            i = 0
            while i < len(midi_filenames_from_json):
                duration = self.maestroJSON['duration'][f'{i}']
                if duration < min_duration:
                    print(f"Removed file {midi_filenames_from_json[i]}, Duration of is {duration}, which is less than {min_duration}")
                    del midi_filenames_from_json[i]
                i+=1
            print(f"Number of midi files remaining after removing files less than {min_duration} seconds: {len(midi_filenames_from_json)}")
        
        #Remove midi files that are too short in length by number of events
        if min_event_length is not None:
            i = 0
            while i < len(midi_filenames_from_json):
                with open(midi_filenames_from_json[i], 'rb') as f:
                    data = pickle.load(f)
                if len(data) < min_event_length:
                    print(f"Removed file {midi_filenames_from_json[i]}, Length of is {len(data)}, which is less than {min_event_length}")
                    del midi_filenames_from_json[i]
                i+=1
            print(f"Number of midi files remaining after removing files less than {min_event_length} events: {len(midi_filenames_from_json)}")

        return midi_filenames_from_json

    # def partition_by_filecount(self):
    #     total_num_files = len(self.fileDict)

    #     #gets values as paths, shuffle the file paths
    #     paths = list(self.fileDict.values())
    #     random.shuffle(paths)

    #     #Split paths via list splicing into train, validation and test sets
    #     train = paths[:int(total_num_files * 0.8)]
        
    #     validation = paths[int(total_num_files * 0.8):int(total_num_files * 0.9)]
        
    #     test = paths[int(total_num_files * 0.9):]

    #     assert len(train) + len(validation) + len(test) == total_num_files, \
    #         f"Number of files in train, validation and test sets does not match total number of files: {len(train)} + {len(validation)} + {len(test)} != {total_num_files}"
        
    #     self.model_data["train"] = train
    #     self.model_data["validation"] = validation
    #     self.model_data["test"] = test
    
    def retrieve_files_by_maestro_split(self):
        for i in self.fileDict.keys():
            if self.maestroJSON['split'][f'{i}'] == self.mode:
                self.data.append(FileData(self.fileDict[i]))
            
    def get_batch(self, batch_size, length):
        selected = []

        #select k files from the list of files - allows us to grab a batch from the same file multiple times
        for _ in range(batch_size):
            selected.append(random.choice(self.data))

        #from each file, extract a random sequence of length
        data = [self.extract_sequence_v2(file, length) for file in selected]
        return np.array(data,int)
    
    # def extract_sequence(self, fname, length, mode):
    #     #Grab a random sample of length len from a while
    #     with open(fname, 'rb') as f:
    #         data = pickle.load(f)
    #     if length <= len(data):
    #         #If in test mode, always start from the beginning of the file, else start from a random index
    #         if mode == "test":
    #             start = 0
    #         else:
    #             start = random.randrange(0,len(data) - length)
    #         #extract a sequence of length len from the file and append the EOS token to the end
    #         data = data[start:start + length-2]
    #         data = np.append(data, self.params.token_eos)
    #     else:
    #         #if there is not enough data in the file, start from the beginning of the file
    #         start = 0
    #         #concat EOS tokens to the end of the sequence
    #         data = np.append(data, self.params.token_eos)
    #         #pad the sequence with pad tokens until the length is equal to the length parameter
    #         while len(data) < length-1:
    #             data = np.append(data, self.params.pad_token)

    #     # if self.params.record_data_stats:
    #     #     self.record_stats(fname, start, length, len(data),mode)

    #     return data
    
    #v2 of extract_sequence, grabs a sequence of size 'length' from file, starting at the start index 0. Then, shifts the sequence down by 1
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
        self.complete_files.append(file_data)
        self.data.remove(file_data)

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
    
    #Reset all indices
    def on_epoch_end(self):
        print("Epoch ended, resetting fileData objects")
        self.data += self.complete_files
        self.complete_files = []
        for file in self.data:
            file.current_note_index = 0

    #Construct a tf dataset directly from the midi files - loads everything into memory                
    # def construct_tf_dataset(self,mode,seq_len,stride = 1,for_baseline=False,num_files=None):
    #     current_file_index = 0
    #     current_note_index = 0
    #     if num_files is not None:
    #         paths = self.model_data[mode][:num_files]
    #     else:
    #         paths = self.model_data[mode]
    #     x, y = [],[]
    #     #while looping through list of files
    #     while current_file_index < len(paths):
    #         file_data = paths[current_file_index]
    #         # print(fp)
    #         with open(file_data.file_path, 'rb') as f: 
    #             test_data = pickle.load(f)

    #         #while looping through each file, grab a sequence of length seq_len 
    #         #stride sets the distance between each grabbed sequence
    #         #This while loop ensures we don't grab sequences that are too short, they are discarded
    #         while current_note_index + seq_len*2 + 1 < len(test_data):
    #             x_start = current_note_index
    #             x_end = current_note_index + seq_len
    #             sequence_x = test_data[x_start:x_end]

    #             y_start = x_end 
    #             y_end = x_end + seq_len
    #             sequence_y = test_data[y_start:y_end-1]
    #             sequence_y = np.insert(sequence_y,0,self.params.token_sos)
    #             sequence_y = np.append(sequence_y,self.params.token_eos)
    #             x.append(sequence_x)
    #             y.append(sequence_y)
    #             current_note_index += stride
            
    #         print(f'File {current_file_index} at path {file_data.file_path} complete: {current_note_index} notes processed out of {len(test_data)}')
    #         #move to next file
    #         current_file_index += 1
    #         current_note_index = 0

    #     # assert len(x) == len(y), f"num of x samples should equal to y samples, {len(x)} != {len(y)}"
    #     # for i in range(len(x)):
    #     #     assert len(x[i]) == len(y[i]), f"Length of x elem should be 2 less than y elem: {len(x[i])} != {len(y[i])}"
    #     #     # assert x[i][0] == self.params.token_sos, f"First token of x elem at {i} is not sos token: {x[i][0]} != {self.params.token_sos}"
    #     #     assert y[i][0] == self.params.token_sos, f"First token of y elem at {i} is not sos token: {y[i][0]} != {self.params.token_sos}"
    #     #     #assert x[i][-1] == self.params.token_eos, f"Last token of x elem at {i} is not eos token: {x[i][-1]} != {self.params.token_eos}"
    #     #     assert y[i][-1] == self.params.token_eos, f"Last token of y elem at {i} is not eos token: {y[i][-1]} != {self.params.token_eos}"
    #     #     assert (x[i][2:-1] == y[i][1:-2]).all(), f"Sequence of x elem at {i} is not equal to sequence of y elem at {i}: {x[i][2:-1]} != {y[i][1:-2]}"

    #     dataset = tf.data.Dataset.from_tensor_slices((x,y))
    #     dataset_path = f"./data/tf_midi_" + mode + f"_{seq_len}_{stride}"
    #     print("Tf dataset constructed")
    #     if for_baseline == True:
    #         path += "_baseline"
    #         dataset = dataset.map(self.__format_dataset)
 
    #     if os.path.exists(path):
    #         print('Dataset already exists - do you want to overwrite? Y/N')
    #         char = input().lower()
    #         while char not in ['y','n']:
    #             print('Invalid input - do you want to overwrite? Y/N')
    #             char = input().lower()
    #         if char == 'n':
    #             #Append a random string and save this data set so as to not overwrite the original
    #             res = ''.join(random.choices(string.ascii_uppercase +
    #                          string.digits, k=7))
    #             path = path + res
    #             tf.data.Dataset.save(dataset, path)
    #         else:
    #             tf.data.Dataset.save(dataset, path)
            
    #         print(f"Dataset saved at {path}")
        
    #     return path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    '''Construct and load dataset with tf.data.Dataset'''
    p = Params(midi_test_params_v2)

    data = CustomDataset(p, 'train', min_event_length=p.max_seq_len)
    print(data)

    #============================================================================================
    #Testing out methods required for keras.utils.sequence
    #__getitem__ and __len__ methods
    #__calculate_num_batches method caluculates the number of batches in the dataset, which is needed for __len__
    #__getitem__ returns a batch of data, given an idx
    #============================================================================================
    num_batches = data.calculate_num_batches(p.encoder_seq_len, 1)
    time.sleep(2)
    for i in range(num_batches+1):
        print(i,"===============================")
        batch = data.__getitem__(i)
        print(f'encoder_input shape: {batch[0]["encoder_inputs"].shape}')
        print(f'decoder_input shape: {batch[0]["decoder_inputs"].shape}')
        print(f'decoder_output shape: {batch[1].shape}')
        for file in data.data:
            print(file.path,file.current_note_index)
        print(len(data.complete_files))
    print(len(data.complete_files))

    # num_train_files = len(data.model_data['train'])
    # print(f"Number of Train files: {num_train_files}")
    # num_batches_generated = 0
    # while len(data.complete_files) < num_train_files:
    #     batch = data.__getitem__(0)
    #     print(f"Batch shape: {batch[0]['encoder_inputs'].shape}")
    #     print(f"Batch shape: {batch[0]['decoder_inputs'].shape}")
    #     print(f"Batch shape: {batch[1].shape}")
    #     num_batches_generated += 1
    #     print(len(data.complete_files))

    # print(f"Number of batches generated: {num_batches_generated}")

    #============================================================================================
    # Testing out construct_tf_dataset - very memory intensive to store all possible sequences in memory directly
    #============================================================================================
    # train_path = data.construct_tf_dataset(f'train', p.encoder_seq_len, 1,for_baseline=True,num_files=10)
    # val_path = data.construct_tf_dataset(f'validation', p.encoder_seq_len, 1,for_baseline=True,num_files=2)
    # test_path = data.construct_tf_dataset(f'test', p.encoder_seq_len, 1,for_baseline=True,num_files=2)

    # train = tf.data.Dataset.load(train_path)
    # val = tf.data.Dataset.load(val_path)
    # test = tf.data.Dataset.load(test_path)

    # print("TRAIN DATASET------------------")
    # for inputs, targets in train.take(1):
    #     print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    #     print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    #     print(f"targets.shape: {targets.shape}")

    # print("VALIDATION DATASET------------------")
    # for inputs, targets in val.take(1):
    #     print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    #     print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    #     print(f"targets.shape: {targets.shape}")

    # print("TEST DATASET------------------")
    # for inputs, targets in test.take(1):
    #     print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    #     print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    #     print(f"targets.shape: {targets.shape}")

    #============================================================================================
    # Testing out slide_seq2seq - checking average number of events per midi file
    #============================================================================================
    # for step in range(len(dataset.fileDict) // p.batch_size):
    #     data, train_batchX,train_batchY = dataset.seq2seq_batch(1, 10,'train',1)
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
    # print(sorted(lengths))

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
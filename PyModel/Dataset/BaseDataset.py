import os
import json
import random
import numpy as np
import pickle
from CustomTransformer.params import midi_test_params_v2, Params
import json
from time import time

class BaseDataset():
    def __init__(self, p:Params,
                path="./data/processed",
                data_format="pickle",
                min_duration=None,
                min_event_length=None,
                logger = None):
        
        self.logger = logger
        self.data_format = data_format
        # if self.data_format == 'npy':
        #     path += "_numpy"

        self.maestroJSON = self.get_maestroJSON()
        self.fileDict = self.get_encoded_files(path,min_duration,min_event_length)
        self.params = p
        
        random.seed(self.params.seed)


    def get_maestroJSON(self, path="./data/raw/maestro-v3.0.0.json"):
        with open(path) as f:
            data = json.load(f)
        return data
    
    def get_encoded_files(self,path,min_duration,min_event_length):
        if not os.path.exists(path):
            os.mkdir(path)

        #strip the year from the midi filename path, add .pickle to the end and add to base path
        lambda_func = lambda x: os.path.join(path, x.split('/')[-1] +'.' + 'pickle')

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
                    self.log_or_print(f"Removed file {midi_filenames_from_json[i]}, Duration of is {duration}, which is less than {min_duration}")
                    del midi_filenames_from_json[i]
                i+=1

            self.log_or_print(f"Number of midi files remaining after removing files less than {min_duration} seconds: {len(midi_filenames_from_json)}")
        
        #Remove midi files that are too short in length by number of events
        if min_event_length is not None:
            i = 0
            while i < len(midi_filenames_from_json):
                with open(midi_filenames_from_json[i], 'rb') as f:
                    if self.data_format == "pickle":
                        data = pickle.load(f)
                    else:
                        data = np.load(f,allow_pickle=True)
                if len(data) < min_event_length:
                    self.log_or_print(f"Removed file {midi_filenames_from_json[i]}, Length of is {len(data)}, which is less than {min_event_length}")
                    del midi_filenames_from_json[i]
                i+=1
            self.log_or_print(f"Number of midi files remaining after removing files less than {min_event_length} events: {len(midi_filenames_from_json)}")

        return midi_filenames_from_json
    
    def log_or_print(self,log_str,isWarning=False):
        if self.logger:
            if isWarning:
                self.logger.warning(log_str)
            else:
                self.logger.info(log_str)
        else:
            print(log_str)
    
    def format_dataset(self,x, y):
        return (
            {
                "encoder_inputs": x,
                "decoder_inputs": y[:, :-1],
            },
            y[:, 1:],
        )

if __name__ == "__main__":
    #============================================================================================
    #Testing creating base dataset - testing speed of np files vs lists
    #============================================================================================
    p = Params(midi_test_params_v2)

    start = time()
    dataset = BaseDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2)
    end = time()
    print(f"Time taken to create dataset with npy format: {end-start}")  

    start = time()
    dataset = BaseDataset(p, data_format='pickle', min_event_length=p.encoder_seq_len*2)
    end = time()
    print(f"Time taken to create dataset with pickle format: {end-start}")

    
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
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
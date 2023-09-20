import os
import json
import random
import itertools
import numpy as np
import pickle

class CustomDataset:
    def __init__(self) -> None:
        self.maestroJSON = self.get_maestroJSON()
        self.fileDict = self.get_encoded_files()
        self.model_data = {
            "train": [],
            "validation": [],
            "test": []
        }

    def get_maestroJSON(self, path="./data/raw/maestro-v3.0.0/maestro-v3.0.0.json") -> list:
        with open(path) as f:
            data = json.load(f)
        return data
    
    def get_encoded_files(self, path="./data/processed") -> list:
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
   
    #TODO: Implement partitioning by duration
    def partition_by_duration(self):
        #Assert that number of files listed in the json = number of processed files
        assert len(self.maestroJSON['duration']) == len(self.fileDict), \
            f"Number of files in json: {len(self.maestroJSON['duration'])} does not match number of processed files: {len(self.fileDict)}"
        total_duration = sum(self.maestroJSON['duration'].values())

        train_duration = total_duration * 0.8
        validation_duration = test_duration = total_duration * 0.1

    def partition_by_filecount(self):
        total_num_files = len(self.fileDict)

        #gets values as paths, shuffle the file paths
        paths = list(self.fileDict.values())
        random.shuffle(paths)

        train = paths[:int(total_num_files * 0.8)]
        
        validation = paths[int(total_num_files * 0.8):int(total_num_files * 0.9)]
        
        test = paths[int(total_num_files * 0.9):]

        assert len(train) + len(validation) + len(test) == total_num_files, \
            f"Number of files in train, validation and test sets does not match total number of files: {len(train_dict)} + {len(validation_dict)} + {len(test_dict)} != {total_num_files}"
        
        self.model_data["train"] = train
        self.model_data["validation"] = validation
        self.model_data["test"] = test
    
    #supply batch size, length and mode (train, validation, test)
    def get_batch(self, batch_size, length, mode):
        #select files based on training mode
        files = self.model_data[mode]

        #select k files from the list of files
        files = random.sample(files, batch_size)

        data = [self.extract_sequence(file, length) for file in files]

        #Possibly convert to tensor directly?
        return np.array(data)

    # def extract_sequence(self,file, length):
    #     with open(file, 'rb') as f:
    #         encoded_midi_data = pickle.load(f)
        
    #     if length == None:
    #         return encoded_midi_data
    #     else:

        
dataset = CustomDataset()
dataset.partition_by_filecount()

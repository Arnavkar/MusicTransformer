import tensorflow as tf
import random
import pickle
import numpy as np
from .BaseDataset import BaseDataset
from CustomTransformer.params import Params, midi_test_params_v2
import resource
import sys
import time
import os

MAJOR_SCALE = [24, 26, 28, 29, 31, 33, 35]
MINOR_SCALE = [24, 26, 27, 29, 31, 32, 34]
MAJOR_ARPEGGIO_7 = [24, 28, 31, 35]
MINOR_ARPEGGIO_7 = [24, 27, 31, 34]
MAX_VAL = 127


def memory_limit(percent):
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * percent), hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory  # KiB

class TestDataset(BaseDataset):
    def __init__(self, 
                p:Params,
                data_format="pickle",
                min_duration=None,
                min_event_length=None,
                logger=None,
                num_files_by_split=None):
        
        super().__init__(
            p=p,  
            data_format=data_format,
            min_duration=min_duration, 
            min_event_length=min_event_length, 
            logger=logger)
        
        self.data = {
            'train':[],
            'validation':[],
            'test':[]
        }
        self.retrieve_files_by_maestro_split()

        if num_files_by_split != None:
            self.data['train'] = self.data['train'][0:num_files_by_split['train']]
            self.data['validation'] = self.data['validation'][0:num_files_by_split['validation']]
            self.data['test'] = self.data['test'][0:num_files_by_split['test']]
        
    def constructScales(self,scale):
        num_iterations = (MAX_VAL - scale[-1]) // 12
        single_scale = []

        for i in range(num_iterations):
            for note in scale:
                single_scale.append(note + i*12)

        all_scales = [[note + i for note in single_scale] for i in range(12)]
        return single_scale, all_scales
    
    def mockTfDataset_from_scale(self, scale, seq_len, stride=1):
        single_scale, all_scales  = self.constructScales(scale)
        all_sequences = self.rolling_window(all_scales, seq_len*2, stride)
        random.shuffle(all_sequences)
        x, y = [], []
        for seq in all_sequences:
            x.append(seq[:seq_len])
            y.append([1] + seq[seq_len:] + [2])

        split_1 = int(0.8*len(x))
        split_2 = int(0.9*len(x))

        train_x , val_x, test_x = x[0:split_1], x[split_1:split_2], x[split_2:]
        train_y , val_y, test_y = y[0:split_1], y[split_1:split_2], y[split_2:]
        
        train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        return train,val,test

    def rolling_window(self,sequence, seq_len, stride=1):
        ls = []
        for i in range(0,len(sequence) - seq_len + 1,stride):
            ls.append(sequence[i: i + seq_len])
        return ls

    def get_all_sequences_by_split(self,split, seq_len,stride=1):
        print("Getting all sequences for split:{}".format(split))
        all_sequences = []
        counter = 0
        for path in self.data[split]:
            with open(path, 'rb') as f:
                event_sequence = pickle.load(f)
            sequences = self.rolling_window(event_sequence, seq_len * 2, stride)
            all_sequences.extend(sequences)
            counter+=1
            print("Finished processing {} files".format(counter))

        return all_sequences
    
    def rolling_window_np(self,seq,seq_len, stride=1):
        # Convert each list to a NumPy array of type uint16
        ls = []
        #if sequence is a list and not a numpy array , convert to numpy array
        if type(seq) == list:
            seq = np.array(seq, dtype=np.uint16)

        for i in range(0,len(seq) - seq_len + 1,stride):
            ls.append(seq[i: i + seq_len])
        return np.array(ls)
    
    def get_all_sequences_by_split_np(self,split, seq_len,stride=1):
        print("Getting all sequences for split:{}".format(split))
        all_sequences = []
        counter = 0
        for path in self.data[split]:
            with open(path, 'rb') as f:
                event_sequence = np.load(f,allow_pickle=True)
            sequences = self.rolling_window_np(event_sequence, seq_len * 2, stride)
            all_sequences.append(sequences)
            counter+=1
            print("Finished processing {} files".format(counter))

        # Concatenate all sequences and shuffle
        all_sequences_np = np.concatenate(all_sequences, axis=0)
        return all_sequences_np
    
    def mockTfDataset_from_encoded_midi_np(self, stride=1):
        datasets = {}

        for key in self.data.keys():
            sequences = self.get_all_sequences_by_split_np(key, self.params.encoder_seq_len, stride)
            x = sequences[:, :self.params.encoder_seq_len]
            y = np.pad(sequences[:, self.params.encoder_seq_len:], 
                       ((0, 0), (1, 1)), 
                       mode='constant', 
                       constant_values=(1, 2))

            datasets[key] = tf.data.Dataset.from_tensor_slices((x, y))

        return datasets['train'], datasets['validation'], datasets['test']

    def mockTfDataset_from_encoded_midi_pickle(self, stride=1):
        datasets = {}

        for key in self.data.keys():
            sequences = self.get_all_sequences_by_split(key, self.params.encoder_seq_len, stride)
            x = [seq[:self.params.encoder_seq_len] for seq in sequences]
            y = [[1] + seq[self.params.encoder_seq_len:] + [2] for seq in sequences]
            datasets[key] = tf.data.Dataset.from_tensor_slices((x, y))
        
        return datasets['train'], datasets['validation'], datasets['test']
    
    def mockTfDataset_from_encoded_midi(self, stride=1):
        if self.data_format == 'pickle':
            return self.mockTfDataset_from_encoded_midi_pickle(stride)
        elif self.data_format == 'npy':
            return self.mockTfDataset_from_encoded_midi_np(stride)
        else:
            raise Exception("Invalid data format")
        
    def mockTfDataset_from_encoded_midi_path(self, path ,stride=1):
        with open(path, 'rb') as f:
            event_sequence = np.load(f,allow_pickle=True)
        sequences = self.rolling_window_np(event_sequence, self.params.encoder_seq_len * 2, stride)
        x = [seq[:self.params.encoder_seq_len] for seq in sequences]
        y = [[1] + seq[self.params.encoder_seq_len:] + [2] for seq in sequences]
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        return dataset
    
    def retrieve_files_by_maestro_split(self):
        for i in self.fileDict.keys():
            if self.maestroJSON['split'][f'{i}'] == 'train':
                self.data['train'].append(self.fileDict[i])
            elif self.maestroJSON['split'][f'{i}'] == 'validation':
                self.data['validation'].append(self.fileDict[i])
            elif self.maestroJSON['split'][f'{i}'] == 'test':
                self.data['test'].append(self.fileDict[i])
            else:
                raise Exception(f"Invalid mode found: {self.maestroJSON['split'][f'{i}']}")
        
    def __repr__(self) -> str:
        return "<TestDataset has {} files for training, {} files for validation, {} files for testing>".format(len(self.data['train']),len(self.data['validation']),len(self.data['test']))


if __name__ == '__main__':
    p = Params(midi_test_params_v2)
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    # ==================Test creating scales for easier problem==================
    # dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2)
    # train,val,test, = dataset.mockTfDataset_from_scale(MAJOR_SCALE, 12, 2)

    # ==================Test rolling window efficiency (numpy vs list)==================
    #Compare time between rolling window and rolling window npwith encoded midi data

    dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2)
    with open('./data/processed/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle', 'rb') as f:
        event_sequence = pickle.load(f)

    start = time.time()
    all_sequences = dataset.rolling_window(event_sequence, 1000,1)
    end = time.time()
    print("Time taken for rolling window with encoded midi data:{}".format(end - start))

    start = time.time()
    all_sequences_np = dataset.rolling_window_np(event_sequence, 1000,1)
    end = time.time()
    print("Time taken for rolling window np with encoded midi data:{}".format(end - start))

    # ==================Test tf dataset from single file==================
    # dataset = TestDataset(p, data_format='npy')

    # p.encoder_seq_len = 128
    # train = dataset.mockTfDataset_from_encoded_midi_path('./data/processed_numpy/piano_train.mid.npy', 1)
    # val = dataset.mockTfDataset_from_encoded_midi_path('./data/processed_numpy/piano_test.mid.npy', 1)
    # print(len(train))
    # print(len(val))

    # # ==================Test tf dataset==================
    # start = time.time()
    # dataset = TestDataset(p, data_format='pickle', min_event_length=p.encoder_seq_len*2, num_files_by_split={'train':8,'validation':2,'test':2})
    # #limit num files considered
    # print(dataset)
    # train,val,test = dataset.mockTfDataset_from_encoded_midi(2)
    # end = time.time()
    # print("Time taken for creating tf dataset:{}".format(end - start))

    # start = time.time()
    # dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2, num_files_by_split={'train':8,'validation':2,'test':2})

    # print(dataset)
    # train_np,val_np,test_np = dataset.mockTfDataset_from_encoded_midi_np(2)
    # end = time.time()
    # print("Time taken for creating tf dataset:{}".format(end - start))

    # # train = train.shuffle(len(train))
    # train = train.batch(2, drop_remainder=True)
    # train = train.map(dataset.format_dataset)

    # # val = val.shuffle(len(val))
    # val = val.batch(2, drop_remainder=True)
    # val = val.map(dataset.format_dataset)

    # # train_np = train_np.shuffle(len(train_np))
    # train_np = train_np.batch(2, drop_remainder=True)
    # train_np = train_np.map(dataset.format_dataset)

    # # val_np = val_np.shuffle(len(val_np))
    # val_np = val_np.batch(2, drop_remainder=True)
    # val_np = val_np.map(dataset.format_dataset)

    # for x , x_np in zip(train.take(len(train)), train_np.take(len(train_np))):
    #     print(x), print(x_np)
    #     print("encoder Inputs are equal:{}".format(np.array_equal(x[0]['encoder_inputs'],x_np[0]['encoder_inputs'])))
    #     print("decoder inputs are equal:{}".format(np.array_equal(x[0]['decoder_inputs'],x_np[0]['decoder_inputs'])))
    #     print("Targets are equal:{}".format(np.array_equal(x[1],x_np[1])))

    # print(len(train))
    # print(len(train_np))

    # #Check that the output is infact the same by taking 1 batch from each dataset
    # for (inputs, targets),(inputs_np,targets_np) in zip(train.take(1), train_np.take(1)):
        
    #     print("Inputs are equal:{}".format(np.array_equal(inputs,inputs_np)))
    #     print("Targets are equal:{}".format(np.array_equal(targets,targets_np)))

    
    #========Test if we can construct fully in memory dataset with np using stride of 2================
    # memory_limit(0.8)
    # strategy = tf.distribute.MirroredStrategy()
    
    # with strategy.scope():
    #     try:
    #         dataset = TestDataset(p, data_format='npy', min_event_length=p.encoder_seq_len*2, num_files_by_split={'train':120,'validation':15,'test':15})

    #         train_np,val_np,test_np = dataset.mockTfDataset_from_encoded_midi_np(2)
    #     except MemoryError:
    #         print("Memory error raised")
    #         print("Memory usage:{}".format(get_memory()))
    #         sys.exit(1)







    # train,val,test = mockTfDataset_from_encoded_midi('./data/processed/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle', 128)
    # train = train.shuffle(len(train))
    # train = train.batch(16, drop_remainder=True)
    # train = train.map(format_dataset)

    # # val = val.shuffle(len(test))
    # val = val.batch(16, drop_remainder=True)
    # val = val.map(format_dataset)

    # # test = test.shuffle(len(test))
    # test = test.batch(16, drop_remainder=True)
    # test = test.map(format_dataset)

    # print("Train dataset==============")
    # for inputs, targets in train.take(1):
    #     test_sequences = inputs["encoder_inputs"]
    #     actual_sequences = targets
    #     for i in range(3):
    #         print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))

    # print("Val dataset==============")
    # for inputs, targets in val.take(1):
    #     test_sequences = inputs["encoder_inputs"]
    #     actual_sequences = targets
    #     for i in range(3):
    #         print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))

    # print("Test dataset==============")
    # for inputs, targets in test.take(1):
    #     test_sequences = inputs["encoder_inputs"]
    #     actual_sequences = targets
    #     for i in range(3):
    #         print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))


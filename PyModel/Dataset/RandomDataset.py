from .BaseDataset import BaseDataset
from CustomTransformer.params import Params, midi_test_params_v2
import tensorflow as tf
import numpy as np
import random
import pickle

class RandomCropDataset(BaseDataset):
    def __init__(self, p: Params, mode, min_duration=None, min_event_length=None, num_files_to_use=None, logger=None):
        super().__init__(p=p, min_duration=min_duration, min_event_length=min_event_length, logger=logger)
        self.data = []
        self.mode = mode
        self.num_files_to_use = num_files_to_use
        self.retrieve_files_by_maestro_split()
        self.stats = {}

        random.shuffle(self.data)
        if num_files_to_use is not None:
            self.data = self.data[:num_files_to_use]

    def retrieve_files_by_maestro_split(self):
        if self.mode not in ["train", "validation", "test"]:
            raise Exception(f"Invalid mode passed: {self.mode}")

        for i in self.fileDict.keys():
            if self.maestroJSON['split'][f'{i}'] == self.mode:
                self.data.append(self.fileDict[i])

    def make_gen_callable(self,_gen):
        def gen():
            for x,y in _gen:
                 yield x,y
        return gen
                
    def get_batch(self,length, collect_stats=False):
        batch_data = []
        for _ in range(self.params.batch_size):
            file_data = random.choice(self.data)
            sequence = self.extract_random_crop(file_data,length, collect_stats=collect_stats)
            batch_data.append(sequence)
        return np.array(batch_data,int)
    
    def batch_generator(self,length,collect_stats=False):
        return self.make_gen_callable(self._batch_generator(length,collect_stats=collect_stats))
    
    def _batch_generator(self,length, num_tokens_to_predict=None, collect_stats=False):
        if num_tokens_to_predict is None:
            num_tokens_to_predict = length
        while True:
            data = self.get_batch(length+num_tokens_to_predict, collect_stats=collect_stats)

            x = data[:, 0:length]
            y = data[:, length:length+num_tokens_to_predict]

            #attach start and end tokens to each y sequence
            y = np.array([[self.params.token_sos] + list(seq) + [self.params.token_eos] for seq in y])
            yield self.format_dataset(x,y)

    def extract_random_crop(self, file_data, length, collect_stats=False):
        with open(file_data, 'rb') as f:
            data = pickle.load(f)

        start_index = random.randint(0, len(data))
        end_index = start_index + length

        # Handle the padding scenario
        #If the end index is greater than the length of the file and  the pad length is less than half the window size, pad the sequence with 0s
        #Otherwise, recursively find a new random crop
        if end_index > len(data):
            pad_length = end_index - len(data)
            if pad_length <= length / 2:
                sequence = np.pad(data[start_index:], (0, pad_length), 'constant', constant_values=0)
            else:
                return self.extract_random_crop(file_data,length, collect_stats)  # Recursively find a new random crop
        else:
            sequence = data[start_index:end_index]

        if collect_stats:
            if file_data not in self.stats:

                self.stats[file_data] = [len(data),(start_index, end_index)]
            else:
                self.stats[file_data] = self.stats[file_data].append((start_index, end_index))

        return sequence

    def __repr__(self):
        return "<RandomCropDataset_{} has {} files>".format(self.mode, len(self.data))
    
if __name__ == "__main__":
    p = Params(midi_test_params_v2)
    p.batch_size =2

    train_dataset = RandomCropDataset(p, 'train', min_event_length=p.encoder_seq_len*2,logger=None)

    train = train_dataset._batch_generator(10)
    for _ in range(5): 
        example = next(train)
        print(example)
        print(example[0]["encoder_inputs"].shape)
        print(example[0]["decoder_inputs"].shape)
        print(example[1].shape)

    train = train_dataset.batch_generator(10)

    #convert to tf.data.Dataset
    train = tf.data.Dataset.from_generator(
            train,
            output_signature=(
            {
                'encoder_inputs':tf.TensorSpec(shape=(p.batch_size,10), dtype=tf.int32),
                'decoder_inputs':tf.TensorSpec(shape=(p.batch_size,11), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(p.batch_size,11), dtype=tf.int32))

        )
    for ex in train.take(5):
        print(ex)

    #=======================
    #Test to check stats
    #=======================
    train = train_dataset._batch_generator(1000,collect_stats=True)
    for _ in range(1000): 
        example = next(train)
    
    for k,v in train_dataset.stats.items():
        if v == None:
            print(k)


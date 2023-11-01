from itertools import islice
from CustomDataset import CustomDataset
from Transformer.params import baseline_test_params, midi_test_params_v1, Params
import pickle


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result

    for elem in it:
        result = result[1:] + (elem,)
        yield result

# class sequenceGenerator:
#     def __init__(self,data,seq_len,stride):
#         self.data = data
#         self.stride = stride
#         self.seq_len = seq_len
#         self.max = return math.ceil(le))

#     def __iter__(self):
#         self





if __name__ == "__main__":
    p = Params(midi_test_params_v1)
    dataset = CustomDataset(p)
    batch_size = 2
    seq_len = 5

    current_file_index = 0
    current_note_index = 0

    test_paths = dataset.model_data["train"][0:10]
    print(test_paths)
    batch_x, batch_y = [],[]

            
    # for test_path in test_paths:
    #     with open(test_path, 'rb') as f:
    #         test_data = pickle.load(f)
    #     print(test_path)
    #     print(len(test_data))

    # for i in window(test_data,5):
    #    pass
    
    while True and current_file_index < len(test_paths):
        fp = test_paths[current_file_index]
        print(fp)
        with open(fp, 'rb') as f: 
            test_data = pickle.load(f)
            if current_file_index == 0:
                test_data = test_data[len(test_data)-7:]

        while current_note_index + seq_len + 1 < len(test_data):
            sequence_x = test_data[current_note_index:current_note_index+seq_len]
            sequence_y = test_data[current_note_index+1:current_note_index+seq_len+1]
            batch_x.append(sequence_x)
            batch_y.append(sequence_y)
            current_note_index += seq_len

            if len(batch_x) == batch_size:
                break
        
        if len(batch_x) == batch_size:
            break
        else:
            current_file_index += 1
            current_note_index = 0
            print(f'current_file_index: {current_file_index}')

    print(test_paths[:2])
    with open(test_paths[0], 'rb') as f1: 
        test_data1 = pickle.load(f1)

    with open(test_paths[1], 'rb') as f2: 
        test_data2 = pickle.load(f2)
    
    print(f'test data_1: {test_data1[len(test_data1)-7:]}')
    print(f'test data_2: {test_data2[:15]}')
    print(f'batch_x: {batch_x}')
    print(f'batch_y: {batch_y}')
    print(len(test_data1))
    print(len(test_data2))
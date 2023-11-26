import tensorflow as tf
import random
import pickle

MAJOR_SCALE = [24, 26, 28, 29, 31, 33, 35]
MINOR_SCALE = [24, 26, 27, 29, 31, 32, 34]
MAJOR_ARPEGGIO_7 = [24, 28, 31, 35]
MINOR_ARPEGGIO_7 = [24, 27, 31, 34]
MAX_VAL = 127

def constructScales(scale):
    num_iterations = (MAX_VAL - scale[-1]) // 12
    single_scale = []

    for i in range(num_iterations):
        for note in scale:
            single_scale.append(note + i*12)

    all_scales = [[note + i for note in single_scale] for i in range(12)]
    return single_scale, all_scales

def rolling_window(sequence_batch, seq_len):
    ls = []
    for seq in sequence_batch:
        for i in range(len(seq) - seq_len + 1):
            ls.append(seq[i: i + seq_len])
    return ls

def mockTfDataset(scale, seq_len):
    random.seed(42)
    single_scale, all_scales  = constructScales(scale)
    all_sequences = rolling_window(all_scales, seq_len*2)
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

def mockTfDataset_from_encoded_midi(path, seq_len):
    random.seed(42)
    with open(path, 'rb') as f:
        event_sequence = pickle.load(f)
    print("Len of event sequence:{}".format(len(event_sequence)))
    all_sequences = rolling_window([event_sequence], seq_len*2)
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

def format_dataset(x, y):
    return (
        {
            "encoder_inputs": x,
            "decoder_inputs": y[:, :-1],
        },
        y[:, 1:],
    )

if __name__ == '__main__':
    single_scale, all_scales = constructScales(MAJOR_SCALE)
    train,val,test, = mockTfDataset(MAJOR_SCALE, 12)
    train,val,test = mockTfDataset_from_encoded_midi('./data/processed/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle', 128)
    # train = train.shuffle(len(train))
    train = train.batch(16, drop_remainder=True)
    train = train.map(format_dataset)

    # val = val.shuffle(len(test))
    val = val.batch(16, drop_remainder=True)
    val = val.map(format_dataset)

    # test = test.shuffle(len(test))
    test = test.batch(16, drop_remainder=True)
    test = test.map(format_dataset)

    print("Train dataset==============")
    for inputs, targets in train.take(1):
        test_sequences = inputs["encoder_inputs"]
        actual_sequences = targets
        for i in range(3):
            print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))

    print("Val dataset==============")
    for inputs, targets in val.take(1):
        test_sequences = inputs["encoder_inputs"]
        actual_sequences = targets
        for i in range(3):
            print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))

    print("Test dataset==============")
    for inputs, targets in test.take(1):
        test_sequences = inputs["encoder_inputs"]
        actual_sequences = targets
        for i in range(3):
            print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))


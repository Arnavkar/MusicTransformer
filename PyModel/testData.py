import tensorflow as tf

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
    single_scale, all_scales  = constructScales(scale)
    all_sequences = rolling_window(all_scales, seq_len*2)
    x, y = [], []
    for seq in all_sequences:
        x.append(seq[:seq_len])
        y.append([1] + seq[seq_len:] + [2])
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    return dataset

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
    dataset = mockTfDataset(MAJOR_SCALE, 12)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.map(format_dataset)

    for inputs, targets in dataset.take(1):
        test_sequences = inputs["encoder_inputs"]
        actual_sequences = targets
        for i in range(10):
            print('test sequence:{}, actual sequence:{}'.format(test_sequences[i],actual_sequences[i]))


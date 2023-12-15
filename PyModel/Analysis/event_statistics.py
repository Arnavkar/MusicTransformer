import numpy as np
from Dataset.TestDataset import TestDataset
from Transformer.params import midi_test_params_v2, Params
import pickle
import json

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

range_note_on = range(0, RANGE_NOTE_ON)
range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF,RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)

def collect_split_statistics(dataset):
    split_data = {}
    all_lengths = []
    for split in dataset.data.keys():
        total_events = 0
        split_lengths = []
        for file in dataset.data[split]:
            with open(file, 'rb') as f:
                encoded_midi_data = pickle.load(f)
                length = len(encoded_midi_data)
                split_lengths.append(length)
                all_lengths.append(length)
            total_events += length

        split_data[split] = {
            "avg_events_per_file": total_events // len(dataset.data[split]),
            "max_events": max(split_lengths),
            "min_events": min(split_lengths),
            "std_dev": np.std(split_lengths)
        }
    
    split_data["All"] = {
        "avg_events_per_file": sum(all_lengths) // len(dataset.fileDict),
        "max_events": max(all_lengths),
        "min_events": min(all_lengths),
        "std_dev": np.std(all_lengths)

    }
    return split_data

def decode_event(event):

    valid_value = event

    if event in range_note_on:
        return f'note_on_{valid_value}'
    
    elif event in range_note_off:
        valid_value -= RANGE_NOTE_ON
        return f'note_off_{valid_value}'
    
    elif event in range_time_shift:
        valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
        return f'time_shift_{valid_value}'
    
    else:
        valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
        return f'velocity_{valid_value}'
    
def collect_event_count_statistics(dataset):
    event_count_dict = {}

    for i in range(RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL):
        event_count_dict[i] = 0

    for file in dataset.fileDict.values():
        with open(file, 'rb') as f:
            encoded_midi_data = pickle.load(f)
            for event in encoded_midi_data:
                event_count_dict[event] += 1
    
    event_count_dict = {decode_event(k): v for k, v in sorted(event_count_dict.items(), key=lambda item: item[0], reverse=True)}
    return event_count_dict

def check_event_count_dict(event_count_dict):
    for i in range(128):
        assert event_count_dict[f'note_on_{i}'] == event_count_dict[f'note_off_{i}']
        if i < 21 or i > 108:
            assert event_count_dict[f'note_on_{i}'] == 0
    
if __name__ == '__main__':
    p = Params(midi_test_params_v2)
    dataset = TestDataset(p, data_format='pickle')
    split_data = collect_split_statistics(dataset)
    print('Split data: ', json.dumps(split_data,indent=4))

    event_dict = collect_event_count_statistics(dataset)
    check_event_count_dict(event_dict)
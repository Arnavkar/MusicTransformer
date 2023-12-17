import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import json
from midi_neural_preprocessor.processor import encode_midi, decode_midi

test_input_path = './samples/baseline_80_files_np_28-11-2023_16-34-13/input.midi'
test_input_path = './samples/big_model_v2_07-11-2023_21-10-22/input.mid'
test_input_seq = encode_midi(test_input_path)

test_actual_path = './samples/baseline_80_files_np_28-11-2023_16-34-13/actual.midi'
test_actual_path = './samples/big_model_v2_07-11-2023_21-10-22/actual.mid'
test_actual_seq = encode_midi(test_actual_path)

test_output_path = './samples/baseline_80_files_np_28-11-2023_16-34-13/output.midi'
test_output_path = './samples/big_model_v2_07-11-2023_21-10-22/output.mid'
test_output_seq = encode_midi(test_output_path)

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

range_note_on = range(0, RANGE_NOTE_ON)
range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF,RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)

class Analysis:
    def __init__(self):
        self.seq_data = {
            'input': [],
            'output': [],
            'actual': []
        }
        self.pitch_list = [ "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G","G#/Ab", "A", "A#/Bb", "B"]
        self.pitch_class_dict = {i: self.pitch_list[i] for i in range(len(self.pitch_list))}
    
    def load_encoded_MIDI_seq(self, input_seq, output_seq, actual_seq):
        self.seq_data['input'] = input_seq
        self.seq_data['output'] = output_seq
        self.seq_data['actual'] = actual_seq
        # self.seq_data['input'].append(input_seq)
        # self.seq_data['output'].append(output_seq)
        # self.seq_data['actual'].append(actual_seq)

    def decode_single_event(self,event):
        valid_value = event
        if event in range_note_on:
            return ('note_on', valid_value)
        
        elif event in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return ('note_off', valid_value)
        
        elif event in range_time_shift:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
            return ('time_shift', valid_value)
        
        else:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
            return ('velocity', valid_value)

    def get_histograms(self):
        histograms_from_seq = {key: self.calculate_event_distribution(self.seq_data[key]) for key in self.seq_data}
        return histograms_from_seq
    
    def calculate_event_distribution(self,seq):
        note_on_counts = [0] * 12
        note_off_counts = [0] * 12
        time_shift_counts = [0] * 100
        velocity_counts = [0] * 32

        for event in seq:
            event_type, event_value = self.decode_single_event(event)
            if event_type == 'note_on':
                pitch_class = event_value % 12
                note_on_counts[pitch_class] += 1
            elif event_type == 'note_off':
                pitch_class = event_value % 12
                note_off_counts[pitch_class] += 1
            elif event_type == 'time_shift':
                time_shift_counts[event_value] += 1
            elif event_type == 'velocity':
                velocity_counts[event_value] += 1
            else:
                raise Exception("Invalid event type decoded")
        #normalize by sum of all notes - to turn into a probability distribution
        note_on_counts = np.array(note_on_counts) / np.sum(note_on_counts)
        note_off_counts = np.array(note_off_counts) / np.sum(note_off_counts)
        time_shift_counts = np.array(time_shift_counts) / np.sum(time_shift_counts)
        velocity_counts = np.array(velocity_counts) / np.sum(velocity_counts)

        #create dictionaries to be used for histogram
        note_on_dict = {self.pitch_class_dict[i]: note_on_counts[i] for i in range(len(note_on_counts))}
        note_off_dict = {self.pitch_class_dict[i]: note_off_counts[i] for i in range(len(note_off_counts))}

        time_shift_dict = {i: time_shift_counts[i] for i in range(len(time_shift_counts))}
        velocity_dict = {i: velocity_counts[i] for i in range(len(velocity_counts))}
        return {'note_on': note_on_dict, 'note_off': note_off_dict, 'time_shift': time_shift_dict, 'velocity': velocity_dict}
    
    def get_entropies(self,histograms):
        input_histograms, output_histograms, actual_histograms = histograms['input'], histograms['output'], histograms['actual']
        all_entropies = {
            'input': {},
            'output': {},
            'actual': {}
        }
        for key,histogram in input_histograms.items():
            all_entropies["input"][key] = sum([histogram[k] * np.log2(histogram[k]) for k in histogram.keys() if histogram[k] > 0])
        for key,histogram in output_histograms.items():
            all_entropies["output"][key] = sum([histogram[k] * np.log2(histogram[k]) for k in histogram.keys() if histogram[k] > 0])
        for key,histogram in actual_histograms.items():
            all_entropies["actual"][key] = sum([histogram[k] * np.log2(histogram[k]) for k in histogram.keys() if histogram[k] > 0])
        return all_entropies

    def plot_values(self,histograms,entropies,name):
        for key,histogram in histograms.items():
            plt.bar(histogram.keys(), histogram.values())
            plt.title(f'{name}-{key} (Entropy value: {entropies[key]})')
            plt.show()

if __name__ == '__main__':
    analysis = Analysis()
    analysis.load_encoded_MIDI_seq(test_input_seq, test_output_seq, test_actual_seq)
    histograms = analysis.get_histograms()
    entropies = analysis.get_entropies(histograms)
    print(json.dumps(entropies, indent=4))

    # analysis.plot_values(pitch_histograms, pitch_entropies,"Pitch")
    # analysis.plot_values(timeshift_histograms, timeshift_entropies,"Timeshift")
    # analysis.plot_values(velocity_histograms, velocity_entropies,"Velocity")

    
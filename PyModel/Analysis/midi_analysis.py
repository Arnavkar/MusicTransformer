import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import json
from midi_neural_preprocessor.processor import encode_midi, decode_midi

test_input_path = './samples/baseline_80_files_np_28-11-2023_16-34-13/input.midi'
test_input_seq = encode_midi(test_input_path)

test_actual_path = './samples/baseline_80_files_np_28-11-2023_16-34-13/actual.midi'
test_actual_seq = encode_midi(test_actual_path)

test_output_path = './samples/baseline_80_files_np_28-11-2023_16-34-13/output.midi'
test_output_seq = encode_midi(test_output_path)

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

range_note_on = range(0, RANGE_NOTE_ON)
range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF,RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)

class ModelAnalysis:
    def __init__(self):
        self.midi_data = None
        self.seq_data = None
        self.pitch_list = [ "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G","G#/Ab", "A", "A#/Bb", "B"]
        self.pitch_class_dict = {i: self.pitch_list[i] for i in range(len(self.pitch_list))}
    
    def load_MIDI_paths(self, inputMIDI, outputMIDI, actualMIDI):
        self.midi_data = {
            'input': pretty_midi.PrettyMIDI(inputMIDI),
            'output': pretty_midi.PrettyMIDI(outputMIDI),
            'actual': pretty_midi.PrettyMIDI(actualMIDI)
        }
        self.seq_data = {
            'input': encode_midi(inputMIDI),
            'output': encode_midi(outputMIDI),
            'actual': encode_midi(actualMIDI)
        }
    
    def load_encoded_MIDI_seq(self, inputMIDI, outputMIDI, actualMIDI):
        self.seq_data = {
            'input': inputMIDI,
            'output': outputMIDI,
            'actual': actualMIDI
        }

        self.midi_data = {
            'input': decode_midi(inputMIDI),
            'output': decode_midi(outputMIDI),
            'actual': decode_midi(actualMIDI)
        }

    def get_pitch_histograms(self):
        histograms_from_midi = {}
        histograms_from_seq = {}
        for key in self.midi_data:
            histograms_from_midi[key] = self.calculate_pitch_distribution_midi(self.midi_data[key])
            histograms_from_seq[key] = self.calculate_pitch_distribution_seq(self.seq_data[key])
        #Check equality
        for key in self.midi_data:
            assert histograms_from_midi[key] == histograms_from_seq[key]
        # print(json.dumps(histograms_from_midi, indent=4))
        # print(json.dumps(histograms_from_seq, indent=4))
        return histograms_from_seq
    
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
    
    def calculate_pitch_distribution_midi(self,midi):
        pitch_counts = [0] * 12
        for instrument in midi.instruments: # get our piano instrument
            if not instrument.is_drum: #ensure it is not of type drum (in case)
                for note in instrument.notes:
                    pitch_class = note.pitch % 12
                    pitch_counts[pitch_class] += 1
        #normalize by sum of all notes - to turn into a probability distribution
        pitch_counts = np.array(pitch_counts) / np.sum(pitch_counts)
        #convert to dictionary using pitch class names
        pitch_counts = {self.pitch_class_dict[i]: pitch_counts[i] for i in range(len(pitch_counts))}
        return pitch_counts
    
    def calculate_pitch_distribution_seq(self,seq):
        pitch_counts = [0] * 12
        for event in seq:
            event_type, event_value = self.decode_single_event(event)
            if event_type == 'note_on':
                pitch_class = event_value % 12
                pitch_counts[pitch_class] += 1
        #normalize by sum of all notes - to turn into a probability distribution
        pitch_counts = np.array(pitch_counts) / np.sum(pitch_counts)
        #convert to dictionary using pitch class names
        pitch_counts = {self.pitch_class_dict[i]: pitch_counts[i] for i in range(len(pitch_counts))}
        return pitch_counts
    
    def get_pitch_entropy(self):
        histograms = self.get_pitch_histograms()
        entropies = {}
        for key in histograms:
            entropies[key] = self.calculate_entropy(histograms[key])
        return entropies

    def calculate_entropy(self, histogram):
        entropy = sum([histogram[key] * np.log2(histogram[key]) for key in histogram.keys() if histogram[key] > 0])
        return -entropy
    
    def plot_pitch_histograms(self):
        histograms = self.get_pitch_histograms()
        for key in histograms:
            plt.bar(histograms[key].keys(), histograms[key].values())
            plt.title(key)
            plt.show()

if __name__ == '__main__':
    analysis = ModelAnalysis()
    analysis.load_MIDI_paths(test_input_path, test_output_path, test_actual_path)
    histograms = analysis.get_pitch_histograms()
    entropies = analysis.get_pitch_entropy()
    print(entropies)

    
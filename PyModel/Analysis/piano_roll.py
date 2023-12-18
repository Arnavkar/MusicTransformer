import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

def create_piano_roll(midi_file, start_time=0, end_time=None, fs=100):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # If end_time is not specified, use the end time of the last note
    if end_time is None:
        end_time = midi_data.get_end_time()

    # Create a piano roll matrix
    piano_roll = midi_data.get_piano_roll(fs=fs, times=np.arange(start_time, end_time, 1.0/fs))
    piano_roll = np.clip(piano_roll, 0, 127)

    # Plot the piano roll
    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(piano_roll, aspect='auto', origin='lower', cmap='magma_r', interpolation='nearest')
    
    # Add color bar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Velocity')

    ax.set_yticks(np.arange(0, 128, 12))
    ax.set_yticklabels(['C{}({})'.format(x,x*12) for x in range(11)])
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Note Number')
    plt.title('Piano Roll')
    plt.tight_layout()
    plt.show()

def plot_chroma(midi_file):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Extract chroma feature
    chroma = midi_data.get_chroma()

    # Plot the chroma feature
    plt.figure(figsize=(12, 4))
    plt.imshow(chroma, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.ylabel('Pitch Class')
    plt.xlabel('Time (frames)')
    plt.yticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    plt.title('Chroma Feature')
    plt.tight_layout()
    plt.show()


midi_file = './samples/baseline_80_files_np_28-11-2023_16-34-34/input.midi'
create_piano_roll(midi_file)
plot_chroma(midi_file)

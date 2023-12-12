import pretty_midi

def calculate_total_notes(midi):
    return sum(len(instrument.notes) for instrument in midi.instruments if not instrument.is_drum)

def calculate_pitch_distribution(midi):
    pitch_counts = [0] * 12  # 12 pitches in an octave
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_class = note.pitch % 12
                pitch_counts[pitch_class] += 1
    return pitch_counts

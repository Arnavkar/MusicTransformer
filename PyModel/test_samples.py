import pretty_midi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input_seq_path", type=str,required= True)
parser.add_argument('-o',"--output_seq_path", type=str,required= True)
args = parser.parse_args()
input_midi = pretty_midi.PrettyMIDI(args.input_seq_path)
output_midi = pretty_midi.PrettyMIDI(args.output_seq_path)

print(input_midi.get_onsets())
print(output_midi.get_onsets())



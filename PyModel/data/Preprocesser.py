import pickle #used for serialization and desrialization of data
from midi_neural_preprocessor.processor import encode_midi, decode_midi
from concurrent.futures import ThreadPoolExecutor
import os

class Preprocesser:

    def __init__(self, raw_data_path = "./data/raw/", encoded_data_path = "./data/processed") -> None:
        self.raw_data_path = raw_data_path
        self.encoded_data_path = encoded_data_path

    def encode_midi(self, midi_path) -> list:
        return encode_midi(midi_path) #Use midiprocesser library that encodes midi values into single integer value
    
    def decode_midi(self, array) -> list:
        return decode_midi(array) #Use midiprocesser library that decodes encoded midi values
    
    def get_midi_files(self) -> list:
        midi_files = []
        for root, _, files in os.walk(self.raw_data_path):
            for file in files:
                if file.endswith(".mid") or file.endswith(".midi"):
                    midi_files.append(os.path.join(root, file))
        return midi_files
    
    def process_midi_files(self):
        if not os.path.exists(self.encoded_data_path):
            os.mkdir(self.encoded_data_path)

        midi_file_paths = self.get_midi_files()
        print(midi_file_paths)
        if not midi_file_paths or len(midi_file_paths) == 0: 
            raise Exception("No midi files found")
        
        print(f'Encoding midi files... {len(midi_file_paths)} files found')
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.process_midi_file, midi_file_paths)
        print("Done encoding midi files – saved as bytestream pickle files")

    def process_midi_file(self,path_to_midi):
        try:
            encoded = encode_midi(path_to_midi)
            name = path_to_midi.split('/')[-1]
            with open(f'{self.encoded_data_path}/{name}.pickle', 'wb') as f:
                pickle.dump(encoded, f)
        except Exception as e:
            print(f'Error processing {path_to_midi}: {e}')

if __name__ == '__main__':
    #Process all midi files
    process_midi = Preprocesser()
    process_midi.process_midi_files()    
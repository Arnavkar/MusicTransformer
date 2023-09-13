import os
import pickle #used for serialization and desrialization of data
from midi_processor.processor import encode_midi, decode_midi
from concurrent.futures import ThreadPoolExecutor

class DataHandler:

    def __init__(self) -> None:
        pass

    def encode_midi(self, midi_path) -> list:
        return encode_midi(midi_path) #Use midiprocesser library that encodes midi values into single integer value
    
    def get_midi_files(self, path="./data/raw/maestro-v3.0.0") -> list:
        midi_files = []
        for root, _, files in os.walk(path):
            print(files)
            for file in files:
                if file.endswith(".mid") or file.endswith(".midi"):
                    midi_files.append(os.path.join(root, file))
        return midi_files
    
    def process_midi_files(self, output_dir = "./data/processed"):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        midi_file_paths = self.get_midi_files()
        print(midi_file_paths)
        if not midi_file_paths or len(midi_file_paths) == 0: 
            raise Exception("No midi files found")
        
        print(f'Encoding midi files... {len(midi_file_paths)} files found')
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.process_midi_file, midi_file_paths)
        print("Done encoding midi files – saved as bytestream pickle files")

    def process_midi_file(self,path, output_dir = "./data/processed"):
        try:
            encoded = encode_midi(path)
            name = path.split('/')[-1]
            with open(f'{output_dir}/{name}.pickle', 'wb') as f:
                pickle.dump(encoded, f)
        except Exception as e:
            print(f'Error processing {path}: {e}')


    def get_encoded_files(self, path="./data/processed") -> list:
        if not os.path.exists(path):
            os.mkdir(path)

        encoded_files = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".pickle"):
                    encoded_files.append(os.path.join(root, file))
        return encoded_files

        
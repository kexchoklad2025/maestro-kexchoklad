import logging
import os
import pretty_midi

DATASET_FOLDER_PATH = '.'

logging.basicConfig(level=logging.INFO,format="%(levelname)s:  %(message)s")

def get_subfolder_names(path_rel: str) -> list[str]:
    return [d for d in os.listdir(path_rel) if os.path.isdir(os.path.join(path_rel, d))]

def get_subfile_names(path_rel: str) -> list[str]:
    return [d for d in os.listdir(path_rel) if not os.path.isdir(os.path.join(path_rel, d))]


def main():
    dataset_folders: list[str] = get_subfolder_names(DATASET_FOLDER_PATH)

    for folder in dataset_folders:
        logging.info(f'ANNOTATION STARTED FOR FOLDER `{folder}`')

        folder_files: list[str] = get_subfile_names(os.path.join(
                                    DATASET_FOLDER_PATH, folder))
        midi_files: list[str] = list(filter(lambda x: x.endswith('.midi'), folder_files))

        for midi_file in midi_files:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(os.path.join(DATASET_FOLDER_PATH, folder, midi_file))

            ann_file: str = os.path.join(DATASET_FOLDER_PATH, folder,
                                         f"{midi_file.rsplit('.')[0]}.txt")

            with open(ann_file, 'w') as f:
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        onset = note.start
                        offset = note.end
                        pitch = note.pitch
                        velocity = note.velocity
                        f.write(f"{onset:.6f}\t{offset:.6f}\t{pitch}\n")


if __name__ == '__main__':
    main()

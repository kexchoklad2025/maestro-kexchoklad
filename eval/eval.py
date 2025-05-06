from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import mir_eval
import csv

MAX_WORKERS: int = 2
EVAL_FOLDER_PATH: str = './'
DATASET_FOLDER_PATH: str = '../dataset/'
CSV_SUMMARY_PATH: str ='./summary.csv'

logging.basicConfig(level=logging.INFO,format="%(levelname)s:  %(message)s")


def get_subfolder_names(path_rel: str) -> list[str]:
    return [d for d in os.listdir(path_rel) if os.path.isdir(os.path.join(path_rel, d))]

def get_subfile_names(path_rel: str) -> list[str]:
    return [d for d in os.listdir(path_rel) if not os.path.isdir(os.path.join(path_rel, d))]

def estimate(folder: str, model: str, csv_writer, lock: Lock):
    estimated_files: list[str] = get_subfile_names(os.path.join(EVAL_FOLDER_PATH, folder, model))

    for estimated_file in estimated_files:
        estimated_file_path = os.path.join(EVAL_FOLDER_PATH, folder, model, estimated_file)
        reference_file_path = os.path.join(DATASET_FOLDER_PATH, folder, estimated_file)

        ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(reference_file_path)
        est_intervals, est_pitches = mir_eval.io.load_valued_intervals(estimated_file_path)

        onset_tolerance = 0.05 # default: 0.05 s = 50 ms
        pitch_tolerance = 50.0 # default: 50.0 cents
        offset_ratio = None    # default: 0.2
        precision,recall, f_measure, avg_overlap_ratio \
        = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance,
            pitch_tolerance,
            offset_ratio)

        lock.acquire()
        csv_writer.writerow({
                'folder': folder,
                'model': model,
                'file': estimated_file,
                'precision': precision,
                'recall': recall,
                'f1': 2 * (precision * recall ) / (precision + recall)
            })
        lock.release()

def main():
    eval_folders: list[str] = get_subfolder_names(EVAL_FOLDER_PATH)
    dataset_folders: list[str] = get_subfolder_names(DATASET_FOLDER_PATH)

    if not eval_folders:
        logging.error('NO EVALUATION FOLDERS WERE FOUND!')
        return

    if not dataset_folders:
        logging.error('NO DATASET FOLDERS WERE FOUND!')
        return

    non_existent_folders: list[str] = list(
        filter(lambda x: x not in dataset_folders, eval_folders))

    if non_existent_folders:
        logging.error(f'EVALUATION FOLDERS: `{",".join(non_existent_folders)}` WERE NOT FOUND IN THE ORIGINAL DATASET!')
        return


    # ===================================================================================
    models: list[str] = get_subfolder_names(os.path.join(EVAL_FOLDER_PATH, eval_folders[0]))

    csv_f = open(CSV_SUMMARY_PATH, 'w')
    csv_fieldnames: list[str] = ['folder', 'model', 'file', 'precision', 'recall', 'f1']

    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    lock = Lock()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for folder in eval_folders:
            logging.info(f'ESTIMATION STARTED FOR FOLDER `{folder}`')

            for model in models:
                logging.info(f'ESTIMATION STARTED FOR MODEL `{model}`')
                executor.submit(estimate, folder, model, csv_writer, lock)



    logging.info(f'ESTIMATION SUMMARY DUMPED TO `{CSV_SUMMARY_PATH}`')
    csv_f.close()

if __name__ == '__main__':
    main()

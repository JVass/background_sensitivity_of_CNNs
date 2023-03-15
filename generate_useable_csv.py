import os
import torch
from torch.utils.data import Dataset
import torchaudio as taudio
import numpy as np
from tqdm import tqdm
from global_vars import *
import pandas as pd
import os
from tqdm import tqdm

def generate_path_to_songs(path_to_songs = PATH_TO_DATASET):
        # Get the annotations and parse them to a list
        annotations_csv = pd.read_csv(PATH_TO_DATASET + "metadata/UrbanSound8K.csv")
        annotations_csv = annotations_csv[["slice_file_name","classID","fold","class"]]
        annotations_csv["start"] = 'None'
        annotations_csv["path"] = 'None'
        annotations_csv["sample_rate"] = 'None'
        annotations_csv["length"] = "None"
        annotations_csv["seconds"] = "None"

        song_names = annotations_csv["slice_file_name"].tolist()

        for song_index, song_name in enumerate(song_names):
            path_to_song = PATH_TO_DATASET + "audio/" + "fold" + str(annotations_csv.iloc[song_index]["fold"]) + "/" + song_name

            if not(os.path.exists(path_to_song)):
                print(f"{path_to_song} is not valid")

            annotations_csv.loc[song_index, "path"] = path_to_song
        
        annotations_csv = annotations_csv.drop("slice_file_name", axis = 1)

        return annotations_csv

def generate_timestamps_for_audio(annotations):
    path_to_songs = annotations["path"].to_list()

    for song in tqdm(path_to_songs):

        metadata = taudio.info(song)
        song_length = metadata.num_frames
        sampling_rate = metadata.sample_rate

        annotations.loc[annotations["path"] == song, "sample_rate"] = sampling_rate
        annotations.loc[annotations["path"] == song, "seconds"] = song_length / sampling_rate
        annotations.loc[annotations["path"] == song, "length"] = song_length

    return annotations

if __name__ == "__main__":
    chunk_size = int(CHUNK_SIZE*SR)
    path_to_songs = None
    annotations_of_chunks = None
    songs_and_timestamps = None

    annotations = generate_path_to_songs()
    annotations = generate_timestamps_for_audio(annotations)

    annotations.to_csv("functional_urban8k.csv")
    print("Finished")
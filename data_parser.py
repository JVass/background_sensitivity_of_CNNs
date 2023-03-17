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

class UrbanSound8K_parser(Dataset):
    def __init__(self, chunk_size = 4*SR, annotation_csv = "functional_urban8k.csv"):
        # round to closest value of a multiple of 256
        self.chunk_size = (chunk_size // NOVERLAP + 1) * NOVERLAP
        self.all_annotations = pd.read_csv(annotation_csv)

    def __len__(self):
        return len(self.annotations.index)

    def __getitem__(self, index):
        audio, sr = taudio.backend.sox_io_backend.load(self.annotations.loc[index, "path"])

        # Resampling
        if sr != SR:
            audio = taudio.functional.resample(audio, sr, SR)

        # Length normalization
        if audio.shape[-1] < self.chunk_size:
            audio = torch.cat((audio, torch.zeros(size = (audio.shape[0], self.chunk_size - audio.shape[-1]))), dim = 1)
        elif audio.shape[-1] > self.chunk_size:
            audio = audio[:, 0:self.chunk_size]

        annotation = torch.Tensor([self.annotations.loc[index, "classID"]]) 

        return audio[0,:].to(self.device), annotation.to(self.device)

    def prepare_folds(self, test_fold_no = 1):
        self.annotations = self.all_annotations.loc[self.all_annotations["fold"] != test_fold_no, :]
        self.test_annotations = self.all_annotations.loc[self.all_annotations["fold"] == test_fold_no, :]

        self.annotations = self.annotations.set_index(pd.Index(np.arange(0, len(self.annotations.index))))
        self.test_annotations = self.test_annotations.set_index(pd.Index(np.arange(0, len(self.test_annotations.index))))

    def set_as_annotations(self, annotations):
        self.annotations = annotations

    def set_device(self, device_name = "cpu"):
        self.device = device_name
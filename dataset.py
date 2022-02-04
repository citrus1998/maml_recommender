import os, sys
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MovieRatings(Dataset):
    def __init__(self, dat_file):
        dfr = pd.read_csv(dat_file, header=0)
        dfr = dfr.drop(columns=['timestamp'])
        print(dfr)
        self.samples = torch.from_numpy(dfr.values.astype('int').astype('float64'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # every sample is a tensor with 3 values: user ID, movie ID, and rating
        return self.samples[idx]


class JesterRatings(Dataset):
    def __init__(self, dat_file):
        dfr = pd.read_csv(dat_file, header=0)
        #dfr = dfr.fillna(99)
        self.samples = torch.from_numpy(dfr.values.astype('int').astype('float64'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # every sample is a tensor with 3 values: user ID, movie ID, and rating
        return self.samples[idx]


class JesterText(Dataset):
    def __init__(self, text_file):
        text = pd.read_csv(text_file, header=0)

        if not os.path.exists('datas/all_jester'):
            os.mkdir('datas/all_jester')

        self.samples = torch.from_numpy(text.values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # every sample is a tensor with 3 values: user ID, movie ID, and rating
        return self.samples[idx]
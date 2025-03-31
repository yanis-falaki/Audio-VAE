import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class SpectrogramDataset(Dataset):
    def __init__(self, dataset_path):
        self.spectrograms = []

        for root, _, file_names in os.walk(dataset_path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                spectrogram = np.load(file_path) # (n_bins, n_frames)
                self.spectrograms.append(spectrogram)

        self.spectrograms = torch.from_numpy(np.array(self.spectrograms)).unsqueeze(1) # Add channel dimension
        print(self.spectrograms.shape)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    dataset_path = "./processed_data/spectrograms"
    SpectrogramDataset(dataset_path)
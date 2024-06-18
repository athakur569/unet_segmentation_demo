import torch
from torch.utils.data import Dataset
import numpy as np

class ToySegmentationDataset(Dataset):
    def __init__(self, n_samples=100, image_size=64):
        self.images = np.random.rand(n_samples, 3, image_size, image_size).astype(np.float32)
        self.masks = np.random.randint(0, 2, (n_samples, 1, image_size, image_size)).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.masks[idx])
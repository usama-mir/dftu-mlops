import click
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """Custom MNIST dataset"""

    def __init__(self, data, labels):
        self.images = torch.load(data)
        self.labels = torch.load(labels)

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


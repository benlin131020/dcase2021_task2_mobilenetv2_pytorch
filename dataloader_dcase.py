import enum
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, condition):
        self.data = data
        self.condition = condition

    def __len__(self):
        return len(self.condition)

    def __getitem__(self, idx):
        
        return self.data[idx], self.condition[idx]

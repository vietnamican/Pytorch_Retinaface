import sys
import numpy as np
import cv2
from torch.utils import data
import torch
import glob
import random
import os


class ConcatDataset(data.Dataset):
    def __init__(self, dataset_a, dataset_b, is_return_dataset_index=False):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        self.is_return_dataset_index = is_return_dataset_index

    def __getitem__(self, index):
        dataset_index = random.randint(0, 1)
        if dataset_index == 0:
            if index >= self.len_a:
                index = round((index - self.len_a) / self.len_b * self.len_a)
                if index >= self.len_a:
                    index = self.len_a - 1
            if self.is_return_dataset_index:
                return self.dataset_a.__getitem__(index), dataset_index
            else:
                return self.dataset_a.__getitem__(index)
        else:
            if index >= self.len_b:
                index = round((index - self.len_b) / self.len_a * self.len_b)
                if index >= self.len_b:
                    index = self.len_b - 1
            if self.is_return_dataset_index:
                return self.dataset_b.__getitem__(index), dataset_index
            else:
                return self.dataset_b.__getitem__(index)

    def __len__(self):
        return self.len_a + self.len_b

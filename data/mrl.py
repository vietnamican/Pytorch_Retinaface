import sys
import numpy as np
import cv2
from torch.utils import data
import torch
import glob
import random
import os

class MRLEyeDataset(data.Dataset):
    def __init__(self, data_dir, set_type='all', augmentation=True):
        self.EYE_IMAGE_SHAPE = (112, 112)
        self.data_dir = data_dir
        self.set_type = set_type
        assert self.set_type in ['train', 'val', 'all'], f'Set_type should belong to either "train"/"val"/"all" '
        # Train set belong to s0001 ---> s0025, val set belong to s0026->s0037, all set belong to s0001 --> s0037
        self.files = self.__load_files()

    def __load_files(self):
        files = []
        if self.set_type == "train":
            folder_indice = list(range(1, 26))
        elif self.set_type == "val":
            folder_indice = list(range(26, 38))
        else:
            folder_indice = list(range(1, 38))
        for i in folder_indice:
            files_part_i = glob.glob(f'{self.data_dir}/s{i:04}/*.png')
            files += files_part_i
        random.shuffle(files)
        print(f'Len files : {len(files)}')
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        f = self.files[index]
        # Eye image
        eye = cv2.imread(f)
        eye = cv2.resize(eye, self.EYE_IMAGE_SHAPE)
        # print("eye shape: ", eye.shape)
        # Label
        eyestate_label = int(os.path.basename(f).split("_")[4])
        return eye, torch.FloatTensor([eyestate_label])

if __name__ == "__main__":
    mrl = MLREyeDataset('data/mrleye')
    
import numpy as np
import cv2
from torch.utils import data
import torch
import glob
import random
import os
import albumentations as A
import albumentations.pytorch as AP
import torchvision.transforms as T
from torch.utils.data import DataLoader

transformerr = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=1),
        A.Normalize(),
        AP.ToTensorV2(),
    ],
)

def augment(img):
    out = transformerr(image=img)
    return out['image']

class MRLEyeDataset(data.Dataset):
    def __init__(self, data_dir, set_type='all', augmentation=True):
        self.EYE_IMAGE_SHAPE = (32, 32)
        self.data_dir = data_dir
        self.set_type = set_type
        assert self.set_type in ['train', 'val', 'all'], f'Set_type should belong to either "train"/"val"/"all" '
        self.image_paths = []
        self._load_files()

    def _load_files(self):
        image_paths = []
        folder_indice = list(range(1, 38))
        for i in folder_indice:
            part_dir = os.path.join(self.data_dir, 's{:04}'.format(i))
            for image_file in os.listdir(part_dir):
                full_image_path = os.path.join(part_dir, image_file)
                image_paths.append(full_image_path)

        if self.set_type == 'train':
            i = 0
            for image_path in image_paths:
                if i == 9:
                    i = 0
                    continue
                self.image_paths.append(image_path)
                i += 1
        elif self.set_type == 'val':
            self.image_paths = image_paths[9::10]
        else:
            self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        f = self.image_paths[index]
 
        # Eye image
        eye = cv2.imread(f)
        eye = cv2.resize(eye, self.EYE_IMAGE_SHAPE)
        eye = augment(eye)
        # eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        # Label
        eyestate_label = int(os.path.basename(f).split("_")[4])
        glass = int(os.path.basename(f).split("_")[3])
        reflection = int(os.path.basename(f).split("_")[5])
        lighting = int(os.path.basename(f).split("_")[6])
        return eye, eyestate_label, glass, reflection, lighting

class LaPa(data.Dataset):
    def __init__(self, image_dir, label_dir):
        self.EYE_IMAGE_SHAPE = (32, 32)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = []
        self._load_files()

    def _load_files(self):
        for image_path in os.listdir(self.image_dir):
            self.image_paths.append(os.path.join(self.image_dir, image_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        f = self.image_paths[index]
 
        # Eye image
        eye = cv2.imread(f)
        eye = cv2.resize(eye, self.EYE_IMAGE_SHAPE)
        eye = augment(eye)
        # eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        # Label
        base = os.path.basename(f)
        image_name, extension = base.split('.')
        eye_state = 1 if 'positive' in image_name.split('_')[-1] else 0
        return eye, eye_state


if __name__ == "__main__":
    dataset = LaPa('../LaPa_negpos_fusion/test/images', '../LaPa_negpos_fusion/test/labels') 
    loader = DataLoader(dataset, batch_size=2, pin_memory=True, num_workers=12)
    for eye, eye_state in loader:
        print(eye_state)
    # print(image)
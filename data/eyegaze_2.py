import os
import re
from functools import partial
from tqdm import tqdm
# from collections import defaultdict
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import shape
import torch
import cv2
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import random
import albumentations as A
import albumentations.pytorch as AP

transformerr = {
    'train': A.Compose(
    [
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.Normalize(),
        AP.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc')),
    'val': A.Compose(
    [
        A.Normalize(),
        AP.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc'))
} 

def _augment(transformer, img, boxes=[[0,0,1,1,1]]):
    out = transformer(image=img, bboxes=boxes)
    image = out['image']
    return image, out['bboxes']


def load_txt(file_path):
    headposes = list()
    gazes = list()
    paths = list()
    with open(file_path, "r") as file:
        for line in file:
            raw = line.strip().split("\t")
            path = raw[0]
            labels = [float(num) for num in raw[1].split(' ')]
            pose = labels[:3]
            gaze = labels[3:]
            # pose = np.array([float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", raw[1])])
            # gaze = np.array([float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", raw[2])])

            paths.append(path)
            headposes.append(pose)
            gazes.append(gaze)
    return paths, headposes, gazes


def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                        tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color, thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


class EyeGaze(Dataset):
    def __init__(self, root, set_type='train', target_size=96, augment=False, preload=False, to_gray=False) -> None:
        super().__init__()
        self.target_size = target_size
        self.rgb = not to_gray
        # Load labels
        self.paths = []
        self.headposes = []
        self.gazes = []
        self.imgs = []
        self.split_ratio = 0.8
        self.set_type = set_type
        self._parse_data(root)
        self.augment_function = partial(_augment, transformerr[set_type]) 
        self.preload = preload
        if self.preload:
            self.__load_images()

    def __getitem__(self, idx):
        if self.preload:
            img= self.imgs[idx]
        else:
            img = cv2.imread(self.paths[idx])
        h, w, _ = img.shape
        w2 = int(w / 2)
        left_img = img[:, :w2]
        right_img = img[:, w2:]
        # left eye
        left_img = cv2.resize(left_img, (self.target_size, self.target_size))
        if not self.rgb:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            left_img = cv2.normalize(left_img, left_img, alpha=0,
                                beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Right eye
        right_img = cv2.resize(right_img, (self.target_size, self.target_size))
        if not self.rgb:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            right_img = cv2.normalize(right_img, right_img,
                                alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        

        gaze = self.gazes[idx]
        # Normalize to unit vector
        gaze = gaze/np.linalg.norm(gaze)
        gaze[0], gaze[1] = gaze[1], gaze[0]
        gaze = np.arcsin(gaze)
        gaze[0] = -gaze[0]
        gaze = -gaze

         # Quick check data shape
        data_ok = gaze.shape==(3,)
        if not data_ok:
            print(self.paths[idx])
            print(gaze)
            print(f'There is wrong format data, just skip!!!')
            return self.__getitem__(random.randint(0, self.__len__()-1))

        img = np.concatenate([left_img, right_img], axis=0)
        img, _ = self.augment_function(img)
        return img, torch.FloatTensor(gaze)

    def __len__(self):
        return len(self.headposes)

    def _load_data_in_one_dir(self, root):
        paths, headposes, gazes = load_txt(os.path.join(root, 'labels', 'log.txt'))
        paths = [os.path.join(root, path) for path in paths]
        split_index = round(len(paths) * self.split_ratio)
        if self.set_type == 'train':
            print('set type train')
            paths = paths[:split_index]
            headposes = headposes[:split_index]
            gazes = gazes[:split_index]
        else:
            paths = paths[split_index:]
            headposes = headposes[split_index:]
            gazes = gazes[split_index:]
        self.paths.extend(paths)
        self.headposes.extend(headposes)
        self.gazes.extend(gazes)

    def _parse_data(self, root):
        if isinstance(root, list):
            for r in root:
                self._load_data_in_one_dir(r)
        else:
            self._load_data_in_one_dir(root)

    def __load_images(self):
        for image_path in tqdm(self.paths):
            self.imgs.append(cv2.imread(image_path))


# def load_data(root, batch_size=16, to_gray=False):
#     data = EyeGaze(root, 'train', to_gray=to_gray)
#     return data


# dataset = load_data(['../../datasets/data_gaze/ir/normal', '../../datasets/data_gaze/rgb/normal', '../../datasets/data_gaze/rgb/glance'])
# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# for image, gaze in loader:
#     print(gaze)
#     break
# print(dataset.__len__())

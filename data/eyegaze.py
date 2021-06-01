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

def get_all_image_paths(rootdir):
    filenames = list()
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith("jpg") or filename.endswith("png") or filename.endswith("jpeg"):
                filenames.append(filename)
    return [os.path.join(root, filename) for filename in sorted(filenames, key=lambda path: int(re.findall(r'\d+', path)[0]))]


def load_txt(file_path):
    out = list()
    idx = list()
    with open(file_path, "r") as file:
        for line in file:
            raw = line.strip().split("\t")
            idx.append(raw[0])
            temp_out = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", raw[1])]
            out.append([temp_out[0], temp_out[1], temp_out[2]])
    return out, idx


def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                    tuple(
                        np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out


class EyeGaze(Dataset):
    def __init__(self, root, set_type='train', target_size=112, augment=False, preload=False, to_gray=False) -> None:
        super().__init__()
        self.img_paths = get_all_image_paths(os.path.join(root, 'images'))
        self.target_size = target_size
        self.headposes, self.headposes_frame = load_txt(os.path.join(root, 'labels', 'headpose.txt'))
        self.gazes, self.gazes_frame = load_txt(os.path.join(root, 'labels', 'gaze.txt'))

        split_ratio = 0.8
        split_index = round(len(self.img_paths) * split_ratio)
        if set_type == 'train':
            self.img_paths = self.img_paths[:split_index]
            self.headposes, self.headposes_frame = self.headposes[:split_index], self.headposes_frame[:split_index]
            self.gazes, self.gazes_frame = self.gazes[:split_index], self.gazes_frame[:split_index]
        else:
            self.img_paths = self.img_paths[split_index:]
            self.headposes, self.headposes_frame = self.headposes[split_index:], self.headposes_frame[split_index:]
            self.gazes, self.gazes_frame = self.gazes[split_index:], self.gazes_frame[split_index:]

        self.rgb = not to_gray
        self.imgs = []
        self.augment_function = partial(_augment, transformerr[set_type])
        self.preload = preload
        if self.preload:
            self.__load_image()

    def __getitem__(self, idx):
        if self.preload:
            img = self.imgs[idx]
        else:
            img = cv2.imread(self.img_paths[idx])
        h, w, _ = img.shape
        w2 = int(w / 2)
        left_img = img[0:h, 0:w2]
        right_img = img[0:h, w2:w]
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
        # headpose = self.headposes[idx]
        gaze = self.gazes[idx]
        # Normalize to unit vector
        gaze = gaze/np.linalg.norm(gaze)
        gaze[0], gaze[1] = gaze[1], gaze[0]
        gaze = np.arcsin(gaze)
        gaze[0] = -gaze[0]
        gaze = -gaze
        chosen_left = random.randint(0, 1)
        if chosen_left:
            img = left_img
        else:
            img = right_img
        img, _ = self.augment_function(img)
        return img, torch.FloatTensor(gaze)

    def __len__(self):
        return len(self.headposes)

    def __load_image(self):
        for image_path in tqdm(self.img_paths):
            self.imgs.append(cv2.imread(image_path))
    


# def load_data(root, batch_size=16, to_gray=False):
#     data = EyeGazeData(root, 'val', to_gray=to_gray)
#     return data


# dataset = load_data('../../datasets/eyegazedata')
# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# for image, gaze in loader:
#     print(image)
#     break
# print(dataset.__len__())

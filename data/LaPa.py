import os
import os.path
import random
import torch
import torch.utils.data as data
import cv2
import numpy as np
import albumentations as A
import albumentations.pytorch as AP
from tqdm import tqdm
from functools import partial
from torch.nn import functional as F
import glob
# Declare an augmentation pipeline
# category_ids = [0]
transformerr = {
    'train': A.Compose(
    [
        A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=1),
        # A.Normalize(),
        A.Normalize(std=(1/255.0, 1/255.0, 1/255.0)),
        AP.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc')),
    'val': A.Compose(
    [
        # A.Normalize(),
        A.Normalize(std=(1/255.0, 1/255.0, 1/255.0)),
        AP.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc'))
}   

def _augment(transformer, img, boxes=[[0,0,1,1,1]]):
    out = transformer(image=img, bboxes=boxes)
    image = out['image']
    # convert_to_gray_matrix = torch.Tensor([[[[0.114]], [[0.587]], [[0.299]]]])
    # image = F.conv2d(image.unsqueeze(0), convert_to_gray_matrix).squeeze(0)
    # return out['image'], out['bboxes']
    return image, out['bboxes']

def is_valid_annotation(annotations, height, width):
    left, top, w, h = annotations
    right, bottom = left + w, top + h

    left_invalid_condition = left < 0 or left >= right or left > width - 1
    right_invalid_condition = right < 0 or right <= left or right > width - 1
    top_invalid_condition = top < 0 or top >= bottom or top > height - 1
    bottom_invalid_condition = bottom < 0 or bottom <= top or bottom > height - 1
    if left_invalid_condition or right_invalid_condition or top_invalid_condition or bottom_invalid_condition:
        return False
    return True

class LaPa(data.Dataset):
    def __init__(self, img_dirs, set_type='train', augment=False, preload=False, preload_image=False, to_gray=False):
        self.augment = augment
        self.preload = preload
        self.preload_image = preload_image
        self.to_gray = to_gray
        # self.imread_type = cv2.IMREAD_COLOR if self.to_gray else cv2.IMREAD_COLOR
        self.imread_type = cv2.IMREAD_COLOR
        self.img_paths = []
        self.imgs = []
        self.labels = []
        self.negpos = []
        self.img_dirs = img_dirs
        self.augment_function = partial(_augment, transformerr[set_type])
        # print(self.img_dirs)
        self.__parse_file()

    def __len__(self):
        return len(self.img_paths)

    def _load_item(self, index):
        if self.preload:
            img = self.imgs[index]
            label = self.labels[index]
            negpos = self.negpos[index]
        else:
            img_path = self.img_paths[index]
            img = cv2.imread(img_path, self.imread_type)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label_file_path = img_path.replace('images', 'labels').replace('jpg', 'txt')
            with open(label_file_path, 'r') as f:
                content = f.read().split('\n')[0:1]
                label = np.loadtxt(content)
                label = label[1:]
                if not is_valid_annotation(label, 96, 96):
                    return None, None, None
            if 'negative' in img_path:
                negpos = 0
            else:
                negpos = 1
        return img, label, negpos

    def __getitem__(self, index):
        img, label, negpos = self._load_item(index)
        if img is None:
            return self.__getitem__(random.randint(0, self.__len__()-1))
        
        height, width = img.shape[:2]
        annotations = np.zeros((0, 5))
        annotation = np.zeros((1, 5))
        if negpos == 0:
            # print('negative')
            annotation[0, 0] = 0
            annotation[0, 1] = 0
            annotation[0, 2] = 0
            annotation[0, 3] = 0
            annotation[0, 4] = 0
            annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)

            if self.augment:
                img, _ = self.augment_function(img)
        else:
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            annotation[0, 4] = -1
            annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)
            # print(target)
            if self.augment:
                img, bboxes = self.augment_function(img, target)
                bboxes = np.array(bboxes)
                try:
                    target[:, :4] = bboxes[:, :4]
                except:
                    raise Exception("Wrong data format exception in {} with bbox {}".format(self.img_paths[index], bboxes))
                # print(target.shape, target)

        target[:, (0,2)] /= width
        target[:, (1,3)] /= height
        return img, torch.FloatTensor(target)
    
    def __load_data_in_one_dir(self, img_dir):
        img_paths = glob.glob(img_dir + '**/*.jpg', recursive=True)
        if self.preload:
            for full_img_path in tqdm(img_paths):
                label_file_path = full_img_path.replace('images', 'labels').replace('jpg', 'txt')
                with open(label_file_path, 'r') as f:
                    content = f.read().split('\n')[0:1]
                    label = np.loadtxt(content)
                    label = label[1:]
                    if not is_valid_annotation(label, 96, 96):
                        continue
                    self.labels.append(label)
                if self.preload_image:
                    self.imgs.append(cv2.cvtColor(cv2.imread(full_img_path, self.imread_type), cv2.COLOR_BGR2RGB))
                if 'negative' in full_img_path:
                    self.negpos.append(0)
                else:
                    self.negpos.append(1)
                self.img_paths.append(full_img_path)
        else:
            self.img_paths.extend(img_paths)
    def __parse_file(self):
        if isinstance(self.img_dirs, str):
            self.__load_data_in_one_dir(self.img_dirs)
        else:
            for img_dir in self.img_dirs:
                self.__load_data_in_one_dir(img_dir)



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        image, target = sample
        if image is not None:
            imgs.append(image)
            targets.append(target)

                # elif isinstance(tup, type(np.empty(0))):
                #     annos = torch.from_numpy(tup).float()
                #     targets.append(annos)

    return (torch.stack(imgs, 0), targets)


if __name__ == '__main__':
    img_dim = 96
    rgb_mean = (104, 117, 123)
    dataset = LaPa(
        os.path.join('data', 'train', 'images'), 
        os.path.join('data', 'train', 'labels'), 
        # preproc(img_dim, rgb_mean)
        )
    dataloader = data.DataLoader(dataset, 2, shuffle=True, num_workers=1)
    for img, target in dataloader:
        print(target.shape)
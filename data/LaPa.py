import os
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import albumentations as A
import albumentations.pytorch as AP
from tqdm import tqdm
from functools import partial
from torch.nn import functional as F
# Declare an augmentation pipeline
# category_ids = [0]
transformerr = {
    'train': A.Compose(
    [
        A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=1),
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
    # convert_to_gray_matrix = torch.Tensor([[[[0.114]], [[0.587]], [[0.299]]]])
    # image = F.conv2d(image.unsqueeze(0), convert_to_gray_matrix).squeeze(0)
    # return out['image'], out['bboxes']
    return image, out['bboxes']

def is_valid_annotation(annotations, height, width):
    left = annotations[:, 0]
    top = annotations[:, 1]
    right = annotations[:, 2]
    bottom = annotations[:, 3]

    if np.any(left<0) or np.any(top<0) or any(right>width-1) or any(bottom>height-1) or any(left>=right) or any(top>=bottom):
        return False
    return True

class LaPa(data.Dataset):
    def __init__(self, img_dirs, label_dirs, set_type='train', augment=False, preload=False, to_gray=False):
        self.augment = augment
        self.preload = preload
        self.to_gray = to_gray
        # self.imread_type = cv2.IMREAD_COLOR if self.to_gray else cv2.IMREAD_COLOR
        self.imread_type = cv2.IMREAD_COLOR
        self.img_paths = []
        self.imgs = []
        self.labels = []
        self.negpos = []
        self.img_dirs = img_dirs
        self.label_dirs = label_dirs
        self.augment_function = partial(_augment, transformerr[set_type])
        # print(self.img_dirs)
        self.__parse_file()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.preload:
            img = self.imgs[index]
        else:
            img = cv2.imread(self.img_paths[index], self.imread_type)
        
        height, width = img.shape[:2]
        label = self.labels[index]
        annotations = np.zeros((0, 5))
        annotation = np.zeros((1, 5))
        if self.negpos[index] == 0:
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
            if not is_valid_annotation(annotation, height, width):
                # print(annotation)
                return None, None
            annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)
            # print(target)
            if self.augment:
                img, bboxes = self.augment_function(img, target)
                bboxes = np.array(bboxes)
                target[:, :4] = bboxes[:, :4]
                # print(target.shape, target)

        target[:, (0,2)] /= width
        target[:, (1,3)] /= height
        return img, torch.FloatTensor(target)
    
    def __load_data_in_one_dir(self, img_dir, label_dir):
        # num_batch = 100
        label_extension = 'txt'
        for img_file in tqdm(os.listdir(img_dir)):
        # for img_file in tqdm(os.listdir(img_dir)[:num_batch]):
            img_name, img_extension = os.path.splitext(img_file)
            full_img_path = os.path.join(img_dir, img_file)
            if self.preload:
                self.imgs.append(cv2.imread(full_img_path, self.imread_type))
            self.img_paths.append(full_img_path)
            if 'negative' in full_img_path:
                self.negpos.append(0)
            else:
                self.negpos.append(1)
            label_file_path = os.path.join(label_dir, img_name + "." + label_extension)
            with open(label_file_path, 'r') as f:
                content = f.read().split('\n')[0:1]
                label = np.loadtxt(content)
                label = label[1:]
                self.labels.append(label)


    def __parse_file(self):
        if isinstance(self.img_dirs, str):
            self.__load_data_in_one_dir(self.img_dirs, self.label_dirs)
        else:
            for img_dir, label_dir in zip(self.img_dirs, self.label_dirs):
                self.__load_data_in_one_dir(img_dir, label_dir)



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
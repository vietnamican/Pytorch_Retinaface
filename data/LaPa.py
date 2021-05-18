import os
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
# Declare an augmentation pipeline
# category_ids = [0]
transformerr = A.Compose(
    [
        A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2),
        A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.ColorJitter (brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.25, rotate_limit=30, interpolation=1, border_mode=4, always_apply=False, p=1)
       
    ],
    bbox_params=A.BboxParams(format='pascal_voc')
)

def augment(img, boxes):
    out = transformerr(image=img, bboxes=boxes)
    return out['image'], out['bboxes']

class LaPa(data.Dataset):
    def __init__(self, img_dirs, label_dirs, preproc=None, augment=False, preload=False):
        self.preproc = preproc
        self.augment = augment
        self.preload = preload
        self.img_paths = []
        self.imgs = []
        self.labels = []
        self.img_dirs = img_dirs
        self.label_dirs = label_dirs
        print(self.img_dirs)
        self.__parse_file()

    def __len__(self):
        if self.preload:
            return len(self.imgs)
        else:
            return len(self.img_paths)

    def __getitem__(self, index):
        if self.preload:
            img = self.imgs[index]
        else:
            img = cv2.imread(self.img_paths[index])

        label = self.labels[index]
        annotations = np.zeros((0, 5))
        annotation = np.zeros((1, 5))
        # bbox
        annotation[0, 0] = label[0]  # x1
        annotation[0, 1] = label[1]  # y1
        annotation[0, 2] = label[0] + label[2]  # x2
        annotation[0, 3] = label[1] + label[3]  # y2
        annotation[0, 4] = -1
        annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.augment:
            try:
                img, bboxes = augment(img, target[:, (0,1,2,3,4)])
                bboxes = np.array(bboxes)
                target[:, (0,1,2,3)] = bboxes[:, (0,1,2,3)]
            except:
                pass
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return torch.from_numpy(img), target
    
    def __parse_file(self):
        label_extension = 'txt'
        if isinstance(self.img_dirs, str):
            for img_file in tqdm(os.listdir(self.img_dirs)):
                img_name, img_extension = os.path.splitext(img_file)
                if self.preload:
                    self.imgs.append(cv2.imread(os.path.join(self.img_dirs, img_file), cv2.IMREAD_COLOR))
                else:
                    self.img_paths.append(os.path.join(self.img_dirs, img_file))
                label_file_path = os.path.join(self.label_dirs, img_name + "." + label_extension)
                with open(label_file_path, 'r') as f:
                    content = f.read().split('\n')[0:1]
                    label = np.loadtxt(content)
                    label = label[1:]
                    self.labels.append(label)
        else:
            for img_dir, label_dir in zip(self.img_dirs, self.label_dirs):
                for img_file in tqdm(os.listdir(img_dir)):
                    img_name, img_extension = os.path.splitext(img_file)
                    if self.preload:
                        self.imgs.append(cv2.imread(os.path.join(img_dir, img_file), cv2.IMREAD_COLOR))
                    else:
                        self.img_paths.append(os.path.join(img_dir, img_file))
                    label_file_path = os.path.join(label_dir, img_name + '.' + label_extension)
                    with open(label_file_path, 'r') as f:
                        content = f.read().split('\n')[0:1]
                        label = np.loadtxt(content)
                        label = label[1:]
                        self.labels.append(label)



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
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

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
from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from tqdm import tqdm
# from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from data import LaPa, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--img_dir', default='./data/train/images',
                    help='Training images directory')
parser.add_argument('--label_dir', default='./data/train/labels',
                    help='Training labels directory')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--log_frequent', default=1000,
                    help='Location to save checkpoint models')

args = parser.parse_args()
cfg = cfg_mnet
rgb_mean = (104, 117, 123)
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
img_dir = args.img_dir
label_dir = args.label_dir
save_folder = args.save_folder


dataset = LaPa(img_dir, label_dir, preproc(img_dim, rgb_mean), augment=True)
dataloader = data.DataLoader(
    dataset, batch_size, shuffle=True, collate_fn=detection_collate)

for img, target in tqdm(dataloader):
    pass

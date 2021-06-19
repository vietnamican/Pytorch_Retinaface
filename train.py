import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from data import LaPa, detection_collate, preproc, cfg_mnet, cfg_re50, cfg_re34, ConcatDataset
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from model import Model

pl.seed_everything(42)
# from models import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--network', default='resnet34',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--train_batch_size', default=32,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', default=32,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='logs/resnet34_logs',
                    help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
elif args.network == "resnet34":
    cfg = cfg_re34


cudnn.benchmark = True


def load_trainer(logdir, device, max_epochs, checkpoint=None):

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name=logdir,
    )
    lr_monitor = LearningRateMonitor(log_momentum=False)
    loss_callback = ModelCheckpoint(
        save_last=True
        # monitor='val_loss',
        # filename='checkpoint-{epoch:02d}',
        # save_top_k=10,
        # mode='min',
    )
    callbacks = [loss_callback, lr_monitor]
    resume_from_checkpoint = checkpoint
    if device == 'tpu':
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            tpu_cores=8,
            resume_from_checkpoint=resume_from_checkpoint
        )
    elif device == 'gpu':
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            gpus=[1,],
            accelerator=None,
            resume_from_checkpoint=resume_from_checkpoint
        )
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=resume_from_checkpoint
        )

    return trainer


def load_data(args, val_only=False):
    train_image_dir = '../datasets/LaPa_negpos_fusion_cleaned/train/images'
    train_label_dir = '../datasets/LaPa_negpos_fusion_cleaned/train/labels'
    val_image_dir = '../datasets/LaPa_negpos_fusion_cleaned/val/images'
    val_label_dir = '../datasets/LaPa_negpos_fusion_cleaned/val/labels'

    train_ir_image_dirs = [
        '../datasets/ir_negpos/positive/images/out2',
        '../datasets/ir_negpos/positive/images/out22',
        '../datasets/ir_negpos/positive/images/out222',
        '../datasets/ir_negpos/negative/images/out2',
        '../datasets/ir_negpos/negative/images/out22',
        '../datasets/ir_negpos/negative/images/out222'
    ]

    train_ir_label_dirs = [
        '../datasets/ir_negpos/positive/labels/out2',
        '../datasets/ir_negpos/positive/labels/out22',
        '../datasets/ir_negpos/positive/labels/out222',
        '../datasets/ir_negpos/negative/labels/out2',
        '../datasets/ir_negpos/negative/labels/out22',
        '../datasets/ir_negpos/negative/labels/out222'
    ]

    val_ir_image_dirs = [
        '../datasets/tatden/positive/images',
        '../datasets/tatden/negative/images',
    ]

    val_ir_label_dirs = [
        '../datasets/tatden/positive/labels',
        '../datasets/tatden/negative/labels',
    ]

    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    
    lapatraindataset = LaPa(train_image_dir, train_label_dir,
                            'train', augment=True, preload=True, to_gray=False)
    irtraindataset = LaPa(train_ir_image_dirs, train_ir_label_dirs,
                          'train', augment=True, preload=True, to_gray=False)
    traindataset = ConcatDataset(lapatraindataset, irtraindataset)
    print(len(traindataset))
    print(len(irtraindataset))
    print(len(lapatraindataset))
    trainloader = DataLoader(traindataset, batch_size=train_batch_size,
                             pin_memory=True, num_workers=num_workers, shuffle=True, collate_fn=detection_collate)
    lapavaldataset = LaPa(val_image_dir, val_label_dir, 'val',
                          augment=True, preload=True, to_gray=False)
    irvaldataset = LaPa(val_ir_image_dirs, val_ir_label_dirs, 'val', augment=True, preload=True, to_gray=False)
    valdataset = ConcatDataset(lapavaldataset, irvaldataset)
    print(len(valdataset))
    print(len(irvaldataset))
    print(len(lapavaldataset))
    valloader = DataLoader(valdataset, batch_size=val_batch_size,
                           pin_memory=True, num_workers=num_workers, collate_fn=detection_collate)
    if not val_only:
        return traindataset, trainloader, valdataset, valloader
    return valdataset, valloader


if __name__ == '__main__':
    _, trainloader, _, valloader = load_data(args)
    num_training_steps = len(trainloader)
    net = Model(cfg=cfg, args=args, num_training_steps=num_training_steps)
    print("Printing net...")
    print(net)
    trainer = load_trainer(args.save_folder, 'gpu', cfg['max_epochs'])
    trainer.fit(net, trainloader, valloader)

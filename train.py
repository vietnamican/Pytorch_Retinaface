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
        # version=1
    )
    lr_monitor = LearningRateMonitor(log_momentum=False)
    loss_callback = ModelCheckpoint(
        save_last=True,
        
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
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
            gpus=1,
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

if __name__ == '__main__':
    net = Model(cfg=cfg, args=args)
    print("Printing net...")
    print(net)
    # trainer = load_trainer(args.save_folder, 'gpu', cfg['max_epochs'], checkpoint='logs/resnet34_logs/version_1/checkpoints/last.ckpt')
    trainer = load_trainer(args.save_folder, 'gpu', cfg['max_epochs'], checkpoint=None)
    trainer.fit(net)

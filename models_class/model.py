import torch
from torch import nn
from torchmetrics import Accuracy

from .base import ConvBatchNormRelu
from .base import Base

class Config(object):
    dataroot = 'data/mrleye'
    train_image_dir = '../LaPa_negpos_fusion/train/images'
    train_label_dir = '../LaPa_negpos_fusion/train/labels'
    val_image_dir = '../LaPa_negpos_fusion/val/images'
    val_label_dir = '../LaPa_negpos_fusion/val/labels'
    batch_size = 512
    pin_memory=  True
    num_workers = 6
    device = 'gpu'
    max_epochs = 200
    steps = [0.5, 0.7, 0.9]

cfg = Config()

class IrisModel(Base):
    def __init__(self, cfg=cfg):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(3, 10, kernel_size=3, padding=1, with_relu=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvBatchNormRelu(10, 20, kernel_size=3, padding=1, with_relu=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = ConvBatchNormRelu(20, 50, kernel_size=3, padding=1, with_relu=False)
        self.conv4 = ConvBatchNormRelu(50, 2, kernel_size=1, padding=0, with_relu=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.val_acc = Accuracy()
    
    def forward(self, x):
        x = self.relu1(self.maxpool1(self.conv1(x)))
        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return x
    
    def _shared_step(self, batch, batch_dix):
        eye, label, *_ = batch
        logit = self.forward(eye)
        loss = self.criterion(logit, label)
        return loss, logit

    def training_step(self, batch, batch_dix):
        _, label, *_ = batch
        loss, logit = self._shared_step(batch, batch_dix)
        pred = logit.argmax(dim=1)
        self.log('train_acc', self.val_acc(pred, label))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_dix):
        _, label, *_ = batch
        loss, logit = self._shared_step(batch, batch_dix)
        pred = logit.argmax(dim=1)
        self.log('val_acc', self.val_acc(pred, label))
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        max_epochs = self.cfg.max_epochs
        step0, step1, step2 = self.cfg.steps
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [max_epochs*step0, max_epochs*step1, max_epochs*step2], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

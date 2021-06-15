import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from .net import MobileNetV1 as MobileNetV1
from .net import FPN as FPN
from .net import SSH as SSH
from .net import conv_dw as conv_dw
from .base import Base



class ClassHead(Base):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(Base):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class RetinaFace(Base):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            self.body = MobileNetV1()
        in_channels_stage = cfg['in_channel']
        in_channels_list = [
            in_channels_stage * 2,
            in_channels_stage * 4,
            in_channels_stage * 8,
        ]
        out_channels = cfg['out_channel']
        self.ssh_eyestate = SSH(in_channels_list[0], out_channels)
        self.ssh_eyegaze = nn.Sequential(
            SSH(in_channels_list[2], out_channels),
            nn.Conv2d(out_channels, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Tanh()
        )

        self.ClassHead = self._make_class_head(fpn_num=1, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=1, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        if len(classhead) == 1:
            return classhead[0]
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        if len(bboxhead) == 1:
            return bboxhead[0]
        return bboxhead


    def forward(self,inputs):
        out = self.body(inputs)

        # SSH eyegaze
        feature_eyegaze = self.ssh_eyegaze(out[2])

        # SSH
        feature_eyestate = self.ssh_eyestate(out[0])
        bbox_regressions = self.BboxHead(feature_eyestate)
        classifications = self.ClassHead(feature_eyestate)

        if self.phase == 'train':
            output = (bbox_regressions, classifications), feature_eyegaze 
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1)), feature_eyegaze
        return output


import torch
from itertools import product as product
import numpy as np
from math import ceil, sqrt


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.ratios = cfg['ratios'] if 'ratio' in cfg else [1]
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.stride = cfg['stride_prior'] if 'stride_prior' in cfg else 1
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            stride = self.stride
            for i, j in product(range(0, f[0], stride), range(0, f[1], stride)):
                for min_size in min_sizes:
                    for ratio in self.ratios:
                        min_size_x = min_size * sqrt(ratio)
                        min_size_y = min_size / sqrt(ratio)
                        s_kx = min_size_x / self.image_size[1]
                        s_ky = min_size_y / self.image_size[0]
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors.append([cx, cy, s_kx, s_ky])

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32]],
    'ratios': [1, 1.5, 2], # width/height
    'steps': [8],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 96,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    # 'stride_prior': 1
}

if __name__ == '__main__':
    prior = PriorBox(cfg=cfg, image_size=(96,96))
    out = prior.forward()

    print(out.shape)
    print(out[:5])
    # for o in out:
    #     print(o)
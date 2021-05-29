import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        # print(self.feature_maps)
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # print(dense_cx, dense_cy)
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32]],
    'steps': [16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'ngpu': 1,
    'max_epochs': 250,
    'warmup_epochs' : 5,
    'decay_steps': [0.76, 0.88],
    'image_size': 96,
    'pretrain': False,
    'return_layers': {'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

if __name__ == '__main__':
    prior = PriorBox(cfg=cfg, image_size=(96,96))
    out = prior.forward()
    # print(out.shape)
    # for o in out:
    #     print(o)
import os
from numpy.core.fromnumeric import prod
import torch
import albumentations as A
import albumentations.pytorch as AP
import cv2
from tqdm import tqdm
import numpy as np
from time import time
from itertools import product
import json

from models import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode
from data.config import cfg_mnet as cfg

transformerr = A.Compose(
    [
        A.Normalize(),
        AP.ToTensorV2()
    ],
)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        print('Load to cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    state_dict = pretrained_dict['state_dict']
    model.migrate(state_dict, force=True)
    return model

def transforms(img):
    out = transformerr(image=img)
    return out['image']

device = 'cpu'
priorbox = PriorBox(cfg, image_size=(96, 96))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data

def detect_iris(img, net):
    img = transforms(img).unsqueeze(0)
    img = img.to(device)

    loc, conf = net(img)  # forward pass

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    ind = scores.argmax()

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes.cpu().numpy()
    scores = scores[ind:ind+1]
    boxes = boxes[ind:ind+1]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    return dets

def filter_iris_box(dets, threshold):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) >threshold
    dets = dets.copy()[mask]
    return dets

def filter_confident_box(dets, threshold):
    dets = dets.copy()
    return dets[dets[:, 4] > threshold]

def paint_bbox(image, bboxes):
    for b in bboxes:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] - 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

if __name__ == '__main__':
    # net_path = os.path.join('weights_negpos', 'mobilenet0.25_Final.pth')
    # net_path = os.path.join('weights_negpos_cleaned', 'mobilenet0.25_Final.pth')
    net_path = 'training_16x_featuremap/version_0/checkpoints/checkpoint-epoch=247-val_loss=6.2495.ckpt'
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, net_path, device)
    net.eval()
    labels = []
    predicted = []

    # rgb_image_dir = os.path.join('dataset', 'eyestate_label', 'outputir', 'Open')
    rgb_image_dir = os.path.join('..', 'LaPa_negpos', 'positive_data', 'val', 'images')
    # out_open_image_dir = os.path.join('out', 'outputir', 'open_open')
    # out_close_image_dir = os.path.join('out', 'outputir', 'open_close')
    # if not os.path.isdir(out_open_image_dir):
    #     os.makedirs(out_open_image_dir)
    # if not os.path.isdir(out_close_image_dir):
    #     os.makedirs(out_close_image_dir)
    listdir = os.listdir(rgb_image_dir)
    for image_file in tqdm(listdir):
        image = cv2.imread(os.path.join(rgb_image_dir, image_file))
        dets = detect_iris(image, net)
        labels.append(1)
        predicted.append(dets)

    # rgb_image_dir = os.path.join('dataset', 'eyestate_label', 'outputir', 'Close')
    rgb_image_dir = os.path.join('..', 'LaPa_negpos', 'negative_data', 'val', 'images')
    # out_open_image_dir = os.path.join('out', 'outputir', 'close_open')
    # out_close_image_dir = os.path.join('out', 'outputir', 'close_close')
    # if not os.path.isdir(out_open_image_dir):
    #     os.makedirs(out_open_image_dir)
    # if not os.path.isdir(out_close_image_dir):
    #     os.makedirs(out_close_image_dir)
    listdir = os.listdir(rgb_image_dir)
    listdir = os.listdir(rgb_image_dir)
    for image_file in tqdm(listdir):
        image = cv2.imread(os.path.join(rgb_image_dir, image_file))
        dets = detect_iris(image, net)
        labels.append(0)
        predicted.append(dets)

    results = []
    conf_thresholds = np.linspace(0.7, 0.9, num=17)
    width_height_threshold = np.linspace(0.4, 0.7, num=25)
    for conf_threshold, width_height_threshold in tqdm(product(conf_thresholds, width_height_threshold)):
        preds = []
        for (label, pred) in zip(labels, predicted):
            pred = filter_iris_box(pred, width_height_threshold)
            pred = filter_confident_box(pred, conf_threshold)
            if pred.shape[0] > 0:
                preds.append(1)
            else:
                preds.append(0)
        open_open = 1
        open_close = 1
        close_open = 1
        close_close = 1
        for (label, pred) in zip(labels, preds):
            if label == 1 and pred == 1:
                open_open += 1
            elif label == 1 and pred == 0:
                open_close += 1
            elif label == 0 and pred == 1:
                close_open += 1
            else:
                close_close += 1

        report = {
            'conf_threshold': conf_threshold,
            'width_height_threshold': width_height_threshold,
            'open_open': open_open,
            'open_close': open_close,
            'close_open': close_open,
            'close_close': close_close,
            'accuracy': (open_open + close_close) / (open_open + open_close + close_open + close_close),
            'precision open': (open_open) / (open_open + close_open),
            'recall open': (open_open) / (open_open + open_close),
            'precision close': (close_close) / (close_close + open_close),
            'recall close': (close_close) / (close_close + close_open)
        }
        results.append(report)
    with open('data_lapa_val.json', 'w') as f:
        json.dump(results, f)
import os
import torch
import albumentations as A
import albumentations.pytorch as AP
import cv2
from tqdm import tqdm
import numpy as np
from time import time

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

def transforms(img):
    out = transformerr(image=img)
    return out['image']

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
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, device='cuda'):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if device == 'cpu':
        print('load to cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

device = 'cpu'
priorbox = PriorBox(cfg, image_size=(96, 96))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data

def detect_iris(img_raw, net):
    confidence_threshold = 0.02
    top_k = 5
    nms_threshold=0.4
    keep_top_k = 1

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img = transforms(img).unsqueeze(0)
    # img -= (104, 117, 123)
    # img = img.transpose(2, 0, 1)
    # img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time()
    # loc, conf, landms = net(img)  # forward pass
    loc, conf = net(img)  # forward pass
    # print('net forward time: {:.4f}'.format(time() - tic))

    
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / 1
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                         img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                         img.shape[3], img.shape[2]])
    # scale1 = scale1.to(device)

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]

    return dets

def filter_iris_box(dets, threshold):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) > threshold
    dets = dets[mask]
    return dets

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
    net_path = os.path.join('weights_negpos_cleaned', 'mobilenet0.25_Final.pth')
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, net_path, device)
    # net = net.cuda()
    net.eval()
    # if not os.path.isdir(out_open_image_dir):
    #     os.makedirs(out_open_image_dir)
    # label_predict
    open_open = 0
    open_close = 0
    close_open = 0
    close_close = 0
    conf_threshold = 0.80625
    width_height_threshold = 0.4875 

    rgb_image_dir = os.path.join('dataset', 'eyestate_label', 'outputir', 'Open')
    out_open_image_dir = os.path.join('dataset', 'out', 'outputir', 'open_open')
    out_close_image_dir = os.path.join('dataset', 'out', 'outputir', 'open_close')
    if not os.path.isdir(out_open_image_dir):
        os.makedirs(out_open_image_dir)
    if not os.path.isdir(out_close_image_dir):
        os.makedirs(out_close_image_dir)
    listdir = os.listdir(rgb_image_dir)
    for image_file in tqdm(listdir):
        image = cv2.imread(os.path.join(rgb_image_dir, image_file))
        dets = detect_iris(image, net)
        dets = filter_iris_box(dets, width_height_threshold)
        bboxes = dets[dets[:, 4] > conf_threshold]
        if bboxes.shape[0] > 0:
            # print('Predicted Open')
            open_open += 1
            paint_bbox(image, dets)
            cv2.imwrite(os.path.join(out_open_image_dir, image_file), image)
        else:
            # print('Predicted Close')
            open_close += 1
            paint_bbox(image, dets)
            cv2.imwrite(os.path.join(out_close_image_dir, image_file), image)
        # paint_bbox(image, dets)

    rgb_image_dir = os.path.join('dataset', 'eyestate_label', 'outputir', 'Close')
    out_open_image_dir = os.path.join('dataset', 'out', 'outputir', 'close_open')
    out_close_image_dir = os.path.join('dataset', 'out', 'outputir', 'close_close')
    if not os.path.isdir(out_open_image_dir):
        os.makedirs(out_open_image_dir)
    if not os.path.isdir(out_close_image_dir):
        os.makedirs(out_close_image_dir)
    listdir = os.listdir(rgb_image_dir)
    listdir = os.listdir(rgb_image_dir)
    for image_file in tqdm(listdir):
        image = cv2.imread(os.path.join(rgb_image_dir, image_file))
        dets = detect_iris(image, net)
        dets = filter_iris_box(dets, width_height_threshold)
        dets = dets[dets[:, 4] > conf_threshold]
        if dets.shape[0] > 0:
            # print('Predicted Open')
            close_open += 1
            paint_bbox(image, dets)
            cv2.imwrite(os.path.join(out_open_image_dir, image_file), image)
        else:
            # print('Predicted Close')
            close_close += 1
            cv2.imwrite(os.path.join(out_close_image_dir, image_file), image)
        # paint_bbox(image, dets)
        # cv2.imwrite(os.path.join(out_open_image_dir, image_file), image)
    print(open_open, open_close, close_open, close_close)
    # print('Accuracy with open: {}'.format(match/total))


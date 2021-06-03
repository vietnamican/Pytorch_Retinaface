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

def transform(img):
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

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        print('Load to cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    state_dict = pretrained_dict['state_dict']
    # state_dict = model.filter_state_dict_with_prefix(state_dict, 'student_model.model', True)
    print(state_dict.keys())
    model.migrate(state_dict, force=True)
    return model

# def load_model(model, pretrained_path, device='cuda'):
#     print('Loading pretrained model from {}'.format(pretrained_path))
#     if device == 'cpu':
#         print('load to cpu')
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
#     else:
#         device = torch.cuda.current_device()
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
#     if "state_dict" in pretrained_dict.keys():
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#     else:
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#     check_keys(model, pretrained_dict)
#     model.load_state_dict(pretrained_dict, strict=False)
#     return model

device = 'cpu'
priorbox = PriorBox(cfg, image_size=(96, 96))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data

def calculate_box(loc, conf):
    scores = conf[:, 1]
    ind = scores.argmax()
    boxes = decode(loc, prior_data, cfg['variance'])
    scores = scores[ind:ind+1]
    boxes = boxes[ind:ind+1]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    return dets

def detect_iris(img, net):
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    loc, conf = net(img)  # forward pass
    loc = loc.squeeze(0).data.cpu()
    conf = conf.squeeze(0).data.cpu().numpy()
    split_index = loc.shape[0] // 2
    loc_left, loc_right = loc[:split_index], loc[split_index:]
    conf_left, conf_right = conf[:split_index], conf[split_index:]
    return calculate_box(loc_left, conf_left), calculate_box(loc_right, conf_right)

# def detect_iris(img, net):
#     img = transform(img).unsqueeze(0)
#     img = img.to(device)

#     loc, conf = net(img)  # forward pass

#     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
#     ind = scores.argmax()

#     boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#     boxes = boxes.cpu().numpy()
#     scores = scores[ind:ind+1]
#     boxes = boxes[ind:ind+1]

#     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

#     return dets


def filter_iris_box(dets, threshold):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) > threshold
    dets = dets[mask]
    return dets


def filter_conf(dets, threshold):
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
    # net_path = 'distill_logs/version_0/checkpoints/checkpoint-epoch=52-val_loss=7.4488.ckpt'
    # net_path = 'multi_ratio_prior_box_logs/version_0/checkpoints/checkpoint-epoch=99-val_loss=5.1367.ckpt'
    net_path = 'test_logs/version_0/checkpoints/checkpoint-epoch=99-val_loss=4.4232.ckpt'
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

    rgb_image_dir = os.path.join('../datasets', 'eyestate_label', 'outputrgb', 'Open')
    out_open_image_dir = os.path.join('../datasets', 'out', 'outputrgb', 'open_open')
    out_close_image_dir = os.path.join('../datasets', 'out', 'outputrgb', 'open_close')
    if not os.path.isdir(out_open_image_dir):
        os.makedirs(out_open_image_dir)
    if not os.path.isdir(out_close_image_dir):
        os.makedirs(out_close_image_dir)
    listdir = os.listdir(rgb_image_dir)
    listdir = listdir[:len(listdir) // 2 * 2]
    for image_file_one, image_file_two in tqdm(zip(listdir[::2], listdir[1::2])):
        image_one = cv2.imread(os.path.join(rgb_image_dir, image_file_one))
        image_two = cv2.imread(os.path.join(rgb_image_dir, image_file_two))
        image = np.concatenate([image_one, image_two], axis=0)
        one_dets, two_dets = detect_iris(image, net)
        one_dets = filter_iris_box(one_dets, width_height_threshold)
        one_dets = filter_conf(one_dets, conf_threshold)
        two_dets = filter_iris_box(one_dets, width_height_threshold)
        two_dets = filter_conf(two_dets, conf_threshold)

        if one_dets.shape[0] > 0:
            open_open += 1
            # paint_bbox(image, dets)
        else:
            open_close += 1
            # paint_bbox(image, dets)

        if two_dets.shape[0] > 0:
            open_open += 1
            # paint_bbox(image, dets)
        else:
            open_close += 1
            # paint_bbox(image, dets)

    rgb_image_dir = os.path.join('../datasets', 'eyestate_label', 'outputrgb', 'Close')
    out_open_image_dir = os.path.join('../datasets', 'out', 'outputrgb', 'close_open')
    out_close_image_dir = os.path.join('../datasets', 'out', 'outputrgb', 'close_close')
    if not os.path.isdir(out_open_image_dir):
        os.makedirs(out_open_image_dir)
    if not os.path.isdir(out_close_image_dir):
        os.makedirs(out_close_image_dir)
    listdir = os.listdir(rgb_image_dir)
    listdir = listdir[:len(listdir) // 2 * 2]
    for image_file_one, image_file_two in tqdm(zip(listdir[::2], listdir[1::2])):
        image_one = cv2.imread(os.path.join(rgb_image_dir, image_file_one))
        image_two = cv2.imread(os.path.join(rgb_image_dir, image_file_two))
        image = np.concatenate([image_one, image_two], axis=0)
        one_dets, two_dets = detect_iris(image, net)
        one_dets = filter_iris_box(one_dets, width_height_threshold)
        one_dets = filter_conf(one_dets, conf_threshold)
        two_dets = filter_iris_box(one_dets, width_height_threshold)
        two_dets = filter_conf(two_dets, conf_threshold)

        if one_dets.shape[0] > 0:
            close_open += 1
            # paint_bbox(image, dets)
        else:
            close_close += 1
            # paint_bbox(image, dets)

        if two_dets.shape[0] > 0:
            close_open += 1
            # paint_bbox(image, dets)
        else:
            close_close += 1
            # paint_bbox(image, dets)
    print(open_open, open_close, close_open, close_close)
    # print('Accuracy with open: {}'.format(match/total))


    # rgb_image_dir = os.path.join('../datasets', 'eyestate_label', 'outputir', 'Open')
    # out_open_image_dir = os.path.join('../datasets', 'out', 'outputir', 'open_open')
    # out_close_image_dir = os.path.join('../datasets', 'out', 'outputir', 'open_close')
    # if not os.path.isdir(out_open_image_dir):
    #     os.makedirs(out_open_image_dir)
    # if not os.path.isdir(out_close_image_dir):
    #     os.makedirs(out_close_image_dir)
    # listdir = os.listdir(rgb_image_dir)
    # for image_file in tqdm(listdir):
    #     image = cv2.imread(os.path.join(rgb_image_dir, image_file))
    #     dets = detect_iris(image, net)
    #     dets = filter_iris_box(dets, width_height_threshold)
    #     bboxes = dets[dets[:, 4] > conf_threshold]
    #     if bboxes.shape[0] > 0:
    #         # print('Predicted Open')
    #         open_open += 1
    #         paint_bbox(image, dets)
    #         # cv2.imwrite(os.path.join(out_open_image_dir, image_file), image)
    #     else:
    #         # print('Predicted Close')
    #         open_close += 1
    #         paint_bbox(image, dets)
    #         # cv2.imwrite(os.path.join(out_close_image_dir, image_file), image)
    #     # paint_bbox(image, dets)

    # rgb_image_dir = os.path.join('../datasets', 'eyestate_label', 'outputir', 'Close')
    # out_open_image_dir = os.path.join('../datasets', 'out', 'outputir', 'close_open')
    # out_close_image_dir = os.path.join('../datasets', 'out', 'outputir', 'close_close')
    # if not os.path.isdir(out_open_image_dir):
    #     os.makedirs(out_open_image_dir)
    # if not os.path.isdir(out_close_image_dir):
    #     os.makedirs(out_close_image_dir)
    # listdir = os.listdir(rgb_image_dir)
    # listdir = os.listdir(rgb_image_dir)
    # for image_file in tqdm(listdir):
    #     image = cv2.imread(os.path.join(rgb_image_dir, image_file))
    #     dets = detect_iris(image, net)
    #     dets = filter_iris_box(dets, width_height_threshold)
    #     dets = dets[dets[:, 4] > conf_threshold]
    #     if dets.shape[0] > 0:
    #         # print('Predicted Open')
    #         close_open += 1
    #         paint_bbox(image, dets)
    #         # cv2.imwrite(os.path.join(out_open_image_dir, image_file), image)
    #     else:
    #         # print('Predicted Close')
    #         close_close += 1
    #         # cv2.imwrite(os.path.join(out_close_image_dir, image_file), image)
    #     # paint_bbox(image, dets)
    #     # cv2.imwrite(os.path.join(out_open_image_dir, image_file), image)
    # print(open_open, open_close, close_open, close_close)
    # # print('Accuracy with open: {}'.format(match/total))

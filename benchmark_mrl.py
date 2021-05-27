import torch
from time import time
import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A
import albumentations.pytorch as AP

from data.mrl import MRLEyeDataset
from data import cfg_mnet
from models import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode

cfg = cfg_mnet

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

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        print('load to cuda')
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model = model.cuda()
    model.eval()
    print(model)
    return model

def detect_iris(img_raw, net):
    device='cuda'
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

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / 1
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)

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

net_path = os.path.join('weights_negpos_cleaned', 'mobilenet0.25_Final.pth')
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, net_path, False)
net.eval()

if __name__ == '__main__':
    width_height_threshold = 0.4875
    conf_threshold = 0.80625
    dataset_path = os.path.join('dataset', 'mrleye')
    outdir = os.path.join('dataset', 'mrleye', 'out')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    dataset = MRLEyeDataset(dataset_path, set_type='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    # label_predict
    open_open = 0
    open_close = 0
    close_open = 0
    close_close = 0
    i = 0
    for image, label in tqdm(dataloader):
        image = np.array(image[0], dtype=np.uint8)
        label = label[0].item()
        dets = detect_iris(image, net)
        b = filter_iris_box(dets, width_height_threshold)
        b = b[b[:, 4] > conf_threshold]
        predicted = 0
        if b.shape[0] > 0:
            predicted = 1
        
        if label == 1 and predicted == 1:
            open_open += 1
        if label == 1 and predicted == 0:
            open_close += 1
        if label == 0 and predicted == 1:
            close_open += 1
        if label == 0 and predicted == 0:
            close_close += 1
        
        # print('width', b[2]-b[0])
        # print('height', b[3]-b[1])
        # print('predicted', predicted)
        # print('label', label)
        if b.shape[0] > 0:
            b = b[0]
            text = "{:.4f}".format(b[4])
            b = [int(x) for x in b[:4]]
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
            cx = b[0]
            cy = b[1] - 12
            image = cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imwrite(os.path.join(outdir, '{}.jpg'.format(i)), image)
        i += 1
    print(open_open, open_close, close_open, close_close)
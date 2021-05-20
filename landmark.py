import os
import numpy as np

import torch
import cv2
from PIL import Image
from time import time

from torchalign import FacialLandmarkDetector
from data import cfg_mnet
from models_class import IrisModel
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode
import albumentations as A
import albumentations.pytorch as AP

torch.set_grad_enabled(False)

transformerr = A.Compose(
    [
        A.Normalize(),
        AP.ToTensorV2(),
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

def segment_eye(image, lmks, eye='left', ow=96, oh=96, transform_mat=None):
        if eye=='left':
            # Left eye
            x1, y1 = lmks[66]
            x2, y2 = lmks[70]
        else: # right eye
            x1, y1 = lmks[75]
            x2, y2 = lmks[79]


        eye_width = 1.5 * np.linalg.norm(x1-x2)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        
        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        
        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]

        # Get rotated and scaled, and segmented image
        # transform_mat = center_mat * scale_mat * translate_mat
        # Re-centre eye image such that eye fits (based on determined `eye_middle`)
        recentre_mat = np.asmatrix(np.eye(3))

        # Apply transforms
        if transform_mat is None:
            transform_mat =  recentre_mat * center_mat * scale_mat * translate_mat

        eye_image = cv2.warpAffine(image, transform_mat[:2, :], (oh, ow))
        # eye_image = cv2.normalize(eye_image, eye_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return eye_image, transform_mat

def detect_iris(img_raw, net):
    confidence_threshold = 0.02
    top_k = 5000
    nms_threshold=0.4
    keep_top_k = 750

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img = transform(img).unsqueeze(0)
    # img -= (104, 117, 123)
    # img = img.transpose(2, 0, 1)
    # img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time()
    # loc, conf, landms = net(img)  # forward pass
    loc, conf = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time() - tic))

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

def paint_bbox(image, bboxes):
    for b in bboxes:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] - 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

def paint_landmark(image, landmark):
    for (x, y) in landmark:
        cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

def filter_iris_box(dets):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) > 0.5
    print(heights/widths)
    # print(mask)
    dets = dets[mask]
    return dets

if __name__ == '__main__':
    landmark_model_path = os.path.join('landmark_models', 'lapa', 'hrnet18_256x256_p2')
    landmark_model = FacialLandmarkDetector(landmark_model_path)
    landmark_model.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    landmark_model.to(device)

    cfg = cfg_mnet
    # net and model
    net_path = os.path.join('weights_without_prepoc', 'mobilenet0.25_Final.pth')
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, net_path, True)
    net = net.cuda()
    net.eval()
    # net = IrisModel()
    # checkpoint = torch.load('../iris_classification/logs/training/version_0/checkpoints/checkpoint-epoch=13-val_loss=0.2990.ckpt', map_location=torch.device('cpu'))
    # net.load_state_dict(checkpoint['state_dict'])
    # net.eval()

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('ir3/out2.mp4')
    cap = cv2.VideoCapture('record.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_record_detection_negpos.avi', fourcc, 20.0, (320, 240))

    while(cap.isOpened()):
        ret, frame = cap.read()
        print(frame.shape)
        # frame = cv2.resize(frame, (320, 240))
        # frame = cv2.resize(frame, (256, 256))
        image = Image.fromarray(frame[:,:,::-1])
        start = time()
        landmarks = landmark_model(image, None, device=device)[0]
        landmarks = landmarks.cpu()
        end = time()
        print("Landmark time: {}".format(end-start))
        left_eye, left_transform_mat = segment_eye(frame, landmarks, 'left')
        right_eye, right_transform_mat = segment_eye(frame, landmarks, 'right')
        paint_landmark(frame, landmarks)

        # left_eye_resize = cv2.resize(left_eye, (32, 32))
        # cx, cy = left_eye_resize.shape[0] // 2, left_eye_resize.shape[1] // 2
        # left_eye_tensor = transform(left_eye_resize).unsqueeze_(0)
        # # left_eye_tensor = torch.FloatTensor(left_eye_resize.transpose(2, 0, 1)).unsqueeze_(0)
        # start = time()
        # logit = net(left_eye_tensor)[0]
        # end = time()
        # print("Open/Close time: {}".format(end-start))
        # pred = logit.argmax(dim=0)
        # text = 'close' if pred == 0 else 'open'
        # print(logit, text)
        # cv2.putText(left_eye, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # right_eye_resize = cv2.resize(right_eye, (32, 32))
        # cx, cy = right_eye_resize.shape[0] // 2, right_eye_resize.shape[1] // 2
        # right_eye_tensor = transform(right_eye_resize).unsqueeze_(0)
        # # right_eye_tensor = torch.FloatTensor(right_eye_resize.transpose(2, 0, 1)).unsqueeze_(0)
        # logit = net(right_eye_tensor)[0]
        # pred = logit.argmax(dim=0)
        # text = 'close' if pred == 0 else 'open'
        # cv2.putText(right_eye, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        dets = detect_iris(left_eye, net)
        dets = filter_iris_box(dets)
        dets = dets[dets[:, 4] > 0.85]
        paint_bbox(left_eye, dets)
        dets = detect_iris(right_eye, net)
        dets = filter_iris_box(dets)
        dets = dets[dets[:, 4] > 0.85]
        paint_bbox(right_eye, dets)
        left_eye = cv2.resize(left_eye, (48, 48))
        right_eye = cv2.resize(right_eye, (48, 48))
        eyes = np.concatenate([left_eye, right_eye], axis=1)
        h, w = eyes.shape[:2]
        frame[:h, :w] = eyes

        # for point in landmark[0]:
        #     frame = cv2.circle(frame, (point[0], point[1]), 2, (255,0,0), -1)
        # print(landmark.shape)
        out.write(frame)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
        
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

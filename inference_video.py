import os
import numpy as np

import torch
import cv2
from PIL import Image
from time import time
from retinaface import RetinaFace as Detector
import torch.nn.functional as F

from data import cfg_mnet
from models_class import IrisModel
from models import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode
import albumentations as A
import albumentations.pytorch as AP
from models_heatmap.heatmapmodel import HeatMapLandmarker

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

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

def square_box(box, ori_shape):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    w = max(x2-x1, y2-y1)*1.2
    x1 = cx - w//2
    y1 = cy - w//2
    x2 = cx + w//2
    y2 = cy + w//2

    x1 = max(x1, 0)
    y1 = max(y1+(y2-y1)*0, 0)
    x2 = min(x2-(x1-x1)*0, ori_shape[1]-1)
    y2 = min(y2, ori_shape[0]-1)

    return [x1, y1, x2, y2]

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
    model.migrate(state_dict, force=True)
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


priorbox = PriorBox(cfg_mnet, image_size=(96, 96))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to('cpu')
    prior_data = priors.data
def detect_iris(img, net):
    img = transform(img).unsqueeze(0)
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

def filter_iris_box(dets, threshold):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) > threshold
    dets = dets[mask]
    return dets

def filter_conf(dets, threshold):
    return dets[dets[:, 4] > threshold]

def predict_landmark(detector, img, model, device):
    faces = detector.predict(img)
    lmks = None
    if len(faces) !=0 :
        box = [faces[0]['x1'], faces[0]['y1'], faces[0]['x2'], faces[0]['y2']]
        box = square_box(box, img.shape)
        box = list(map(int, box))
        x1, y1, x2, y2 = box

        # Inference lmks
        crop_face = img[y1:y2, x1:x2]
        crop_face = cv2.resize(crop_face, (256, 256))
        img_tensor = transform(crop_face)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # 1x3x256x256

        heatmapPRED, lmks = model(img_tensor.to(device))

        lmks = lmks.cpu().detach().numpy()[0] # 106x2
        lmks = lmks/256.0  # Scale into 0-1 coordination
        lmks[:,0], lmks[:,1] = lmks[: ,0] * (x2-x1) + x1 ,\
                            lmks[:, 1] * (y2-y1) + y1
    return lmks

def load_heatmap_model():
    landmark_model = HeatMapLandmarker()
    model_path = "../heatmap-based-landmarker/ckpt/epoch_80.pth.tar"
    checkpoint = torch.load(model_path, map_location=device)
    landmark_model.load_state_dict(checkpoint['plfd_backbone'])
    landmark_model.to(device)
    landmark_model.eval()
    return landmark_model

if __name__ == '__main__':
    detector = Detector(quality="normal")
    device = 'cpu'
    landmark_model = load_heatmap_model()
    cfg = cfg_mnet
    net_path = 'training_lapa_ir_logs/mobilenet0.25/checkpoints/checkpoint-epoch=13-val_loss=4.6626.ckpt'
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, net_path, True)
    net.eval()
    cap = cv2.VideoCapture('../video/output_tatden5.mkv')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../video/output_tatden5_det_lapa_ir_2.avi', fourcc, 20.0, (1280, 720))

    # i = 0
    conf_threshold = 0.80625
    width_height_threshold = 0.4875

    while(cap.isOpened()):
        ret, frame = cap.read()
        print(frame.shape)
        start = time()
        landmarks = predict_landmark(detector, frame, landmark_model, device)
        if landmarks is not None:
            end = time()
            print("Landmark time: {}".format(end-start))
            left_eye, left_transform_mat = segment_eye(frame, landmarks, 'left')
            right_eye, right_transform_mat = segment_eye(frame, landmarks, 'right')
            paint_landmark(frame, landmarks)

            start = time()
            dets = detect_iris(left_eye, net)
            end = time()
            print("Left eye time: {}".format(end-start))
            dets[:, :4] *= 96
            dets = filter_iris_box(dets, width_height_threshold)
            dets = filter_conf(dets, conf_threshold)
            paint_bbox(left_eye, dets)
            start = time()
            dets = detect_iris(right_eye, net)
            end = time()
            print("Right eye time: {}".format(end-start))
            dets[:, :4] *= 96
            dets = filter_iris_box(dets, width_height_threshold)
            dets = filter_conf(dets, conf_threshold)
            paint_bbox(right_eye, dets)
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

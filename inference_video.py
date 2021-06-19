import os
import numpy as np

import torch
import cv2
from PIL import Image
from time import time
from retinaface import RetinaFace as Detector
import torch.nn.functional as F
from tqdm import tqdm

from data import cfg_mnet, cfg_re34
from models_class import IrisModel
from models import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode
import albumentations as A
import albumentations.pytorch as AP
from models_heatmap.heatmapmodel import HeatMapLandmarker

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
cfg = cfg_re34
device='cuda:1'

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

def load_model(model, pretrained_path, device):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if device ==  'cpu':
        print('Load to cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device(device))
    # state_dict = pretrained_dict
    state_dict = pretrained_dict['state_dict']
    print(state_dict.keys())
    model.migrate(state_dict, force=True)
    model = model.to(device)
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


priorbox = PriorBox(cfg, image_size=(96, 96))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
variance = torch.Tensor(cfg['variance']).to(device)

def calculate_box(loc, conf):
    scores = conf[:, 1]
    ind = scores.argmax()
    boxes = decode(loc, prior_data, variance)
    scores = scores[ind:ind+1]
    boxes = boxes[ind:ind+1]
    dets = np.hstack((boxes.cpu().numpy(), scores.cpu().numpy()[:, np.newaxis])).astype(np.float32, copy=False)
    return dets

# def detect_iris(img, net):
#     img = transform(img).unsqueeze(0)
#     # img = np.float32(img)
#     # img -= (104, 117, 123)
#     # img = img.transpose(2, 0, 1)
#     # img = torch.from_numpy(img).unsqueeze(0)
#     img = img.to(device)
#     loc, conf = net(img)  # forward pass
#     loc = loc.squeeze(0)
#     conf = conf.squeeze(0)
#     split_index = loc.shape[0] // 2
#     print(conf.shape)
#     loc_left, loc_right = loc[:split_index], loc[split_index:]
#     conf_left, conf_right = conf[:split_index], conf[split_index:]
#     print(loc_left.shape, loc_right.shape)
#     print(conf_left.shape, conf_right.shape)
#     return calculate_box(loc_left, conf_left), calculate_box(loc_right, conf_right)

def detect_iris(img, net):
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    loc, conf = net(img)  # forward pass
    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    # print(conf.shape)
    return calculate_box(loc, conf)

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
        cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)

def filter_iris_box(dets, threshold):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) > threshold
    return dets[mask]

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

        _, lmks = model(img_tensor.to(device))

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

def detect_one_video(detector, landmark_model, net, cap_path, out_path):
    cap = cv2.VideoCapture(cap_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (1280, 720))

    conf_threshold = 0.85
    width_height_threshold = 0.6
    print("Predicting {}...".format(cap_path))

    for _ in tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break
        # print(frame.shape)
        start = time()
        landmarks = predict_landmark(detector, frame, landmark_model, device)
        if landmarks is not None:
            end = time()
            # print("Landmark time: {}".format(end-start))
            left_eye, left_transform_mat = segment_eye(frame, landmarks, 'left')
            right_eye, right_transform_mat = segment_eye(frame, landmarks, 'right')
            paint_landmark(frame, landmarks)
            start = time()
            left_dets = detect_iris(left_eye, net)
            right_dets = detect_iris(right_eye, net)
            end = time()
            # print("Eye time: {}".format(end-start))

            left_dets[:, :4] *= 96
            left_dets = filter_iris_box(left_dets, width_height_threshold)
            left_dets = filter_conf(left_dets, conf_threshold)
            paint_bbox(left_eye, left_dets)

            right_dets[:, :4] *= 96
            right_dets = filter_iris_box(right_dets, width_height_threshold)
            right_dets = filter_conf(right_dets, conf_threshold)
            paint_bbox(right_eye, right_dets)

            eyes = np.concatenate([left_eye, right_eye], axis=1)
            h, w = eyes.shape[:2]
            frame[:h, :w] = eyes
        out.write(frame)
        
    cap.release()
    out.release()

def mkdir_if_not(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

if __name__ == '__main__':
    detector = Detector(quality="normal")
    landmark_model = load_heatmap_model()
    net_path = 'logs/resnet34_logs/version_0/checkpoints/last.ckpt'
    net = RetinaFace(cfg=cfg, phase = 'test')
    print(net)
    net = load_model(net, net_path, device)
    net.eval()
    dms_folder = '/vinai/tienpv12/datasets/20201201'
    out_folder = '/vinai/tienpv12/out_1_range/20201201'

    for folder in os.listdir(dms_folder):
        if '.' in folder:
            continue
        rgb_video_path = os.path.join(dms_folder, folder, 'WH_RGB.mp4')
        ir_video_path = os.path.join(dms_folder, folder, 'WH_IR.mp4')
        rgb_out_video_path = rgb_video_path.replace(dms_folder, out_folder).replace('mp4', 'avi')
        ir_out_video_path = ir_video_path.replace(dms_folder, out_folder).replace('mp4', 'avi')
        mkdir_if_not(rgb_out_video_path)
        mkdir_if_not(ir_out_video_path)
        detect_one_video(detector, landmark_model, net, rgb_video_path, rgb_out_video_path)
        detect_one_video(detector, landmark_model, net, ir_video_path, ir_out_video_path)

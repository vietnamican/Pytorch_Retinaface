import torch
import cv2
import albumentations as A
import albumentations.pytorch as AP
import numpy as np
from utils.box_utils import decode

from models import RetinaFace
from layers.functions.prior_box import PriorBox

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32]],
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
    'out_channel': 64
}

checkpoint_path = 'training_lapa_ir_logs/mobilenet0.25/checkpoints/checkpoint-epoch=249-val_loss=5.6218.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
state_dict = checkpoint['state_dict']
model = RetinaFace(cfg=cfg, phase='test')
model.migrate(state_dict, force=True)
model = model.to(device)
model.eval()

transform = A.Compose(
    [
        A.Normalize(),
        AP.ToTensorV2()
    ]
)

# default image_size=(96, 96)
priorbox = PriorBox(cfg, image_size=(96, 96))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data.cpu()

def calculate_box(loc, conf):
    scores = conf[:, 1]
    ind = scores.argmax()
    boxes = decode(loc, prior_data, cfg['variance'])
    scores = scores[ind:ind+1]
    boxes = boxes[ind:ind+1]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    return dets

def detect_iris(img, net):
    img = transform(image=img)['image'].unsqueeze(0)
    img = img.to(device)
    loc, conf = net(img)  # forward pass
    loc = loc.squeeze(0).data.cpu()
    conf = conf.squeeze(0).data.cpu().numpy()
    split_index = loc.shape[0] // 2
    loc_left, loc_right = loc[:split_index], loc[-split_index:]
    conf_left, conf_right = conf[:split_index], conf[-split_index:]
    return calculate_box(loc_left, conf_left), calculate_box(loc_right, conf_right)


def filter_iris_box(dets, threshold):
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    mask = (heights / widths) > threshold
    dets = dets[mask]
    return dets

def filter_conf(dets, threshold):
    return dets[dets[:, 4] > threshold]

def paint_box(image, bboxes):
    for b in bboxes:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] - 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

if __name__ == '__main__':

    # grid search on LaPa val set
    conf_threshold = 0.80625
    width_height_threshold = 0.4875

    image_path = 'sample.jpeg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_fusion = np.concatenate([image, image], axis=0)
    left_dets, right_dets = detect_iris(image_fusion, model)

    # convert ratio to pixel
    height, width = image.shape[:2]
    left_dets[:, (0, 2)]*= width
    left_dets[:, (1, 3)]*= height
    right_dets[:, (0, 2)]*= width
    right_dets[:, (1, 3)]*= height

    dets = filter_conf(left_dets, conf_threshold)
    dets = filter_iris_box(dets, width_height_threshold)
    image_copy = image.copy()
    paint_box(image_copy, dets)
    cv2.imwrite('result_left.jpg', image_copy)

    dets = filter_conf(right_dets, conf_threshold)
    dets = filter_iris_box(dets, width_height_threshold)
    image_copy = image.copy()
    paint_box(image_copy, dets)
    cv2.imwrite('result_right.jpg', image_copy)
    # print(dets)


# # summary(model, (2, 3, 96, 96))

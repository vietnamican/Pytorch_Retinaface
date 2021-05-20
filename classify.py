from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
from models_class import IrisModel
from models_class.datasets import LaPa
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Retinaface')

# parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    # net and model
    checkpoint_path = '../iris_classification/logs/training/version_0/checkpoints/checkpoint-epoch=13-val_loss=0.2990.ckpt'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    net = IrisModel()
    net.load_state_dict(state_dict)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1


    # testing begin
    # for i in range(100):
    # image_path = "./curve/test.jpg"
    image_dir = os.path.join('../LaPa_negpos_fusion', 'val', 'images')
    label_dir = os.path.join('../LaPa_negpos_fusion', 'val', 'labels')
    out_dir = os.path.join('out_augment', 'val', 'images')
    dataset = LaPa(image_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    num_rights = 0
    num_total = 0
    for image, target in tqdm(dataloader):
        image = image.cuda()
        target = target.cuda()
        # target = [t.cuda() for t in target]
        logit = net(image)
        pred = logit.argmax(dim=1)
        # print(pred.shape)
        # print(target.shape)
        num_right = (pred == target).float().sum()
        num_rights += num_right
        num_total += target.shape[0]
        # show image
        # if args.save_image:
        #     for b in dets:
        #         if b[4] < args.vis_thres:
        #             continue
        #         text = "{:.4f}".format(b[4])
        #         b = list(map(int, b))
        #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #         cx = b[0]
        #         cy = b[1] + 12
        #         cv2.putText(img_raw, text, (cx, cy),
        #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        #         # landms
        #         # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        #         # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        #         # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        #         # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        #         # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        #     # save image
        #     name = os.path.join(out_dir, image_file)
        #     # name = "test.jpg"
        #     cv2.imwrite(name, img_raw)
    
    print('accuracy: ', num_rights/num_total)


import numpy as np
import cv2
import sys
from tqdm import tqdm

from torch.utils import data
import glob
import os

import ast
import random
import math
import torch


class LAPA106DataSet(data.Dataset):
    TARGET_IMAGE_SIZE = (256, 256)

    def __segment_eye(self, image, lmks, eye='left', ow=96, oh=96, transform_mat=None):
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
        ih, iw, _ = image.shape
        if eye=='left':
            eyeindices = [66,67,68,69,70,71,72,73]
        elif eye=='right':
            eyeindices = [75,76,77,78,79,80,81,82]
        else:
            raise NotImplementedError()
        
        eye_middle = np.mean(lmks[eyeindices], axis=1)


        if self.augment:
            recentre_mat[0, 2] += random.randint(0, ow//4)  # x
            recentre_mat[1, 2] += random.randint(0, oh//4)  # y

        # Apply transforms
        if transform_mat is None:
            transform_mat =  recentre_mat * center_mat * scale_mat * translate_mat

        eye_image = cv2.warpAffine(image, transform_mat[:2, :], (oh, ow))

        return eye_image, transform_mat

    def __init__(self, img_dir, anno_dir, out_img_dir, out_label_dir, augment=False, transforms=None, imgsize=256):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.out_dir = out_dir
        self.out_label_dir = out_label_dir
        self.transforms = transforms
        self.augment = augment
        self.img_path_list = glob.glob(img_dir + "/*.jpg")
        self.TARGET_IMAGE_SIZE = (imgsize, imgsize)
        if not os.path.isdir(self.out_img_dir):
            os.makedirs(self.out_img_dir)
        if not os.path.isdir(self.out_label_dir):
            os.makedirs(self.out_label_dir)

    def __len__(self):
        return len(self.img_path_list)

    def _get_106_landmarks(self, path):
        file1 = open(path, 'r') 
        ls = file1.readlines() 
        ls = ls[1:] # Get only lines that contain landmarks

        lm = []
        for l in ls:
            l = l.replace("\n","")
            a = l.split(" ")
            a = [float(i) for i in a]
            lm.append(a)
        
        lm = np.array(lm)
        assert len(lm)==106, "There should be 106 landmarks. Get {len(lm)}"
        return lm


    def __get_default_item(self):
        return self.__getitem__(0)

    def __draw_boundingbox(self, image, landmark):
        # landmark order like
        #       2
        #    1     3
        # 0     8     4
        #    7     5
        #       6
        landmark = landmark[0]
        top1, top2 = landmark[1], landmark[3]
        bottom1, bottom2 = landmark[7], landmark[5]
        width = max((top2-top1)[0], (bottom2 - bottom1)[0])
        width = width * 3 / 4
        height = (landmark[6] - landmark[2])[1]
        center_x, center_y = landmark[8]
        left, top, right, bottom = center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2
        left, top, right, bottom = round(left), round(top), round(right), round(bottom)
        return image

    def __write_label(self, out_label_dir, img_name, landmark):
        if len(landmark == 1):
            landmark = landmark[0]
        top1, top2 = landmark[1], landmark[3]
        bottom1, bottom2 = landmark[7], landmark[5]
        width = max((top2-top1)[0], (bottom2 - bottom1)[0]) * 3 / 4
        height = (landmark[6] - landmark[2])[1]
        center_x, center_y = landmark[8]
        left, top = center_x - width / 2, center_y - height / 2
        left, top, width, height = round(left), round(top), round(width), round(height)
        line = "{} {} {} {} {}".format(0, left, top, width, height)
        with open(os.path.join(out_label_dir, img_name + ".txt"), 'w') as f:
            f.write(line)

    def create_mask(self, img, lmks):
        mask = np.zeros((img.shape[1], img.shape[0], 1))
        lmks = np.int32(lmks)
        mask = cv2.fillConvexPoly(mask, lmks, 1)

        return mask

    def  __getitem__(self, index):
        outdir = 'Label'
        f = self.img_path_list[index]
        img = cv2.imread(f)
        replacing_extension = ".txt"
        self.landmark = self._get_106_landmarks(f.replace(self.img_dir, self.anno_dir).replace(".jpg", replacing_extension))

        image = img.copy()

        # return eye_cropped and transform matrix
        left_eye, transform_mat_left = self.__segment_eye(img, self.landmark, eye='left', ow=self.TARGET_IMAGE_SIZE[1], oh=self.TARGET_IMAGE_SIZE[0])
        right_eye, transform_mat_right = self.__segment_eye(img, self.landmark, eye='right', ow=self.TARGET_IMAGE_SIZE[1], oh=self.TARGET_IMAGE_SIZE[0])
        
        # using transform matrix to transform landmark points
        left_eye_landmarks = np.array([self.landmark[[66,67,68,69,70,71,72,73,104]]])
        right_eye_landmarks = np.array([self.landmark[[75,76,77,78,79,80,81,82,105]]])
        if math.isnan(transform_mat_left[0, 0]) or math.isnan(transform_mat_right[0, 0]):
            return None, None, None
        left_eye_landmarks = cv2.transform(left_eye_landmarks, transform_mat_left[:2 :])
        right_eye_landmarks = cv2.transform(right_eye_landmarks, transform_mat_right[:2 :])
        
        img_name = os.path.basename(f)
        img_name = img_name.split('.')[0]
        left_eye_rgb = left_eye.copy()
        right_eye_rgb = right_eye.copy()
       

        # Calulate EARs
        v1 = np.linalg.norm(self.landmark[67]-self.landmark[73])
        v2 = np.linalg.norm(self.landmark[68]-self.landmark[72])
        v3 = np.linalg.norm(self.landmark[69]-self.landmark[71])
        h = np.linalg.norm(self.landmark[66]-self.landmark[70])
        left_ear = np.max([v1,v2,v3])/(h+0.00001)
        left_ear = min(left_ear,1.0)

        v1 = np.linalg.norm(self.landmark[76]-self.landmark[82])
        v2 = np.linalg.norm(self.landmark[77]-self.landmark[81])
        v3 = np.linalg.norm(self.landmark[78]-self.landmark[80])
        h = np.linalg.norm(self.landmark[75]-self.landmark[79])
        right_ear = np.max([v1,v2,v3])/(h+0.000001)
        right_ear = min(right_ear,1.0)

        if not left_ear < 0.15:
            cv2.imwrite(os.path.join(self.out_img_dir, img_name + "_left.jpg"), left_eye_rgb)
            self.__write_label(self.out_label_dir, img_name + "_left", left_eye_landmarks)
        if not right_ear < 0.15:
            cv2.imwrite(os.path.join(self.out_img_dir, img_name + "_right.jpg"), right_eye_rgb)
            self.__write_label(self.out_label_dir, img_name + "_right", right_eye_landmarks)

        return None, None, None

if __name__ == "__main__":
   
    lapa = LAPA106DataSet(img_dir=os.path.join('LaPa', 'test', 'images'),
                          anno_dir=os.path.join('LaPa', 'test', 'landmarks'), augment=True, imgsize=96)
    
    for eye, ear, mask in tqdm(lapa):
        pass
        

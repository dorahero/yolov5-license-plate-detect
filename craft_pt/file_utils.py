# -*- coding: utf-8 -*-
import time
import math
import glob
import pandas as pd
import re
import os
import numpy as np
import cv2
import craft_pt.imgproc as imgproc
from PIL import Image
   

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='/home/ub/red/CRAFT-pytorch-master/result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        color = (255, 255, 255)
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        # res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        word_list = []
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            # print(box)
            word_list.append(poly)
            # strResult = ','.join([str(p) for p in poly]) + '\r\n'
            # f.write(strResult)
            
            poly = poly.reshape(-1, 2)
            # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            cv2.fillPoly(img, [poly.reshape((-1, 1, 2))], color)
            # ptColor = (0, 255, 255)
            # if verticals is not None:
            #     if verticals[i]:
            #         ptColor = (255, 0, 0)

            # if texts is not None:
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.5
            #     cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
            #     cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
        # word_list = sorted(word_list, key=lambda x: x[0])  #按X坐標排序
        # # print(word_list)
        # thresh_list = []
        # for j, w in enumerate(word_list): 
        #     print(w)
        #     word = img[w[1]: w[-1], w[0]: w[2]]
        #     # cv2.imwrite(res_img_file, word)
        #     # gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)  # 灰階
        #     # _,thresh = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY) #轉為黑白
        #     # gray = cv2.GaussianBlur(gray, (5, 5), 10)
        #     _,thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白   
        #     # cv2.imshow('test', word)
        #     t_h, t_w = thresh.shape
        #     # print(thresh.shape)
        #     #  and 2 < w[0] and w[2] < 278
        #     if 80 > t_h > 35 and 58 > t_w > 10:
        #         # poly_0 = w.reshape(-1, 2)
        #         # cv2.polylines(img, [poly_0.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        #         thresh_list.append(thresh)
        #         # cv2.imwrite(dirname + '/word/' + filename + "_" + str(count_num) + '.jpg', thresh)
        #         # count_num += 1
        #     elif 80 > t_h > 35 and 90 > t_w >= 58:
        #         # cv2.polylines(img, [poly_0.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        #         thresh_1 = thresh[:, :int(t_w/2)]
        #         thresh_2 = thresh[:, int(t_w/2):]
        #         # print(dirname)
        #         thresh_list.append(thresh_1)
        #         thresh_list.append(thresh_2)
        #         # cv2.imwrite(dirname + '/word/' + filename + "_" + str(count_num) + '.jpg', thresh_1)
        #         # count_num += 1
        #         # cv2.imwrite(dirname + '/word/' + filename + "_" + str(count_num) + '.jpg', thresh_2)
        #         # count_num += 1
        #     else:
        #         continue
        #     # print(thresh.shape)
        # # Save result image
        # # if len(thresh_list) >= 4:
        # #     for x, t in enumerate(thresh_list):
        # #         cv2.imwrite(dirname + '/word/' + filename + "_" + str(x) + '.jpg', t)
        # #     return img, True
        if len(boxes) > 0: 
            # cv2.imwrite(dirname + str(time.time()) + '.jpg', img)
            return img, False
        else:
            img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, -1)
            # cv2.imwrite(dirname + str(time.time()) + '.jpg', img)
            return img, False


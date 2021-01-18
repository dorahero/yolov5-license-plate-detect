"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_pt.craft_utils as craft_utils
import craft_pt.imgproc as imgproc
import craft_pt.file_utils as file_utils
import json
import zipfile

from craft_pt.craft import CRAFT

from collections import OrderedDict
import pandas as pd
import os
import shutil

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
# parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=560, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

# args = parser.parse_args()


# """ For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

# result_folder = '/home/ub/red/CRAFT-pytorch-master/result/'
# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net, args):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    # print(img_resized.shape)
    # print(target_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    # print(x.shape)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    # print(x.shape)
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # print(x.shape)
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()

    # print(score_text.shape)
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # print(boxes)
    # print(len(polys), '========')
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # print(len(polys), '========')
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    # print(polys)
    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    # render_img = score_link.copy()
    # render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    # print(ret_score_text.shape)
    # gray = cv2.cvtColor(ret_score_text, cv2.COLOR_BGR2GRAY)  # 灰階
    # _,thresh = cv2.threshold(ret_score_text, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白
    # # _,thresh = cv2.threshold(thresh, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白
    # # print(thresh.shape)
    # # thresh = thresh[5:95, :]
    # contours1 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #尋找輪廓
    # # print(contours1)
    # contours = contours1[0]   #取得輪廓
    # letter_image_regions = [] #文字圖形串列
    # # print(contours)
    # for i, contour in enumerate(contours):  #依序處理輪廓
    #     (x, y, w, h) = cv2.boundingRect(contour)  #單一輪廓資料
    #     letter_image_regions.append((x, y, w, h))
    # letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])  #按X坐標排序
    # for box in letter_image_regions:  #依序處理輪廓資料
    #     # print(box['coord'])
    #     x, y, w, h = box       
    #     cv2.rectangle(thresh, (x, y), (x+w, y+h), (0, 255, 255), 2)
    # cv2.imshow('123', thresh)
    # cv2.imshow('2', ret_score_text)
    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def lp_pre(net, img, args):
    image = imgproc.loadImage(img)
    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net=None, args=args)
    
    # save score text
    # filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)
    try:
        img, tof = file_utils.saveResult('', image, polys)
        # print("elapsed time : {}s".format(time.time() - t))
        # if img is not None:
        #     img_dict.update({'{}'.format(image_path): img})
        # print(tof)
        return img
    except Exception as e:
        print(e)
        return cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)

if __name__ == '__main__':
    # load net
    if os.path.exists('./result'):
        shutil.rmtree('./result')  # delete output folder
    os.makedirs('./result')  # make new output folder
    img_dict = main()
    for img in img_dict:
        cv2.imwrite('./result/' + img.split('/')[-1], img_dict[img])

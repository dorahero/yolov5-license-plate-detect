# -*- coding: utf-8 -*-      
import time
import math
import glob
import pandas as pd
import re
import os
import numpy as np
import cv2
import craft_pt.imgproc_lp as imgproc
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
nn = 0
def bgx(img):
    def area(row, col):
        global nn
        if bg[row][col] != 255:
            pass
        bg[row][col] = lifearea #記錄生命區的編號
        if col>1: #左方
            if bg[row][col-1]==255:
                nn +=1
                area(row,col-1)
        if col< w-1: #右方
            if bg[row][col+1]==255:
                nn +=1
                area(row,col+1)             
        if row>1: #上方
            if bg[row-1][col]==255:
                nn+=1            
                area(row-1,col)
        if row<h-1: #下方
            if bg[row+1][col]==255:
                nn+=1            
                area(row+1,col)      
    width = 25
    height = 35
    dim = (width, height)
    thresh = cv2.resize(img, dim)
    h, w = thresh.shape
    for i in range(h):  #i為高度
        for j in range(w): #j為寬度  
            if thresh[i][j] == 255:     #顏色為白色
                count = 0 
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        try:
                            if thresh[i + k][j + l] == 255: #若是白點就將count加1
                                count += 1
                        except IndexError:
                            pass
                if count <= 6:  #週圍少於等於6個白點
                    thresh[i][j] = 0  #將白點去除 
    bg = thresh.copy()
    lifearea=0 # 生命區塊
    global nn
    nn = 0       # 每個生命區塊的生命數
    life=[]    # 記錄每個生命區塊的生命數串列            
    for row in range(0,h):
        for col in range(0,w):
            if bg[row][col] == 255:
                nn = 1  #生命起源
                lifearea = lifearea + 1  #生命區塊數
                area(row,col)  #以生命起源為起點探索每個生命區塊的總生命數
                life.append(nn)
    # print(max(life))
    try:
        print(max(life))
        maxlife=max(life) #找到最大的生命數
    except ValueError as e:
        print(e)
        return None
    indexmaxlife=life.index(maxlife) #找到最大的生命數的區塊編號          
        
    for row in range(0,h):
        for col in range(0,w):
            if bg[row][col] == indexmaxlife+1:
                bg[row][col]=255
            else:
                bg[row][col]=0  
    # print(bg)   
    if max(life) > 100:
        print('====')
        # cv2.imwrite('./word_bg/' + f.split('/')[-1], bg)
        return bg
    else:
        return None

def deskew(img):
    m = cv2.moments(img)
    h, w = img.shape
    # print(m)
    if abs(m['mu02']) < 1e-2:    
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*h*skew], [0,1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    # print(img.shape)
    # cv2.imshow('3', img)
    black = np.zeros((h+10, w+10), dtype=np.uint8)
    black[5:h+5, 5:w+5] = img
    _,thresh = cv2.threshold(black, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白
    thresh = cv2.resize(thresh, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    # dig_2 = cv2.GaussianBlur(thresh, (5, 5), 10)
    # _,thresh = cv2.threshold(dig_2, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白
    # _,thresh = cv2.threshold(thresh, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白
    return thresh

def openball(img):
    kernel_1 = np.ones((3, 3), dtype=np.uint8)
    kernel_2 = np.ones((5, 5), dtype=np.uint8)
    dilate = cv2.dilate(img, kernel_1, 1)
    erosion = cv2.erode(dilate, kernel_1, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_2, 5)  
    blur = cv2.blur(opening, (3, 3))
    _,thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY) #轉為黑白
    bg = np.zeros((90, 400), np.uint8)
    bg.fill(255)
    print(thresh.shape)
    print(bg.shape)
    t_h, t_w = thresh.shape
    b_h, b_w = bg.shape
    bg[int((b_h-t_h)/2): t_h + int((b_h-t_h)/2), int((b_w-t_w)/2): t_w + int((b_w-t_w)/2)] = thresh
    return bg

def saveResult(img_file, img, boxes, verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        # filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        # res_file = dirname + "res_" + filename + '.txt'
        # res_img_file = dirname + "res_" + filename + '.jpg'

        # if not os.path.isdir(dirname):
        #     os.mkdir(dirname)
        word_list = []
        # with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            # print(box)
            word_list.append(poly)
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            # f.write(strResult)
            
            poly = poly.reshape(-1, 2)
            # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            # ptColor = (0, 255, 255)
            # if verticals is not None:
            #     if verticals[i]:
            #         ptColor = (255, 0, 0)

            # if texts is not None:
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.5
            #     cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
            #     cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
        word_list = sorted(word_list, key=lambda x: x[0])  #按X坐標排序
        # print(word_list)
        thresh_list = []
        for j, w in enumerate(word_list): 
            word = img[w[1]: w[-1], w[0]: w[2]]
            print(w)
            gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)  # 灰階
            # _,thresh = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY) #轉為黑白
            _,thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV) #轉為黑白   
            # cv2.imshow('test', word)
            t_h, t_w = thresh.shape
            # print(thresh.shape)
            #  and 2 < w[0] and w[2] < 278
            if 80 > t_h > 35 and 59 > t_w > 10:
                # poly_0 = w.reshape(-1, 2)
                # cv2.polylines(img, [poly_0.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                thresh_list.append(thresh)
            elif 80 > t_h > 35 and 90 > t_w >= 59:
                # cv2.polylines(img, [poly_0.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                thresh_1 = thresh[:, :int(t_w/2)]
                thresh_2 = thresh[:, int(t_w/2):]
                # print(dirname)
                thresh_list.append(thresh_1)
                thresh_list.append(thresh_2)
            else:
                continue
            # print(thresh.shape)
        # Save result image
        count_num = 0
        dig_list = []
        if len(thresh_list) >= 4:
            for x, t in enumerate(thresh_list):
                t = bgx(t)
                if t is not None:
                    dig = deskew(t)
                    # cv2.imwrite(dirname + '/word/' + filename + "_" + str(x) + '.jpg', dig)
                    dig_list.append(dig)
                    count_num += 1
        im0 = dig_list[0]
        for im in dig_list[1:]:
            im0 = np.hstack((im0, im))
        im0 = openball(im0)
        # cv2.imwrite(res_img_file, im0)
        if count_num >= 5:
            return im0, True
        else:
            return None, False


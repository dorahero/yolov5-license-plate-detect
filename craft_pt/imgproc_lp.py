"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2

def contrast_brightness_demo(img, c, b):
    h, w = img.shape
    blank = np.zeros([h, w], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c , b)
    # cv2.imshow("contrast_brightness_demo", dst)
    return dst
def find_light(img):
    if '/' in img:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    list_a = hist.tolist()
    b_sum = 0
    for i, b in enumerate(list_a):
        b_sum += i*b[0]
    # print(b_sum/sum(hist)) 
    if b_sum/sum(hist) < 130 or b_sum/sum(hist) > 150:
        result = contrast_brightness_demo(img, float(140/(b_sum/sum(hist))), -10)
        return result
    else:
        return img

def transform(img):
    r = 2
    width = 140*r
    height = 50*r
    dim = (width, height)
    img = cv2.resize(img, dim)
    # print(img.shape)
    # img = io.imread(img_file, 0)           # RGB order
    _, thresh = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
    contours1 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1[0]   # 取得輪廓
    letter_image_regions = [] # 文字圖形串列
    for contour in contours:  # 依序處理輪廓
        (x, y, w, h) = cv2.boundingRect(contour)  # 單一輪廓資料
        letter_image_regions.append((x, y, w, h)) # 輪廓資料加入串列
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0]) # 按X坐標排序
    letterlist = [] # 儲存擷取的字元坐標 
    rows, cols = img.shape
    for box in letter_image_regions:  # 依序處理輪廓資料
        x, y, w, h = box        
        if x>=5*r and (x+w)<=135*r and w>=3.5*r and w<=24*r and h>=20*r and h<29*r:
            letterlist.append((x, y, w, h)) # 儲存擷取的字元
        try:
            x_1, y_1, w_1, h_1 = letterlist[0]
            x_n, y_n, w_n, h_n = letterlist[-1]
            slope = -(y_n-y_1)/(x_n-x_1)
            print(slope)
            if abs(slope) > 0.015: # 針對每個字元做轉正的動作
                pts1 = np.float32([[x_1, y_1], [x_1+h_1*slope, y_1+h_1], [x_n+w_n, y_n]])
                pts2 = np.float32([[x_1, rows/2-h_1/2], [x_1, rows/2+h_1/2], [x_n+w_n, rows/2-h_1/2]])
                matrix = cv2.getAffineTransform(pts1, pts2)
                result = cv2.warpAffine(img, matrix, (cols, rows))
                return result
            else:
                return img

        except Exception as e:
            print(e)
            return img
def loadImage(img_file):
    img = find_light(img_file)
    img = transform(img)
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    # print(target_h, target_w)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    # return resized, ratio, size_heatmap
    return proc, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    # img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

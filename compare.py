import argparse
import time
from pathlib import Path
import os
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box

from utils.torch_utils import select_device, load_classifier, time_synchronized

from craft_pt.pre_lp import lp_pre
from craft_pt.craft import CRAFT
from collections import OrderedDict
# from craft_pt.basenet.vgg16_bn import vgg16_bn, init_weights
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

def detect_lp_num_o(img0, opt, model, half, webcam, device):
    out, source, view_img, save_txt, imgsz, save_img= \
        opt.output, img0, False, False, opt.img_size, True
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 固定車牌為紅色
    colors = [[0, 0, 255]]

    # Run inference
    t0 = time.time()
    # Padded resize
#     ret, img0 = cv2.threshold(img0, 110, 255, cv2.THRESH_BINARY)
#     img0 = cv2.GaussianBlur(img0, (3, 3), 0)
#     cv2.imwrite('./target/result.jpg', img0)
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    vid_cap = None
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, img0)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            r_dict = {}
            result = []
            conf_l = []
            for *xyxy, conf, cls in reversed(det):
                # save target image
                target_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                # cv2.imwrite('./target/lp.jpg'.format(int(xyxy[1])), target_img)
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
#                     plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=3)
                    result.append((float(xyxy[0]), names[int(cls)]))
                    conf_l.append(conf)
            result_l = [s[1] for s in sorted(result, key=lambda x: x[0])]
            result_s = ''
            conf_avg = sum(conf_l) / len(conf_l)
            for r in result_l:
                result_s += r
            print(result_s)
            print(conf_avg)
            if len(result_s) in [5, 6, 7] and conf_avg > 0.80:
                print('-------------------------------------' + result_s)
                print('-------------------------------------', conf_avg)
                return result_s, conf_avg
    fake_lp = ''
    fake_conf = 0.0
    return fake_lp, fake_conf

def detect_lp_num(img0, opt, model, half, webcam, device):
    out, source, view_img, save_txt, imgsz, save_img= \
        opt.output, img0, False, False, opt.img_size, True
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 固定車牌為紅色
    colors = [[0, 0, 255]]

    # Run inference
    t0 = time.time()
    # Padded resize
#     ret, img0 = cv2.threshold(img0, 110, 255, cv2.THRESH_BINARY)
#     img0 = cv2.GaussianBlur(img0, (3, 3), 0)
#     cv2.imwrite('./target/result.jpg', img0)
    imgsz = 640
    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    print(img0.shape)
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    vid_cap = None
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = time_synchronized()
    opt.iou_thres = 0.0
    opt.agnostic_nms = True
    opt.augment = True
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, img0)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            r_dict = {}
            result = []
            conf_l = []
            for *xyxy, conf, cls in reversed(det):
                # save target image
                target_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                # cv2.imwrite('./target/lp.jpg'.format(int(xyxy[1])), target_img)
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
#                     plot_one_box(xyxy, img0, label=label, color=colors[0], line_thickness=3)
                    result.append((float(xyxy[0]), names[int(cls)]))
                    conf_l.append(conf)
            result_l = [s[1] for s in sorted(result, key=lambda x: x[0])]
            result_s = ''
            conf_avg = sum(conf_l) / len(conf_l)
            for r in result_l:
                result_s += r
            print(result_s)
            print(conf_avg)
            if len(result_s) in [5, 6, 7] and conf_avg > 0.80:
                if len(result_s) == 7:
                    result_s = result_s[:3].replace('2', 'S').replace('5', 'S').replace('4', 'A').replace('8', 'B').replace('0', 'D').replace('3', 'E').replace('7', 'F').replace('6', 'G').replace('1', 'N')\
                        + result_s[3:].replace('A', '4').replace('B', '8').replace('D', '0').replace('E', '3').replace('F', '7').replace('Q', '0').replace('G', '6').replace('Z', '2')
                print('-------------------------------------' + result_s)
                print('-------------------------------------', conf_avg)
                return result_s, conf_avg
    fake_lp = ''
    fake_conf = 0.0
    return fake_lp, fake_conf

def detect_lp(opt, model, model_1, model_2, half, webcam, device, net):
    out, source, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.view_img, opt.save_txt, opt.img_size
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[0, 0, 255]]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # save target image
                    if int(cls) == 0:
                        o_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])].copy()
                        target_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        target_img = lp_pre(net, target_img, opt)
                        if target_img is None:
                            continue
                        # else:
                        #     cv2.imwrite('./target/lp{}.jpg'.format(int(xyxy[1])), target_img)
                        label_num, conf_num = detect_lp_num(img0=target_img, opt=opt, model=model_1, half=half, webcam=webcam, device=device)
                        label_num_o, conf_num_o = detect_lp_num_o(img0=o_img, opt=opt, model=model_2, half=half, webcam=webcam, device=device)
    #                     print('im0_1', im0.shape)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            im0_o = im0.copy()
                            if len(label_num) >= 5:
                                label = '%s %.2f' % (label_num, conf_num)
                            else:
                                label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[0], line_thickness=3)
                            if len(label_num_o) >= 5:
                                label_o = '%s %.2f' % (label_num_o, conf_num_o)
                            else:
                                label_o = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0_o, label=label_o, color=colors[0], line_thickness=3)
                            # cv2.imwrite('./target/lp_detect_{}.jpg'.format(int(xyxy[1])), im0)

            # Print time (inference + NMS)
            t2 = time_synchronized()
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.namedWindow(p, cv2.WINDOW_AUTOSIZE)
                imS = cv2.resize(im0, (640, 480)) 
                cv2.imshow(p, imS)
                cv2.moveWindow(p, 0, 0)
                cv2.namedWindow('o', cv2.WINDOW_AUTOSIZE)
                im0_o = cv2.resize(im0_o, (640, 480)) 
                cv2.imshow('o', im0_o)
                cv2.moveWindow('o', 640, 0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # save_img = False
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    # if save_txt or save_img:
    #     print('Results saved to %s' % Path(out))
    #     if platform.system() == 'Darwin' and not opt.update:  # MacOS
    #         os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

def set_model(save_img=False):
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + opt.trained_model + ')')
    if opt.cuda:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model, map_location='cpu')))

    if opt.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    out, source, weights_0, weights_1, weights_2, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights_0, opt.weights_1, opt.weights_2, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_0 = attempt_load(weights_0, map_location=device)  # lp model
    model_1 = attempt_load(weights_1, map_location=device)  # lp content model
    model_2 = attempt_load(weights_2, map_location=device) 
    detect_lp(opt=opt, model=model_0, model_1=model_1, model_2=model_2, half=half, webcam=webcam, device=device, net=net)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_0', nargs='+', type=str, default='weights/lp_1125.pt', help='model.pt path(s)')
    parser.add_argument('--weights_1', nargs='+', type=str, default='weights/lp_num_new_best.pt', help='model.pt path(s)')
    parser.add_argument('--weights_2', nargs='+', type=str, default='weights/lp_num_best_b.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--trained_model', default='craft_pt/weights/craft_ic15_20k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.95, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=560, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                set_model()
                strip_optimizer(opt.weights)
        else:
            set_model()

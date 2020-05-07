# -*- coding: utf-8 -*-
import os
import pickle
import random

import torch
import argparse

from data.coco import COCODetection
from yolact import Yolact
from utils.augmentations import BaseTransform
from utils.aptools import calc_mAP,APDataObject
from utils.functions import MovingAverage,ProgressBar

# 全局静态变量
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

# 全局动态变量
coco_cats = {}  # 将coco标注类别映射为连续的数字，用以模型计算，Call prep_coco_cats to fill this
coco_cats_inverse = {}  # 将模型中的类和映射回原始coco标注 Call prep_coco_cats to fill this


# 功能函数
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_label_map():
    if COCO_LABEL_MAP:
        return COCO_LABEL_MAP
    else:
        return {x + 1: x + 1 for x in range(len(COCO_CLASSES))}


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    #主要是准备class label的映射字典
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inverse[coco_cat_id] = transformed_cat_id

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


# 参数部分
parser = argparse.ArgumentParser(
    description='YOLACT COCO Evaluation')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to evaulate model')
parser.add_argument('--display', dest='display', action='store_true',
                    help='Display qualitative results instead of quantitative ones.')

#我想删掉关于benchmark的所有逻辑，如果只是考虑出不出图片的话，可以在输出端加一个控制，而不是加参数量。
parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                    help='Equivalent to running display mode but without displaying an image.')

parser.add_argument('--trained_model',
                    default=None, type=str,
                    help='Trained state_dict file path to open.')

parser.add_argument('--fast_nms', default=True, type=str2bool,
                    help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                    help='Whether compute NMS cross-class or per-class.')



#设置display True时
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                    help='Do not sort images by hashed image ID.')

args = parser.parse_args()

# 数据集与标签
valid_dataset = COCODetection(image_path='./data/coco/images/val2017/',
                              info_file='./data/coco/annotations/instances_val2017.json',
                              transform=BaseTransform(),
                              has_gt=True
                              )
prep_coco_cats()

# 模型
print('Loading model...', end='')
model = Yolact()
model.load_weights(args.trained_model)
model.eval()
model = model.cuda() if args.cuda else model.cpu()
print(' Done.')


def evaluate(net: Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    #cfg.mask_proto_debug = args.mask_proto_debug

    # # TODo0 Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    # if args.image is not None:
    #     if ':' in args.image:
    #         inp, out = args.image.split(':')
    #         evalimage(net, inp, out)
    #     else:
    #         evalimage(net, args.image)
    #     return
    # elif args.images is not None:
    #     inp, out = args.images.split(':')
    #     evalimages(net, inp, out)
    #     return
    # elif args.video is not None:
    #     if ':' in args.video:
    #         inp, out = args.video.split(':')
    #         evalvideo(net, inp, out)
    #     else:
    #         evalvideo(net, args.video)
    #     return

    frame_times = MovingAverage()
    dataset_size = len(dataset) #if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)



    print()

    if not args.display and not args.benchmark:
            #不显示，直接算分
            # For each class and iou, stores tuples (score, isPositive)
            # Index ap_data[type][iouIdx][classIdx]
            ap_data = {
                'box' : [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds],
                'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds]
            }
            detections = Detections()

    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    #再else就什么也不做


# 核心入口
with torch.no_grad():
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.cuda:
        torch.backends.cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.resume and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_mAP(ap_data)
        exit()

    evaluate(model, valid_dataset)

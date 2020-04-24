# -*- coding: utf-8 -*-
import torch
import argparse

import os

from yolact import Yolact
from data.coco import COCODetection
from utils.augmentations import SSDAugmentation
from utils.models.multibox_loss import MultiBoxLoss

parser = argparse.ArgumentParser()
parser.description = "qq_test_1.0"
parser.add_argument('--batch_size',type=int,default=5,help='Batch size for training')
parser.add_argument('--save_folder', default='weights/',help='Directory for saving logs.')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser.add_argument('--cuda',type=str2bool,default=True,help='Use CUDA to train model')


args = parser.parse_args()

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

#------------------------
lr = 1e-3
decay = 5e-4
momentum = 0.9
gamma = 0.1
lr_steps =  (280000, 600000, 700000, 750000)
max_iter = 800000

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
#--------------------------
# train_dataset = COCODetection(image_path='./data/coco/images/train2017/',
#                         info_file='./data/coco/annotations/instances_train2017.json',
#                         transform=SSDAugmentation(mean=MEANS,std=STD))


#1.数据
valid_dataset = COCODetection(image_path='./data/coco/images/val2017/',
                        info_file='./data/coco/annotations/instances_val2017.json',
                        transform=SSDAugmentation(mean=MEANS,std=STD))


#2.模型
model = Yolact()
net = model
#3.优化器
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=decay)

#4.损失
criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,#0.5
                             neg_threshold=cfg.negative_iou_threshold,#0.4
                             negpos_ratio=cfg.ohem_negpos_ratio)#3


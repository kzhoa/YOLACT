# -*- coding: utf-8 -*-
import torch
import argparse

from data.coco import COCODetection
from yolact import Yolact
from utils.augmentations import SSDAugmentation

valid_dataset = COCODetection(image_path='./data/coco/images/val2017/',
                        info_file='./data/coco/annotations/instances_val2017.json',
                        transform=SSDAugmentation(mean=MEANS,std=STD))


model = Yolact()

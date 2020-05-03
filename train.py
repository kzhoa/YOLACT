# -*- coding: utf-8 -*-
import time

import torch
import argparse

import os
import math, random

from yolact import Yolact
from data.coco import COCODetection, detection_collate
from utils.augmentations import SSDAugmentation
from utils.models.multibox_loss import MultiBoxLoss
from utils.functions import SavePath
from utils.logger import Log
from utils import timetools

# import eval as eval_script #从eval.py里面拿函数，可是这样的话为什么不把公用函数单独提出来呢???

# ---2个工具类----
class NetLoss(torch.nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net: Yolact, criterion: MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses


class CustomDataParallel(torch.nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out


# --功能函数-------------------

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr


def gradinator(x):
    """set_requires_grad_to_false"""
    x.requires_grad = False
    return x


def prepare_data(datum, devices: list = None, allocation: list = None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)  # 最差情况下*0返回空列表[]
            allocation.append(args.batch_size - sum(allocation))  # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        # 下面这个过程本来有个bug,
        # 在"for _ in range(alloc):"这句，原作者默认了dataset是可以被batchsize整除的。
        # 不然的话，最后一个incomplete_batch里，可能会出现cur_idx越界的情况。
        # 例如batchsize设为5，只有1个device,最后一个batch只有3张图，但是"+=1"会循环5次，于是cur_idx=3的时候报越界错误。
        # 解决方案是，(1)在dataloader里，设置drop_last=True。
        # (2)在下面这个过程里，判断是否越界。
        # 根据低耦合原则，不要让函数外面的使用者过分关心这种细节，我选择方案(2)。
        cur_idx = 0
        num_imgs = len(images)
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx] = gradinator(masks[cur_idx].to(device))
                cur_idx += 1
                # 因为有2层循环，需要break2次
                if cur_idx >= num_imgs:
                    break

            if cur_idx >= num_imgs:
                break

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx] = torch.stack(images[cur_idx:cur_idx + alloc], dim=0)
            split_targets[device_idx] = targets[cur_idx:cur_idx + alloc]
            split_masks[device_idx] = masks[cur_idx:cur_idx + alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx + alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds


def compute_validation_map(epoch, iteration, model, dataset, log: Log = None):
    with torch.no_grad():
        model.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(model, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        model.train()


# --------参数区域-----------
parser = argparse.ArgumentParser()
parser.description = "qq_test_1.0"
parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
parser.add_argument('--save_folder', default='weights/',type=str, help='Directory for saving logs.')
parser.add_argument('--cuda', type=str2bool, default=True, help='Use CUDA to train model')  # 引用str2bool函数
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--resume', default=None, type=str,
                    help='Number of workers used in dataloading')

parser.add_argument('--batch_alloc', default=None, type=str,  # 暂时没什么用的参数
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')

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

# --显式参数--
lr = 1e-3
cur_lr = 1e-3
decay = 5e-4
momentum = 0.9
gamma = 0.1
lr_steps = (280000, 600000, 700000, 750000)
max_iter = 800000

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
# --------------------------

# 1.数据

train_dataset = COCODetection(image_path='./data/coco/images/train2017/',
                              info_file='./data/coco/annotations/instances_train2017.json',
                              transform=SSDAugmentation(mean=MEANS, std=STD))

# valid_dataset = COCODetection(image_path='./data/coco/images/val2017/',
#                               info_file='./data/coco/annotations/instances_val2017.json',
#                               transform=SSDAugmentation(mean=MEANS, std=STD))

# if args.validation_epoch > 0:
#     setup_eval()
#     valid_dataset = COCODetection(image_path='./data/coco/images/val2017/',
#                                   info_file='./data/coco/annotations/instances_val2017.json',
#                                   transform=SSDAugmentation(mean=MEANS, std=STD))

data_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=True,
                                          collate_fn=detection_collate,
                                          pin_memory=True,
                                          drop_last=True)

# 2.模型
yolact_model = Yolact()
net = yolact_model  # 这步的作用是给model加一个别名net,后续net会被包装。

if args.resume is not None:
    yolact_model.load_weights(args.resume)

# 3.优化器
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                            weight_decay=decay)

# 4.损失
criterion = MultiBoxLoss(num_classes=81,
                         pos_threshold=0.5,  # 0.5
                         neg_threshold=0.4,  # 0.4
                         negpos_ratio=3)  # 3

# 包装net，改变net的指向，但不影响yolact_model的指向
net = CustomDataParallel(NetLoss(net, criterion))
if args.cuda:
    net = net.cuda()


num_epochs = 10
iteration = 0
save_interval = 1000
step_index = 0

last_time = time.time()

# 写死学习率
set_lr(optimizer, 1e-5)

#仿照原作者，更换epoch时不重置iteration
num_iters_per_epoch = len(train_dataset)//args.batch_size #由于我开启了drop_last所以地板除是精确的
total_iters = num_epochs * num_iters_per_epoch

for epoch in range(num_epochs):
    # # Resume from start_iter
    # if (epoch+1)*epoch_size < iteration:
    #     continue
    for datum in data_loader:
        # Adjust the learning rate at the given iterations, but also if we resume from past that iteration

        # while step_index < len(lr_steps) and iteration >= lr_steps[step_index]:
        #     step_index += 1
        #     set_lr(optimizer, lr * (gamma ** step_index))

        # Zero the grad to get ready to compute gradients
        optimizer.zero_grad()

        # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
        losses = net(datum)

        losses = {k: (v).mean() for k, v in losses.items()}  # Mean here because Dataparallel
        loss = sum([losses[k] for k in losses])

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        # Backprop
        loss.backward()  # Do this to free up vram even if loss is not finite
        if torch.isfinite(loss).item():
            optimizer.step()

        iteration += 1

        if iteration % 10 == 0:
            cur_time = time.time()
            elapsed = cur_time - last_time  # 经过的时间
            last_time = cur_time
            avg_time = elapsed/10 #每个iter消耗的秒数
            iter_persec = 1/avg_time #每秒运算的iter数
            eta_time = (total_iters - iteration)*avg_time #预计剩余秒数
            eta_str = timetools.seconds2str(eta_time,reduce=True)

            fmt_str = "[{:d}] {:0>6d} || total_loss:{:.2f} |" + " {}: {:.4f} |"*len(losses) + " eta:{} | iter/s:{:.3f}"
            loss_labels = sum([[k,losses[k].detach().cpu().numpy()] for k in losses],[])
            data = [epoch, iteration, loss]+loss_labels +[eta_str,iter_persec]
            print(fmt_str.format(*data),flush=True)

            # print(eta_str,'\n',iter_persec)
            # stt= " eta:{} | iter/s:{:.3f}"
            # rs = stt.format(eta_str,iter_persec)
            # print(rs)

        if iteration % save_interval == 0:
            print('Saving state, iter:', iteration)

            # 特意写个Savepath类，感觉作用不大。
            yolact_model.save_weights(SavePath('yolact_base', epoch, iteration).get_path(root=args.save_folder))

#     # This is done per epoch
#     if args.validation_epoch > 0:
#         if epoch % args.validation_epoch == 0 and epoch > 0:
#             compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
#
# # Compute validation mAP after training is finished
# compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

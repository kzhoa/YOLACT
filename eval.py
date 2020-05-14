# -*- coding: utf-8 -*-
import os
import pickle
import random
import argparse

import torch
import matplotlib.pyplot as plt


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


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections: Detections = None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes, gt_boxes = split(gt_boxes)
                crowd_masks, gt_masks = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop,
                                                    score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    # timer.stop('Main loop')


#--- 参数部分
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
    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
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

    #我们去掉了args.max_images之后，这句也可以不要了。不过以免万一先做保留。
    dataset_indices = dataset_indices[:dataset_size]

    # Main eval loop
    for it, image_idx in enumerate(dataset_indices):

        #Load Data
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

        batch = torch.autograd.Variable(img.unsqueeze(0))
        if args.cuda:
            batch = batch.cuda()

        #送入网络'Network Extra'
        preds = net(batch)

        # Perform the meat of the operation here depending on our mode.
        if args.display:
            img_numpy = prep_display(preds, img, h, w) #我们不搞display
        elif args.benchmark:
            prep_benchmark(preds, h, w) #我们也不搞这个
        else:
            prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)

        # First couple of images take longer because we're constructing the graph.
        # Since that's technically initialization, don't include those in the FPS calculations.
        # if it > 1:
        #     frame_times.add(timer.total_time())

        if args.display:
            if it > 1:
                print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
            plt.imshow(img_numpy)
            plt.title(str(dataset.ids[image_idx]))
            plt.show()
        elif not args.no_bar:
            if it > 1:
                fps = 1 / frame_times.get_avg()
            else:
                fps = 0
            progress = (it + 1) / dataset_size * 100
            progress_bar.set_val(it + 1)
            print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                  % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')


    if not args.display and not args.benchmark:
        print()
        if args.output_coco_json:
            print('Dumping detections...')
            if args.output_web_json:
                detections.dump_web()
            else:
                detections.dump()
        else:
            if not train_mode:
                print('Saving data...')
                with open(args.ap_data_file, 'wb') as f:
                    pickle.dump(ap_data, f)

            return calc_map(ap_data)
    elif args.benchmark:
        print()
        print()
        print('Stats for the last frame:')
        timer.print_stats()
        avg_seconds = frame_times.get_avg()
        print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000 * avg_seconds))


if __name__ == '__main__':

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

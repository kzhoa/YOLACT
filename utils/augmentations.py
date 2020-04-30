import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from math import sqrt


# 从SSD开源代码里改来的
# this file is adapted from https://github.com/amdegroot/ssd.pytorch/utils/augmentations.py


# -----------------------
# 功能函数
# -----------------------

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """

    def intersect(box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class PrepareMasks(object):
    """
    Prepares the gt masks for use_gt_bboxes by cropping with the gt box
    and downsampling the resulting mask to mask_size, mask_size. This
    function doesn't do anything if cfg.use_gt_bboxes is False.
    """

    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, image, masks, boxes, labels=None):
        if not self.use_gt_bboxes:
            return image, masks, boxes, labels

        height, width, _ = image.shape

        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))

            # +1 So that if y1=10.6 and y2=10.9 we still have a bounding box
            cropped_mask = masks[i, y1:(y2 + 1), x1:(x2 + 1)]
            scaled_mask = cv2.resize(cropped_mask, (self.mask_size, self.mask_size))

            new_masks[i, :] = scaled_mask.reshape(1, -1)

        # Binarize
        new_masks[new_masks > 0.5] = 1
        new_masks[new_masks <= 0.5] = 0

        return image, new_masks, boxes, labels


# ----------------
# 基础变换
# ---------------

class ConvertFromInts(object):
    """从np.int矩阵变成np.float32"""

    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image.astype(np.float32), masks, boxes, labels


class ToCV2Image(object):
    """convert tensor to cv2img"""

    def __call__(self, tensor, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, labels


class ToTensor(object):
    """convert cv2img to tensor"""

    def __call__(self, cvimage, masks=None, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, labels


class ToAbsoluteCoords(object):
    """从百分比(x1,y1,x2,y2)变成绝对坐标"""

    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, masks, boxes, labels


class ToPercentCoords(object):
    """从绝对坐标(x1,y1,x2,y2)变成百分比坐标"""

    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, masks, boxes, labels


class ConvertColor(object):
    """改变颜色空间,基于opencv-python"""

    def __init__(self, current='BGR', transform='HSV'):
        self.current = current
        self.transform = transform

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, masks, boxes, labels


# --------------------------------------------------------------
# 图像进阶处理
# ---------------------------------------------------------------

# 形状处理

class Expand(object):
    """将原图片的高宽乘以ratio，将原图片放在扩张后图片的中间，其他位置像素值使用均值填充，相应的bbox也进行移动"""

    def __init__(self, mean):
        self.mean = mean  # shape=(3)

    def __call__(self, image, masks, boxes, labels):
        if np.random.randint(2):
            return image, masks, boxes, labels

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 4)  # 在[1,4]随机一个实数
        left = np.random.uniform(0, width * ratio - width)  # 确定原图的新位置-左
        top = np.random.uniform(0, height * ratio - height)  # 确定原图的新位置-上

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        # 处理变换后的mask
        expand_masks = np.zeros(
            (masks.shape[0], int(height * ratio), int(width * ratio)),
            dtype=masks.dtype)
        expand_masks[:, int(top):int(top + height),
        int(left):int(left + width)] = masks
        masks = expand_masks

        # 处理变换后的box框
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, masks, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, masks, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                return image, masks, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing:
                # https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1 - crowd_mask[mask]) == 0:
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                # We now might have fewer crowd annotations
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                return current_image, current_masks, current_boxes, current_labels


class Resize(object):
    """改变图像尺寸，基于opencv-python"""
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True, max_size=550):
        self.resize_gt = resize_gt
        self.max_size = max_size
        self.preserve_aspect_ratio = False  # 1.0默认参数

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape

        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        # 修改图像尺寸
        image = cv2.resize(image, (width, height))

        # mask与box相应修改
        if self.resize_gt:

            # 把mask(1,w,h)转置后当成图片去resize
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))

            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # box用浮点数？
            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        # 默认参数
        discard_box_width = 4 / 550,
        discard_box_height = 4 / 550,

        keep = (w > discard_box_width) * (h > discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """

    def __init__(self, width, height, mean=(103.94, 116.78, 123.68), pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, labels=None):
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:, :im_h, :im_w] = masks
            masks = expand_masks

        return expand_image, masks, boxes, labels


class RandomMirror(object):
    """随机水平镜像"""

    def __call__(self, image, masks, boxes, labels):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, masks, boxes, labels


class RandomFlip(object):
    """随机上下翻转"""

    def __call__(self, image, masks, boxes, labels):
        height, _, _ = image.shape
        if np.random.randint(2):
            image = image[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
        return image, masks, boxes, labels


class RandomRot90(object):
    """随机旋转90度"""

    def __call__(self, image, masks, boxes, labels):
        old_height, old_width, _ = image.shape
        k = np.random.randint(4)
        image = np.rot90(image, k)
        masks = np.array([np.rot90(mask, k) for mask in masks])
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2], box[3], old_width - 1 - box[0]] for box in boxes])
            old_width, old_height = old_height, old_width
        return image, masks, boxes, labels


# RGB空间颜色处理
class RandomContrast(object):
    """随机对比度，图像随机乘以一个系数"""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, masks, boxes, labels


class RandomBrightness(object):
    """随机亮度，图片整体加减一个随机实数"""

    def __init__(self, delta=32):
        # 默认delta=32，delta的范围要在0-255之间
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks, boxes, labels


class RandomLightingNoise(object):
    """随机通道变换，以0.5的概率触发shuffle，每次shuffle从6种排列中抽取一种"""

    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, masks=None, boxes=None, labels=None):
        # 下面英文部分是yolact作者吐槽ssd作者，我要笑死了，你不喜欢这个通道变换为啥不直接删掉，留着也就罢了还要吐槽人家-by qq
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)
        return image, masks, boxes, labels


# -----
# HSV空间处理,请搭配ColorConvert食用
# ----
class RandomHue(object):
    """随机色调，H维度上加减一个随机系数"""

    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, masks, boxes, labels


class RandomSaturation(object):
    """随机饱和度，在HSV空间的S维度上乘以一个随机系数"""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, masks, boxes, labels


# -----------------------------------------------------------
# 图像高阶处理
# -----------------------------------------------------------

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, masks, boxes, labels):
        im = image.copy()
        im, masks, boxes, labels = self.rand_brightness(im, masks, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, masks, boxes, labels = distort(im, masks, boxes, labels)
        return self.rand_light_noise(im, masks, boxes, labels)


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    """修改了，不用cfg传参"""

    def __init__(self,
                 mean,
                 std,
                 normalize=True,
                 subtract_means=False,  # 是否减去均值
                 to_float=False,  # 是否除以255
                 in_channel_order='bgr',
                 out_channel_order='rgb'):

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize = normalize
        self.subtract_means = subtract_means
        self.to_float = to_float

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order.lower())}
        self.channel_permutation = [self.channel_map[c] for c in out_channel_order.lower()]

    def __call__(self, img, masks=None, boxes=None, labels=None):

        img = img.astype(np.float32)

        if self.normalize:
            img = (img - self.mean) / self.std
        elif self.subtract_means:
            img = (img - self.mean)
        elif self.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32), masks, boxes, labels


# class BaseTransform(object):
#     """ Transorm to be used when evaluating. """
#
#     def __init__(self, mean=MEANS, std=STD):
#         self.augment = Compose([
#             ConvertFromInts(),
#             Resize(resize_gt=False),
#             BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
#         ])
#
#     def __call__(self, img, masks=None, boxes=None, labels=None):
#         return self.augment(img, masks, boxes, labels)



# class FastBaseTransform(torch.nn.Module):
#     """
#     Transform that does all operations on the GPU for super speed.
#     This doesn't suppport a lot of config settings and should only be used for production.
#     Maintain this as necessary.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#         self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
#         self.std = torch.Tensor(STD).float().cuda()[None, :, None, None]
#         self.transform = cfg.backbone.transform
#
#     def forward(self, img):
#         self.mean = self.mean.to(img.device)
#         self.std = self.std.to(img.device)
#
#         # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
#         if cfg.preserve_aspect_ratio:
#             _, h, w, _ = img.size()
#             img_size = Resize.calc_size_preserve_ar(w, h, cfg.max_size)
#             img_size = (img_size[1], img_size[0])  # Pytorch needs h, w
#         else:
#             img_size = (cfg.max_size, cfg.max_size)
#
#         img = img.permute(0, 3, 1, 2).contiguous()
#         img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)
#
#         if self.transform.normalize:
#             img = (img - self.mean) / self.std
#         elif self.transform.subtract_means:
#             img = (img - self.mean)
#         elif self.transform.to_float:
#             img = img / 255
#
#         if self.transform.channel_order != 'RGB':
#             raise NotImplementedError
#
#         img = img[:, (2, 1, 0), :, :].contiguous()
#
#         # Return value is in channel order [n, c, h, w] and RGB
#         return img


# --------------------------------------------------------

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels


class SSDAugmentation(object):
    """ Transform to be used when training. """

    # These means and stds are in BGR and are for ImageNet
    def __init__(self, mean=(103.94, 116.78, 123.68), std=(57.38, 57.12, 58.40)):
        # self.augment = Compose([
        #     ConvertFromInts(),
        #     ToAbsoluteCoords(),
        #     enable_if(cfg.augment_photometric_distort, PhotometricDistort()),  # True
        #     enable_if(cfg.augment_expand, Expand(mean)),  # True
        #     enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),  # True
        #     enable_if(cfg.augment_random_mirror, RandomMirror()),  # True
        #     enable_if(cfg.augment_random_flip, RandomFlip()),  # False
        #     enable_if(cfg.augment_random_flip, RandomRot90()),  # False
        #     Resize(),
        #     enable_if(not cfg.preserve_aspect_ratio, Pad(cfg.max_size, cfg.max_size, mean)),#False
        #     ToPercentCoords(),
        #     PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
        #     BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        # ])
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(mean),
            RandomSampleCrop(),
            RandomMirror(),
            # RandomFlip(),
            # RandomRot90(),
            Resize(),
            # Pad(550, 550, mean),
            ToPercentCoords(),
            PrepareMasks(mask_size=16, use_gt_bboxes=False),
            BackboneTransform(
                mean=mean,
                std=std,
                normalize=True,
                subtract_means=False,  # 是否减去均值
                to_float=False,  # 是否除以255
                in_channel_order='BGR',
                out_channel_order='RGB')
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, center_size, crop, elemwise_mask_iou, elemwise_box_iou


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes

        self.pos_threshold = pos_threshold  # 0.5
        self.neg_threshold = neg_threshold  # 0.4
        self.negpos_ratio = negpos_ratio  # 3

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        # False
        self.use_class_balanced_conf = False
        if self.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

    def forward(self, net, predictions, targets, masks, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.

            * Only if mask_type == lincomb
        """

        loc_data = predictions['loc']  # (bz, numpriors, 4)
        conf_data = predictions['conf']  # (bz, numpriors, num_classes)
        mask_data = predictions['mask']  # (bz, numpriors, 32)
        priors = predictions['priors']  # (numpriors, 4) ，（x,y,w,h）

        # 默认就是lincomb,linearcombination
        # if self.mask_type == mask_type.lincomb:
        proto_data = predictions['proto']  # (bz,maskh,maskw,maskdim)

        self.use_mask_scoring = False
        self.use_instance_coeff = False

        score_data = predictions['score'] if self.use_mask_scoring else None  # False->None
        inst_data = predictions['inst'] if self.use_instance_coeff else None  # False->None

        # Used in sem segm loss #这里len()的返回是batchsize
        labels = [None] * len(targets)

        batch_size = loc_data.size(0)
        num_priors = priors.size(0)  # num_priors = sum(num_prior*hi*wi),i=3~7

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        # new的作用是复制type和device，所以下面这4个的值都是随机初始化的，之后要赋值。
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()

        for idx in range(batch_size):
            # 牢记idx指batch_idx
            # [:-1]表示从开始到倒数第2个，最后一个不取，即(x,y,w,h)
            truths = targets[idx][:, :-1].detach()  # num_objs*[x1,y1,x2,y2]
            labels[idx] = targets[idx][:, -1].detach().long()  # num_objs*[class_label]

            # Split the crowd annotations because they come bundled in
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths = split(truths)

                # We don't use the crowd labels or masks
                _, labels[idx] = split(labels[idx])  # [num_objs-cur_crowds]
                _, masks[idx] = split(masks[idx])  # [num_objs-cur_crowds]

            else:
                crowd_boxes = None

            # 这个没有返回值的函数，利用可变参数传引用，原地修改
            # 对之前随机初始化的4个东西match后赋值
            match(self.pos_threshold, self.neg_threshold,
                  truths, priors.data, labels[idx], crowd_boxes,
                  loc_t, conf_t, idx_t, idx, loc_data[idx])

            gt_box_t[idx, :, :] = truths[idx_t[idx]]  # 这不就是match函数的中间结果matches吗？(x1,y1,x2,y2)

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)

        pos = conf_t > 0  # (bz,numpriors),大于0这个条件同时过滤-1中性与0背景。
        num_pos = pos.sum(dim=1, keepdim=True)  # (bz,1)

        # Shape: [batch,num_priors,4]
        # unsqueeze在最后一个位置上扩充维度1，变成(bz,num_priors,1),然后expand到4
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        losses = {}
        self.bbox_alpha = 1.5  # 默认参数
        # 默认cfg.train_box总为True
        # 1. Localization Loss (Smooth L1)
        loc_p = loc_data[pos_idx].view(-1, 4)  # 推测格式为(x1,y1,x2,y2)
        loc_t = loc_t[pos_idx].view(-1, 4)  # (x1,y1,x2,y2)
        losses['B'] = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') * self.bbox_alpha

        # 总是train_mask, type为lincomb
        # 2.mask_loss
        ret = self.lincomb_mask_loss(pos, idx_t,
                                     loc_data,  # 注意这里是pred出来的loc_data,不是表征gt坐标的loc_t
                                     mask_data, priors,
                                     proto_data,
                                     masks,  # 这是标注给的mask ,[bz][numobj,imh,imw]
                                     gt_box_t,  # match函数的中间结果matches,[bz,numprior,x1y1x2y2]
                                     score_data, inst_data,  # 默认2个None
                                     labels  # (bz,numobj)
                                     )

        # 默认False，++版本True
        self.use_maskiou = False
        if self.use_maskiou:
            loss, maskiou_targets = ret
        else:
            loss = ret
        losses.update(loss)  # 返回的loss是一个字典，用原生字典的update方法更新到主字典losses中。

        # 3.Confidence loss
        # 默认False
        self.use_focal_loss = False
        self.use_sigmoid_focal_loss = False  # 默认参数
        self.use_objectness_score = False
        if self.use_focal_loss:
            if self.use_sigmoid_focal_loss:
                losses['C'] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            elif self.use_objectness_score:
                losses['C'] = self.focal_conf_objectness_loss(conf_data, conf_t)
            else:
                losses['C'] = self.focal_conf_loss(conf_data, conf_t)
        else:
            if self.use_objectness_score:
                losses['C'] = self.conf_objectness_loss(conf_data, conf_t, batch_size, loc_p, loc_t, priors)
            else:
                # 默认来到这里
                # 总结一下，ohem_loss，就是取真实label为0（背景)的priors的num_classes个输出结果，与真实label做CE。
                losses['C'] = self.ohem_conf_loss(conf_data,  # (bz, numpriors, num_classes) 预测出的每个prior的结果
                                                  conf_t,  # [bz, numpriors]标注的每个prior的label
                                                  pos,
                                                  batch_size)

        # Mask IoU Loss
        # 默认False，++版本True
        if self.use_maskiou and maskiou_targets is not None:
            losses['I'] = self.mask_iou_loss(net, maskiou_targets)

        # These losses also don't depend on anchors
        # self.use_class_existence_loss =False#默认参数
        # if self.use_class_existence_loss:
        #     losses['E'] = self.class_existence_loss(predictions['classes'], class_existence_t)

        self.use_semantic_segmentation_loss = True  # 默认参数
        if self.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(predictions['segm'],
                                                          masks,  # 这是标注给的mask ,[bz][numobj,imh,imw]
                                                          labels,  # (bz,numobj)
                                                          )

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.data.sum().float()
        for k in losses:
            if k not in ('P', 'E', 'S'):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size

        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - P: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - E: Class Existence Loss
        #  - S: Semantic Segmentation Loss
        return losses

    def class_existence_loss(self, class_data, class_existence_t):
        class_existence_alpha = 1.0  # 默认参数
        return class_existence_alpha * F.binary_cross_entropy_with_logits(class_data, class_existence_t,
                                                                          reduction='sum')

    def semantic_segmentation_loss(self, segment_data,
                                   mask_t, #[bz][numobj,imgh,imgw]
                                   class_t,
                                   interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0

        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]

            with torch.no_grad():
                #mask_t
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float() #二值化处理

                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]],
                                                                downsampled_masks[obj_idx])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')

        self.semantic_segmentation_alpha = 1  # 1.0参数
        return loss_s / mask_h / mask_w * self.semantic_segmentation_alpha

    def ohem_conf_loss(self,
                       conf_data,  # (bz,numpriors,num_classes)
                       conf_t, #(bz,numpriors)
                       pos,
                       num  # forward里传进来的num就是batchsize，不知道怎么吐槽这个变量命名习惯。我怕他自己以后没注释都看不懂。
                       ):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)  # (bz*numpriors,num_classes)，代表每个类别的预测分

        self.ohem_use_most_confident = False  # 默认参数
        if self.ohem_use_most_confident:
            # i.e. max(softmax) along classes > 0
            batch_conf = F.softmax(batch_conf, dim=1)
            loss_c, _ = batch_conf[:, 1:].max(dim=1)  # 第0类是背景，所以拿掉了。
        else:
            # i.e. -softmax(class 0 confidence)
            # 上面是同batch每个prior取最大的类别得分，下面这个是每个prior对每个类别得分做类似softmax的操作后加总所有类别。
            loss_c = log_sum_exp(batch_conf) - batch_conf[:,0]
            # 以(x_max-背景类conf)为基石，其他类按得分接近x_max的程度获得(0,log2]范围的bonus。

        # 这个算法有点神奇。
        # 我们任选一个真实标签为0的prior,因为默认参数是走log_sum_exp的。
        # 对于这个prior而言，模型生成的conf=[1,0,0,0,...,0]和[0,1,0,0,...,0]，在log_sum_exp函数中获得的结果是完全一样的。
        # 区别仅仅在于，出来之后，减去batch_conf[:,0] ,会让前者的loss更小一点。
        # 值得注意的是，此处的loss_c仅仅为一个临时的定性loss，而非定量loss，
        # 之后会被覆盖掉，不用于求导，所以不必追求精确，只要能表达位次关系就好。

        # loss_c.shape=[bz*numpriors]
        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)  # (bz,num_priors)
        loss_c[pos] = 0  # 将真实标签不为背景的prior，loss置0。pos=conf_t>0，所以参数只需要一个conf_t,根本没必要传个pos进来啊。
        loss_c[conf_t < 0] = 0  # 将真实标签为中性（-1）的prior，loss置为0。
        # 经过上面这两步，只剩下对应gt的label为背景的那些prior有loss。
        # 所以...所谓的困难样本neg，就是处理真实标签为0的样本？？？这就是困难样本:ma:
        _, loss_idx = loss_c.sort(1, descending=True)  # 利用torch.sort,返回loss最大的prior的序号，(bz,num_priors)
        _, idx_rank = loss_idx.sort(1)  # 二次升序sort的效果是，知道第idx个框在原来batch中的排名，(bz,num_priors)
        num_pos = pos.long().sum(1, keepdim=True)  # (bz,1),每个batch内有几个正框
        num_neg = torch.clamp(self.negpos_ratio * num_pos,
                              max=pos.size(1) - 1)  # (bz,1),根据正框数量，扩张一个比例，默认3.0，注意每个bz的num_neg不一样
        neg = idx_rank < num_neg.expand_as(idx_rank)  # (bz,num_priors),bool型索引，选中loss_c数值最大的前num_neg个框为1。

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos] = 0  # 如果前num_neg个框中有pos框，置为0
        neg[conf_t < 0] = 0  # 同样地，中性neural框也置为0

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # (bz,numpriors,num_classes)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # (bz,numpriors,num_classes)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)  # (num_pos+num_neg,num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]  # [bz,num_priors] -> (num_pos+num_neg),每个框的真实label
        # 因为正负框都没有包含label为-1的prior, 所以选中的target中也不会包含-1，下一步计算CE不会有问题。
        # CE之前不需要softmax
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')  # (num_pos+num_neg),

        # 默认参数False
        self.use_class_balanced_conf = False
        if self.use_class_balanced_conf:
            # Lazy initialization
            if self.class_instances is None:
                self.class_instances = torch.zeros(self.num_classes, device=targets_weighted.device)

            classes, counts = targets_weighted.unique(return_counts=True)

            for _cls, _cnt in zip(classes.cpu().numpy(), counts.cpu().numpy()):
                self.class_instances[_cls] += _cnt

            self.total_instances += targets_weighted.size(0)

            weighting = 1 - (self.class_instances[targets_weighted] / self.total_instances)
            weighting = torch.clamp(weighting, min=1 / self.num_classes)

            # If you do the math, the average weight of self.class_instances is this
            avg_weight = (self.num_classes - 1) / self.num_classes

            loss_c = (loss_c * weighting).sum() / avg_weight
        else:
            # 默认直接加总
            loss_c = loss_c.sum()

        self.conf_alpha = 1.0

        # 总结一下，ohem_loss，就是根据pos的数量，生成neg的数量，取真实label为0（背景)的priors的loss倒排中，前num_neg个输出结果对应的索引，
        # 再与所有pos框的索引结合成总索引，取出num_classes个预测结果,与真实label做CE。
        return self.conf_alpha * loss_c

    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        Note that this uses softmax and not the original sigmoid from the paper.
        """
        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(-1, conf_data.size(-1))  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # so that gather doesn't drum up a fuss

        logpt = F.log_softmax(conf_data, dim=-1)
        logpt = logpt.gather(1, conf_t.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        # I adapted the alpha_t calculation here from
        # https://github.com/pytorch/pytorch/blob/master/modules/detectron/softmax_focal_loss_op.cu
        # You'd think you want all the alphas to sum to one, but in the original implementation they
        # just give background an alpha of 1-alpha and each forground an alpha of alpha.
        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # See comment above for keep
        return cfg.conf_alpha * (loss * keep).sum()

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        num_classes = conf_data.size(-1)

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(-1, num_classes)  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # can't mask with -1, so filter that out

        # Compute a one-hot embedding of conf_t
        # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t = conf_one_t * 2 - 1  # -1 if background, +1 if forground for specific class

        logpt = F.logsigmoid(conf_data * conf_pm_t)  # note: 1 - sigmoid(x) = sigmoid(-x)
        pt = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0  # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)

        return cfg.conf_alpha * loss.sum()

    def focal_conf_objectness_loss(self, conf_data, conf_t):
        """
        Instead of using softmax, use class[0] to be the objectness score and do sigmoid focal loss on that.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.

        If class[0] = 1 implies forground and class[0] = 0 implies background then you achieve something
        similar during test-time to softmax by setting class[1:] = softmax(class[1:]) * class[0] and invert class[0].
        """

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(-1, conf_data.size(-1))  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # so that gather doesn't drum up a fuss

        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

        logpt = F.logsigmoid(conf_data[:, 0]) * (1 - background) + F.logsigmoid(-conf_data[:, 0]) * background
        pt = logpt.exp()

        obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # All that was the objectiveness loss--now time for the class confidence loss
        pos_mask = conf_t > 0
        conf_data_pos = (conf_data[:, 1:])[pos_mask]  # Now this has just 80 classes
        conf_t_pos = conf_t[pos_mask] - 1  # So subtract 1 here

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

        return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())

    def conf_objectness_loss(self, conf_data, conf_t, batch_size, loc_p, loc_t, priors):
        """
        Instead of using softmax, use class[0] to be p(obj) * p(IoU) as in YOLO.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.
        """

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(-1, conf_data.size(-1))  # [batch_size*num_priors, num_classes]

        pos_mask = (conf_t > 0)
        neg_mask = (conf_t == 0)

        obj_data = conf_data[:, 0]
        obj_data_pos = obj_data[pos_mask]
        obj_data_neg = obj_data[neg_mask]

        # Don't be confused, this is just binary cross entropy similified
        obj_neg_loss = - F.logsigmoid(-obj_data_neg).sum()

        with torch.no_grad():
            pos_priors = priors.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 4)[pos_mask, :]

            boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
            boxes_targ = decode(loc_t, pos_priors, cfg.use_yolo_regressors)

            iou_targets = elemwise_box_iou(boxes_pred, boxes_targ)

        obj_pos_loss = - iou_targets * F.logsigmoid(obj_data_pos) - (1 - iou_targets) * F.logsigmoid(-obj_data_pos)
        obj_pos_loss = obj_pos_loss.sum()

        # All that was the objectiveness loss--now time for the class confidence loss
        conf_data_pos = (conf_data[:, 1:])[pos_mask]  # Now this has just 80 classes
        conf_t_pos = conf_t[pos_mask] - 1  # So subtract 1 here

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

        return cfg.conf_alpha * (class_loss + obj_pos_loss + obj_neg_loss)

    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]

                # Shape: [num_priors, 4], decoded predicted bboxes
                pos_bboxes = decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors)
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]

                # Convert bboxes to absolute coordinates
                num_pos, img_height, img_width = pos_masks.size()

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)

                # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
                # Note that each bounding box crop is a different size so I don't think we can vectorize this
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]

                    # Restore any dimensions we've left out because our bbox was 1px wide
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)

                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                    scaled_masks.append(new_mask.view(1, -1))

                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float()  # Threshold downsampled mask

            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += F.binary_cross_entropy(torch.clamp(pos_mask_data, 0, 1), mask_t, reduction='sum') * cfg.mask_alpha

        return loss_m

    def lincomb_mask_loss(self, pos, idx_t,
                          loc_data,  # 注意这里是pred出来的loc_data,(格式推测x1y1x2y2)
                          mask_data,  # pred出来的mask，对应(bz,numpriors,maskdim=32)
                          priors,  # (numpriors,xywh)
                          proto_data,  # (bz,70,70,32)
                          masks,  # list<tensor>,[batch_size][num_objs,im_height,im_width]
                          gt_box_t,  # match函数的中间结果matches,[bz,numprior,x1y1x2y2]
                          score_data, inst_data,  # 默认2个None
                          labels,  # (bz,numobj)
                          interpolation_mode='bilinear'):
        """简化版本"""
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        self.mask_proto_normalize_emulate_roi_pooling = True  # Normalize the mask loss to emulate roi pooling's affect on loss.
        self.mask_proto_crop = True  # If True, crop the mask with the predicted bbox during training.
        process_gt_bboxes = self.mask_proto_normalize_emulate_roi_pooling or self.mask_proto_crop  # True

        loss_m = 0

        # 遍历batch
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                # 此处unsqueeze是为了方便调用interpolate函数，对末2位进行修改。
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0),  # [1,num_objs,im_height,im_width]
                                                  (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()  # (maskh,maskw,num_objs)

                self.mask_proto_binarize_downsampled_gt = True  # Binarize GT after dowsnampling during training?
                if self.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()  # 利用torch.greaterthan()函数二值化gt_mask

            cur_pos = pos[idx]  # pos = conf_t > 0,[num_priors],pos代表每个bz下类别标签不为中性or背景的那些prior的bool型索引,即正样本prior的索引
            pos_idx_t = idx_t[idx, cur_pos]  # idx_t表示每个bz下，每个prior对应的obj的索引值。
            # pos_idx_t表示本batch中正样本prior对应的gtbox的索引(对应哪个obj),(num_pos,1)

            # True
            if process_gt_bboxes:
                # Note: this is in point-form
                self.mask_proto_crop_with_pred_box = False
                if self.mask_proto_crop_with_pred_box:
                    pos_gt_box_t = decode(loc_data[idx, :, :], priors.data, use_yolo_regressors=False)[cur_pos]
                else:
                    # pos_gtbox_t 表示本batch中正样本对应的gtbox的坐标，(num_pos,x1y1x2y2)
                    pos_gt_box_t = gt_box_t[idx, cur_pos]

            # 若本batch下没有正样本，则跳过本batch
            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]  # (maskh,maskw,32)
            proto_coef = mask_data[idx, cur_pos, :]  # (num_pos,32)

            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            self.masks_to_train = 100  # yolact_base里100,在im700里是300
            if old_num_pos > self.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))  # 将[0,size(0)-1]范围内的整数随机打乱排列
                select = perm[:self.masks_to_train]  # 这两句组合在一起，就是随机选择masks_to_train数量的正样本(指prior)

                proto_coef = proto_coef[select, :]
                pos_idx_t = pos_idx_t[select]

                # True
                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[select, :]

                # 默认False
                if self.use_mask_scoring:
                    mask_scores = mask_scores[select, :]

            # 更新正样本数量
            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]  # (maskh,maskw,num_objs)->(maskh,maskw,num_pos)
            label_t = labels[idx][pos_idx_t]  # (bz,numobj) -> (num_pos)

            # [mask_h,mask_w,mask_dim]*[mask_dim,num_pos] = [mask_h, mask_w, num_pos]
            # 艾特@符号表示矩阵乘法
            pred_masks = proto_masks @ proto_coef.t()
            pred_masks = F.sigmoid(pred_masks)

            # 默认True
            if self.mask_proto_crop:
                # 在pred_masks上，将所有pos_gtbox之外的点都置为0
                pred_masks = crop(pred_masks, pos_gt_box_t)

            # 默认True,计算mask交叉熵
            # 理论上已经sigmoid过了，这个clamp没有意义。
            # pred_masks,[mask_h, mask_w, num_pos]
            # mask_t,[maskh,maskw,num_pos]
            # 因为reduction='none',结果保留形状[maskh,maskw,num_pos]
            pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')

            # 默认True
            if self.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if self.mask_proto_crop else 1  # True,个人感觉这里为False的时候设为1会让梯度有点略显微小。
                pos_gt_csize = center_size(pos_gt_box_t)  # 变成(cx, cy, w, h)格式,shape=[num_pos,4]
                gt_box_width = pos_gt_csize[:, 2] * mask_w  # [num_pos]
                gt_box_height = pos_gt_csize[:, 3] * mask_h  # [num_pos]
                # 经过sum变成[num_pos],然后除以相应的gt_w与gt_h
                pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight
                # 上面这种写法也冗余啊。按下面这样写不是2行搞定吗。
                # pre_loss = pre_loss.sum(dim=(0,1))/pos_gt_csize[:, 2]/pos_gt_csize[:, 3]
                # pre_loss = pre_loss/mask_h/mask_w if self.mask_proto_crop else pre_loss

            # 目前为止，pre_loss.shape=[num_pos]
            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos  # 放大梯度

            # 当前img的loss加到整个batch的累计量中
            loss_m += torch.sum(pre_loss)  # 现在又变成标量了

            # #默认False，++版本True
            # if self.use_maskiou:
            #     self.discard_mask_area = 5*5 #++版本
            #     if self.discard_mask_area > 0:
            #         gt_mask_area = torch.sum(mask_t, dim=(0, 1))
            #         select = gt_mask_area > self.discard_mask_area
            #
            #         if torch.sum(select) < 1:
            #             continue
            #
            #         pos_gt_box_t = pos_gt_box_t[select, :]
            #         pred_masks = pred_masks[:, :, select]
            #         mask_t = mask_t[:, :, select]
            #         label_t = label_t[select]
            #
            #     maskiou_net_input = pred_masks.permute(2, 0, 1).contiguous().unsqueeze(1)
            #     pred_masks = pred_masks.gt(0.5).float()
            #     maskiou_t = self._mask_iou(pred_masks, mask_t)
            #
            #     maskiou_net_input_list.append(maskiou_net_input)
            #     maskiou_t_list.append(maskiou_t)
            #     label_t_list.append(label_t)

        self.mask_alpha = 6.125  # 神奇的经验参数？
        losses = {'M': loss_m * self.mask_alpha / mask_h / mask_w}  # 为什么又要除以mask_h和mask_w啊，在gt_box_width那一步不是除过了。

        # #默认False，++版本True
        # if self.use_maskiou:
        #     # discard_mask_area discarded every mask in the batch, so nothing to do here
        #     if len(maskiou_t_list) == 0:
        #         return losses, None
        #
        #     maskiou_t = torch.cat(maskiou_t_list)
        #     label_t = torch.cat(label_t_list)
        #     maskiou_net_input = torch.cat(maskiou_net_input_list)
        #
        #     num_samples = maskiou_t.size(0)
        #     self.maskious_to_train = -1 #默认参数
        #     if self.maskious_to_train > 0 and num_samples > self.maskious_to_train:
        #         perm = torch.randperm(num_samples)
        #         select = perm[:self.masks_to_train]
        #         maskiou_t = maskiou_t[select]
        #         label_t = label_t[select]
        #         maskiou_net_input = maskiou_net_input[select]
        #
        #     return losses, [maskiou_net_input, maskiou_t, label_t]

        return losses


def _mask_iou(self, mask1, mask2):
    intersection = torch.sum(mask1 * mask2, dim=(0, 1))
    area1 = torch.sum(mask1, dim=(0, 1))
    area2 = torch.sum(mask2, dim=(0, 1))
    union = (area1 + area2) - intersection
    ret = intersection / union
    return ret


def mask_iou_loss(self, net, maskiou_targets):
    maskiou_net_input, maskiou_t, label_t = maskiou_targets

    maskiou_p = net.maskiou_net(maskiou_net_input)

    label_t = label_t[:, None]
    maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)

    loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')

    return loss_i * cfg.maskiou_alpha

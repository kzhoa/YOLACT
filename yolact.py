import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
from backbone import ResNetBackboneGN
import itertools

"""
原作者在整个工程里采用了torch.jit,
为了精简工作，专注内核，已删去相关内容。
"""


class FPN(nn.Module):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf
    3+2=5层结构
    """

    def __init__(self, in_dims: list, out_dim=256, num_downsample=2):
        """
        核心问题在于如何使这个网络泛用，支持不定长的层数。
        """

        # in_dims按传入顺序为[c3,c4,c5]，但处理顺序为c5,c4,c3
        # 先1x1
        self.lat_layers = nn.ModuleList([
            nn.Conv2d(in_dm, out_dim, kernel_size=1) for in_dm in reversed(in_dims)])
        # 再3x3
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1) for _ in in_dims])

        self.num_downsample = int(num_downsample)
        if self.num_downsample > 0:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
                for _ in range(num_downsample)
            ])
        else:
            self.downsample_layers = None

        # 原作者代码里留下的可调节参数，暂时不准备改动它们。
        self.interpolation_mode = 'bilinear'
        self.relu_downsample_layers = False
        self.relu_pred_layers = True

    def forward(self, inpt: list[torch.Tensor]):
        """
        传入顺序为[c3,c4,c5]，但处理顺序为c5,c4,c3
        """
        outs = []
        x = torch.zeros(1, device=inpt[0].device)

        for idx, c_i in enumerate(reversed(inpt)):
            # 以c5c4c3的顺序处理
            _, _, h, w = c_i.shape
            x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x = x + self.lat_layers[idx](c_i)  # 循环中累加的部分不经过pred_layers

            prd = self.pred_layers[idx](x)  # 输出的部分要经过pred_layers
            if self.relu_pred_layers:
                F.relu(prd, inplace=True)
            outs.append(prd)

        # 此时outs的顺序为p5,p4,p3，需要反向为p3p4p5
        outs.reverse()

        # 再增加downsample部分(根据原作者参数，不使用relu)
        if self.num_downsample > 0:
            for dsamp in self.downsample_layers:
                outs.append(dsamp(outs[-1]))

        return outs


class InterpolateModule(nn.Module):
    """
    This is a module version of F.interpolate (rip nn.Upsampling).
    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


class ProtoNet(nn.Module):
    def __init__(self, in_dim, inner_dim=256, out_dim=32):
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, inner_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            InterpolateModule(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, out_dim, kernel_size=3, padding=1),
            # 最后一层不需要relu
        )

    def forward(self, x):
        x = self.layers(x)  # (bz, 32,70,70)
        x = x.permute(0, 2, 3, 1).contiguous()  # (bz,70,70,32)
        return x


class Prediction(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution.
    摘要部分搬运，后面自己写
    """

    def __init__(self,
                 in_dim,
                 out_dim=256,
                 mask_dim=32,
                 num_classes=81,
                 aspect_ratios=[[1, 1 / 2, 2]],
                 scales=[1],  # [24]/[48]/[96]/[192]/[384]]
                 index=0,
                 num_heads=5):
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        self.num_priors = sum(len(x) * len(scales) for x in aspect_ratios)  # 3个ratio*1个sclae=3
        self.parent = None
        self.index = index
        self.num_heads = num_heads
        self.aspect_ratios = aspect_ratios
        self.scales = scales

        # 可选:默认为false
        # 1.不同head分割prototype
        # self.mask_dim = self.mask_dim // self.num_heads
        # 2.prototypes_as_features
        # in_channels += self.mask_dim

        # 原作者默认设为True
        self.extra_head_net = True
        if self.extra_head_net:
            self.upfeature = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        # 原作者默认设为False
        self.use_prediction_module = True
        if self.use_prediction_module:
            self.block = Bottleneck(out_dim, out_dim // 4)  # 除4因为自带expansion=4
            self.conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)
            self.bn = nn.BatchNorm2d(out_dim)

        # 原作者默认设为True
        self.eval_mask_branch = True

        self.bbox_layer = nn.Conv2d(out_dim, self.num_priors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_dim, self.num_priors * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_dim, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        # use_mask_scoring:false
        # use_instance_coeff:false
        # and no extra_layers
        # no gate_layers

        self.last_img_size = None
        self.last_conv_size = None
        self._tmp_img_w, self._tmp_img_h = None, None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        src = self if self.parent is None else self.parent
        conv_h = x.size(2)
        conv_w = x.size(3)

        if self.extra_head_net:
            x = src.upfeature(x)

        if self.use_prediction_module:
            a = src.block(x)

            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)

            x = a + b

        # no extra layers

        # (bz,num_priors*4,conv_h,conv_w)->(bz,num_priors*conv_h*conv_w,4)
        # (bz,num_priors*num_classes,conv_h,conv_w)->(bz,num_priors*conv_h*conv_w,num_classes)
        bbox = src.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        # Default True
        if self.eval_mask_branch:
            mask = src.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if self.eval_mask_branch:
            # 默认mask_type=lincomb,使用torch.tanh,反之若=direct,则用sigmoid
            mask = torch.tanh(mask)

        priors = self.make_priors(conv_h, conv_w, x.device)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}

        return preds

    def make_priors(self, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        size = (conv_h, conv_w)
        if self.last_img_size != (self._tmp_img_w, self._tmp_img_h):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in itertools.product(range(conv_h), range(conv_w)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                # aspect_ratios: [[1, 1/2, 2]], shape=(1,3)
                # scales: [24] or [48] or [96] or [192] or [384],shape=(1)
                for ars in self.aspect_ratios:
                    for scale in self.scales:
                        for ar in ars:

                            # Defalut True,max_size=550
                            self.use_pixel_scales = True
                            self.max_size = 550
                            if self.use_pixel_scales:
                                w = scale * ar / self.max_size
                                h = scale / ar / self.max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h

                            # This is for backward compatability with a bug where I made everything square by accident
                            self.backbone.use_square_anchors = True
                            if self.backbone.use_square_anchors:
                                # True,只使用正方形anchor
                                h = w

                            # 这个函数写的冗余，wh都是复用的。
                            prior_data += [x, y, w, h]

            self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach()
            self.priors.requires_grad = False
            self.last_img_size = (self._tmp_img_w, self._tmp_img_h)
            self.last_conv_size = (conv_w, conv_h)

            # prior_cache[size] = None
            # 为什么要强行置为空呢？如果过去曾经生成过这个size，为什么不从过去的cache里提取？就这也配叫cache吗？
            # 删除关于prior_cache的部分也不影响运行。

        elif self.priors.device != device:
            # 如果本batch的尺寸同上batch，则不必生成新的先验框。
            self.priors = self.priors.to(device)

            # 这个并行部分看的不是很明白，感觉有些冗余。
            # # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
            # if prior_cache[size] is None:
            #     prior_cache[size] = {}
            #
            # if device not in prior_cache[size]:
            #     prior_cache[size][device] = self.priors

        return self.priors


class Detection(nn.Module):
    def __init__(self):
        pass


class Yolact(nn.Module):
    """
    implemented by qq
    """

    def __init__(self,

                 selected_layers=[1, 2, 3],
                 fpn_dim=256,
                 num_downsample=2,
                 ):

        self.mask_dim = 32
        self.num_grids = 0  # 不使用grid
        self.selected_layers = selected_layers
        self.fpn_dim = fpn_dim
        self.num_downsample = num_downsample

        # 5个主要结构
        # 1.backbone
        self.backbone = ResNetBackboneGN([3, 4, 23, 3])

        # 2.fpn
        src_channels = self.backbone.channels  # [64*4,128*4,256*4,512*4]
        # 取[1,2,3],得到[128*4,256*4,512*4]
        self.fpn = FPN([src_channels[i] for i in self.selected_layers],
                       out_dim= fpn_dim, num_downsample=num_downsample)

        #经过fpn后，[0,1,2,3,4] 现在有5个层
        self.selected_layers = list(range(len(selected_layers)+num_downsample))

        # 3.protonet
        self.proto_net = ProtoNet(in_dim=fpn_dim, inner_dim=256, out_dim=32)

        # 4.prediction_module
        # 原作者在yolact1.0中采用共享，yolact++中关闭共享。所以需要同时支持开关。
        # 他那种写法不太符合我的习惯，这里改写一下。
        self.share_prediction_module = True
        if self.share_prediction_module:
            #开启共享时，本质上只有一层。
            self.prediction_layers = Prediction(in_dim=fpn_dim,out_dim=256,mask_dim=32,num_classes=81)
        else:
            #关闭共享时，就是一个list对list了
            self.prediction_layers = nn.ModuleList()


        # 5.detection
        self.detection

    def forward(self, x):
        pass

    # 其他功能函数
    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn(enable=mode)

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                            all_in(module.__dict__['_constants_set'], conv_constants)
                            and all_in(conv_constants, module.__dict__['_constants_set']))

            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0] = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0] = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

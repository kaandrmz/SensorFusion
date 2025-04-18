# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Changqian Yu (y-changqian@outlook.com)
import torch.nn as nn
from .cond_aspp_head import CondASPPHead
from .cond_sep_aspp_head import CondDepthwiseSeparableASPPHead
# from .cond_uper_head import CondUPerHead
from .fcnhead import FCNHead
from .resnet import ResNetV1c
import torch.nn.functional as F
class CondNet_50(nn.Module):
    def __init__(self, numclass):
        super().__init__()
        self.backbone = ResNetV1c(depth=50,
                                num_stages=4,
                                out_indices=(0, 1, 2, 3),
                                dilations=(1, 1, 2, 4),
                                strides=(1, 2, 1, 1),
                                norm_cfg=dict(type='BN', requires_grad=True),
                                norm_eval=False,
                                style='pytorch',
                                contract_dilation=True)
        self.decode_head = CondDepthwiseSeparableASPPHead(in_channels=2048,
                                in_index=3,
                                channels=512,
                                dilations=(1, 12, 24, 36),
                                num_cond_layers=1,
                                cond_layer_with_bias=True,
                                c1_in_channels=256,
                                c1_channels=48,
                                dropout_ratio=0.1,
                                num_classes=numclass,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
        self.auxiliary_head = FCNHead(
                                in_channels=1024,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=numclass,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))

    def forward(self, x):
        backbone_out = self.backbone(x)
        out = self.decode_head(backbone_out)
        out_aux = self.auxiliary_head(backbone_out)
        out_aux = F.interpolate(out_aux,(out_aux.shape[2]*2,out_aux.shape[3]*2),mode="bilinear", align_corners=True)
        return out
class CondNet_101(nn.Module):
    def __init__(self, numclass):
        super().__init__()
        self.backbone = ResNetV1c(depth=101,
                                num_stages=4,
                                out_indices=(0, 1, 2, 3),
                                dilations=(1, 1, 2, 4),
                                strides=(1, 2, 1, 1),
                                norm_cfg=dict(type='BN', requires_grad=True),
                                norm_eval=False,
                                style='pytorch',
                                contract_dilation=True)
        self.decode_head = CondDepthwiseSeparableASPPHead(in_channels=2048,
                                in_index=3,
                                channels=512,
                                dilations=(1, 12, 24, 36),
                                num_cond_layers=1,
                                cond_layer_with_bias=True,
                                c1_in_channels=256,
                                c1_channels=48,
                                dropout_ratio=0.1,
                                num_classes=numclass,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
        self.auxiliary_head = FCNHead(
                                in_channels=1024,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=numclass,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))

    def forward(self, x):
        backbone_out = self.backbone(x)
        out = self.decode_head(backbone_out)
        out_aux = self.auxiliary_head(backbone_out)
        out_aux = F.interpolate(out_aux,(out_aux.shape[2]*4,out_aux.shape[3]*4),mode="bilinear", align_corners=True)
        return out



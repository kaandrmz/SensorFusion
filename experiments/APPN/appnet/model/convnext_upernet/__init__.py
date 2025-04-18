from mimetypes import init
import torch
import torch.nn as nn
from .convnext import ConvNeXt_B,ConvNeXt_S,ConvNeXt_T,ConvNeXt_L,ConvNeXt_XL
from .uperhead import UPerHead
from .fcnhead import FCNHead
import torch.nn.functional as F
class ConvNextUPNet_T(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = ConvNeXt_T()
        self.decode_head = UPerHead(in_channels=[96, 192, 384, 768],  # tiny的参数
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=channels,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.auxiliary_head = FCNHead(
                                in_channels=384,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
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
class ConvNextUPNet_S(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = ConvNeXt_S()
        self.decode_head = UPerHead(in_channels=[96, 192, 384, 768],  # Small的参数
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=channels,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.auxiliary_head = FCNHead(
                                in_channels=384,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
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
class ConvNextUPNet_B(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = ConvNeXt_B()
        self.decode_head = UPerHead(in_channels=[128, 256, 512, 1024],  # Base的参数
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=channels,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.auxiliary_head = FCNHead(
                                in_channels=512,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
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
class ConvNextUPNet_L(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = ConvNeXt_L()
        self.decode_head = UPerHead(in_channels=[192, 384, 768, 1536],  # Large的参数
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=channels,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.auxiliary_head = FCNHead(
                                in_channels=768,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
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
class ConvNextUPNet_XL(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = ConvNeXt_XL()
        self.decode_head = UPerHead(in_channels=[256, 512, 1024, 2048],  # Large的参数
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=channels,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                align_corners=False,)
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.auxiliary_head = FCNHead(
                                in_channels=1024,
                                in_index=2,
                                channels=256,
                                num_convs=1,
                                concat_input=False,
                                dropout_ratio=0.1,
                                num_classes=out_chans,
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
# if __name__ == '__main__':
#     from torchsummary import summary
#
#     data = torch.randn(2, 1, 256, 256).cuda()
#     a = upernet_convnext_tiny(1, 4).cuda()
#     s = a(data)
#     print(s.shape)
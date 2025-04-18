from mimetypes import init
import torch
import torch.nn as nn
from .mask2former_head import Mask2FormerHead
from .msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from .transformer import DetrTransformerEncoder,DetrTransformerDecoder,DetrTransformerDecoderLayer
from .positional_encoding import SinePositionalEncoding
from .vit_adapter import Adapter_T,Adapter_B,Adapter_L
from .beit_adapter import BeiTadapter_L
from .uperhead import UPerHead
from .fcnhead import FCNHead
import torch.nn.functional as F
class ViTAdapter_T(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = Adapter_T()
        self.decode_head = UPerHead(in_channels=[192, 192, 192, 192],  # tiny的参数
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
                                in_channels=192,
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
class ViTAdapter_B(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = Adapter_B()
        self.decode_head = UPerHead(in_channels=[768, 768, 768, 768],  # tiny的参数
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
class ViTAdapter_L(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = Adapter_L()
        self.decode_head = UPerHead(in_channels=[1024, 1024, 1024, 1024],  # tiny的参数
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
class BeiTAdapter_L(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = BeiTadapter_L()
        self.decode_head = UPerHead(in_channels=[1024, 1024, 1024, 1024],  # tiny的参数
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=1024,
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

class MaskAdapter_L(nn.Module):
    def __init__(self, out_chans,channels):
        super().__init__()
        self.backbone = BeiTadapter_L()
        self.decode_head = Mask2FormerHead(in_channels=[1024, 1024, 1024, 1024],
                                           feat_channels=1024,
                                           out_channels=1024,
                                           in_index=[0, 1, 2, 3],
                                           num_things_classes=8,
                                           num_stuff_classes=11,
                                           num_queries=100,
                                           num_transformer_feat_level=3,
                                           pixel_decoder=MSDeformAttnPixelDecoder(
                                               # type='MSDeformAttnPixelDecoder',
                                               num_outs=3,
                                               norm_cfg=dict(type='GN', num_groups=32),
                                               act_cfg=dict(type='ReLU'),
                                               encoder=DetrTransformerEncoder(
                                                   # type='DetrTransformerEncoder',
                                                   num_layers=6,
                                                   transformerlayers=dict(
                                                       type='BaseTransformerLayer',
                                                       attn_cfgs=dict(
                                                           type='MultiScaleDeformableAttention',
                                                           embed_dims=1024,
                                                           num_heads=32,
                                                           num_levels=3,
                                                           num_points=4,
                                                           im2col_step=64,
                                                           dropout=0.0,
                                                           batch_first=False,
                                                           norm_cfg=None,
                                                           init_cfg=None),
                                                       ffn_cfgs=dict(
                                                           type='FFN',
                                                           embed_dims=1024,
                                                           feedforward_channels=4096,
                                                           num_fcs=2,
                                                           ffn_drop=0.0,
                                                           with_cp=False,  # set with_cp=True to save memory
                                                           act_cfg=dict(type='ReLU', inplace=True)),
                                                           operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                                                           init_cfg=None),
                                               positional_encoding=SinePositionalEncoding(num_feats=512, normalize=True),
                                               init_cfg=None),
                                           positional_encoding=SinePositionalEncoding(num_feats=512, normalize=True),
                                           transformer_decoder=DetrTransformerDecoder(
                                               # type='DetrTransformerDecoder',
                                               return_intermediate=True,
                                               num_layers=9,
                                               transformerlayers=dict(
                                                   type='DetrTransformerDecoderLayer',
                                                   attn_cfgs=dict(
                                                       type='MultiheadAttention',
                                                       embed_dims=1024,
                                                       num_heads=32,
                                                       attn_drop=0.0,
                                                       proj_drop=0.0,
                                                       dropout_layer=None,
                                                       batch_first=False),
                                                   ffn_cfgs=dict(
                                                       embed_dims=1024,
                                                       feedforward_channels=4096,
                                                       num_fcs=2,
                                                       act_cfg=dict(type='ReLU', inplace=True),
                                                       ffn_drop=0.0,
                                                       dropout_layer=None,
                                                       with_cp=True,  # set with_cp=True to save memory
                                                       add_identity=True),
                                                   feedforward_channels=4096,
                                                   operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                                                 'ffn', 'norm')),
                                               init_cfg=None)
                                    )
                                # loss_decode=dict(
                                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        # self.auxiliary_head = FCNHead(
        #                         in_channels=1024,
        #                         in_index=2,
        #                         channels=256,
        #                         num_convs=1,
        #                         concat_input=False,
        #                         dropout_ratio=0.1,
        #                         num_classes=out_chans,
        #                         norm_cfg=dict(type='BN', requires_grad=True),
        #                         align_corners=False,)
        #                         # loss_decode=dict(
        #                         #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))

    def forward(self, x):
        backbone_out = self.backbone(x)
        out = self.decode_head.forward_test(backbone_out,1)
        # out_aux = self.auxiliary_head(backbone_out)
        # out_aux = F.interpolate(out_aux,(out_aux.shape[2]*4,out_aux.shape[3]*4),mode="bilinear", align_corners=True)
        return out


# if __name__ == '__main__':
#     from torchsummary import summary
#
#     data = torch.randn(2, 1, 256, 256).cuda()
#     a = upernet_convnext_tiny(1, 4).cuda()
#     s = a(data)
#     print(s.shape)
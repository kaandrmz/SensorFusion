import torch
import torch.nn as nn
from .base import BN_MOMENTUM, BatchNorm2d, Bottleneck
from appnet.utils.geometry import calculate_certainty
from appnet.utils.SignReLU import SignReLU
import torch.nn.functional as F
from appnet.model.convnext_upernet.uperhead import UPerHead
from mmseg.ops import resize
import math

class RefinementAppNet(nn.Module):
    def __init__(self, n_classes, inputsize, k_size=3, use_bn=False):
        super().__init__()

        self.use_bn = use_bn
        self.w = int(inputsize[0] / 2)
        self.h = int(inputsize[1] / 2)
        # else:
        #     self.w = int(inputsize[0] / 8)
        #     self.h = int(inputsize[1] / 8)
        self.in_index = [0, 1, 2, 3]
        self.input_transform = 'resize_concat'
        if use_bn:
            self.bn0 = BatchNorm2d(n_classes*2, momentum=BN_MOMENTUM)

        # pixel-aware attention

        # channel attention moudle
        self.adpool = nn.AdaptiveAvgPool2d((int(self.h/4), int(self.w/4)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 2 conv layers
        self.conv1 = nn.Conv2d(n_classes*2, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(96, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(96, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # spatial attention moudle
        self.conv_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2,bias=False)
        self.conv_5 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=(5 - 1)//2,bias=False)
        self.conv_7 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2,bias=False)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)


        #class-aware attention
        self.class_head = MultiLabelHead(num_classes=n_classes,d_encoder=256,hidden_dim=256,n_heads=4,d_ff=1024,dropout=0.1,share_embedding=False,downsample=16)
        # 2 residual blocks
        self.residual = self._make_layer(Bottleneck, 96, 64, 2)

        # Prediction head
        self.seg_conv = nn.Conv2d(256, 9, kernel_size=1, stride=1, padding=0, bias=False)
        self.edge_conv = nn.Conv2d(150, 2, kernel_size=1, stride=1, padding=0, bias=False)
        # # edge prediciton
        self.mask_conv = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.mask_conv2 = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        """Make residual block"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fi_segmentation, co_segmentation, aug=False):
        co = F.interpolate(co_segmentation, (self.w,self.h), mode="bilinear", align_corners=False)
        fi = F.interpolate(fi_segmentation, (self.w,self.h), mode="bilinear", align_corners=False)
        x = torch.cat([fi.softmax(1),co.softmax(1)],dim=1)
        if self.use_bn:
            x = self.bn0(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        y = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        s3 = self.conv_3(y)
        s5 = self.conv_5(y)
        s7 = self.conv_7(y)
        s = torch.cat([s3, s5, s7], dim=1)
        out = self.conv3(s)
        out1 = self.avg_pool(x)
        out1 = self.conv(out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = out*out1
        out = self.sigmoid(out)
        x = x*out
        x = self.residual(x)
        class_x = self.class_head(x)
        class_x = self.sigmoid(class_x)
        alter = self.mask_conv(x)
        alter = self.sigmoid(alter)
        sum_weights = (class_x.unsqueeze(2).unsqueeze(3) * alter)
        refine_output  = fi*sum_weights+co*(1-sum_weights)
        

        if aug is True:
            return fi,refine_output,co
        else:
            return refine_output















class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class MultiLabelHead(nn.Module):
    def __init__(
        self,
        num_classes,
        d_encoder,
        hidden_dim,
        n_heads,
        d_ff,
        dropout,
        share_embedding,
        downsample=None,
        mlp=True,
        droppath=0,
    ):
        super().__init__()
        self.share_embedding = share_embedding
        self.mlp = mlp
        self.block = Block(hidden_dim, n_heads, d_ff, dropout, droppath)
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_classes = num_classes
        self.fc = GroupWiseLinear(num_classes, hidden_dim)

        if not share_embedding:
            self.cls_emb = nn.Parameter(torch.randn(1, num_classes, hidden_dim))
            from torch.nn.init import trunc_normal_

            trunc_normal_(self.cls_emb, std=0.02)
        self.scale = hidden_dim ** -0.5

        self.proj_dec = nn.Linear(d_encoder, hidden_dim)
        self.downsample = downsample
        if downsample:
            self.pooling = nn.AdaptiveAvgPool2d(downsample)

    def forward(self, x):
        if self.share_embedding:
            x, cls_emb = x
            cls_emb = cls_emb.unsqueeze(0)
        else:
            cls_emb = self.cls_emb
        if self.downsample:
            x = self.pooling(x)

        B, C = x.size()[:2]
        x = x.view(B, C, -1).permute(0,2,1)
        x = self.proj_dec(x)

        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        x = self.block(x)
        x = self.norm(x)
        cls_emb = x[:, -self.num_classes :]
        img_pred = self.fc(cls_emb)

        return img_pred


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        B, HW, C = x.size()
        x = x.view(B, GS, HW // GS, C).permute(0, 3, 1, 2)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, int(hidden_features/10))
        self.act2 = act_layer()
        self.fc3 = nn.Linear(int(hidden_features / 10), out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
from torch import tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchinfo import summary
import sys,os
from einops import rearrange
from archs.odconv import ODConv2d
import numbers
    

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
## ODConv2d
def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

    
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv_dwconv = odconv3x3(dim*3, dim*3, reduction=0.0625, kernel_num=1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class DualModalFusionSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias"):
        super(DualModalFusionSelfAttention, self).__init__()

        # Self-Attention and FeedForward for modality 1 (e.g., infrared)
        self.norm1_mod1 = LayerNorm(dim, LayerNorm_type)
        self.attn_mod1 = Attention(dim, num_heads, bias)
        self.norm2_mod1 = LayerNorm(dim, LayerNorm_type)
        self.ffn_mod1 = FeedForward(dim, ffn_expansion_factor, bias)

        # Self-Attention and FeedForward for modality 2 (e.g., visible)
        self.norm1_mod2 = LayerNorm(dim, LayerNorm_type)
        self.attn_mod2 = Attention(dim, num_heads, bias)
        self.norm2_mod2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_mod2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x1, x2):
        # Process modality 1 (e.g., infrared)
        x1 = x1 + self.attn_mod1(self.norm1_mod1(x1))
        x1 = x1 + self.ffn_mod1(self.norm2_mod1(x1))

        # Process modality 2 (e.g., visible)
        x2 = x2 + self.attn_mod2(self.norm1_mod2(x2))
        x2 = x2 + self.ffn_mod2(self.norm2_mod2(x2))

        return x1, x2

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False, LayerNorm_type="WithBias"):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Conv layers for processing Q
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Conv layers for processing K, V
        self.kv_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # LayerNorm for both Q and KV
        self.norm_q = LayerNorm(dim, LayerNorm_type)
        self.norm_kv = LayerNorm(dim * 2, LayerNorm_type)

    def forward(self, q, kv):
        # Apply LayerNorm to Q and KV
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Process Q
        q = self.q_dwconv(self.q_conv(q))

        # Process K and V
        kv = self.kv_dwconv(self.kv_conv(kv))
        k, v = kv.chunk(2, dim=1)

        # Attention calculation
        b, _, h, w = q.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out


class ProgressiveFusionCrossAttention(nn.Module):
    def __init__(self, dim=36, num_heads=6, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias"):
        super(ProgressiveFusionCrossAttention, self).__init__()

        # Cross-attention mechanism for modality 1 and modality 2
        self.cross_attention_mod1 = CrossAttention(dim, num_heads, bias, LayerNorm_type)
        self.cross_attention_mod2 = CrossAttention(dim, num_heads, bias, LayerNorm_type)

        # LayerNorm and FeedForward for modality 1
        self.norm2_mod1 = LayerNorm(dim, LayerNorm_type)
        self.ffn_mod1 = FeedForward(dim, ffn_expansion_factor, bias)

        # LayerNorm and FeedForward for modality 2
        self.norm2_mod2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_mod2 = FeedForward(dim, ffn_expansion_factor, bias)

        # Output projection layer
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, d1, d2, s1, s2):
        # Process deep features and shallow features for modality 1
        ds1 = torch.cat([d1, s1], dim=1)
        ds2 = torch.cat([d2, s2], dim=1)

        # Apply cross-attention and projection for modality 1
        d1 = d1 + self.project_out(self.cross_attention_mod1(d1, ds2))
        d1 = d1 + self.ffn_mod1(self.norm2_mod1(d1))

        # Apply cross-attention and projection for modality 2
        d2 = d2 + self.project_out(self.cross_attention_mod2(d2, ds1))
        d2 = d2 + self.ffn_mod2(self.norm2_mod2(d2))

        return d1, d2
    
class ConcatFusion(nn.Module):
    def __init__(self, feature_dim):
        super(ConcatFusion, self).__init__()
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            nn.Hardswish(inplace=False),
            #nn.LeakyReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
        )
                    
    def forward(self, modality1, modality2):
        concat_features = torch.cat([modality1, modality2], dim=1)
        return self.dim_reduce(concat_features)

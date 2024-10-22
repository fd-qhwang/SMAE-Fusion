
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import math
import numbers
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from einops import rearrange
import sys,os
from torchinfo import summary
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
from archs.attn_utils import LayerNorm, FeedForward, Attention
    
##########################################################################
class HATBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, reduction, kernel_size):
        super(HATBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)  
        self.norm2 = LayerNorm(dim, LayerNorm_type)  
        self.self_attn = Attention(dim, num_heads, bias) 
        self.cbam = CBAMBlock(channel=dim, reduction=reduction, kernel_size=kernel_size)  
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias) 
        self.conv1_1 = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x)
        cbam_out = self.cbam(norm_x)
        combined_out = self.conv1_1(torch.cat([cbam_out, attn_out], dim=1))
        x = x + combined_out  

        x = x + self.ffn(self.norm2(x)) 
        return x

#########################################################################
    
class PatchEmbed(nn.Module):
    def __init__(self, input_channels=1, output_channels=64, layer_norm='WithBias'):
        super(PatchEmbed, self).__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            LayerNorm(output_channels, LayerNorm_type=layer_norm) if layer_norm else nn.Identity(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        
        return x

##---------- Deep Extractor -----------------------
class Encoder(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, reduction, kernel_size):
        super(Encoder, self).__init__()

        self.blocks = nn.ModuleList([
            HATBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, reduction, kernel_size)
            for _ in range(num_blocks)
        ])
        
        self.final_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        residual = x  
        for block in self.blocks:
            x = block(x)

        return self.final_conv(x) + residual

##---------- Share Decoder -----------------------
class Decoder(nn.Module):
    def __init__(self, num_blocks, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, reduction, kernel_size):
        super(Decoder, self).__init__()
        
        self.blocks = nn.ModuleList([
            HATBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, reduction, kernel_size)
            for _ in range(num_blocks)
        ])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),  
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1),  
        )
        
        self.tanh = nn.Tanh()

    def forward(self, x, inp_img=None):
        residual = x  
        
        for block in self.blocks:
            x = block(x)

        x = x + residual

        x = self.decoder(x)

        if inp_img is not None:
            x = x + inp_img

        return (self.tanh(x) + 1) / 2

        
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual
    

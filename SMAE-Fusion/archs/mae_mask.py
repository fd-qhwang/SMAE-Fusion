import torch 
import torch.nn as nn 
import random 
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def ShuffleIndex(index_tensor: torch.Tensor, sample_ratio: float):

    num_total = len(index_tensor)
    
    num_mask = int(round(num_total * sample_ratio))
    
    perm = torch.randperm(num_total)
    
    mask_index = perm[:num_mask]  
    sample_index = perm[num_mask:]  
    
    return mask_index, sample_index  

def MaskEmbed(token_emb, mask_ratio):
    B, D, H, W = token_emb.shape  
    
    token_emb = token_emb.view(B, D, H * W)
    
    total_tokens = H * W
    token_index = torch.arange(total_tokens)  
    
    mask_index, sample_index = ShuffleIndex(token_index, mask_ratio)
    
    mask = torch.ones((B, D, H * W), device=token_emb.device, dtype=torch.bool)
    
    mask[:, :, mask_index] = 0.0
    
    token_emb[:, :, mask_index] = 0.0

    token_emb = token_emb.view(B, D, H, W)
    mask = mask.view(B, D, H, W)
    
    return token_emb, mask  


class UnMaskEmbed(nn.Module):

    def __init__(self, embed_dim, in_chans, patch_size):
        super(UnMaskEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        
        self.raw_input = torch.ones((in_chans, 1, 1)) * 127.0 / 255.0  # 全1张量，值为127/255
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=1, padding=patch_size // 2)

    def forward(self, enc_emb, mask):

        b, c, h, w = enc_emb.shape
        device = enc_emb.device
        
        raw_inputs = self.raw_input.expand(b, -1, h, w).to(device)
        mask_token_embedding = self.proj(raw_inputs)  # [B, embed_dim, H, W]
        
        combined_features = torch.where(mask > 0, enc_emb, mask_token_embedding)

        return combined_features
    
    
def SAMaskEmbed(token_emb, sa_map, mask_ratio):

    b, c, h, w = token_emb.shape
    num_pixels = h * w
    num_to_mask = int(round(mask_ratio * num_pixels))
    
    noise = torch.rand_like(sa_map) * 0.5  
    
    sa_map_with_noise = sa_map + noise
    flat_sa_map = sa_map_with_noise.view(b, -1)  # [b, h*w]

    _, sorted_indices = torch.sort(flat_sa_map, dim=1, descending=True)
    
    mask_indices = sorted_indices[:, :num_to_mask]  

    mask = torch.ones(b, h*w, device=token_emb.device, dtype=torch.bool)

    mask.scatter_(1, mask_indices, 0)
    
    mask = mask.view(b, 1, h, w).expand(-1, c, -1, -1) 
    
    masked_emb = token_emb * mask
    
    return masked_emb, mask

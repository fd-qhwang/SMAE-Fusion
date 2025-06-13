import torch
import torch.nn as nn
import numbers
# This line resolves absolute import issues but may cause relative import problems
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from archs.arch_utils import Encoder, Decoder, PatchEmbed
from archs.mae_mask import MaskEmbed, UnMaskEmbed, SAMaskEmbed
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys,os
from torchinfo import summary
from thop import profile
from thop import clever_format

@ARCH_REGISTRY.register()
class SMAEPretrain(nn.Module):
    def __init__(self, 
                 inp_channels=1, 
                 dim=64,
                 img_size=128,
                 mask_ratio=0.75,
                 encoder_configs={},
                 decoder_configs={},
                 ):
        super(SMAEPretrain, self).__init__()

        self.inp_channels = inp_channels
        self.dim = dim
        self.img_size = img_size
        self.mask_ratio = mask_ratio

        # Step 1: Embedding for both modalities (VIS and IR)
        self.VIS_embed = PatchEmbed(input_channels=inp_channels, output_channels=dim)
        self.IR_embed = PatchEmbed(input_channels=inp_channels, output_channels=dim)

        # Step 2: Feature Extractors for both VIS and IR
        self.VFE = Encoder(**encoder_configs)
        self.IFE = Encoder(**encoder_configs)

        # Step 3: Unmasking Modules for both VIS and IR
        self.VIS_UNMASK = UnMaskEmbed(dim, dim, 3)
        self.IR_UNMASK = UnMaskEmbed(dim, dim, 3)

        # Step 4: Decoders for VIS and IR
        self.VIS_Dec = Decoder(**decoder_configs)
        self.IR_Dec = Decoder(**decoder_configs)

        # Step 5: Position Embedding
        self.pos_embed_vis = nn.Parameter(torch.zeros(1, dim, img_size, img_size))  # Adjust based on input image size
        self.pos_embed_ir = nn.Parameter(torch.zeros(1, dim, img_size, img_size))
        trunc_normal_(self.pos_embed_vis, std=.02)
        trunc_normal_(self.pos_embed_ir, std=.02)

    def forward(self, modality1, modality2, map1=None, map2=None):
        # Step 1: Apply Patch Embedding
        vis_enc = self.VIS_embed(modality1)
        ir_enc = self.IR_embed(modality2)

        # Step 3: Apply Masking for MAE Pretraining using SAMaskEmbed
        vis_masked, vis_mask = SAMaskEmbed(vis_enc, map1, mask_ratio=self.mask_ratio)
        ir_masked, ir_mask = SAMaskEmbed(ir_enc, map2, mask_ratio=self.mask_ratio)
        #vis_masked, vis_mask = MaskEmbed(vis_enc, mask_ratio=self.mask_ratio)
        #ir_masked, ir_mask = MaskEmbed(ir_enc, mask_ratio=self.mask_ratio)
        

        # Step 4: Deep Feature Extraction
        vis_feat = self.VFE(vis_masked)  # Visual feature extraction from masked VIS embedding
        ir_feat = self.IFE(ir_masked)    # Infrared feature extraction from masked IR embedding

        # Step 5: Unmasking
        vis_dec = self.VIS_UNMASK(vis_feat, vis_mask)
        ir_dec = self.IR_UNMASK(ir_feat, ir_mask)

        # Step 6: Decoding
        vis_dec = self.VIS_Dec(vis_dec)
        ir_dec = self.IR_Dec(ir_dec)
        
        return vis_dec, ir_dec, vis_mask, ir_mask  # Return the reconstructed VIS/IR images and masks

import torch
import torch.nn as nn
import numbers
# This line resolves absolute import issues but may cause relative import problems
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from archs.arch_utils import Encoder, Decoder, PatchEmbed
from archs.attn_utils import DualModalFusionSelfAttention, ProgressiveFusionCrossAttention, ConcatFusion
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys,os
from torchinfo import summary
from thop import profile
from thop import clever_format

@ARCH_REGISTRY.register()
class SMAEFinetune(nn.Module):
    def __init__(self, 
                 inp_channels=1,  # Input channels (1 for grayscale, 3 for RGB)
                 dim=64,          # Feature embedding dimension
                 img_size=128,    # Input image size
                 encoder_configs={},  # Encoder configuration for MAE pretrained model settings
                 decoder_configs={},  # Decoder configuration for image reconstruction
                 ):
        super(SMAEFinetune, self).__init__()

        self.inp_channels = inp_channels  # Set input channels
        self.dim = dim  # Feature embedding dimension
        self.img_size = img_size  # Input image size

        # Step 1: Patch Embedding for both modalities (VIS and IR)
        self.VIS_embed = PatchEmbed(input_channels=inp_channels, output_channels=dim)
        self.IR_embed = PatchEmbed(input_channels=inp_channels, output_channels=dim)

        # Step 2: Encoders for VIS and IR
        self.VFE = Encoder(**encoder_configs)
        self.IFE = Encoder(**encoder_configs)

        # Step 3: Deep Feature Interaction (Self-Attention and Cross-Attention)
        self.SAB = DualModalFusionSelfAttention(dim=dim, num_heads=8)
        self.CAB = ProgressiveFusionCrossAttention(dim=dim, num_heads=8)
        self.fused = ConcatFusion(feature_dim=dim)

        # Step 4: Decoders for VIS and IR
        self.Rec = Decoder(**decoder_configs)



    def forward(self, modality1, modality2):
        # Step 1: Apply Patch Embedding 
        vis_enc = self.VIS_embed(modality1)  
        ir_enc = self.IR_embed(modality2)    
        

        # Step 3: Deep Feature Extraction
        vis_feat = self.VFE(vis_enc)  
        ir_feat = self.IFE(ir_enc)    

        # Step 4: Deep Feature Interaction
        vis_feat, ir_feat = self.SAB(vis_feat, ir_feat)
        vis_feat, ir_feat = self.CAB(vis_feat, ir_feat, vis_enc, ir_enc)

        # Step 5: Feature Fusion
        fused_features = self.fused(vis_feat, ir_feat)

        # Step 6: Decoding
        output = self.Rec(fused_features, modality1)
        
        return output  

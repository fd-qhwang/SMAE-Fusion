import torch
import torch.nn as nn
import numbers
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from archs.arch_utils import Encoder, Decoder, PatchEmbed
from archs.attn_utils import DualModalFusionSelfAttention, ProgressiveFusionCrossAttention, ConcatFusion
import sys,os


#@ARCH_REGISTRY.register()
class SMAEFinetune(nn.Module):
    def __init__(self, 
                 inp_channels=1, 
                 dim=64,         
                 img_size=128,    
                 encoder_configs={},  
                 decoder_configs={},  
                 ):
        super(SMAEFinetune, self).__init__()

        self.inp_channels = inp_channels  
        self.dim = dim 
        self.img_size = img_size 

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

        self.Rec = Decoder(**decoder_configs)

        #self.pos_embed_vis = nn.Parameter(torch.zeros(1, dim, img_size, img_size))  
        #self.pos_embed_ir = nn.Parameter(torch.zeros(1, dim, img_size, img_size))   
        #trunc_normal_(self.pos_embed_vis, std=.02)  
        #trunc_normal_(self.pos_embed_ir, std=.02)  


    def forward(self, modality1, modality2):
        # Step 1: Apply Patch Embedding
        vis_enc = self.VIS_embed(modality1) 
        ir_enc = self.IR_embed(modality2)    
        
        # Step 2: Add Position Embedding
        #vis_enc = vis_enc + self.pos_embed_vis  
        #ir_enc = ir_enc + self.pos_embed_ir    

        vis_feat = self.VFE(vis_enc) 
        ir_feat = self.IFE(ir_enc)    

        vis_feat, ir_feat = self.SAB(vis_feat, ir_feat)  
        vis_feat, ir_feat = self.CAB(vis_feat, ir_feat, vis_enc, ir_enc)  # 交叉注意力交互

        fused_features = self.fused(vis_feat, ir_feat)  # 将 VIS 和 IR 特征融合

        output = self.Rec(fused_features, modality1)  
        
        return output  
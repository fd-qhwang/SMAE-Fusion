from collections import OrderedDict
import torch
import torch.nn as nn
from archs import build_network
from losses import build_loss
from models import lr_scheduler as lr_scheduler
from utils import get_root_logger
from utils.registry import MODEL_REGISTRY
import os
from os import path as osp
from .base_model import BaseModel

@MODEL_REGISTRY.register()
class SMAEPreModel(BaseModel):
    """Model for SMAE pretraining only."""
    ###The remaining code is in progress and will be uploaded shortly###
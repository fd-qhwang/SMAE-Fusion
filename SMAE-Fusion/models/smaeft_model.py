from collections import OrderedDict
import torch
import cv2
import numpy as np
from archs import build_network
from losses import build_loss
from models import lr_scheduler as lr_scheduler
from metrics import calculate_metric
from utils import get_root_logger
from utils import get_root_logger, imwrite, tensor2img
from tqdm import tqdm
from utils.registry import MODEL_REGISTRY
import os
from os import path as osp
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SMAEFtModel(BaseModel):
    """Finetuning model with encoder initialized from pretrained SMAE."""
    ###The remaining code is in progress and will be uploaded shortly###
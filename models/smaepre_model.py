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

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        # Create generator network
        self.net_g = build_network(opt['network_g']).to(self.device)

        # Load pretrained model weights if available
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, strict=self.opt['path'].get('strict_load_g', True), param_key=param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        """Initialize training settings including loss functions, optimizers and learning rate schedulers."""
        train_opt = self.opt['train']

        # Loss function: mask pixel loss
        self.cri_mask_pixel = build_loss(train_opt['mask_pixel_opt']).to(self.device)

        # Optimizer and scheduler setup
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """Set up optimizers."""
        train_opt = self.opt['train']
        optim_params = self.net_g.parameters()
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers = [self.optimizer_g]

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def feed_data(self, data):
        """Get input visible (VIS) and infrared (IR) image data."""
        self.VIS = data['VIS'].to(self.device)
        self.IR = data['IR'].to(self.device)
        self.VISMAP = data.get('VIS_MAP', None).to(self.device) if 'VIS_MAP' in data else None
        self.IRMAP = data.get('IR_MAP', None).to(self.device) if 'IR_MAP' in data else None

    def optimize_parameters(self, current_iter):
        """Optimize network parameters, execute one training step per iteration."""
        self.optimizer_g.zero_grad()

        # Forward pass, get network output: reconstructed images and masks
        self.VIS_hat, self.IR_hat, self.VIS_Mask, self.IR_Mask = self.net_g(self.VIS, self.IR, self.VISMAP, self.IRMAP)

        # Calculate mask pixel loss, only compute pixel loss for masked regions
        loss_mask_VIS = self.cri_mask_pixel(self.VIS_hat * (~self.VIS_Mask), self.VIS * (~self.VIS_Mask))
        loss_mask_IR = self.cri_mask_pixel(self.IR_hat * (~self.IR_Mask), self.IR * (~self.IR_Mask))

        # Total loss is the sum of both
        total_loss = loss_mask_VIS + loss_mask_IR

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer_g.step()

        # Record losses
        self.log_dict = {
            'loss_mask_VIS': loss_mask_VIS.item(),
            'loss_mask_IR': loss_mask_IR.item(),
            'total_loss': total_loss.item()
        }

    def test(self):
        """Test phase."""
        self.net_g.eval()
        with torch.no_grad():
            self.VIS_hat, self.IR_hat, _, _ = self.net_g(self.VIS, self.IR, self.VISMAP, self.IRMAP)
        self.net_g.train()

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load pretrained model weights."""
        logger = get_root_logger()
        logger.info(f'Loading model for {net.__class__.__name__} [{param_key}] from {load_path}.')
        state_dict = torch.load(load_path, map_location=self.device)
        if param_key is not None:
            state_dict = state_dict[param_key]
        net.load_state_dict(state_dict, strict=strict)

    def get_current_visuals(self):
        """Get current visual outputs (visible and infrared reconstructed images)."""
        return OrderedDict({
            'VIS': self.VIS_hat.detach().cpu(),
            'IR': self.IR_hat.detach().cpu()
        })

    def get_current_log(self):
        """Get current training log information."""
        return self.log_dict

    def save(self, epoch, current_iter):
        """Save model and current training state."""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)



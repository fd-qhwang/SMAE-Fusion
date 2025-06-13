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

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

        # Create generator network
        self.net_g = build_network(opt['network_g']).to(self.device)

        # Load encoder part weights from pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path:
            self.load_encoder_from_pretrained(load_path)

        if self.is_train:
            self.init_training_settings()

    def load_encoder_from_pretrained(self, load_path):
        """Load only encoder weights from pretrained model and initialize encoder part of current model."""
        logger = get_root_logger()
        logger.info(f'Loading pretrained encoder from {load_path}.')

        # Load pretrained model
        state = torch.load(load_path, map_location=self.device)

        # Check if pretrained model contains 'params', extract if available
        if 'params' in state:
            state_dict = state['params']
        else:
            state_dict = state

        # Print pretrained model parameter names
        logger.info("Pretrained model parameter names:")
        for k in state_dict.keys():
            pass
            #logger.info(k)

        # Remove possible prefix, e.g., 'module.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            else:
                new_k = k
            new_state_dict[new_k] = v

        # Get current model state dict
        model_dict = self.net_g.state_dict()

        # Print current model parameter names
        logger.info("Current model parameter names:")
        for k in model_dict.keys():
            pass
            #logger.info(k)

        # Filter based on actual parameter names
        pretrained_dict = {k: v for k, v in new_state_dict.items() if ('VFE' in k or 'IFE' in k)}

        # Filter parameters that exist in current model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Print parameter names to be loaded
        logger.info("Encoder parameters to be loaded:")
        for k in pretrained_dict.keys():
            logger.info(k)

        # Update model state dict
        model_dict.update(pretrained_dict)
        self.net_g.load_state_dict(model_dict)

        # Log number of successfully loaded parameters
        num_loaded_params = len(pretrained_dict)
        if num_loaded_params == 0:
            logger.warning('No encoder parameters were loaded. Please check the parameter names and loading conditions.')
        else:
            logger.info(f'Successfully loaded {num_loaded_params} encoder parameters from the pretrained model.')

        # Optional: freeze encoder weights
        #if self.opt['train'].get('freeze_encoder', False):
            for name, param in self.net_g.named_parameters():
                if 'VFE' in name or 'IFE' in name:
                    param.requires_grad = False
            logger.info('Encoder weights have been frozen.')



    def init_training_settings(self):
        """Initialize training settings including loss functions, optimizers and learning rate schedulers."""
        train_opt = self.opt['train']
        # Define SSIM loss function
        self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device) if train_opt.get('ssim_opt') else None

        # Define loss functions
        self.cri_content_mask = build_loss(train_opt['content_mask_opt']).to(self.device) if train_opt.get('content_mask_opt') else None
        #self.cri_align = build_loss(train_opt['align_opt']).to(self.device) if train_opt.get('align_opt') else None
        self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device) if train_opt.get('edge_opt') else None

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
        self.MASK = data.get('MASK', None).to(self.device) if 'MASK' in data else None
        
    def optimize_parameters(self, current_iter):
        """Optimize network parameters, execute one training step per iteration."""
        self.optimizer_g.zero_grad()

        # Forward pass, get network output
        self.data_fusion = self.net_g(self.VIS, self.IR)

        # Calculate losses
        loss_content_mask = self.cri_content_mask(self.IR, self.VIS, self.data_fusion, self.MASK) if self.cri_content_mask else 0
        #loss_align = self.cri_align(self.IR, self.VIS, self.data_fusion) if self.cri_align else 0
        loss_edge = self.cri_edge(self.IR, self.VIS, self.data_fusion) if self.cri_edge else 0
        loss_ssim = self.cri_ssim(self.IR, self.VIS, self.data_fusion) if self.cri_ssim else 0


        # Total loss is the sum of all losses
        total_loss = loss_content_mask + loss_edge + loss_ssim

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer_g.step()

        # Record losses
        self.log_dict = {
            'loss_content_mask': loss_content_mask.item() if loss_content_mask else 0,
            #'loss_align': loss_align.item() if loss_align else 0,
            'loss_edge': loss_edge.item() if loss_edge else 0,
            'loss_ssim': loss_ssim.item() if loss_ssim else 0,
            'total_loss': total_loss.item()
        }

    def test(self):
        """Test phase."""
        self.net_g.eval()
        with torch.no_grad():
            self.data_fusion = self.net_g(self.VIS, self.IR)
        self.net_g.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Distributed validation. Call non-distributed validation function if main process (rank=0)"""
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Non-distributed validation"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics')
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['IR_path'][0]))[0]
            self.feed_data(val_data)  # Input data
            self.test()  # Forward pass to get fused image
            visuals = self.get_current_visuals()
            
            # Get current results and post-process to get RGB image
            fusion_img_rgb = self.postprocess_fusion(val_data)

            # Save image
            if save_img:
                self.save_fused_image(fusion_img_rgb, img_name, current_iter, dataset_name)
            # Calculate and accumulate metrics
            if with_metrics:
                metric_data = self.process_visuals_for_metrics(visuals)
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        # Calculate and record average metrics
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def process_visuals_for_metrics(self, visuals):
        """
        Process visual data for metric calculation.
        Extract fused grayscale image, VIS, IR grayscale images from visual results and process them.
        """
        # Convert network output results to image format
        VIS_img = tensor2img(self.VIS.detach().cpu())  # Visible image
        IR_img = tensor2img([visuals['IR']])  # Infrared image
        fusion_img = tensor2img([visuals['result']])  # Fused image
        fusion_img = (fusion_img - np.min(fusion_img)) / (np.max(fusion_img) - np.min(fusion_img))  # Normalization
        fusion_img, VIS_img, IR_img = fusion_img * 255, VIS_img * 255, IR_img * 255  # Scale to 0-255 range

        # Return fused image, visible image, infrared image needed for metric calculation
        metric_data = {
            'img_fusion': fusion_img.squeeze(0),  # Fused image
            'img_A': VIS_img.squeeze(0),  # Visible image
            'img_B': IR_img.squeeze(0)  # Infrared image
        }

        return metric_data


    def postprocess_fusion(self, val_data):
        """Process fused image to RGB image"""
        # Get fused result
        fusion_img = tensor2img(self.get_current_visuals()['result'])
        fusion_img = (fusion_img - np.min(fusion_img)) / (np.max(fusion_img) - np.min(fusion_img))  # Normalize to 0-1
        fusion_img = (fusion_img * 255).astype(np.uint8)  # Convert to uint8 type

        # Get visible image path and read
        vis_path = val_data['VIS_path'][0]
        img_vi = cv2.imread(vis_path, flags=cv2.IMREAD_COLOR)  # Load visible image
        vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)

        # Get Cb and Cr channels of visible image
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]

        # Combine fused image with Cb and Cr channels to generate RGB image
        fused_ycbcr = np.stack([fusion_img.squeeze(0), vi_cb, vi_cr], axis=2).astype(np.uint8)
        fused_rgb = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)

        return fused_rgb

    def save_fused_image(self, fusion_img_rgb, img_name, current_iter, dataset_name):
        """Save fused image"""
        save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
        imwrite(save_path, fusion_img_rgb)  # Corrected parameter order: path first, then image data

    def log_validation_metrics(self, current_iter, dataset_name, tb_logger):
        """Calculate and record validation metrics"""
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            self.metric_results[metric] /= (current_iter + 1)
            log_str += f'\t # {metric}: {self.metric_results[metric]:.4f}\n'
            
            # Record to Tensorboard
            if tb_logger:
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', self.metric_results[metric], current_iter)

        logger = get_root_logger()
        logger.info(log_str)


    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load pretrained model weights."""
        logger = get_root_logger()
        logger.info(f'Loading model for {net.__class__.__name__} [{param_key}] from {load_path}.')
        state_dict = torch.load(load_path, map_location=self.device)
        if param_key is not None:
            state_dict = state_dict[param_key]
        net.load_state_dict(state_dict, strict=strict)

    def save(self, epoch, current_iter):
        """Save model and current training state."""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        
    def get_current_visuals(self):
        out_dict = OrderedDict()
        #out_dict['VIS'] = self.VIS.detach().cpu()
        out_dict['VIS'] = (self.VIS.detach().cpu())
        out_dict['IR'] = self.IR.detach().cpu()
        # 使用hasattr函数检查类实例中是否有这些属性，并进行赋值
        if hasattr(self, 'CBCR'):
            out_dict['CBCR'] = self.CBCR.detach().cpu().squeeze(0).numpy()
        if hasattr(self, 'data_fusion'):
            out_dict['result'] = self.data_fusion.detach().cpu()

        return out_dict
    
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            best_metric_value = self.best_metric_results[dataset_name][metric]["val"]
            best_metric_iter = self.best_metric_results[dataset_name][metric]["iter"]
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
            if (value==best_metric_value) and (self.is_train==True):
                print(f'Saving best %s models and training states.' % metric)
                self.save_best(metric, best_metric_iter)

        logger = get_root_logger()
        #logger.info('mIou: {:.4f}, Acc: {:.4f}\n'.format(self.mIoU, self.Acc))
        if not self.is_train:
            print(log_str) # For some reason, print is required during test phase
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

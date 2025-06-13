import torch
from torch import nn as nn
from torch.nn import functional as F

from archs.vgg_arch import VGGFeatureExtractor
from utils.registry import LOSS_REGISTRY
import kornia
import kornia.losses
import torch.nn.functional as F
from .loss_util import weighted_loss
import numpy as np
import math
from math import exp

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')



@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        #return torch.abs(sobelx) + torch.abs(sobely)
        return torch.abs(sobelx), torch.abs(sobely)

@LOSS_REGISTRY.register()
class SobelLoss1(nn.Module):
    def __init__(self, reduction='mean',loss_weight=1.0):
        super(SobelLoss1, self).__init__()
        self.sobelconv = Sobelxy()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, image_1, image_2, generate_img):
        grad_1 = self.sobelconv(image_1)
        grad_2 = self.sobelconv(image_2)
        generate_img_grad = self.sobelconv(generate_img)

        # Taking the element-wise maximum of the gradients of the two input images
        x_grad_joint = torch.max(grad_1, grad_2)

        # Computing the L1 loss between the maximum gradient and the gradient of the generated image
        loss = F.l1_loss(x_grad_joint, generate_img_grad,reduction=self.reduction)

        return loss * self.loss_weight
    
@LOSS_REGISTRY.register()
class SobelLoss2(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SobelLoss2, self).__init__()
        self.sobel_conv = Sobelxy()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, img_vi, img_ir, img_fu):
        vi_grad_x, vi_grad_y = self.sobel_conv(img_vi)
        ir_grad_x, ir_grad_y = self.sobel_conv(img_ir)
        fu_grad_x, fu_grad_y = self.sobel_conv(img_fu)
        
        grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
        grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
        
        loss = F.l1_loss(grad_joint_x, fu_grad_x, reduction=self.reduction) + \
               F.l1_loss(grad_joint_y, fu_grad_y, reduction=self.reduction)
        
        return loss * self.loss_weight



@LOSS_REGISTRY.register()
class SSIMLoss2(nn.Module):
    """
    SSIM loss class based on kornia's implementation.

    Args:
        window_size (int): The size of the window to compute SSIM. Default: 11.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, window_size=11, reduction='mean', loss_weight=1.0):
        super(SSIMLoss2, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ssim_loss_fn = kornia.losses.SSIMLoss(window_size, reduction=reduction)

    def forward(self, img1, img2, fusion_img):
        """
        Args:
            img1 (torch.Tensor): The first image tensor.
            img2 (torch.Tensor): The second image tensor.
            fusion_img (torch.Tensor): The fused image tensor.

        Returns:
            torch.Tensor: The combined SSIM loss between the images and the fused image.
        """
        loss1 = self.ssim_loss_fn(img1, fusion_img)
        loss2 = self.ssim_loss_fn(img2, fusion_img)
        combined_loss = (loss1 + loss2) / 2
        return combined_loss * self.loss_weight

@LOSS_REGISTRY.register()
class NCCLoss1(nn.Module):
    """
    Local (over window) normalized cross correlation loss for 2D images.
    """
    def __init__(self, win_size=9, reduction='mean', loss_weight=1.0):
        super(NCCLoss1, self).__init__()
        self.win_size = win_size
        self.sum_filt = torch.ones(1, 1, win_size, win_size).cuda()  # filter for sum operation
        self.padding = win_size // 2
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, y_true, y_pred):
        I = y_true
        J = y_pred

        # Compute squared and product terms
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # Compute local sums via convolution
        I_sum = F.conv2d(I, self.sum_filt, padding=self.padding)
        J_sum = F.conv2d(J, self.sum_filt, padding=self.padding)
        I2_sum = F.conv2d(I2, self.sum_filt, padding=self.padding)
        J2_sum = F.conv2d(J2, self.sum_filt, padding=self.padding)
        IJ_sum = F.conv2d(IJ, self.sum_filt, padding=self.padding)

        # Compute local means
        win_size = self.win_size * self.win_size
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # Compute cross-correlation term
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size

        # Compute variances
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Compute normalized cross-correlation
        cc = cross * cross / (I_var * J_var + 1e-5)

        # Modified loss computation to make it positive
        loss = 1 - cc

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class NCCLoss2(nn.Module):
    """
    Local (over window) normalized cross correlation loss for 2D images.
    Modified to compute the loss for two pairs of images and return their sum.
    """
    def __init__(self, win_size=9, reduction='mean', loss_weight=1.0):
        super(NCCLoss2, self).__init__()
        self.win_size = win_size
        self.sum_filt = torch.ones(1, 1, win_size, win_size).cuda()  # filter for sum operation
        self.padding = win_size // 2
        self.reduction = reduction
        self.loss_weight = loss_weight

    def compute_loss(self, I, J):
        # Compute squared and product terms
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # Compute local sums via convolution
        I_sum = F.conv2d(I, self.sum_filt, padding=self.padding)
        J_sum = F.conv2d(J, self.sum_filt, padding=self.padding)
        I2_sum = F.conv2d(I2, self.sum_filt, padding=self.padding)
        J2_sum = F.conv2d(J2, self.sum_filt, padding=self.padding)
        IJ_sum = F.conv2d(IJ, self.sum_filt, padding=self.padding)

        # Compute local means
        win_size = self.win_size * self.win_size
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # Compute cross-correlation term
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size

        # Compute variances
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Compute normalized cross-correlation
        cc = cross * cross / (I_var * J_var + 1e-5)

        # Modified loss computation to make it positive
        return 1 - cc

    def forward(self, img1, img2, fusion_img):
        loss_img1_fusion = self.compute_loss(img1, fusion_img)
        loss_img2_fusion = self.compute_loss(img2, fusion_img)

        combined_loss = (loss_img1_fusion + loss_img2_fusion) / 2

        if self.reduction == 'mean':
            loss = torch.mean(combined_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(combined_loss)
        else:
            loss = combined_loss

        return loss * self.loss_weight

    
# Moving the helper functions outside the class:

def gaussian(window_size, sigma):
    """Generate a 1D Gaussian tensor based on the given parameters."""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    """Generate a 2D Gaussian window based on the given parameters."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def mse(img1, img2, window_size=9):
    """Compute the local mean squared error between two images."""
    padd = window_size // 2
    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2
    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)
    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))

    return res

def std(img, window_size=9):
    """Compute the local standard deviation of an image."""
    padd = window_size // 2
    (_, channel, _, _) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

        
@LOSS_REGISTRY.register()
class MaskLoss(nn.Module):
    """
    Computes the mask-based MSE loss based on the given logic.

    Args:
        window_size (int): The size of the window to compute local operations. Default: 9.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, window_size=9, reduction='mean', loss_weight=1.0):
        super(MaskLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, img_ir, img_vis, img_fuse, mask=None):
        # 计算最大值
        max_val = torch.max(img_ir, img_vis)
        # 计算最大值损失，忽略通道维度，保持与max_mask维度一致
        max_loss = F.mse_loss(img_fuse, max_val, reduction='none')
        avg_loss = (F.mse_loss(img_fuse, img_ir, reduction='none') + F.mse_loss(img_fuse, img_vis, reduction='none')) / 2
        
        max_mask = mask
        avg_mask = 1 - mask
        
        res = max_mask * max_loss + avg_mask * avg_loss
        #print(res.shape)
        
        if self.reduction == 'mean':
            return (res.mean() * self.loss_weight)
        elif self.reduction == 'sum':
            return (res.sum() * self.loss_weight)
        else:
            return res * self.loss_weight
        
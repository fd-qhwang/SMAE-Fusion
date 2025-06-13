import torch
import torchvision.transforms.functional as TF
import random

def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor to [0, 1] range using min-max normalization.
    Args:
        tensor (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Normalized tensor
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val) if max_val > min_val else tensor

def bgr_to_ycrcb(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a BGR image tensor to YCrCb.
    Args:
        image (torch.Tensor): Tensor of shape (C, H, W) with values in range [0, 1].
    Returns:
        torch.Tensor: Converted YCrCb image.
    """
    b, g, r = image[0], image[1], image[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return torch.stack([y, cr, cb], dim=0)

def random_flip(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply random horizontal and vertical flip to a given tensor.
    Args:
        tensor (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Flipped tensor
    """
    if torch.rand(1) > 0.5:
        tensor = TF.hflip(tensor)
    if torch.rand(1) > 0.5:
        tensor = TF.vflip(tensor)
    return tensor

def random_rot(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply random rotation (90, 180, or 270 degrees) to a given tensor.
    Args:
        tensor (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Rotated tensor
    """
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    return TF.rotate(tensor, angle)

def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor to grayscale Y channel.
    Args:
        img (torch.Tensor): RGB image tensor of shape (3, H, W).
    Returns:
        torch.Tensor: Y channel tensor of shape (1, H, W).
    """
    y = img[0:1, :, :] * 0.299 + img[1:2, :, :] * 0.587 + img[2:3, :, :] * 0.114
    return y 
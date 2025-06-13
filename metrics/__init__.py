from copy import deepcopy

from utils.registry import METRIC_REGISTRY
#from .psnr_ssim import calculate_psnr, calculate_ssim
from .fusion import calculate_entropy

#__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']
__all__ = ['calculate_entropy']

def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric


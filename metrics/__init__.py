from copy import deepcopy

from utils.registry import METRIC_REGISTRY
#from .psnr_ssim import calculate_psnr, calculate_ssim
from .fusion import calculate_entropy
from .seg import calculate_miou

#__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']
__all__ = ['calculate_entropy']

'''
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
'''
def calculate_metric(data, opt):
    """Calculate metric from data and options."""
    opt = deepcopy(opt)

    metric_type = opt.pop('type')

    if metric_type == "calculate_miou" or metric_type == "calculate_iou":
        seg_pred = data.get('seg_pred')
        seg_label = data.get('seg_label')
        if seg_pred is None or seg_label is None:
            raise ValueError("For mIoU calculation, seg_pred and seg_label should be in data.")
        return calculate_miou(seg_pred, seg_label, **opt)
    else:
        # Use a shallow copy for computation without modifying the original data
        data_copy = data.copy()
        data_copy.pop('seg_pred', None)
        data_copy.pop('seg_label', None)
        return METRIC_REGISTRY.get(metric_type)(**data_copy, **opt)


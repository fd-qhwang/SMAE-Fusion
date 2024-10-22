import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
from natsort import natsorted
import random
from utils.registry import DATASET_REGISTRY

# Helper functions
def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
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
    """
    if torch.rand(1) > 0.5:
        tensor = TF.hflip(tensor)
    if torch.rand(1) > 0.5:
        tensor = TF.vflip(tensor)
    return tensor

def random_rot(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply random rotation (90, 180, or 270 degrees) to a given tensor.
    """
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    return TF.rotate(tensor, angle)

@DATASET_REGISTRY.register()
class MSHATDataset(Dataset):
    def __init__(self, opt: dict):
        super(MSHATDataset, self).__init__()
        self.opt = opt
        self.is_train = self.opt['is_train']
        self.img_size = int(self.opt.get('img_size', 128))
        self.is_RGB = self.opt.get('is_RGB', False)
        
        # Dataset paths
        self.vis_folder = self._check_path('dataroot_source2')
        self.ir_folder = self._check_path('dataroot_source1')

        # Optional paths for training
        self.mask_folder = self._check_path('mask_path')
        self.ir_map_folder = self._check_path('ir_map_path')
        self.vis_map_folder = self._check_path('vis_map_path')

        # Image lists
        self.ir_list = natsorted(os.listdir(self.ir_folder))

        # Data augmentation
        self.crop = torchvision.transforms.RandomCrop(self.img_size)

    def __getitem__(self, index: int):
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)

        vis = self._imread(vis_path, is_visible=True) # (3,h,w)
        ir = self._imread(ir_path, is_visible=False) # (1,h,w)
        
        mask = ir_map = vis_map = None
        vis_channels = vis.shape[0]
        if self.is_train:
            mask = self._load_optional(self.mask_folder, image_name, vis.shape[1:])
            ir_map = self._load_optional(self.ir_map_folder, image_name, vis.shape[1:])
            vis_map = self._load_optional(self.vis_map_folder, image_name, vis.shape[1:])

            combined = torch.cat((vis, ir, mask, ir_map, vis_map), dim=0)

            # 只有在图像尺寸小于目标尺寸时才进行resize
            if combined.shape[-1] <= self.img_size or combined.shape[-2] <= self.img_size:
                combined = TF.resize(combined, [self.img_size, self.img_size])

            combined = random_flip(combined)  # 随机翻转
            combined = random_rot(combined)  # 随机旋转
            patch = self.crop(combined)
        
            # 使用基于通道数的拆分
            split_sizes = [vis_channels, 1, 1, 1, 1]  # 根据是否为RGB或灰度图像调整
            vis, ir, mask, ir_map, vis_map = torch.split(patch, split_sizes, dim=0)

            mask = mask.long()

            if not self.is_RGB:
                vis = self.rgb_to_y(vis)
            return {
                'VIS': vis,
                'IR': ir,
                'MASK': mask,
                'IR_MAP': ir_map,
                'VIS_MAP': vis_map,
                'IR_path': ir_path,
                'VIS_path': vis_path,
            }
        else:
            vis = self.rgb_to_y(vis)

            data = {
                'IR': ir,
                'VIS': vis,
                'IR_path': ir_path,
                'VIS_path': vis_path,
            }

            if self.is_RGB:
                cbcr = bgr_to_ycrcb(vis)
                data['CBCR'] = cbcr
                print(cbcr.shape)

            return data

    def __len__(self):
        return len(self.ir_list)

    def _check_path(self, key: str):
        """
        Check if the provided path exists in the configuration and is valid.
        If the path is not set or is invalid, return None.
        """
        path = self.opt.get(key)
        return path if path and path not in [None, '~', ''] else None

    @staticmethod
    def _imread(path: str, is_visible: bool) -> torch.Tensor:
        img = Image.open(path).convert('RGB' if is_visible else 'L')
        img_ts = TF.to_tensor(img)
        return min_max_normalize(img_ts)

    def _load_optional(self, folder: str, image_name: str, shape=None) -> torch.Tensor:
        if folder:
            path = os.path.join(folder, image_name)
            return self._imread(path, is_visible=False)
        else:
            return torch.zeros(1, *shape) if shape else None

    @staticmethod
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

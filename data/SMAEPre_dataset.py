import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF
from natsort import natsorted
from utils.registry import DATASET_REGISTRY
from .utils import min_max_normalize, bgr_to_ycrcb, random_flip, random_rot, rgb_to_y

@DATASET_REGISTRY.register()
class SMAEPreDataset(Dataset):
    def __init__(self, opt: dict):
        super(SMAEPreDataset, self).__init__()
        self.opt = opt
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

        mask = self._load_optional(self.mask_folder, image_name, vis.shape[1:])
        ir_map = self._load_optional(self.ir_map_folder, image_name, vis.shape[1:])
        vis_map = self._load_optional(self.vis_map_folder, image_name, vis.shape[1:])

        combined = torch.cat((vis, ir, mask, ir_map, vis_map), dim=0)

        if combined.shape[-1] <= self.img_size or combined.shape[-2] <= self.img_size:
            combined = TF.resize(combined, [self.img_size, self.img_size])

        combined = random_flip(combined)  # 随机翻转
        combined = random_rot(combined)  # 随机旋转
        patch = self.crop(combined)
    
        split_sizes = [vis_channels, 1, 1, 1, 1]  
        vis, ir, mask, ir_map, vis_map = torch.split(patch, split_sizes, dim=0)

        mask = mask.long()

        if not self.is_RGB:
            vis = rgb_to_y(vis)
        return {
            'VIS': vis,
            'IR': ir,
            'MASK': mask,
            'IR_MAP': ir_map,
            'VIS_MAP': vis_map,
            'IR_path': ir_path,
            'VIS_path': vis_path,
        }

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
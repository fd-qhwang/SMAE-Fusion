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
    b, g, r = image[0], image[1], image[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return torch.stack([y, cr, cb], dim=0)

@DATASET_REGISTRY.register()
class SMAEValDataset(Dataset):
    def __init__(self, opt: dict):
        super(SMAEValDataset, self).__init__()
        self.opt = opt
        self.img_size = int(self.opt.get('img_size', 128))
        self.is_RGB = self.opt.get('is_RGB', False)
        self.is_train = False  # 明确表示这是验证/测试集

        # Dataset paths
        self.vis_folder = self._check_path('dataroot_source2')
        self.ir_folder = self._check_path('dataroot_source1')

        # Image lists
        self.ir_list = natsorted(os.listdir(self.ir_folder))

    def __getitem__(self, index: int):
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)

        vis = self._imread(vis_path, is_visible=True)  # (C, H, W)
        ir = self._imread(ir_path, is_visible=False)   # (1, H, W)

        if not self.is_RGB and vis.shape[0] == 3:
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

        return data

    def __len__(self):
        return len(self.ir_list)

    def _check_path(self, key: str):
        path = self.opt.get(key)
        return path if path and path not in [None, '~', ''] else None

    @staticmethod
    def _imread(path: str, is_visible: bool) -> torch.Tensor:
        img = Image.open(path).convert('RGB' if is_visible else 'L')
        img_ts = TF.to_tensor(img)
        return min_max_normalize(img_ts)

    @staticmethod
    def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
        y = img[0:1, :, :] * 0.299 + img[1:2, :, :] * 0.587 + img[2:3, :, :] * 0.114
        return y
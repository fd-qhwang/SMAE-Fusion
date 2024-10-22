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

def random_flip(tensor: torch.Tensor) -> torch.Tensor:
    if torch.rand(1) > 0.5:
        tensor = TF.hflip(tensor)
    if torch.rand(1) > 0.5:
        tensor = TF.vflip(tensor)
    return tensor

def random_rot(tensor: torch.Tensor) -> torch.Tensor:
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    return TF.rotate(tensor, angle)

@DATASET_REGISTRY.register()
class SMAETrainDataset(Dataset):
    def __init__(self, opt: dict):
        super(SMAETrainDataset, self).__init__()
        self.opt = opt
        self.img_size = int(self.opt.get('img_size', 128))
        self.is_RGB = self.opt.get('is_RGB', False)
        self.stride = self.opt.get('stride', 64)
        self.is_train = True  # 明确表示这是训练集

        # Dataset paths
        self.vis_folder = self._check_path('dataroot_source2')
        self.ir_folder = self._check_path('dataroot_source1')

        # Optional paths for training
        self.mask_folder = self._check_path('mask_path')
        self.ir_map_folder = self._check_path('ir_map_path')
        self.vis_map_folder = self._check_path('vis_map_path')

        # Image lists
        self.ir_list = natsorted(os.listdir(self.ir_folder))

        # Load images and pre-process
        self.vis_images = []
        self.ir_images = []
        self.mask_images = []
        self.ir_map_images = []
        self.vis_map_images = []

        for image_name in self.ir_list:
            vis_path = os.path.join(self.vis_folder, image_name)
            ir_path = os.path.join(self.ir_folder, image_name)

            vis = self._imread(vis_path, is_visible=True)  # (C, H, W)
            ir = self._imread(ir_path, is_visible=False)   # (1, H, W)

            self.vis_images.append(vis)
            self.ir_images.append(ir)

            if self.mask_folder:
                mask = self._load_optional(self.mask_folder, image_name, vis.shape[1:])
                self.mask_images.append(mask)
            else:
                self.mask_images.append(torch.zeros(1, *vis.shape[1:]))

            if self.ir_map_folder:
                ir_map = self._load_optional(self.ir_map_folder, image_name, vis.shape[1:])
                self.ir_map_images.append(ir_map)
            else:
                self.ir_map_images.append(torch.zeros(1, *vis.shape[1:]))

            if self.vis_map_folder:
                vis_map = self._load_optional(self.vis_map_folder, image_name, vis.shape[1:])
                self.vis_map_images.append(vis_map)
            else:
                self.vis_map_images.append(torch.zeros(1, *vis.shape[1:]))

        # Calculate the number of patches per image
        _, H, W = self.vis_images[0].shape
        self.patch_per_line = (W - self.img_size) // self.stride + 1
        self.patch_per_column = (H - self.img_size) // self.stride + 1
        self.patches_per_image = self.patch_per_line * self.patch_per_column
        self.total_patches = self.patches_per_image * len(self.ir_list)

    def __getitem__(self, idx: int):
        # Determine which image and which patch
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        h_idx = patch_idx // self.patch_per_line
        w_idx = patch_idx % self.patch_per_line

        # Get the images
        vis = self.vis_images[img_idx]
        ir = self.ir_images[img_idx]
        mask = self.mask_images[img_idx]
        ir_map = self.ir_map_images[img_idx]
        vis_map = self.vis_map_images[img_idx]

        # Crop the patch
        top = h_idx * self.stride
        left = w_idx * self.stride

        vis_patch = vis[:, top:top + self.img_size, left:left + self.img_size]
        ir_patch = ir[:, top:top + self.img_size, left:left + self.img_size]
        mask_patch = mask[:, top:top + self.img_size, left:left + self.img_size]
        ir_map_patch = ir_map[:, top:top + self.img_size, left:left + self.img_size]
        vis_map_patch = vis_map[:, top:top + self.img_size, left:left + self.img_size]

        # Data augmentation
        combined = torch.cat((vis_patch, ir_patch, mask_patch, ir_map_patch, vis_map_patch), dim=0)
        combined = random_flip(combined)
        combined = random_rot(combined)

        # Split back into individual components
        # 使用从数据中获取的通道数
        vis_channels = vis.shape[0]  # 从 vis 图像获取通道数
        split_sizes = [vis_channels, 1, 1, 1, 1]
        vis_patch, ir_patch, mask_patch, ir_map_patch, vis_map_patch = torch.split(combined, split_sizes, dim=0)

        mask_patch = mask_patch.long()

        if not self.is_RGB and vis_channels == 3:
            vis_patch = self.rgb_to_y(vis_patch)

        return {
            'VIS': vis_patch,
            'IR': ir_patch,
            'MASK': mask_patch,
            'IR_MAP': ir_map_patch,
            'VIS_MAP': vis_map_patch,
            'Index': idx
        }

    def __len__(self):
        return self.total_patches

    def _check_path(self, key: str):
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
        y = img[0:1, :, :] * 0.299 + img[1:2, :, :] * 0.587 + img[2:3, :, :] * 0.114
        return y
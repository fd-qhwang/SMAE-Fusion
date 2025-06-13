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
class SMAEFtDataset(Dataset):
    def __init__(self, opt: dict):
        super(SMAEFtDataset, self).__init__()
        self.opt = opt
        self.img_size = int(self.opt.get('img_size', 128))
        self.is_RGB = self.opt.get('is_RGB', False)
        self.is_train = self.opt.get('is_train', True)  # 默认为训练模式
        self.stride = self.opt.get('stride', 64) if self.is_train else self.img_size  # 训练时使用stride，验证时使用完整图像

        # Dataset paths
        self.vis_folder = self._check_path('dataroot_source2')
        self.ir_folder = self._check_path('dataroot_source1')

        # Optional paths for training
        self.mask_folder = self._check_path('mask_path')
        self.ir_map_folder = self._check_path('ir_map_path')
        self.vis_map_folder = self._check_path('vis_map_path')

        # Image lists
        self.ir_list = natsorted(os.listdir(self.ir_folder))

        if self.is_train:
            # 训练模式：预加载所有图像
            self.vis_images = []
            self.ir_images = []
            self.mask_images = []
            self.ir_map_images = []
            self.vis_map_images = []

            for image_name in self.ir_list:
                vis_path = os.path.join(self.vis_folder, image_name)
                ir_path = os.path.join(self.ir_folder, image_name)

                vis = self._imread(vis_path, is_visible=True)
                ir = self._imread(ir_path, is_visible=False)

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

            # 计算训练时的patch数量
            _, H, W = self.vis_images[0].shape
            self.patch_per_line = (W - self.img_size) // self.stride + 1
            self.patch_per_column = (H - self.img_size) // self.stride + 1
            self.patches_per_image = self.patch_per_line * self.patch_per_column
            self.total_patches = self.patches_per_image * len(self.ir_list)

    def __getitem__(self, idx: int):
        if self.is_train:
            # 训练模式：返回图像块
            img_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
            h_idx = patch_idx // self.patch_per_line
            w_idx = patch_idx % self.patch_per_line

            vis = self.vis_images[img_idx]
            ir = self.ir_images[img_idx]
            mask = self.mask_images[img_idx]
            ir_map = self.ir_map_images[img_idx]
            vis_map = self.vis_map_images[img_idx]

            top = h_idx * self.stride
            left = w_idx * self.stride

            vis_patch = vis[:, top:top + self.img_size, left:left + self.img_size]
            ir_patch = ir[:, top:top + self.img_size, left:left + self.img_size]
            mask_patch = mask[:, top:top + self.img_size, left:left + self.img_size]
            ir_map_patch = ir_map[:, top:top + self.img_size, left:left + self.img_size]
            vis_map_patch = vis_map[:, top:top + self.img_size, left:left + self.img_size]

            # 数据增强
            combined = torch.cat((vis_patch, ir_patch, mask_patch, ir_map_patch, vis_map_patch), dim=0)
            combined = random_flip(combined)
            combined = random_rot(combined)

            # 分离回各个组件
            vis_channels = vis.shape[0]
            split_sizes = [vis_channels, 1, 1, 1, 1]
            vis_patch, ir_patch, mask_patch, ir_map_patch, vis_map_patch = torch.split(combined, split_sizes, dim=0)

            mask_patch = mask_patch.long()

            if not self.is_RGB and vis_channels == 3:
                vis_patch = rgb_to_y(vis_patch)

            return {
                'VIS': vis_patch,
                'IR': ir_patch,
                'MASK': mask_patch,
                'IR_MAP': ir_map_patch,
                'VIS_MAP': vis_map_patch,
                'Index': idx
            }
        else:
            # 验证模式：返回完整图像
            image_name = self.ir_list[idx]
            vis_path = os.path.join(self.vis_folder, image_name)
            ir_path = os.path.join(self.ir_folder, image_name)

            vis = self._imread(vis_path, is_visible=True)
            ir = self._imread(ir_path, is_visible=False)

            if not self.is_RGB and vis.shape[0] == 3:
                vis = rgb_to_y(vis)

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
        if self.is_train:
            return self.total_patches
        else:
            return len(self.ir_list)

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
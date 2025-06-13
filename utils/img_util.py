import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor):
    """Convert a tensor or a list of 4D tensors into a numpy array or a list of 2D/3D numpy arrays.

    Args:
        tensor (Tensor or list of Tensors): A tensor or a list containing tensors, where each tensor is of shape (1 x C x H x W).
                                            C can be 1 (for grayscale) or 3 (for RGB).

    Returns:
        ndarray or list of ndarray: A 2D/3D array of shape (H x W) or (C x H x W) and type float32, or a list of such 2D/3D arrays.
    """

    if torch.is_tensor(tensor):
        tensor = [tensor]

    if not isinstance(tensor, list) or not all(torch.is_tensor(t) for t in tensor):
        raise TypeError('A tensor or a list of tensors is expected.')

    result = []

    for t in tensor:
        # Check tensor dimensions
        if t.dim() != 4 or t.shape[0] != 1 or (t.shape[1] != 1 and t.shape[1] != 3):
            raise ValueError(f'Expected tensor of shape (1, C, H, W) where C can be 1 or 3, but got {t.shape}')

        # Convert tensor to numpy float32
        img_np = t.squeeze(0).cpu().numpy().astype(np.float32)
        '''
        # If it's a grayscale image, further squeeze the channel dimension
        if img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)
        '''
        result.append(img_np)

    if len(result) == 1:
        result = result[0]

    return result

def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(file_path, img, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]




"""
Define task metrics, loss functions and model trainer here.
"""
def randrot(img):
    mode = np.random.randint(0, 4)  # 0, 1, 2, 3
    return rot(img, mode)

def randfilp(img):
    mode = np.random.randint(0, 3)  # 0, 1, 2
    return flip(img, mode)

def rot(img, rot_mode):
    if rot_mode == 1:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 3:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:  # 原始图像，不进行翻转
        pass
    elif flip_mode == 1:
        img = img.flip(-2)  # 水平翻转
    elif flip_mode == 2:
        img = img.flip(-1)  # 垂直翻转
    return img


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

def RGB2GRAY(rgb_image):
    """
    将RGB格式转换为GRAY格式。
    :param rgb_image: RGB格式的图像数据, 其shape为[B, C, H, W], 其中C=3。
    :return: 灰度图像
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]

    # 使用上述公式计算灰度值
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    # 限制值在[0,1]范围内
    gray = gray.clamp(0.0, 1.0)

    return gray


def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    person = [64, 64, 0]
    bike = [0, 128, 192]
    curve = [0, 0, 192]
    car_stop = [128, 128, 0]
    guardrail = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump = [192, 64, 0]
    palette = np.array(
        [
            unlabelled,
            car,
            person,
            bike,
            curve,
            car_stop,
            guardrail,
            color_cone,
            bump,
        ]
    )
    return palette

def seg_visualize(predictions, save_name):
    palette = get_palette()
    pred = predictions[0].data.cpu().numpy()
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(1, int(predictions.max())):
        img[pred == cid] = palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(save_name)

def RGB2YCrCb_np(rgb_image):
    """
    Converts an RGB image to YCrCb format.
    :param rgb_image: RGB image data with shape [C, H, W]
    :return: Y, Cr, Cb
    """
    R = rgb_image[0:1, :, :]
    G = rgb_image[1:2, :, :]
    B = rgb_image[2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = np.clip(Y, 0.0, 1.0)
    Cr = np.clip(Cr, 0.0, 1.0)
    Cb = np.clip(Cb, 0.0, 1.0)
    return Y, Cr, Cb

def YCbCr2RGB_np(Y, Cr, Cb):
    """
    Converts YCrCb format to RGB format.
    :param Y, Cr, Cb: Y, Cr, Cb channels
    :return: RGB image
    """
    ycrcb = np.concatenate([Y, Cr, Cb], axis=0)
    H, W = Y.shape[1], Y.shape[2]
    im_flat = ycrcb.transpose(1, 2, 0).reshape(-1, 3)
    mat = np.array([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]])
    bias = np.array([0.0 / 255, -0.5, -0.5])
    temp = np.dot(im_flat + bias, mat.T)
    out = temp.reshape(H, W, 3).transpose(2, 0, 1)
    out = np.clip(out, 0, 1.0)
    return out

def RGB2GRAY_np(rgb_image):
    """
    Converts an RGB image to grayscale format.
    :param rgb_image: RGB image data with shape [C, H, W]
    :return: Grayscale image
    """
    R = rgb_image[0:1, :, :]
    G = rgb_image[1:2, :, :]
    B = rgb_image[2:3, :, :]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    gray = np.clip(gray, 0.0, 1.0)
    return gray




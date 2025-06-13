import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import sobel

from utils.registry import METRIC_REGISTRY
import warnings

def minmax_img(img):
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    return img

def check_img(img):
    img = minmax_img(img)
    img = img*255.0
    img = img.astype(np.uint8)
    return img

@METRIC_REGISTRY.register()
def calculate_entropy(img_fusion, img_A=None, img_B=None):
    """Calculate Entropy (EN) of the fused image.

    Args:
        img_fusion (ndarray): A 2D grayscale fused image or 3D RGB fused image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: Entropy of the fused image.
    """
    img_fusion = img_fusion.astype(np.uint8)
    # Compute entropy
    a = np.uint8(np.round(img_fusion)).flatten()
    h = np.bincount(a) / a.shape[0]
    entropy = -sum(h * np.log2(h + (h == 0)))

    return entropy


@METRIC_REGISTRY.register()
def calculate_average_gradient(img_fusion, img_A=None, img_B=None):
    """Calculate Average Gradient (AG) of the fused image.

    Args:
        img_fusion (ndarray): A 2D grayscale fused image or 3D RGB fused image.

    Returns:
        float: Average gradient of the fused image.
    """

    if len(img_fusion.shape) == 2:
        # If grayscale image, add an additional dimension
        img_fusion = img_fusion[:, :, np.newaxis]

    # Initialize a list to store average gradient for each channel
    avg_gradients = []

    for channel in range(img_fusion.shape[2]):
        band = img_fusion[:, :, channel]

        dzdx, dzdy = np.gradient(band)

        s = np.sqrt((dzdx**2 + dzdy**2) / 2)

        avg_gradient_channel = np.mean(s)
        avg_gradients.append(avg_gradient_channel)

    # Return the average of avg_gradients
    return np.mean(avg_gradients)



@METRIC_REGISTRY.register()
def calculate_sd(img_fusion, img_A=None, img_B=None):
    """Calculate Standard Deviation (SD) of the fused image.

    Args:
        img_fusion (ndarray): A 2D grayscale fused image or 3D RGB fused image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: Standard Deviation of the fused image.
    """
    sd_value = np.std(img_fusion)
    return sd_value

@METRIC_REGISTRY.register()
def calculate_sf(img_fusion, img_A=None, img_B=None):
    """Calculate Spatial Frequency (SF) of the fused image.

    Args:
        img_fusion (ndarray): A 2D grayscale fused image or 3D RGB fused image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: Spatial Frequency of the fused image.
    """

    # Calculate spatial frequency
    delta_x = img_fusion[:, 1:] - img_fusion[:, :-1]
    delta_y = img_fusion[1:, :] - img_fusion[:-1, :]
    sf_value = np.sqrt(np.mean(delta_x ** 2) + np.mean(delta_y ** 2))
    return sf_value

@METRIC_REGISTRY.register()
def calculate_mutual_information(img_fusion, img_A, img_B):
    """Calculate Mutual Information (MI) of the fused image with respect to the source images.

    Args:
        img_fusion (ndarray): A 2D grayscale fused image.
        img_A (ndarray): A 2D grayscale source image.
        img_B (ndarray): A 2D grayscale source image.

    Returns:
        float: Mutual Information of the fused image.
    """
    img_fusion = img_fusion.astype(np.uint8)
    img_A = img_A.astype(np.uint8)
    img_B = img_B.astype(np.uint8)
    mi_with_img_A = skm.mutual_info_score(img_fusion.flatten(), img_A.flatten())
    mi_with_img_B = skm.mutual_info_score(img_fusion.flatten(), img_B.flatten())

    return mi_with_img_A + mi_with_img_B

@METRIC_REGISTRY.register()
def calculate_scd(img_fusion, img_A, img_B):
    """
    Calculate Sum of the Correlations of Differences (SCD) between fused image and input images.

    Args:
        img_fusion (ndarray): Fused image.
        img_A (ndarray): Source image A.
        img_B (ndarray): Source image B.

    Returns:
        float: SCD score.
    """

    def correlation(x, y):
        """Compute the correlation between two images."""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
        return numerator / denominator

    imgF_A = img_fusion - img_A
    imgF_B = img_fusion - img_B

    corr1 = correlation(img_A, imgF_B)
    corr2 = correlation(img_B, imgF_A)

    return corr1 + corr2


@METRIC_REGISTRY.register()
def calculate_ssim(img_fusion, img_A=None, img_B=None):
    """
    Calculate Structural Similarity Index (SSIM) between fused image and input images.

    Args:
        img_fusion (ndarray): Fused image.
        img_A (ndarray): Source image A.
        img_B (ndarray): Source image B.

    Returns:
        float: SSIM score.
    """
    # Determine the data range based on all images
    overall_min = min(img_fusion.min(), img_A.min(), img_B.min())
    overall_max = max(img_fusion.max(), img_A.max(), img_B.max())
    data_range_value = overall_max - overall_min

    ssim_A = ssim(img_fusion, img_A, data_range=data_range_value)
    ssim_B = ssim(img_fusion, img_B, data_range=data_range_value)

    return ssim_A + ssim_B


@METRIC_REGISTRY.register()
def calculate_psnr(img_fusion, img_A=None, img_B=None, data_range=255):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between fused image and input images.

    Args:
        img_fusion (ndarray): Fused image.
        img_A (ndarray): Source image A.
        img_B (ndarray): Source image B.
        data_range (float, optional): The data range of the input image. Default is 255.

    Returns:
        float: PSNR score.
    """
    img_A = check_img(img_A)
    img_B = check_img(img_B)
    img_fusion = check_img(img_fusion)
    psnr_A = psnr(img_fusion, img_A)
    psnr_B = psnr(img_fusion, img_B)

    return psnr_A + psnr_B   # Taking average of the two PSNR values.


@METRIC_REGISTRY.register()
def calculate_vif(img_fusion, img_A, img_B):
    """Calculate Visual Information Fidelity (VIF) for fused images.

    Args:
        img_fusion (ndarray): Fused image.
        img_A (ndarray): Source image A.
        img_B (ndarray): Source image B.

    Returns:
        float: VIF score.
    """
    def compare_vif(ref, dist):
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den
        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    vif_value = compare_vif(img_A, img_fusion) + compare_vif(img_B, img_fusion)

    return vif_value

@METRIC_REGISTRY.register()
def calculate_qabf(img_fusion, img_A, img_B):
    """Calculate QABF metric for fused image."""

    # Model parameters
    L = 1; Tg = 0.9994; kg = -15; Dg = 0.5; Ta = 0.9879; ka = -22; Da = 0.8

    def get_g_a(img):
        # Sobel Operator
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        Sx = convolve2d(img, h3, mode='same')
        Sy = convolve2d(img, h1, mode='same')
        g = np.sqrt(Sx**2 + Sy**2)
        a = np.zeros_like(img)
        a[Sx == 0] = np.pi / 2
        a[Sx != 0] = np.arctan(Sy[Sx != 0] / Sx[Sx != 0])
        return g, a

    def get_qabf_values(g1, a1, gF, aF):
        M, N = g1.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G = np.where(g1 > gF, gF / g1, np.where(g1 == gF, gF, g1 / gF))
            A = 1 - np.abs(a1 - aF) / (np.pi / 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Qg = Tg / (1 + np.exp(kg * (G - Dg)))
            Qa = Ta / (1 + np.exp(ka * (A - Da)))
        return Qg * Qa

    gA, aA = get_g_a(img_A)
    gB, aB = get_g_a(img_B)
    gF, aF = get_g_a(img_fusion)

    QAF = get_qabf_values(gA, aA, gF, aF)
    QBF = get_qabf_values(gB, aB, gF, aF)

    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)

    return nume / deno



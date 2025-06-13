import os
import cv2
import kornia
import torch
import numpy as np

def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def bgr_to_ycrcb(path):
    one = cv2.imread(path,1)
    one = one.astype('float32')
    (B, G, R) = cv2.split(one)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr,Cb])

def ycrcb_to_bgr(one):
    one = one.astype('float32')
    Y, Cr, Cb = cv2.split(one)
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    return cv2.merge([B, G, R])

def calculate_weight1(map_T, map_RGB):
    # 独立归一化 map_T 和 map_RGB
    map_T_normalized = (map_T - map_T.min()) / (map_T.max() - map_T.min())
    map_RGB_normalized = (map_RGB - map_RGB.min()) / (map_RGB.max() - map_RGB.min())

    # 计算权重
    w_T = 0.5 + 0.5 * (map_T_normalized - map_RGB_normalized)
    w_RGB = 0.5 + 0.5 * (map_RGB_normalized - map_T_normalized)

    # 确保权重在 0 到 1 之间
    w_T = np.clip(w_T, 0, 1)
    w_RGB = np.clip(w_RGB, 0, 1)

    return w_T, w_RGB

def calculate_weight2(map_T, map_RGB):

    # 计算权重
    w_T = 0.5 + 0.5 * (map_T - map_RGB)
    w_RGB = 0.5 + 0.5 * (map_RGB - map_T)

    # 确保权重在 0 到 1 之间
    w_T = np.clip(w_T, 0, 1)
    w_RGB = np.clip(w_RGB, 0, 1)

    return w_T, w_RGB

def softmax(map1, map2, c):
    exp_x1 = np.exp(map1*c)
    exp_x2 = np.exp(map2*c)
    exp_sum = exp_x1 + exp_x2
    map1 = exp_x1/exp_sum
    map2 = exp_x2/exp_sum
    return map1, map2

def vsm(img):
    his = np.zeros(256, np.float64)
    for i in range(img.shape[0]): # 256
        for j in range(img.shape[1]): # 256
            his[img[i][j]] += 1
    sal = np.zeros(256, np.float64)
    for i in range(256):
        for j in range(256):
            sal[i] += np.abs(j - i) * his[j]
    map = np.zeros_like(img, np.float64)
    for i in range(256):
        map[np.where(img == i)] = sal[i]
    if map.max() == 0:
        return np.zeros_like(img, np.float64)
    return map / (map.max())


def torch_vsm(img):
    his = torch.zeros(256,  dtype=torch.float32).cuda()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i, j].item()] += 1
    sal = torch.zeros(256, dtype=torch.float32).cuda()
    for i in range(256):
        for j in range(256):
            sal[i] += abs(j - i) * his[j].item()
    map = torch.zeros_like(img, dtype=torch.float32)
    for i in range(256):
        map[torch.where(img == i)] = sal[i]
    if map.max() == 0:
        return torch.zeros_like(img, dtype=torch.float32)
    return map / (map.max())

if __name__ == '__main__':


    T_path = "./datasets/FMB/train/ir/"
    RGB_path = "./datasets/FMB/train/vi/"

    T_file_list = sorted(os.listdir(T_path))
    RGB_file_list = sorted(os.listdir(RGB_path))

    T_map_path = "./datasets/FMB/train/max_mask/"
    RGB_map_path = "./datasets/FMB/train/avg_mask/"
    Fusion_map_path = "./datasets/FMB/train/fusion_mask/"

    if not os.path.exists(T_map_path):
        os.makedirs(T_map_path)
    if not os.path.exists(RGB_map_path):
        os.makedirs(RGB_map_path)
    if not os.path.exists(Fusion_map_path):
        os.makedirs(Fusion_map_path)

    for idx, (T_filename, RGB_filename) in enumerate(zip(T_file_list, RGB_file_list)):

        T_filepath = os.path.join(T_path, T_filename)
        RGB_filepath = os.path.join(RGB_path, RGB_filename)

        img_T = cv2.imread(T_filepath, cv2.IMREAD_GRAYSCALE)  # uint8 (256, 256)
        img_VIS = cv2.imread(RGB_filepath, cv2.IMREAD_COLOR)  # uint8 (256, 256)
        img_RGB = cv2.imread(RGB_filepath, cv2.IMREAD_GRAYSCALE)  # uint8 (256, 256)
        map_T = vsm(img_T)
        map_RGB = vsm(img_RGB)

        # get cb and cr channels of the visible image
        vi_ycbcr = cv2.cvtColor(img_VIS, cv2.COLOR_BGR2YCrCb)
        vi_y  = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]
        #w_T, w_RGB = calculate_weight2(map_T, map_RGB)
        mean_map_T = np.mean(map_T)
        mean_map_RGB = np.mean(map_RGB)
        #w_T, w_RGB = calculate_weight2(saliency_map_ir, saliency_map_vi)
        is_saliency_ir = (map_T > mean_map_T)
        is_saliency_vi = (map_RGB > mean_map_RGB)
        max_mask = (is_saliency_ir | is_saliency_vi).astype(np.uint8)
        avg_mask = (1 - max_mask).astype(np.uint8)
        w_T = max_mask 
        w_RGB = avg_mask
        img_w_T = (w_T * 255).astype(np.uint8)
        img_w_RGB = (w_RGB * 255).astype(np.uint8)
        
        img_fusion = (w_T * img_T + w_RGB * img_RGB).astype(np.uint8)

        # get BGR-fused image
        fused_ycbcr = np.stack([img_fusion, vi_cb, vi_cr], axis=2).astype(np.uint8)
        fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
        

        T_save_name = os.path.join(T_map_path, T_filename)
        RGB_save_name = os.path.join(RGB_map_path, RGB_filename)
        Fusion_save_name = os.path.join(Fusion_map_path, RGB_filename)

        cv2.imwrite(T_save_name, img_w_T)
        cv2.imwrite(RGB_save_name, img_w_RGB)
        
        cv2.imwrite(Fusion_save_name, fused_bgr)



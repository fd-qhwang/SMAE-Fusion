import os
import cv2
import kornia
import torch
import numpy as np


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


###The remaining code is in progress and will be uploaded shortly###
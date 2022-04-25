import cv2
import scipy
import numpy as np
import math
from scipy import signal

img = cv2.imread(r'D:\Files\pix2pix\pix\results\FOLD_AB3_pix2pix\test_latest\images\12_real_A.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def create_window(size, sigma):
    x, y = np.mgrid[-size[0] // 2 + 1:size[0] // 2 + 1, -size[1] // 2 + 1:size[1] // 2 + 1]
    gus = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return gus / gus.sum()


def mssim(img1, img2, size=(11, 11), sigma=1.5, s=False):  # 添加函数的默认值
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    window = create_window(size, sigma)
    K1, K2 = 0.01, 0.03
    L = 255
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    ux = signal.fftconvolve(window, img1, mode='valid')
    uy = signal.fftconvolve(window, img2, mode='valid')
    ux_2 = ux ** 2
    uy_2 = uy ** 2
    ux_uy = ux * uy
    theta_x2 = signal.fftconvolve(window, img1 * img1, mode='valid') - ux_2
    theta_y2 = signal.fftconvolve(window, img2 * img2, mode='valid') - uy_2
    theta_xy = signal.fftconvolve(window, img1 * img2, mode='valid') - ux_uy
    mssim_sum = ((2 * ux_uy + C1) * (2 * theta_xy + C2)) / ((ux_2 + uy_2 + C1) * (theta_y2 + theta_x2 + C2))
    s_sum = (theta_xy + C2 * 0.5) / ((theta_y2 ** 0.5) * (theta_x2 ** 0.5) + C2 * 0.5)
    if s:
        return mssim_sum.mean(), s_sum.mean()
    # 返回S值 结构相似度
    else:
        return mssim_sum.mean()


def mssim_multichannel(img1, img2, structure=False):
    assert (img1.shape == img2.shape)

    if len(img1.shape) == 2:  # 灰度图直接计算返回
        return mssim(img1, img2)

    mssim_values = []  # 三个通道同时计算 计算平均值
    s_values = []
    for i in range(img1.shape[2]):
        mssim_value, s_value = mssim(img1[:, :, i], img2[:, :, i], s=True)
        mssim_values.append(mssim_value)
        s_values.append(s_value)
    if structure:
        return np.mean(mssim_values), np.mean(s_values)
    else:
        return np.mean(mssim_values)

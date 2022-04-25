import lpips
import torch
import cv2
import numpy as np


def transpose_img(img):
    result_img = np.zeros(img.shape, dtype=np.float32)
    cv2.normalize(img, result_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result_img = torch.tensor(result_img)
    result_img = result_img.unsqueeze(0)  # [1,h,w,3]
    result_img = result_img.transpose(1, 3)  # [1,3,w,h]
    result_img = result_img.transpose(2, 3)
    return result_img


def lpi(img0, img1, net='alex'):
    img0, img1 = transpose_img(img0), transpose_img(img1)
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    d = loss_fn_alex(img0, img1)
    return d.item()

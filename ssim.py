import cv2
import scipy
import numpy as np
import math

img=cv2.imread(r'D:\Files\pix2pix\pix\results\FOLD_AB3_pix2pix\test_latest\images\12_real_A.png')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# gaussian
def create_window(size,sigma):
    x,y=np.mgrid[-size[0]//2:size[0]//2+1,-size[1]//2:size[1]//2+1]
    gus=(1/2*sigma)*np.exp(-(x**2+y**2)/2)
    return gus/gus.sum()

def ssim(groundtruth,predction,size,sigma):


    pass


import cv2
import numpy as np
import pandas as pd
import os
from ssim import mssim_multichannel
from matplotlib import pyplot as plt


# [h,w,c]

def mse(pred, groundtruth):  # 计算每一个通道的mse（返回值）
    h, w = pred.shape[0], pred.shape[1]
    assert (h == groundtruth.shape[0] and w == groundtruth.shape[1])
    sqerr = np.square(pred.astype(np.float32) - groundtruth.astype(np.float32))
    return np.sum(sqerr) / (3 * h * w)


def psnr(pred, groundtruth):
    mse_value = mse(pred, groundtruth)
    return 10 * np.log10(255 * 255 / mse_value)


def create_data(colums):
    data = pd.DataFrame(columns=colums)
    return data


def add_data(data, value_dict):
    data_series = pd.Series(value_dict['value'], name=value_dict['index'])
    data = data.append(data_series)
    return data


'''在模型的results中对数据进行评估'''


def make_dir(dirpath):
    try:
        os.makedirs(dirpath)
        print(dirpath + 'make successfully !')
    except:
        print(dirpath + ' make errorly or has made !')
    finally:
        pass


def evalute_psnr_ssim(dirpath, evaluate_name):  # dirpath 为正向传播之后的结果
    dataname = evaluate_name  # 可复用代码
    columns = ['with_scrach2predication', 'with_scrach2groundtruth', 'predication2groundtruth']
    data = create_data(columns)
    real_A_list, real_B_list, fake_B_list = [], [], []

    filenamelist = os.listdir(dirpath)

    for filename in filenamelist:
        if 'real_A' in filename:
            real_A_list.append(filename)
        if 'real_B' in filename:
            real_B_list.append(filename)
        if 'fake' in filename:
            fake_B_list.append(filename)

    for real_A_filename in real_A_list:
        real_A_img = cv2.imread(os.path.join(dirpath, real_A_filename))

        for fake_B_filename in fake_B_list:
            if real_A_filename.split('_')[0] == fake_B_filename.split('_')[0]:
                fake_B_img = cv2.imread(os.path.join(dirpath, fake_B_filename))

        for real_B_filename in real_B_list:
            if real_A_filename.split('_')[0] == real_B_filename.split('_')[0]:
                real_B_img = cv2.imread(os.path.join(dirpath, real_B_filename))

        if dataname == 'psnr':
            with_scrach2predication = psnr(real_A_img, fake_B_img)
            with_scrach2groundtruth = psnr(real_A_img, real_B_img)
            predication2groundtruth = psnr(fake_B_img, real_B_img)

        elif dataname == 'mssim':  # 默认值 直接返回MSSIM
            with_scrach2predication = mssim_multichannel(real_A_img, fake_B_img)
            with_scrach2groundtruth = mssim_multichannel(real_A_img, real_B_img)
            predication2groundtruth = mssim_multichannel(fake_B_img, real_B_img)

        else:  # 默认计算mse
            with_scrach2predication = mse(real_A_img, fake_B_img)
            with_scrach2groundtruth = mse(real_A_img, real_B_img)
            predication2groundtruth = mse(fake_B_img, real_B_img)

        need_add = {'index': real_A_filename.split('_')[0] + '.bmp', 'value': {
            'with_scrach2predication': with_scrach2predication,
            'with_scrach2groundtruth': with_scrach2groundtruth,
            'predication2groundtruth': predication2groundtruth}
                    }
        data = add_data(data, need_add)
    data_csv = data.to_csv(os.path.join(r'dataset/pix2pix_result', dataname + '.csv'))
    return data, data_csv

ssim_data,ssim_dataframe=evalute_psnr_ssim(r'D:\Files\pix2pix\pix\results\FOLD_AB3_pix2pix\test_latest\images','mssim')
print(ssim_data)
ssim_data.plot.bar()
plt.show()

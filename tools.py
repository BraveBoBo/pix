import torch
from torch import nn
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import cv2
import os
import shutil
import numpy as np
import random


def print_bar():
    now_time = datetime.datetime.now().strftime('%H:%M:%S')
    print('=========' * 8 + '%s' % now_time)


'''动态图可以每个epoch画一个点'''


def plot_loss(train_loss, val_loss):
    epochs = [i for i in range(1, len(train_loss) + 1)]
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_loss, c='r', linewidth=5.0, label='train-loss')
    plt.plot(epochs, val_loss, c='b', linewidth=5.0, label='val-loss')
    plt.title('train-loss val-loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def plot_metric(train_metric, val_metric):
    epochs = [i for i in range(1, len(train_metric) + 1)]
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_metric, c='g', linewidth=5.0, label='train-loss')
    plt.plot(epochs, val_metric, c='b', linewidth=5.0, label='val-loss')
    plt.title('train-metric val-metric')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.legend()
    plt.show()


def dropout_layers(x, p):
    assert 0 <= p <= 1
    if p == 1:
        return torch.zeros_like(x)
    if p == 0:
        return x
    mark = (torch.rand(x.shape) > 0.5).float()
    return mark * x / (1 - p)


def photo_add_id(input_dir, output_dir, start_id, end_id, name):
    filename_list = os.listdir(input_dir)
    i = start_id
    print('Start....')
    for filename in os.listdir(input_dir):
        try:
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, name + '-' + str(i) + '.bmp'))
        except:
            print(os.path.join(input_dir, filename) + 'cant copy !')
        finally:
            print(i)
            i += 1
        if end_id == i:
            break
    print('End !')


def resize_img(input_dir, output_dir, size, name):
    print('starting.....')
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        re = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_dir, name + filename), re)
        print(name + filename)
    print('end !')


def add_noise(img, noise_num):
    img_noise = img
    rows, cols, channels = img_noise.shape
    for _ in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


# train_path = r'D:\Files\data\dataset-for-p2p\train\FULL\A'  # todo 改变添加噪声的路径
# filenamelist = os.listdir(train_path)
# filename_buff = []
# filename = random.choice(filenamelist)
#
# for _ in range(250):
#     while filename in filename_buff:
#         filename = random.choice(filenamelist)
#     filename_buff.append(filename)
#     img = cv2.imread(os.path.join(train_path, filename))
#     img_noise = add_noise(img, random.randint(1000, 2000))
#     cv2.imwrite(os.path.join(train_path, filename), img_noise)
#     print(filename)
#
# for _ in range(250):
#     while filename in filename_buff:
#         filename = random.choice(filenamelist)
#     filename_buff.append(filename)
#     img = cv2.imread(os.path.join(train_path, filename))
#     img_noise = add_noise(img, random.randint(2000, 2500))
#     cv2.imwrite(os.path.join(train_path, filename), img_noise)
#     print(filename)
#
# for _ in range(250):
#     while filename in filename_buff:
#         filename = random.choice(filenamelist)
#     filename_buff.append(filename)
#     img = cv2.imread(os.path.join(train_path, filename))
#     img_noise = add_noise(img, random.randint(2500, 3000))
#     cv2.imwrite(os.path.join(train_path, filename), img_noise)
#     print(filename)


def cut_photo(input_dir, output_dir, size):
    print('Start....')
    width, height = size
    print(width, height)
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        w, h, c = img.shape
        w_o = int(w / 2)
        h_o = int(h / 2)
        w_padding = int((w - width) / 2 + 0.5)
        h_padding = int((h - height) / 2 + 0.5)
        new_img = img[w_o - w_padding + 200:w_o + w_padding, h_o - h_padding + 50:h_o + h_padding + 50]
        cv2.imwrite(os.path.join(output_dir, filename), new_img)
    print('Done.')


def flip_photo(input_dir, output_dir, photo_numbers, flag, name):
    print('start...')
    filename_buff = []
    filename = random.choice(os.listdir(input_dir))
    for i in range(photo_numbers):
        while filename in filename_buff:
            filename = random.choice(os.listdir(input_dir))
        filename_buff.append(filename)
        print(filename)
        img = cv2.imread(os.path.join(input_dir, filename))
        filp = cv2.flip(img, flag)
        cv2.imwrite(os.path.join(output_dir, name + '-' + filename), filp)
    print('done.')


def get_mark(input_dir, output_dir, lower, upper):
    print('Start...')
    savepath = os.path.join(output_dir, str(lower[2]) + '-' + str(upper[2]))
    try:
        os.makedirs(savepath)
    except:
        pass
    finally:
        print(savepath + 'make successfully')
    for filename in os.listdir(input_dir):
        cv2.namedWindow('img' + filename, cv2.WINDOW_NORMAL)
        cv2.namedWindow('roi' + filename, cv2.WINDOW_NORMAL)
        img = cv2.imread(os.path.join(input_dir, filename))
        lower_white = np.array(lower)
        upper_white = np.array(upper)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        roi = cv2.inRange(hsv, lower_white, upper_white)
        cv2.imshow('roi' + filename, roi)
        cv2.imshow('img' + filename, img)
        key = cv2.waitKey()
        if key == ord('s'):
            cv2.imwrite(os.path.join(savepath, filename), img)
            print(filename + 'saved !')
            cv2.destroyAllWindows()
        if key == 27:
            cv2.destroyAllWindows()
    print('all photos are look')


def addMark(input_dir, mark_dir, output_dir, number):
    print('Start...')
    marklist = os.listdir(mark_dir)
    i = 0
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        markname = random.choice(marklist)
        mark = cv2.imread(os.path.join(mark_dir, markname))

        x_start = random.randint(50, 100)  # todo 修改位置
        y_start = random.randint(60, 100)

        markHsv = cv2.cvtColor(mark, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([180, 30, 255])
        addpart = cv2.inRange(markHsv, lower_white, upper_white)
        for row in range(addpart.shape[0]):
            for col in range(addpart.shape[1]):
                if (addpart[row, col] != 0):
                    try:
                        img[x_start + row, y_start + col] = mark[row, col]
                    except:
                        pass
        cv2.imwrite(os.path.join(output_dir, filename), img)
        print(i)
        i += 1
        if i == number:
            break
    print('Done.')

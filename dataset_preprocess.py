# -*- coding: utf-8 -*-
'''
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.data as data
import sklearn.metrics as metrics
from netutils import *
from segnet import *
from mytictoc import *
'''
import matplotlib.image as image
import numpy as np
import argparse
"""python dataset_preprocess.py --patch_size 48 --step 48"""
parser = argparse.ArgumentParser()
parser.add_argument('--patch_size','-ps', type = int, default = 32, help = 'the size of train patch and test patch')
parser.add_argument('--step', type = int, default = 32, help = 'crop step, if do not want overlapping, then set it equal to patch size')
#其它之后可能要加的参数：输入img路径，gt路径，训练集块的选择下标
args = parser.parse_args()



#图像和标签切块，训练集和测试集
iput_img_original=image.imread('fcg.jpg')
gt_original=image.imread('gt.jpg')
gt_a = np.asarray(gt_original)
print(gt_original.shape)
h_ipt=gt_original.shape[0]
w_ipt=gt_original.shape[1]
# crop the original input to a set of smaller size images
rownum=12
colnum=12
rowheight = h_ipt // rownum#小块高度
colwidth = w_ipt // colnum#小块宽度
small_input=np.zeros([rownum,colnum,rowheight,colwidth])
small_gt=   np.zeros([rownum,colnum,rowheight,colwidth,3])
# input the parameter of the data
patch_size=args.patch_size
step = args.step


def crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size):
    patch_out=new_input_padding[wi_ipt:wi_ipt+patch_size,hi_ipt:hi_ipt+patch_size]
    return patch_out

for r in range(rownum):
    for c in range(colnum):
        iput_img=small_input[r,c,:,:] = iput_img_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth];
        gt      =small_gt[r,c,:,:,:]  = gt_original      [r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth,:];


#从small_input中取出train_patch_ind中的块，数据和标签切块并保存为训练集
train_patch_ind=[10,32,54,76,88,98,120]
x_num = int((small_input.shape[2]-patch_size)/step)+1#小块中可以切的行数和列数
y_num = int((small_input.shape[3]-patch_size)/step)+1
print(x_num*y_num)
index1 = 0
index2 = 0
index3 = 0

if args.patch_size > 100 and args.patch_size <= 200:
    train_temp = np.zeros((10000,1,patch_size,patch_size))
    test_temp = np.zeros((10000,1,patch_size,patch_size))
elif args.patch_size > 200:
    train_temp = np.zeros((1000,1,patch_size,patch_size))
    test_temp = np.zeros((1000,1,patch_size,patch_size))
else:
    train_temp = np.zeros((100000,1,patch_size,patch_size))
    test_temp = np.zeros((100000,1,patch_size,patch_size))

#图像切块
for ind in range(rownum * colnum):
    if ind in train_patch_ind:
        c = ind%colnum
        r = ind//colnum
        for i in range(1,x_num+1):
            for j in range(1,y_num+1):
                train_temp[index1] = small_input[r,c,(i-1)*step:(i-1)*step+patch_size,(j-1)*step:(j-1)*step+patch_size]
                index1 = index1 + 1
    else:
        c = ind%colnum
        r = ind//colnum
        for i in range(1,x_num+1):
            for j in range(1,y_num+1):
                test_temp[index2] = small_input[r,c,(i-1)*step:(i-1)*step+patch_size,(j-1)*step:(j-1)*step+patch_size]
                index2 = index2 + 1
                
train_temp_rm=train_temp[0:index1,:,:] #remove the empty array X
test_temp_rm=test_temp[0:index2,:,:] #remove the empty array X

train_temp_rm=train_temp_rm/255.
test_temp_rm=test_temp_rm/255.#归一化

print(train_temp_rm.shape,test_temp_rm.shape)
np.save("train_data_x_patch_"+str(patch_size)+".npy", train_temp_rm)
np.save("test_data_x_patch_"+str(patch_size)+".npy", test_temp_rm)

#标签切块
if args.patch_size > 100 and args.patch_size <= 200:
    train_data_y=np.zeros((10000,patch_size,patch_size,3))
    test_data_y=np.zeros((10000,patch_size,patch_size,3))
elif args.patch_size > 200:
    train_data_y=np.zeros((1000,patch_size,patch_size,3))
    test_data_y=np.zeros((1000,patch_size,patch_size,3))
else:
    train_data_y=np.zeros((100000,patch_size,patch_size,3))
    test_data_y=np.zeros((100000,patch_size,patch_size,3))
index1 = 0
index2 = 0
for ind in range(rownum * colnum):
    if ind in train_patch_ind:
        c = ind%colnum
        r = ind//colnum
        for i in range(1,x_num+1):
            for j in range(1,y_num+1):
                train_data_y[index1] = small_gt[r,c,(i-1)*step:(i-1)*step+patch_size,(j-1)*step:(j-1)*step+patch_size,:]
                index1 = index1 + 1
    else:
        c = ind%colnum
        r = ind//colnum
        for i in range(1,x_num+1):
            for j in range(1,y_num+1):
                test_data_y[index2] = small_gt[r,c,(i-1)*step:(i-1)*step+patch_size,(j-1)*step:(j-1)*step+patch_size,:]
                index2 = index2 + 1

train_data_y_rm=train_data_y[0:index1] #remove the empty array X
test_data_y_rm=test_data_y[0:index2] #remove the empty array X
print(train_data_y_rm.shape,test_data_y_rm.shape)

#标签维度暂时不转置，所以是 n,w,h,c

np.save("train_data_y_patch_"+str(patch_size)+".npy", train_data_y_rm)
np.save("test_data_y_patch_"+str(patch_size)+".npy", test_data_y_rm)

#彩色标签转为数字
palette = {0: (0, 0, 0),  # 河流
                1: (255, 0, 0),  # 城区
                2: (255, 255, 0),  # 农田
                3: (255, 255, 255),  # 背景
                4: (0, 255, 0)}  # 不属于图像的部分


invert_palette = {(0, 0, 0):0,
                  (255, 0, 0):1,
                  (255, 255, 0):2,
                  (0, 0, 255):1,#道路归为城区了
                  (255, 255, 255):3,
                  (0, 255, 0):4}
                  

#接受一个w*h*c的块，输出一个w*h的块
def convert_from_color(arr_3d):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.int32)#这里修改了一下，维度改为了第二和第三维

    for c, i in invert_palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d
#把gt从彩色图转为标签数据

train_data_y_c = np.load("train_data_y_patch_"+str(patch_size)+".npy")
test_data_y_c = np.load("test_data_y_patch_"+str(patch_size)+".npy")

train_data_y_g = np.zeros((train_data_y_c.shape[0],1,train_data_y_c.shape[1],train_data_y_c.shape[2]))
test_data_y_g = np.zeros((test_data_y_c.shape[0],1,test_data_y_c.shape[1],test_data_y_c.shape[2]))

for i in range(train_data_y_c.shape[0]):
    train_data_y_g[i,:,:] = convert_from_color(train_data_y_c[i,:,:,:])
for i in range(test_data_y_c.shape[0]):
    test_data_y_g[i,:,:] = convert_from_color(test_data_y_c[i,:,:,:])

print(train_data_y_g.shape,test_data_y_g.shape)

np.save("train_data_y_patch_"+str(patch_size)+"_numerical", train_data_y_g)
np.save("test_data_y_patch_"+str(patch_size)+"_numerical", test_data_y_g)

del  train_temp_rm
del test_temp_rm
del train_data_y_rm
del test_data_y_rm
del train_data_y_g
del test_data_y_g

gt_n = convert_from_color(gt_a)
np.save("gt_numerical", gt_n)

del gt_n
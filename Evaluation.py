import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import argparse
from shutil import copyfile
import glob
import yaml
import cv2
import numpy as np


import py_sod_metrics
# 输入预测图,必须为8位的[0,255]灰度图。形状为 H*W or W*H, 不能有N和C！意味着val_batchsize=1! 格式为numpy，不能是tensor
py_FM = py_sod_metrics.Fmeasure()
py_EM = py_sod_metrics.Emeasure()
py_SM = py_sod_metrics.Smeasure()
py_MAE = py_sod_metrics.MAE()


version = torch.__version__

def normPRED(x): # 拉开0-1分布的距离，使结果更稳定
    MAX = torch.max(x)
    MIN = torch.min(x)
    out = (x - MIN) / (MAX - MIN + 1e-8)
    return out


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='2', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2') # 指定显卡
parser.add_argument('--label_path', default='./Datasets/ORSSD/GT-test/', type=str, help='gt path') # 哪个测试集的GT的路径？
parser.add_argument('--label_ext', default='.png', type=str, help='gt path') # 测试集GT的文件后缀？
parser.add_argument('--name',default='0713_FSMINet_384_e52_b6_3',type=str, help='predicted image path') # 需要测试哪种方法的预测图？
parser.add_argument('--test_set',default='ORSSD',type=str, help='predicted image path') # 具体是哪个测试集的预测图？与label-path相关
parser.add_argument('--image_ext',default='.png',type=str, help='predicted image path') # 预测图片的后缀？
opt = parser.parse_args()


######################################################################
# --------- Load the data ---------
# 统计需要计算定量指标的预测图的文件名，重点是关注什么方法？什么测试集？。并根据预测图片名，加载对应的gt图片
img_name_list = glob.glob('./results/' + opt.name + '/' + opt.test_set + '/' + '*' + opt.image_ext) #./results/writer/ORSSD/
label_name_list = []
# ！！！！根据实际数据集的文件名进行修改
for image_path in img_name_list:
    image_name = image_path.split("/")[-1]
    # image_name = image_path.split("\\")[-1]
    t = image_name.split(".")
    t = t[0:-1]
    imidx = t[0]
    for i in range(1, len(t)):
        imidx = imidx + "." + t[i]
    label_name_list.append(opt.label_path + imidx + opt.label_ext)
print(label_name_list)
print("---")
print("images number: ", len(img_name_list))
print("---")
test_num = len(img_name_list)

MAE = 0.0
SM = 0.0
FM_adp = 0.0
FM_mean = 0.0
FM_max = 0.0
EM_adp = 0.0
EM_mean = 0.0
EM_max = 0.0

for i, img_name in enumerate(img_name_list):
    print(f"[{i}] Processing {img_name}...")
    test_num = i + 1
    # 注意！原图计算定量结果，所以和训练阶段log记录的结果会有偏差，但不重要。趋势是一样的！
    # 思考：如何确保label_name_list[i] 与 img_name_list [i] 读入的图片是一一对应的？简单来说就是图片编号是相同的。
    gt = cv2.imread(label_name_list[i], cv2.IMREAD_GRAYSCALE) # 读图，[W,H]
    pred = cv2.imread(img_name_list[i], cv2.IMREAD_GRAYSCALE) # 读图，[W,H]
    print(label_name_list[i])
    py_MAE.step(pred=pred, gt=gt)
    py_SM.step(pred=pred, gt=gt)
    py_FM.step(pred=pred, gt=gt)
    py_EM.step(pred=pred, gt=gt)

    MAE += py_MAE.get_results()["mae"]
    SM += py_SM.get_results()["sm"]
    FM_adp += (py_FM.get_results()["fm"])['adp']
    FM_mean += (py_FM.get_results()["fm"])['curve'].mean()
    FM_max += (py_FM.get_results()["fm"])['curve'].max()
    EM_adp += (py_EM.get_results()["em"])['adp']
    EM_mean += (py_EM.get_results()["em"])['curve'].mean()
    EM_max += (py_EM.get_results()["em"])['curve'].max()


print('MAE: {:4f} SM: {:4f} FM_adp: {:4f} FM_mean: {:4f} FM_max: {:4f} '
          'EM_adp: {:4f} EM_mean: {:4f} EM_max: {:4f}'.format(MAE / test_num, SM / test_num,
                                                              FM_adp / test_num, FM_mean / test_num, FM_max / test_num,
                                                              EM_adp / test_num, EM_mean / test_num, EM_max / test_num))

# 思考：是什么路径?
log_save_path = './results/' + opt.name + '/' + opt.test_set
with open(log_save_path + '/terminal_log.txt', "a") as terminal_log:
    terminal_log.write(f"MAE: {MAE / test_num:.4f}, SM: {SM / test_num:4f} "
                       f"FM_adp: {FM_adp / test_num:4f} FM_mean: {FM_mean / test_num:4f} FM_max: {FM_max / test_num:4f}"
                       f"EM_adp: {EM_adp / test_num:4f} EM_mean: {EM_mean / test_num:4f} EM_max: {EM_max / test_num:4f} %\n")

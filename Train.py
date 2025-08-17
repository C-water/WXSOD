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
import numpy as np

from Data_loader import Rescale
from Data_loader import Rotate
from Data_loader import ToTensor
from Data_loader import SalObjDataset_train, SalObjDataset_test

from WFANet import WFANet

import pytorch_ssim
import pytorch_iou

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

# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2') # 指定显卡
parser.add_argument('--name', default='0707_FSMINet_384_e200_b8', type=str, help='output model name, save_path = /model/0707_FSMINet_384_e200_b8') # 设置模型参数的文件名
parser.add_argument('--epochs', default=52, type=int, help='training epoch number') # 设置模型训练轮数
parser.add_argument('--batchsize', default=6, type=int, help='batchsize in train stage') # 设置模型训练的batchsize
parser.add_argument('--val_batchsize', default=1, type=int, help='batchsize in val stage, must be 1 if you want val') # 设置模型测试的batchsize
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # 设置训练的初始学习率
parser.add_argument('--resume', action='store_true', help='use resume trainning') # 是否加载模型权重继续训练
parser.add_argument('--image_size', default=384, type=int, help='train image size') # 训练阶段的输入图片尺寸
parser.add_argument('--image_train_dir',default='./',type=str, help='train input path') # 训练集的输入图片路径
parser.add_argument('--image_train_ext',default='.jpg',type=str, help='train input image ext') # 训练集的输入图片后缀
parser.add_argument('--label_train_dir',default='./',type=str, help='train label path') # 训练集的标签图片路径
parser.add_argument('--label_train_ext',default='.png',type=str, help='train label image ext') # 训练集的标签图片后缀
parser.add_argument('--image_test_dir',default='./',type=str, help='train input path') # 测试集的输入图片路径
parser.add_argument('--image_test_ext',default='.jpg',type=str, help='train input image ext') # 测试集的输入图片后缀
parser.add_argument('--label_test_dir',default='./',type=str, help='train label path') # 测试集的标签图片路径
parser.add_argument('--label_test_ext',default='.jpg',type=str, help='train label image ext') # 测试集的标签图片后缀
opt = parser.parse_args()


######################################################################
# ------- set gpu ids -------
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])  #
    cudnn.benchmark = True


######################################################################
# ------- processes the training dataset -------
image_train_name_list = glob.glob(opt.image_train_dir + '*' + opt.image_train_ext)
label_train_name_list = []  # 根据输入图片的文件名，依次加载GT。确保RGB与GT是一一对应的。 如果数据集发生变化，下面这块代码需要微调。
# ！！！！根据实际数据集的文件名进行修改
for image_path in image_train_name_list:
    image_name = image_path.split("/")[-1]
    t = image_name.split(".")
    t = t[0:-1]
    imidx = t[0]
    for i in range(1, len(t)):
        imidx = imidx + "." + t[i]
    label_train_name_list.append(opt.label_train_dir + imidx + opt.label_train_ext)
print("---")
print("train images: ", len(image_train_name_list))
print("train labels: ", len(label_train_name_list))
print("---")
train_num = len(image_train_name_list) # 统计训练图片的数量

salobj_dataset = SalObjDataset_train(
    img_name_list=image_train_name_list,
    lbl_name_list=label_train_name_list,
    transform=transforms.Compose([
        Rescale(opt.image_size),  # 384*384 图片输入
        Rotate(flag=1),
        ToTensor(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=1)


######################################################################
# ------- processes the test dataset -------
image_test_name_list = glob.glob(opt.image_test_dir + '*' + opt.image_test_ext)
label_test_name_list = []  # 根据输入图片的文件名，依次加载GT。确保RGB与GT是一一对应的。 如果数据集发生变化，下面这块代码需要微调。
# ！！！！根据实际数据集的文件名进行修改
for image_path in image_test_name_list:
    image_name = image_path.split("/")[-1]
    t = image_name.split(".")
    t = t[0:-1]
    imidx = t[0]
    for i in range(1, len(t)):
        imidx = imidx + "." + t[i]
    label_test_name_list.append(opt.label_test_dir + imidx + opt.label_test_ext)

print("---")
print("test images: ", len(image_test_name_list))
print("test labels: ", len(label_test_name_list))
print("---")
test_num = len(image_test_name_list) # 统计测试图片的数量

test_salobj_dataset = SalObjDataset_test(img_name_list=image_test_name_list,
                                    lbl_name_list=label_test_name_list,
                                    transform=transforms.Compose([
                                        Rescale(opt.image_size),
                                        ToTensor(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=1)


######################################################################
# ------- define loss function -------
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

# 所采用的loss
def hybrid_loss(pred,target):
    bce_out = bce_loss(pred,target)
    iou_out = iou_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)

    loss = bce_out + iou_out + ssim_out
    return loss

# 目前采用的是单尺度loss
def muti_loss_fusion(d0, labels_v):
    loss0 = hybrid_loss(d0, labels_v)
    loss = loss0
    # print("l0: %.3f, l1: %.3f, l2: %.3f, l3: %.3f, l4: %.3f, l5: %.3f"%(loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()))
    return loss


######################################################################
# ------- define model --------
net = WFANet()
if torch.cuda.is_available():
    net.cuda()


####### classifier loss function ############## 预测天气类别
cls_criterion = nn.CrossEntropyLoss()

######################################################################
# ------- define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

######################################################################
# ------- Decays the lr --------
# Decay LR by a factor of 0.75 every 50 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2) # 40 0.2

######################################################################
# ------- make save path --------
dir_name = os.path.join('./model', opt.name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

# 备份这一次训练模型所用到的文件。便于后期确认模型的训练细节。
copyfile('./runtest.sh', dir_name + '/runtest.sh')
copyfile('Train.py', dir_name + '/Train.py')
copyfile('Test.py', dir_name + '/Test.py')
copyfile('./WFANet.py', dir_name + '/WFANet.py')
copyfile('./PVT_V2.py', dir_name + '/PVT_V2.py')
copyfile('./Data_loader.py', dir_name + '/Data_loader.py')


######################################################################
# ------- training process --------
for epoch in range(0, opt.epochs):
    # save opts，保存路径为/model/0707_FSMINet_384_e200_b8/opts.yaml
    # with open('%s/opts.yaml' % dir_name, 'w') as fp:
    #     yaml.dump(vars(opt), fp, default_flow_style=False)

    #### train stage
    net.train()
    running_loss_total = 0.0
    running_loss_mask = 0.0
    running_loss_weather = 0.0
    # for i, data in enumerate(salobj_dataloader):
    #     inputs, labels_mask, labels_weather = data[0]['image'], data[0]['label'], data[1]
    for data, labels_weather in tqdm(salobj_dataloader):
        inputs, labels_mask = data['image'], data['label']
        labels_weather = labels_weather
        # print('input',inputs.shape)  # input torch.Size([6, 3, 384, 384])
        # print('mask',labels_mask.shape) # input torch.Size([6, 1, 384, 384])
        # print('weather',labels_weather.shape) # input torch.Size([6])

        inputs = inputs.type(torch.FloatTensor)
        labels_mask = labels_mask.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_mask, labels_weather = inputs.cuda().detach(), labels_mask.cuda().detach(), labels_weather.cuda().detach()
        else:
            inputs_v, labels_mask, labels_weather = inputs, labels_mask, labels_weather

        optimizer.zero_grad()
        pred_mask, pred_weather = net(inputs_v)

        # 计算两个分支的loss
        loss_1 = muti_loss_fusion(pred_mask, labels_mask) # 用默认的损失，包括： bce + iou + ssim
        loss_2 = cls_criterion(pred_weather, labels_weather) # 用交叉熵损失，天气类别损失

        # loss求和，
        loss_total = loss_1 + loss_2
        loss_total.backward()
        optimizer.step()

        running_loss_total += loss_total.item()
        running_loss_mask += loss_1.item()
        running_loss_weather += loss_2.item()

        # del d0, d1, d2, d3, d4, d5, loss
        # torch.cuda.empty_cache()

    epoch_loss_total = running_loss_total / train_num
    epoch_loss_mask = running_loss_mask / train_num
    epoch_loss_weather = running_loss_weather / train_num

    # 在终端输出当前epoch的训练loss。随着训练的进行，loss呈现下降趋势才是正确的。
    print("[epoch: %3d/%3d] epoch loss total: %.4f epoch_loss_mask: %.4f  epoch_loss_weather: %.4f "
          % (epoch+1, opt.epochs, epoch_loss_total, epoch_loss_mask, epoch_loss_weather))

    # 动态调整学习率
    exp_lr_scheduler.step()

    #### val stage
    if epoch > 49 and epoch % 1 == 0:
    # if epoch > 0:
        # 每次test，8个定量指标都需要初始化。
        MAE = 0.0
        SM = 0.0
        FM_adp = 0.0
        FM_mean = 0.0
        FM_max = 0.0
        EM_adp = 0.0
        EM_mean = 0.0
        EM_max = 0.0

        print('Val stage begin !!!')
        net.eval()
        # for i, data_test in enumerate(test_salobj_dataloader):
        #     inputs_test, labels_test_mask, labels_test_weather = data_test[0]['image'], data_test[0]['label'], data_test[1]
        for data_test, labels_test_weather in tqdm(test_salobj_dataloader):
            inputs_test, labels_test_mask = data_test['image'], data_test['label']
            labels_test_weather = labels_test_weather

            inputs_test = inputs_test.type(torch.FloatTensor)
            labels_test_mask = labels_test_mask.type(torch.FloatTensor)

            inputs_test = inputs_test.cuda().detach()
            labels_test_mask = labels_test_mask.cuda().detach()
            labels_test_weather = labels_test_weather.cuda().detach()

            with torch.no_grad():
                 pred_mask, pred_weather = net(inputs_test)

            # 处理预测的mask
            pred_mask = pred_mask[0, 0, :, :]  # N=1,C=1,H,W-->H,W， 修改图片的维度
            pred_mask = normPRED(pred_mask)


            # 处理labels_test_mask
            labels_test_mask = labels_test_mask[0, 0, :, :]

            # 把预测图pred和标签GT都转成[0,255]的numpy.uint8格式，维度为[H,W]
            pred_mask_cpu = pred_mask.clamp(0, 1).cpu()
            pred_mask = (pred_mask_cpu*255).numpy().astype(np.uint8)
            labels_test_mask_cpu = labels_test_mask.clamp(0, 1).cpu()
            labels_test_mask = (labels_test_mask_cpu*255).numpy().astype(np.uint8)

            py_MAE.step(pred=pred_mask, gt=labels_test_mask)
            py_SM.step(pred=pred_mask, gt=labels_test_mask)
            py_FM.step(pred=pred_mask, gt=labels_test_mask)
            py_EM.step(pred=pred_mask, gt=labels_test_mask)


            MAE += py_MAE.get_results()["mae"]
            SM += py_SM.get_results()["sm"]
            FM_adp += (py_FM.get_results()["fm"])['adp']
            FM_mean += (py_FM.get_results()["fm"])['curve'].mean()
            FM_max += (py_FM.get_results()["fm"])['curve'].max()
            EM_adp += (py_EM.get_results()["em"])['adp']
            EM_mean += (py_EM.get_results()["em"])['curve'].mean()
            EM_max += (py_EM.get_results()["em"])['curve'].max()

            # del d0, d1, d2, d3, d4, d5
            # del py_MAE, py_SM, py_FM, py_EM
            # torch.cuda.empty_cache()

        # 在服务器终端输出预测图的各项指标
        print('Epoch: {} MAE: {:4f} SM: {:4f} '
              'FM_adp: {:4f} FM_mean: {:4f} FM_max: {:4f} '
              'EM_adp: {:4f} EM_mean: {:4f} EM_max: {:4f} '.format(epoch, MAE/test_num, SM/test_num,
                                                                  FM_adp/test_num, FM_mean/test_num, FM_max/test_num,
                                                                  EM_adp/test_num, EM_mean/test_num, EM_max/test_num))

    # 把结果记录在log文件。
        # 虽然python和matlib测出来的定量结果会有轻微差异，但整体趋势是相同的。用python测出来最好的结果，在matlib也是最好的。
        # with open(dir_name + '/terminal_log.txt', "a") as terminal_log:
            # terminal_log.write(f"Best val Acc:{best_test_recall:.4f},  Best epoch: {best_test_epoch:.4f}%\n")
            # terminal_log.write(f"Epoch:{epoch:},  MAE: {MAE/test_num:.4f}, SM: {SM/test_num:4f} "
            #                    f"FM_adp: {FM_adp/test_num:4f} FM_mean: {FM_mean/test_num:4f} FM_max: {FM_max/test_num:4f}"
            #                    f"EM_adp: {EM_adp/test_num:4f} EM_mean: {EM_mean/test_num:4f} EM_max: {EM_max/test_num:4f} Acc_weather: {Acc_weather/test_num:4f}%\n")

        #### save network
        # 注意：不同的权重保存方式，其加载权重的方式也不同的。
        torch.save(net.state_dict(),  dir_name + '/' + "Net_epoch_%d.pth" % (epoch))  # 保存路径为 /model/xxxxx/Net_epoch_x.pth

    print('Continue training!!!')
    # 恢复模型的训练状态，允许参数变化
    net.train()

print('-------------Congratulations! Training Done!-------------')
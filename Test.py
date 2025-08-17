from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import glob
import os
from tqdm import tqdm
import argparse

from Data_loader import Rescale
from Data_loader import ToTensor
from Data_loader import SalObjDataset_test
from WFANet import WFANet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normPRED(x): # 拉开0-1分布的距离，使结果更稳定
    MAX = torch.max(x)
    MIN = torch.min(x)
    out = (x - MIN) / (MAX - MIN + 1e-8)
    return out

def save_output(image_name, pred, d_dir): #根据实际需求可以修改
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB') #矩阵转RGB格式图片
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    t = img_name.split(".")
    t = t[0:-1]
    imidx = t[0]
    for i in range(1, len(t)):
        imidx = imidx + "." + t[i]

    imo.save(d_dir + imidx + '.png')

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='3', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2') # 指定显卡
parser.add_argument('--name', default='0707_FSMINet_384_e200_b8', type=str, help='output model name, save_path = /model/0707_FSMINet_384_e200_b8') # 设置加载权重的文件名
parser.add_argument('--pth', default='Net_epoch_51.pth', type=str, help='model weight name') # 选择最优模型的权重
parser.add_argument('--save_path', default='0707_FSMINet_384_e200_b8', type=str, help='image save path') # 选择输出图片的保存路径
parser.add_argument('--val_batchsize', default=1, type=int, help='batchsize in val stage, must be 1 if you want val') # 设置模型测试的batchsize
parser.add_argument('--image_size', default=384, type=int, help='train image size') # 训练阶段的输入图片尺寸
parser.add_argument('--test_set', default='EORSSD', type=str, help='Test set name') # 测试集的名称，避免将不同测试集的结果混在一起
parser.add_argument('--image_test_dir',default='./Datasets/VT5000/test/RGB_noise/',type=str, help='train input path') # 测试集的输入图片路径
parser.add_argument('--image_test_ext',default='.jpg',type=str, help='train input image ext') # 测试集的输入图片后缀
opt = parser.parse_args()


######################################################################
# --------- Load the data ---------
# 记录所有输入RGB图片的名称到img_name_list。后续根据图片名，加载对应图片
img_name_list = glob.glob(opt.image_test_dir + '*' + opt.image_test_ext)
test_num = len(img_name_list)

test_salobj_dataset = SalObjDataset_test(img_name_list=img_name_list, lbl_name_list=[],
                                    transform=transforms.Compose([Rescale(opt.image_size), ToTensor(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=1)


######################################################################
# --------- Define the model ---------
# 确定模型权重的完整路径
weight_name = os.path.join('./model', opt.name, opt.pth)  #./model/0707_FSMINet_384_e200_b8/Net_epoch_150.pth
print("...Model weight...", weight_name)

# 加载权重到模型中
print("...load Network...")
net = WFANet()
net.load_state_dict(torch.load(weight_name,map_location=torch.device('cpu')))
if torch.cuda.is_available():
    net.cuda()
net.eval()


######################################################################
# --------- Generate prediction images ---------
Acc_weather = 0.0

for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):
    inputs_test, labels_test_weather = data_test[0]['image'], data_test[1]

    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda().detach())
        labels_test_weather = Variable(labels_test_weather.cuda().detach())
    else:
        inputs_test = Variable(inputs_test)
        labels_test_weather = Variable(labels_test_weather)

    with torch.no_grad():
        pred_mask, pred_weather = net(inputs_test)


    # 处理mask
    pred = pred_mask[:, 0, :, :]  #N C H W  ->  N H W
    pred = normPRED(pred)

    # save results to test_results folder
    # 确定图片的保存路径,同一个模型可能需要测试好几个测试集，所以要用数据集名称区分
    save_dir = os.path.join('./results', opt.save_path, opt.test_set)

    # 保存图片。输出图片名称可能需要调整！ 由于图片输入的时候经过resize操作，因此网络输出需要反向resize成原始分辨率。
    save_output(img_name_list[i_test], pred, save_dir)

    del pred_mask, pred_weather

print('Acc_weather: {:4f}'.format(Acc_weather/test_num))
import torch
from torchvision import models
from thop.profile import profile
# from MINet import *
from thop import clever_format
from WFANet import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CPU算就行
device = torch.device("cpu")

# model_head ='net_orig'
# 选择模型，设置超参数
model =  GPONet_Ours().to(device)

# 定义输入特征
input_1 = (1, 3, 384, 384)
# input_2 = (1, 3, 384, 512)
input_1 = torch.randn(input_1).to(device)
# input_2 = torch.randn(input_2).to(device)

# 单输入模型
total_ops, total_params = profile(model,(input_1.unsqueeze(0)), verbose=False)

# 双输入模型
# total_ops, total_params = profile(model,(input_1,input_2), verbose=False)

# 计算结果
macs, params = clever_format([total_ops, total_params], "%.3f")
# print('Net:',model_head)
print('MACs:',macs)
print('Paras:',params)


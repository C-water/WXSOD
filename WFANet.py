from torch import nn, multiprocessing
from torch.autograd import Variable
from torch.nn import functional as F
import torch
from torch.nn import init
# from Encoder_pvt import Encoder
# from decoder_ours import Decoder
from PVT_V2 import pvt_v2_b2
from torchvision.models import resnet18

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate,num_bottleneck=512):
        super(ClassBlock, self).__init__()
        num_bottleneck = num_bottleneck
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=droprate)]
        add_block += [nn.Linear(num_bottleneck, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder = pvt_v2_b2()
        # self.encoder.load_state_dict(torch.load(r'D:\project\WX_SOD\Ours\pvt_v2_b2.pth', map_location='cpu'), strict=False)
        self.encoder.load_state_dict(torch.load("C:\project\WX_SOD_1222/Ours/pretrained_model/pvt_v2_b2.pth", map_location='cpu'),strict=False)

    def forward(self, x):
        out = self.encoder(x)
        return out[::-1]

class WPM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(WPM, self).__init__()
        self.CNR1 = nn.Sequential(
            nn.Conv2d(512, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.CNR2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.CNR3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.CNR1(x)
        x = self.CNR2(x)
        x = self.CNR3(x)
        return x

class CNR2(nn.Module):
    def __init__(self, in_channels, weather_channels, out_channels):
        super(CNR2, self).__init__()
        # 处理天气特征的分支
        self.weather_branch = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 处理拼接后的融合特征
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, main_feature, weather_feature):
        # 上采样天气特征到主特征的分辨率
        weather_upsampled = nn.functional.interpolate(weather_feature, size=main_feature.shape[2:], mode='bilinear', align_corners=False)
        # 处理天气特征
        weather_out = self.weather_branch(weather_upsampled)
        # 拼接主特征和处理后的天气特征
        fused = torch.cat([main_feature, weather_out], dim=1)
        # 融合特征
        out = self.fusion(fused)
        return out

class GFN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(GFN, self).__init__()
        self.oc = out_channel
        self.squ_1 = nn.Sequential(nn.Conv2d(in_channel_list[0], out_channel, 1), nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
        self.squ_2 = nn.Sequential(nn.Conv2d(in_channel_list[1], out_channel, 1), nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
        self.squ_3 = nn.Sequential(nn.Conv2d(in_channel_list[2], out_channel, 1), nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
        self.squ_4 = nn.Sequential(nn.Conv2d(in_channel_list[3], out_channel, 1), nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))

        self.CNR1 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR4 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.CFM1 = CFM(out_channel)
        self.CFM2 = CFM(out_channel)
        self.CFM3 = CFM(out_channel)


    def forward(self, input_list):
        c1, c2, c3, c4 = input_list[0].shape[1], input_list[1].shape[1], input_list[2].shape[1], input_list[3].shape[1]
        # 将骨干网络提取的多尺度特征的通道进行统一
        if c1 != c2 or c3 != self.oc:
            f1, f2, f3, f4 = self.squ_1(input_list[0]), self.squ_2(input_list[1]), self.squ_3(input_list[2]), self.squ_4(input_list[3])
        else:
            f1, f2, f3, f4 = input_list[0], input_list[1], input_list[2], input_list[3]

        # 多尺度特征交互
        out1 = f1
        out2 = self.CFM1(out1, f2)
        out3 = self.CFM2(out2, f3)
        out4 = self.CFM3(out3, f4)

        out1 = self.CNR1(out1)
        out2 = self.CNR2(out2)
        out3 = self.CNR3(out3)
        out4 = self.CNR4(out4)
        return (out1, out2, out3, out4)

class CFM(nn.Module):
    def __init__(self,out_channel):
        super(CFM, self).__init__()
        self.CNR1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.CNR2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, higher_feat, lower_feat):
        size = lower_feat.shape[2:]
        f1 = F.interpolate(higher_feat,size=size,mode='bilinear')
        g2 = self.CNR1(lower_feat)
        act2 = self.CNR2(g2 * lower_feat)
        f2 = lower_feat + f1 + act2
        return f2

class Decoder(nn.Module):
    def __init__(self, in_channel_list, out_channel, weather_channels):
        super(Decoder, self).__init__()

        self.gfn_gb = GFN(in_channel_list, out_channel)

        # 这里我们定义四条并行处理路径，每条路径有3个卷积操作
        self.conv1 = CNR2(out_channel, weather_channels, out_channel)
        self.conv2 = CNR2(out_channel, weather_channels, out_channel)
        self.conv3 = CNR2(out_channel, weather_channels, out_channel)
        self.final_conv = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))

    def forward(self, gb_list, weather_feat):
        # 从 GFN 模块获取多尺度特征
        fuse_gb_12, fuse_gb_24, fuse_gb_48, fuse_gb_96= self.gfn_gb(gb_list)

        # 按尺度处理主特征并融合天气特征，每个分支的MSFM共享权重。
        out_12 = self.conv1(fuse_gb_12, weather_feat)
        out_12 = self.conv2(out_12, weather_feat)
        out_12 = self.conv3(out_12, weather_feat) + fuse_gb_12
        out_12 = self.final_conv(out_12)

        out_24 = self.conv1(fuse_gb_24, weather_feat)
        out_24 = self.conv2(out_24, weather_feat)
        out_24 = self.conv3(out_24, weather_feat) + fuse_gb_24
        out_24 = self.final_conv(out_24)

        out_48 = self.conv1(fuse_gb_48, weather_feat)
        out_48 = self.conv2(out_48, weather_feat)
        out_48 = self.conv3(out_48, weather_feat) + fuse_gb_48
        out_48 = self.final_conv(out_48)

        out_96 = self.conv1(fuse_gb_96, weather_feat)
        out_96 = self.conv2(out_96, weather_feat)
        out_96 = self.conv3(out_96, weather_feat) + fuse_gb_96
        out_96 = self.final_conv(out_96)

        # 返回多尺度特征
        out_gb_feat = [out_12, out_24, out_48, out_96]
        return out_gb_feat,weather_feat

class Predictor(nn.Module):
    def __init__(self, fuse_channel=256):
        super(Predictor, self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Linear(4, fuse_channel),  # 输入通道数为4
            nn.BatchNorm1d(fuse_channel),
            nn.ReLU(),
            nn.Linear(fuse_channel, 4)  # 输出通道数改为4，与输入匹配
        )

    def forward(self, input):
        B = input.shape[0]

        # get fuse attention
        gap = self.GAP(input).squeeze(-1).squeeze(-1)  # 输出 [B, 4]
        # print(gap.shape)
        fuse_att = self.fuse(gap).view(B, 4, 1, 1)  # 输出 [B, 4, 1, 1]
        # print(fuse_att.shape)
        # fuse from gb&dt out
        fuse = input * fuse_att.expand_as(input)  # 扩展为 [B, 4, 384, 384]

        return fuse

class WFANet(nn.Module):
    def __init__(self, backbone='pvt', d_channel=256, input_size=384):
        super(WFANet, self).__init__()
        if backbone == 'pvt':
            self.in_channel_list = [512, 320, 128, 64]
        elif backbone == 'resnet':
            self.in_channel_list = [2048, 1024, 512, 256]

        # backbone
        self.encoder = Encoder()

        # 天气特征提取器
        self.weather_extractor = resnet18(pretrained=True)
        self.weather_extractor = nn.Sequential(*list(self.weather_extractor.children())[:-2])  # 去掉最后的全连接层和全局平均池化层

        # WPM 模块
        self.wpm = WPM(512, 256)  # ResNet18 的输出通道数为 512，WPB 输出通道数为 256

        # decoder
        weather_channels = 256
        self.decoder1 = Decoder(self.in_channel_list, d_channel, weather_channels)

        self.predictor = Predictor()

        # heads
        self.gb_head1 = nn.Conv2d(d_channel, 1, 3, padding=1)
        self.gb_head2 = nn.Conv2d(d_channel, 1, 3, padding=1)
        self.gb_head3 = nn.Conv2d(d_channel, 1, 3, padding=1)
        self.gb_head4 = nn.Conv2d(d_channel, 1, 3, padding=1)

        self.head = nn.Conv2d(4, 1, 3, padding=1)
        # loss
        self.fuse_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.size_list_per_stage = [input_size // 4, input_size // 8, input_size // 16, input_size // 32]
        self.classifier = ClassBlock(256, 9, 0.5)  # WPB 输出通道数为 256

    def forward(self, x):
        B = x.shape[0]
        s = self.size_list_per_stage

        # 计算骨干网络的4个输出
        f_1, f_2, f_3, f_4 = self.encoder(x)

        # 骨干网络输出特征的形状转换：B*HW*C-->B*C*H*W
        f_1 = f_1.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[0], s[3], s[3])
        f_2 = f_2.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[1], s[2], s[2])
        f_3 = f_3.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[2], s[1], s[1])
        f_4 = f_4.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[3], s[0], s[0])
        feat_list = [f_1, f_2, f_3, f_4]

        # 使用 ResNet18 提取天气特征
        weather_feat = self.weather_extractor(x)  # ResNet18 提取的天气特征 [B, 512, H/32, W/32]
        weather_feat = self.wpm(weather_feat)  # 通过 WPB 进一步提取天气特征 [B, 256, H/32, W/32]

        # 进行特征解码，包括CFM\MSFM\WPM
        out_gb_feat, _ = self.decoder1(feat_list, weather_feat)

        # 计算每个尺度下MSFM的输出mask，并进行拼接
        cat_pred = self.get_stage_pred(out_gb_feat, x)
        fuse = torch.cat(cat_pred, dim=1)

        # 多尺度Mask的自适应融合，输出最终的output_Mask
        fuse = self.predictor(fuse)
        fuse_pred = torch.sigmoid(self.mlp(fuse, x, self.head)) # B*C*H_in*W_in

        # 预测图像的天气噪声类别
        # 计算一维的判别特征向量
        weather_feat_pooled = F.adaptive_avg_pool2d(weather_feat, (1, 1))  # GAP操作
        weather_feat_pooled = weather_feat_pooled.view(weather_feat_pooled.size(0), -1)
        # 进行分类预测
        weather_class = self.classifier(weather_feat_pooled) # torch.Size([B, 9])

        # 返回网络预测的 Mask 和 天气类别特征
        return fuse_pred, weather_class

    def get_stage_pred(self, out_gb_feat, x):
        gb_pre_96 = self.mlp(out_gb_feat[3], x, self.gb_head1)
        gb_pre_48 = self.mlp(out_gb_feat[2], x, self.gb_head2)
        gb_pre_24 = self.mlp(out_gb_feat[1], x, self.gb_head3)
        gb_pre_12 = self.mlp(out_gb_feat[0], x, self.gb_head4)

        cat_pred = [gb_pre_96, gb_pre_48, gb_pre_24, gb_pre_12]

        return  cat_pred

    def up_sample(self, src, target):
        size = target.shape[2:]
        out = F.interpolate(src, size=size, mode='bilinear')
        return out

    def mlp(self, src, tar, head):
        B, _, H, W = tar.shape
        up = self.up_sample(src, tar)
        out = head(up)
        return out



# debug model structure
# Run this code with:
# python model.py
if __name__ == '__main__':
    # multiprocessing.Process()
    # multiprocessing.freeze_support()

    # Test the model, before you train it.
    net = WFANet()
    print(net)
    input = Variable(torch.FloatTensor(6, 3, 384, 384))
    output = net(input)

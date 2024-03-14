import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import math
import random
import numpy as np
from PIL import Image

def pepper_and_salt(img, percentage):
    num=int(percentage*img.shape[0]*img.shape[1])#  椒盐噪声点数量
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)#从0到图像长度之间的一个随机整数,因为是闭区间所以-1
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: #黑白色概率55开
            img2[X,Y] = (255,255,255)#白色
        else:
            img2[X,Y] =(0,0,0)#黑色
    return img2


def torch_add_salt(img, snr = 0.9):
    # 把img转化成ndarry的形式
    img_ = np.array(img).copy()
    h, w, c = img_.shape
    # 原始图像的概率（这里为0.9）
    signal_pct = snr
    # 噪声概率共0.1
    noise_pct = (1 - snr)
    # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
    # 将mask按列复制c遍
    mask = np.repeat(mask, c, axis=2)
    img_[mask == 1] = 255  # 盐噪声
    img_[mask == 2] = 0  # 椒噪声
    return Image.fromarray(img_.astype('uint8')).convert('RGB')  # 转化为PIL的形式




def distillation_loss(source, target, margin):
    # print(source.size(), target.size())

    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")

    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()



        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()
        print(t_channels, s_channels)
        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x,  flag = False, feature_mask = None, poison_feature = None):
        # prerelu 默认 True
        # t_feats, t_out = self.t_net.extract_feature(x, flag, feature_mask, poison_feature, preReLU=True)
        t_feats, t_out = self.t_net.forward_feature(x,  feature_mask, poison_feature, preReLU=True)

        # 学生模型的 X 变为低质量的模态
        x = pepper_and_salt(x, 0.1)

        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)

        feat_num = len(t_feats)
        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])

            # print(t_feats[i].size(), s_feats[i].size())

            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return s_out, loss_distill

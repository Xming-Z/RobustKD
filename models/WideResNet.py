import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # 冻结
        # self.unfreeze(self.block1, self.block2)

    def forward(self, x, flag = False, mask = None, feature = None):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if flag:
            for i in range(len(mask)):
                out  = out - mask[i] * feature[i]
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        return self.fc(out)

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return self.nChannels[1:]

    def extract_feature(self, x,flag = False, mask = None, feature = None, preReLU=False):
        out = self.conv1(x)
        feat1 = self.block1(out)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        if flag:
            for i in range(len(mask)):
                feat3 = feat3 - mask[i] * feature[i]

        out = self.relu(self.bn1(feat3))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        out = self.fc(out)

        if preReLU:
            feat1 = self.block2.layer[0].bn1(feat1)
            feat2 = self.block3.layer[0].bn1(feat2)
            feat3 = self.bn1(feat3)

        return [feat1, feat2, feat3], out

    def unfreeze(self, layer1, layer2 = None):
        for p in self.parameters():
            p.requires_grad = False

        for child in layer1.children():
            for param in child.parameters():
                param.requires_grad = True

        if layer2 != None:
            for child in layer2.children():
                for param in child.parameters():
                    param.requires_grad = True

    def from_features_to_output(self, x):
        out = self.relu(self.bn1(x))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        out = self.fc(out)
        return out


    def forward_feature(self, x,  mask = None, feature = None, preReLU=False):
        out = self.conv1(x)

        if mask == None:
            feat1 = self.block1(out)
            feat2 = self.block2(feat1)
            feat3 = self.block3(feat2)
            out = feat3
        else:
            # length = len(mask)
            # feat1 = self.block1(out)
            # out = feat1 + mask[0] * feature[0]
            #
            # feat2 = self.block2(out)
            # out = feat2 + mask[-2] * feature[-2]
            #
            # feat3 = self.block3(out)
            # out = feat3 + mask[-1] * feature[-1]
            # yes
            out = self.block1(out)
            # feat1 = out + mask[0] * feature[0]
            feat1 = (1 - mask[0]) * out + mask[0] * feature[0]

            out = self.block2(feat1)
            # feat2 = out + mask[-2] * feature[-2]
            feat2 = (1 - mask[-2]) * out + mask[-2] * feature[-2]

            out = self.block3(feat2)
            # feat3 = out + mask[-1] * feature[-1]
            feat3 = (1 - mask[-1]) * out + mask[-1] * feature[-1]

        out = self.relu(self.bn1(feat3))
        # out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])

        if preReLU:
            feat1 = self.block2.layer[0].bn1(feat1)
            feat2 = self.block3.layer[0].bn1(feat2)
            feat3 = self.bn1(feat3)

        return [feat1,feat2,feat3], self.fc(out)
import torch.nn as nn
import math
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


'''class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        #print(downsample)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            #print(x.shape)
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x
        
    def input_to_residual(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        return residual
        
    def residual_to_output(self, residual,conv2):
        x = residual + conv2
        x = self.relu(x)

        return x


    def input_to_conv2(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
        
    def conv2_to_output(self, x, residual):
        x = self.bn2(x)
        x = residual + x
        x = self.relu(x)
        return x
        
    def conv2_to_output_mask(self, x, residual,mask,pattern):
        x = self.bn2(x)
        x = residual + x
        x = (1 - mask) * x + mask * pattern
        x = self.relu(x)
        return x

    def input_to_conv1(self, x):
        x = self.conv1(x)
        return x
        
    def conv1_to_output(self, x, residual):
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x'''
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Added another relu here
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # Modified to use relu2
        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100,in_channels=3):
        self.inplanes = 64
        self.expansion = block.expansion
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.inter_feature = {}
        self.inter_gradient = {}
        self.register_all_hooks()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, flag = False, flip = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if(flag):
            x  = (1 - flip) * x - flip * x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
    def get_fm(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        
        return x

    def from_features_to_output(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    ##########
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        elif isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        else:
            print('ResNet unknown block error !!!')
        # return [bn1, bn2, bn3, bn4]
        return [bn2, bn3, bn4]

    def get_channel_num(self):
        # return [64, 128, 256, 512] * self.expansion
        return [128 * self.expansion, 256 * self.expansion, 512 * self.expansion]

    def extract_feature(self, x, flag = False, feature_mask = None, poison_feature = None, flip = None, preReLU=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feat1 = self.layer1(x)
        # print(feat1.size())
        feat2 = self.layer2(feat1)
        # print(feat2.size())
        feat3 = self.layer3(feat2)
        # print(feat3.size())
        feat4 = self.layer4(feat3)
        if(flip != None):
            feat4 = (1 - flip) * feat4 - flip * feat4
        # print(feat4.size())
        x = self.avgpool(feat4)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)

        # return [feat1, feat2, feat3, feat4], out
        return [ feat2, feat3, feat4], out

    def extract_feature1(self, x,flip, preReLU=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feat1 = self.layer1(x)
        # print(feat1.size())
        feat2 = self.layer2(feat1)
        # print(feat2.size())
        feat3 = self.layer3(feat2)
        # print(feat3.size())
        feat4 = self.layer4(feat3)

        feat4 = (1 - flip) * feat4 - flip * feat4
        # print(feat4.size())
        x = self.avgpool(feat4)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        out = self.fc(x)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)

        # return [feat1, feat2, feat3, feat4], out
        return [ feat2, feat3, feat4], out

    ##########

    def input_to_conv1(self, x):
        x = self.conv1(x)
        
        return x
        
    def conv1_to_output(self, x):
        #x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        
        return x
        
    def layer1_to_output(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        return x
        
    def layer2_to_output(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer3(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
        
    def layer3_to_output(self, x):
        
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer4(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        
    def layer4_to_output(self, x):

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def make_hook(self, name, flag):
        if flag == 'forward':
            def hook(m, input, output):
                self.inter_feature[name] = output
            return hook
        elif flag == 'backward':
            def hook(m, input, output):
                self.inter_gradient[name] = output
            return hook
        else:
            assert False
            
    def register_all_hooks(self):
        self.conv1.register_forward_hook(self.make_hook("Conv1_Conv1_Conv1_", 'forward'))
        self.layer1[0].conv1.register_forward_hook(self.make_hook("Layer1_0_Conv1_", 'forward'))
        self.layer1[0].conv2.register_forward_hook(self.make_hook("Layer1_0_Conv2_", 'forward'))
        self.layer1[1].conv1.register_forward_hook(self.make_hook("Layer1_1_Conv1_", 'forward'))
        self.layer1[1].conv2.register_forward_hook(self.make_hook("Layer1_1_Conv2_", 'forward'))
        
        self.layer2[0].conv1.register_forward_hook(self.make_hook("Layer2_0_Conv1_", 'forward'))
        self.layer2[0].downsample.register_forward_hook(self.make_hook("Layer2_0_Downsample_", 'forward'))
        self.layer2[0].conv2.register_forward_hook(self.make_hook("Layer2_0_Conv2_", 'forward'))
        self.layer2[1].conv1.register_forward_hook(self.make_hook("Layer2_1_Conv1_", 'forward'))
        self.layer2[1].conv2.register_forward_hook(self.make_hook("Layer2_1_Conv2_", 'forward'))
        
        self.layer3[0].conv1.register_forward_hook(self.make_hook("Layer3_0_Conv1_", 'forward'))
        self.layer3[0].downsample.register_forward_hook(self.make_hook("Layer3_0_Downsample_", 'forward'))
        self.layer3[0].conv2.register_forward_hook(self.make_hook("Layer3_0_Conv2_", 'forward'))
        self.layer3[1].conv1.register_forward_hook(self.make_hook("Layer3_1_Conv1_", 'forward'))
        self.layer3[1].conv2.register_forward_hook(self.make_hook("Layer3_1_Conv2_", 'forward'))
        
        self.layer4[0].conv1.register_forward_hook(self.make_hook("Layer4_0_Conv1_", 'forward'))
        self.layer4[0].downsample.register_forward_hook(self.make_hook("Layer4_0_Downsample_", 'forward'))
        self.layer4[0].conv2.register_forward_hook(self.make_hook("Layer4_0_Conv2_", 'forward'))
        self.layer4[1].conv1.register_forward_hook(self.make_hook("Layer4_1_Conv1_", 'forward'))
        self.layer4[1].conv2.register_forward_hook(self.make_hook("Layer4_1_Conv2_", 'forward'))
        
        
        
    '''def get_all_inner_activation(self, x):
        inner_output_index = [0,2,4,8,10,12,16,18]
        inner_output_list = []
        for i in range(23):
            x = self.classifier[i](x)
            if i in inner_output_index:
                inner_output_list.append(x)
        x = x.view(x.size(0), self.num_classes)
        return x,inner_output_list'''

#############################################################################
    def input_to_conv1(self, x):
        x = self.conv1(x)
        return x
        
    def conv1_to_output(self, x):
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#############################################################################
    def input_to_layer1_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1[0].input_to_residual(x)
        
        return x
        
    def layer1_0_residual_to_output(self, residual, conv2):
    
        x = self.layer1[0].residual_to_output(residual,conv2)
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer1_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0].input_to_conv2(x)
        return x
        
    def layer1_0_conv2_to_output(self, x, residual):
        x = self.layer1[0].conv2_to_output(x, residual)
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer1_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0].input_to_conv1(x)
        return x
        
    def layer1_0_conv1_to_output(self, x, residual):
        x = self.layer1[0].conv1_to_output(x, residual)
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#############################################################################

    def input_to_layer1_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0](x)
        x = self.layer1[1].input_to_residual(x)
        
        return x
        
    def input_to_layer1_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0](x)
        x = self.layer1[1].input_to_conv2(x)
        return x
        
    def layer1_1_conv2_to_output(self, x, residual):
        x = self.layer1[1].conv2_to_output(x, residual)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def layer1_1_conv2_to_output_mask(self, x, residual,mask,pattern):
        x = self.layer1[1].conv2_to_output_mask(x, residual,mask,pattern)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer1_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1[0](x)
        x = self.layer1[1].input_to_conv1(x)
        return x
        
    def layer1_1_conv1_to_output(self, x, residual):
        x = self.layer1[1].conv1_to_output(x, residual)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#############################################################################

#############################################################################
    def input_to_layer2_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2[0].input_to_residual(x)
        
        return x
        
    def layer2_0_residual_to_output(self, residual, conv2):
    
        x = self.layer2[0].residual_to_output(residual,conv2)
        x = self.layer2[1](x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer2_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0].input_to_conv2(x)
        return x
        
    def layer2_0_conv2_to_output(self, x, residual):
        x = self.layer2[0].conv2_to_output(x, residual)
        x = self.layer2[1](x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer2_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0].input_to_conv1(x)
        return x
        
    def layer2_0_conv1_to_output(self, x, residual):
        x = self.layer2[0].conv1_to_output(x, residual)
        x = self.layer2[1](x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#############################################################################

    def input_to_layer2_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2[0](x)
        x = self.layer2[1].input_to_residual(x)
        
        return x
        
    def input_to_layer2_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0](x)
        x = self.layer2[1].input_to_conv2(x)
        return x
        
    def layer2_1_conv2_to_output(self, x, residual):
        x = self.layer2[1].conv2_to_output(x, residual)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
        
    def layer2_1_conv2_to_output_mask(self, x, residual,mask,pattern):
        x = self.layer2[1].conv2_to_output_mask(x, residual,mask,pattern)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer2_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2[0](x)
        x = self.layer2[1].input_to_conv1(x)
        return x
        
    def layer2_1_conv1_to_output(self, x, residual):
        x = self.layer2[1].conv1_to_output(x, residual)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#############################################################################

#############################################################################
    def input_to_layer3_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0].input_to_residual(x)
        
        return x
        
    def layer3_0_residual_to_output(self, residual, conv2):
    
        x = self.layer3[0].residual_to_output(residual,conv2)
        x = self.layer3[1](x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer3_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0].input_to_conv2(x)
        return x
        
    def layer3_0_conv2_to_output(self, x, residual):
        x = self.layer3[0].conv2_to_output(x, residual)
        x = self.layer3[1](x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer3_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0].input_to_conv1(x)
        return x
        
    def layer3_0_conv1_to_output(self, x, residual):
        x = self.layer3[0].conv1_to_output(x, residual)
        x = self.layer3[1](x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#############################################################################

    def input_to_layer3_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0](x)
        x = self.layer3[1].input_to_residual(x)
        
        return x
        
    def input_to_layer3_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0](x)
        x = self.layer3[1].input_to_conv2(x)
        return x
        
    def layer3_1_conv2_to_output(self, x, residual):
        x = self.layer3[1].conv2_to_output(x, residual)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def layer3_1_conv2_to_output_mask(self, x, residual,mask,pattern):
        x = self.layer3[1].conv2_to_output_mask(x, residual,mask,pattern)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer3_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3[0](x)
        x = self.layer3[1].input_to_conv1(x)
        return x
        
    def layer3_1_conv1_to_output(self, x, residual):
        x = self.layer3[1].conv1_to_output(x, residual)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#############################################################################
    def input_to_layer4_0_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0].input_to_residual(x)
        
        return x
        
    def layer4_0_residual_to_output(self, residual, conv2):
    
        x = self.layer4[0].residual_to_output(residual,conv2)
        x = self.layer4[1](x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer4_0_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0].input_to_conv2(x)
        return x
        
    def layer4_0_conv2_to_output(self, x, residual):
        x = self.layer4[0].conv2_to_output(x, residual)
        x = self.layer4[1](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer4_0_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0].input_to_conv1(x)
        return x
        
    def layer4_0_conv1_to_output(self, x, residual):
        x = self.layer4[0].conv1_to_output(x, residual)
        x = self.layer4[1](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#############################################################################
    def input_to_layer4_1_residual(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x)
        x = self.layer4[1].input_to_residual(x)
        
        return x
        
    def input_to_layer4_1_conv2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x)
        x = self.layer4[1].input_to_conv2(x)
        return x
        
    def layer4_1_conv2_to_output(self, x, residual):
        x = self.layer4[1].conv2_to_output(x, residual)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def layer4_1_conv2_to_output_mask(self, x, residual,mask,pattern):
        x = self.layer4[1].conv2_to_output_mask(x, residual,mask,pattern)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def input_to_layer4_1_conv1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4[0](x)
        x = self.layer4[1].input_to_conv1(x)
        return x
        
    def layer4_1_conv1_to_output(self, x, residual):
        x = self.layer4[1].conv1_to_output(x, residual)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#############################################################################

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
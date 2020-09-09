'''

New for ResNeXt:

1. Wider bottleneck

2. Add group for conv2

'''
import torch.nn as nn
import torch
import math

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.PReLU = nn.PReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.PReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.PReLU(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4


    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.PReLU = nn.PReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.PReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.PReLU(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.PReLU(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),#in_ch、out_ch是通道数
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU()
            )
    def forward(self,x):
        return self.conv(x)

class ResNeXt(nn.Module):
    def __init__(self, block, layers, in_ch, out_ch, num_group=32):
        self.inplanes = 16
        super(ResNeXt, self).__init__()
        self.conv1 = DoubleConv(in_ch, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.PReLU = nn.PReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.layer1 = self._make_layer(block, 16, layers[0], num_group, stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], num_group, stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], num_group, stride=1)
        self.layer4 = self._make_layer(block, 128, layers[3], num_group, stride=1)
        self.conv5=DoubleConv(512, 512)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        #
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(256+512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(128+256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv10 = DoubleConv(64+128, 64)

        self.up11 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv12 = DoubleConv(32+64, 64)

        self.conv13 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))
        return nn.Sequential(*layers)



    def forward(self, x):
        c0 = self.conv1(x)
        # c0 = self.bn1(c0)
        # c0 = self.PReLU(c0)
        # pool0 = self.maxpool(c0)

        c1 = self.layer1(c0)
        pool1=nn.MaxPool2d(2)(c1)
        c2 = self.layer2(pool1)
        pool2 = nn.MaxPool2d(2)(c2)
        c3 = self.layer3(pool2)
        pool3 = nn.MaxPool2d(2)(c3)
        c4 = self.layer4(pool3)
        pool4 = nn.MaxPool2d(2)(c4)
        c5=self.conv5(pool4)

        up5 = self.up5(c5)
        merge1 = torch.cat([up5, c4], dim=1)
        c6 = self.conv6(merge1)

        up7 = self.up7(c6)
        merge2 = torch.cat([up7, c3], dim=1)
        c8 = self.conv8(merge2)

        up9 = self.up9(c8)
        merge3 = torch.cat([up9, c2], dim=1)
        c10 = self.conv10(merge3)

        up11 = self.up11(c10)
        merge4 = torch.cat([up11, c1], dim=1)
        c12 = self.conv12(merge4)

        c13=self.conv13(c12)

        out = nn.Sigmoid()(c13)

        return out


def resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
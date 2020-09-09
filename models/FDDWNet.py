# FDDWNET: A LIGHTWEIGHT CONVOLUTIONAL NEURAL NETWORK FOR REAL-TIME SEMANTIC SEGMENTATION
# Paper-Link: https://arxiv.org/pdf/1911.00632.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FDDWNet"]


class EERM(nn.Module):
    def __init__(self,
                 chann,
                 kernel_size=3,
                 stride=1,
                 dropprob=0.0,
                 dilated=1,
                 ):
        super(EERM, self).__init__()

        # defaultly inchannels = outchannels
        self.conv1_1 = nn.Conv2d(chann, chann, (kernel_size, 1), stride=(stride, 1),
                                 padding=(int((kernel_size - 1) / 2), 0), dilation=(1, 1),
                                 groups=chann, bias=True)
        self.conv1_1_bn = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv1_2 = nn.Conv2d(chann, chann, (1, kernel_size), stride=(1, stride),
                                 padding=(0, int((kernel_size - 1) / 2)), dilation=(1, 1),
                                 groups=chann, bias=True)
        self.conv1_2_bn = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv1 = nn.Conv2d(chann, chann, 1, padding=0, bias=True)
        self.conv1_bn = nn.BatchNorm2d(chann, eps=1e-3)
        # here is relu

        self.conv2_1 = nn.Conv2d(chann, chann, (kernel_size, 1), stride=(stride, 1),
                                 padding=(int((kernel_size - 1) / 2 * dilated), 0), dilation=(dilated, 1),
                                 groups=chann, bias=True)
        self.conv2_1_bn = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv2_2 = nn.Conv2d(chann, chann, (1, kernel_size), stride=(1, stride),
                                 padding=(0, int((kernel_size - 1) / 2 * dilated)), dilation=(1, dilated),
                                 groups=chann, bias=True)
        self.conv2_2_bn = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv2 = nn.Conv2d(chann, chann, 1, padding=0, bias=True)
        self.conv2_bn = nn.BatchNorm2d(chann, eps=1e-3)
        # self.drop2d = nn.Dropout2d(p=dropprob)
        self.dropout = nn.Dropout2d(dropprob)

        # here is relu
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        residual = x
        main = self.conv1_1(x)
        main = self.conv1_1_bn(main)
        # main = self.relu(main)
        main = self.conv1_2(main)
        main = self.conv1_2_bn(main)
        # main = self.relu(main)
        main = self.conv1(main)
        main = self.conv1_bn(main)
        main = self.relu(main)

        main = self.conv2_1(main)
        main = self.conv2_1_bn(main)
        # main = self.relu(main)
        main = self.conv2_2(main)
        main = self.conv2_2_bn(main)
        # main = self.relu(main)
        main = self.conv2(main)
        main = self.conv2_bn(main)

        if (self.dropout.p != 0):
            main = self.dropout(main)

        main = self.relu(main + residual)
        return main

        # return F.relu(torch.add(main, residual), inplace=True)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Net(nn.Module):
    def __init__(self, classes=19):
        super().__init__()

        # layer 1 - downsampling
        self.down_1 = DownsamplerBlock(3, 16)

        # layer 2 - downsampling
        self.down_2 = DownsamplerBlock(16, 64)

        # layers 3 to 7 - EERM
        self.FDDWC_1 = EERM(chann=64, dropprob=0.03, dilated=1)
        self.FDDWC_2 = EERM(chann=64, dropprob=0.03, dilated=1)
        self.FDDWC_3 = EERM(chann=64, dropprob=0.03, dilated=1)
        self.FDDWC_4 = EERM(chann=64, dropprob=0.03, dilated=1)
        self.FDDWC_5 = EERM(chann=64, dropprob=0.03, dilated=1)

        # layer 8 - downsampling
        self.down_3 = DownsamplerBlock(64, 128)

        # layer 9 to 16 - EERM
        self.FDDWC_6 = EERM(chann=128, dropprob=0.3, dilated=1)
        self.FDDWC_7 = EERM(chann=128, dropprob=0.3, dilated=2)
        self.FDDWC_8 = EERM(chann=128, dropprob=0.3, dilated=5)
        self.FDDWC_9 = EERM(chann=128, dropprob=0.3, dilated=9)
        self.FDDWC_6_1 = EERM(chann=128, dropprob=0.3, dilated=1)
        self.FDDWC_7_1 = EERM(chann=128, dropprob=0.3, dilated=2)
        self.FDDWC_8_1 = EERM(chann=128, dropprob=0.3, dilated=5)
        self.FDDWC_9_1 = EERM(chann=128, dropprob=0.3, dilated=9)

        # layer 17 to 24 - EERM
        self.FDDWC_10 = EERM(chann=128, dropprob=0.3, dilated=2)
        self.FDDWC_11 = EERM(chann=128, dropprob=0.3, dilated=5)
        self.FDDWC_12 = EERM(chann=128, dropprob=0.3, dilated=9)
        self.FDDWC_13 = EERM(chann=128, dropprob=0.3, dilated=17)
        self.FDDWC_10_1 = EERM(chann=128, dropprob=0.3, dilated=2)
        self.FDDWC_11_1 = EERM(chann=128, dropprob=0.3, dilated=5)
        self.FDDWC_12_1 = EERM(chann=128, dropprob=0.3, dilated=9)
        self.FDDWC_13_1 = EERM(chann=128, dropprob=0.3, dilated=17)

        # layer 25 - upsampling
        self.up_1 = UpsamplerBlock(128, 64)

        # layer 26 to 27 - EERM
        self.FDDWC_up_1 = EERM(chann=64, dropprob=0, dilated=1)
        self.FDDWC_up_2 = EERM(chann=64, dropprob=0, dilated=1)

        # layer 28 - upsampling
        self.up_2 = UpsamplerBlock(64, 16)

        # layer 29 to 30 - EERM
        self.FDDWC_up_3 = EERM(chann=16, dropprob=0, dilated=1)
        self.FDDWC_up_4 = EERM(chann=16, dropprob=0, dilated=1)

        # another branch
        self.up_0 = UpsamplerBlock(64, 16)
        self.FDDWC_up_low_2 = EERM(chann=64, dropprob=0, dilated=1)
        self.FDDWC_up_low_3 = EERM(chann=16, dropprob=0, dilated=1)

        self.relu = nn.ReLU6(inplace=True)

        # layer 31 - upsampling
        self.output_conv = nn.ConvTranspose2d(16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.down_1(input)
        output = self.down_2(output)

        output = self.FDDWC_1(output)
        output = self.FDDWC_2(output)
        output = self.FDDWC_3(output)
        output = self.FDDWC_4(output)
        output_0 = self.FDDWC_5(output)
        #
        branch_1 = self.down_3(output_0)
        branch_1 = self.FDDWC_6(branch_1)
        branch_1 = self.FDDWC_7(branch_1)
        branch_1 = self.FDDWC_8(branch_1)
        branch_1 = self.FDDWC_9(branch_1)
        branch_1 = self.FDDWC_6_1(branch_1)
        branch_1 = self.FDDWC_7_1(branch_1)
        branch_1 = self.FDDWC_8_1(branch_1)
        branch_1 = self.FDDWC_9_1(branch_1)

        branch_1 = self.FDDWC_10(branch_1)
        branch_1 = self.FDDWC_11(branch_1)
        branch_1 = self.FDDWC_12(branch_1)
        branch_1 = self.FDDWC_13(branch_1)
        branch_1 = self.FDDWC_10_1(branch_1)
        branch_1 = self.FDDWC_11_1(branch_1)
        branch_1 = self.FDDWC_12_1(branch_1)
        output_1 = self.FDDWC_13_1(branch_1)

        output_1 = self.up_1(output_1)

        output_2 = self.FDDWC_up_low_2(output_0)

        output = self.relu(output_1 + output_2)

        output = self.FDDWC_up_1(output)
        output = self.FDDWC_up_2(output)

        output = self.up_2(output)

        output_3 = self.up_0(output_0)
        output_3 = self.FDDWC_up_low_3(output_3)

        output = self.relu(output + output_3)
        output = self.FDDWC_up_3(output)
        output = self.FDDWC_up_4(output)

        output = self.output_conv(output)

        return nn.Sigmoid()(output)
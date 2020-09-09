import torch.nn as nn
import torch
from models.nonelocallib.non_local_embedded_gaussian import NONLocalBlock2D
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        out_ch_=out_ch//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch, (1, 3), dilation=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()

            )
    def forward(self,x):
        return self.conv(x)

class NeckConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(NeckConv,self).__init__()
        out_ch_=in_ch//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch, (1, 3), dilation=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            )
    def forward(self,x):
        return self.conv(x)

class SingleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SingleConv,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,dilation=2,padding=2),#in_ch、out_ch是通道数
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
    def forward(self,x):
        return self.conv(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False).cuda(),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False).cuda(),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Downsample(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Downsample, self).__init__()
        # out_ch_=out_ch//2
        # self.conv0_0 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch_, (2,2), stride=2, padding=(0,0)),
        #     nn.BatchNorm2d(out_ch_),
        #     nn.PPReLU(num_parameters=1, init=0.25)
        # )
        self.conv0_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch-in_ch, (3, 3), stride=2, dilation=2,padding=(2,2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # self.conv0_2 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch_, (2, 2), stride=2, dilation=2,padding=(1, 1)),
        #     nn.BatchNorm2d(out_ch_),
        #     nn.PPReLU(num_parameters=1, init=0.25)
        # )
        self.maxpool0_3 = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.ext_regul = nn.BatchNorm2d(out_ch)
        self.activation=nn.ReLU()

    def forward(self, x):
        # conv0_0=self.conv0_0(x)
        conv0_1 = self.conv0_1(x)
        # conv0_2 = self.conv0_2(x)
        conv0_3 = self.maxpool0_3(x)

        conv0=torch.cat([conv0_1,conv0_3],dim=1)
        # conv0=conv0_1+conv0_3
        conv0=self.ext_regul(conv0)

        return self.activation(conv0)

class MultiscaleSEResblock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSEResblock,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        out_ch_=out_ch//3

        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU()
        )
        self.conv3_3=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(3,1),dilation=(2,1),padding=(2,0)),#in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(1,3),dilation=(1,2),padding=(0,2)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3, 3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (5, 1), dilation=(2,1),padding=(4, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 5), dilation=(1,2),padding=(0,4)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3,3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        # self.convdilated = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch_, (3,3),dilation=(2,2), padding=(2,2)),  # in_ch、out_ch是通道数
        #     nn.BatchNorm2d(out_ch_),
        #     nn.PPReLU(num_parameters=1, init=0.25)
        # )
        self.conv7_7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (7, 1), dilation=(2,1),padding=(6, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 7), dilation=(1,2),padding=(0,6)),
            nn.BatchNorm2d(out_ch_),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch)
            )
        self.ReLU = nn.ReLU()
        self.conv0 = nn.Conv2d(in_ch, in_ch, (1, 1), padding=(0, 0))
        self.batchnorm0 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        c1_ch1 = self.conv1_1(x)
        c1_ch2=self.conv3_3(x)
        c1_ch3 = self.conv5_5(x)
        # c1_ch4 = self.conv7_7(x)

        c1_concat=torch.cat([c1_ch1,c1_ch2,c1_ch3],dim=1)#按维数1（列）拼接,列增加
        c1_concat=self.batchnorm0(c1_concat)
        # c1_concat += self.extra(x)
        c1_SE=SELayer(self.out_ch, reduction=16)(c1_concat)
        out=c1_SE+c1_concat
        return c1_concat

class MultiscaleSE_block_A(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSE_block_A,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        out_ch_=out_ch//2

        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU()
        )
        self.conv3_3=nn.Sequential(
            # nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(3,1),dilation=(2,1),padding=(2,0)),#in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(1,3),dilation=(1,2),padding=(0,2)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3, 3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (5, 1), dilation=(2,1),padding=(4, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 5), dilation=(1,2),padding=(0,4)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3,3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        # self.convdilated = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch_, (3,3),dilation=(2,2), padding=(2,2)),  # in_ch、out_ch是通道数
        #     nn.BatchNorm2d(out_ch_),
        #     nn.PPReLU(num_parameters=1, init=0.25)
        # )
        self.conv7_7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (7, 1), dilation=(2,1),padding=(6, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 7), dilation=(1,2),padding=(0,6)),
            nn.BatchNorm2d(out_ch_),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch)
            )
        self.ReLU = nn.ReLU()
        self.conv0 = nn.Conv2d(in_ch, in_ch, (1, 1), padding=(0, 0))
        self.batchnorm0 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        c1_ch1 = self.conv1_1(x)
        c1_ch2=self.conv3_3(c1_ch1)
        # c1_ch3 = self.conv5_5(x)
        # c1_ch4 = self.conv7_7(x)

        c1_concat=torch.cat([c1_ch1,c1_ch2],dim=1)#按维数1（列）拼接,列增加
        c1_concat=self.batchnorm0(c1_concat)
        # c1_concat += self.extra(x)
        c1_SE=SELayer(self.out_ch, reduction=16)(c1_concat)
        out=c1_SE+c1_concat
        return c1_concat

class MultiscaleSE_block_B(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Multi2scaleSE_block_B,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        out_ch_=out_ch//2

        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU()
        )
        self.conv3_3=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(3,1),dilation=(2,1),padding=(2,0)),#in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(1,3),dilation=(1,2),padding=(0,2)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3, 3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (5, 1), dilation=(2,1),padding=(4, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 5), dilation=(1,2),padding=(0,4)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3,3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        # self.convdilated = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch_, (3,3),dilation=(2,2), padding=(2,2)),  # in_ch、out_ch是通道数
        #     nn.BatchNorm2d(out_ch_),
        #     nn.PPReLU(num_parameters=1, init=0.25)
        # )
        self.conv7_7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (7, 1), dilation=(2,1),padding=(6, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 7), dilation=(1,2),padding=(0,6)),
            nn.BatchNorm2d(out_ch_),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch)
            )
        self.ReLU = nn.ReLU()
        self.conv0 = nn.Conv2d(in_ch, in_ch, (1, 1), padding=(0, 0))
        self.batchnorm0 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        c1_ch1 = self.conv1_1(x)
        c1_ch2=self.conv3_3(c1_ch1)
        # c1_ch3 = self.conv5_5(x)
        # c1_ch4 = self.conv7_7(x)

        c1_concat=torch.cat([c1_ch1,c1_ch2],dim=1)#按维数1（列）拼接,列增加
        c1_concat=self.batchnorm0(c1_concat)
        # c1_concat += self.extra(x)
        c1_SE=SELayer(self.out_ch, reduction=16)(c1_concat)
        out=c1_SE+c1_concat
        return c1_concat

class MultiscaleSE_block_C(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSE_block_C,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        out_ch_=out_ch//3

        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU()
        )
        self.conv3_3=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(3,1),dilation=(2,1),padding=(2,0)),#in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(1,3),dilation=(1,2),padding=(0,2)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3, 3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (5, 1), dilation=(2,1),padding=(4, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 5), dilation=(1,2),padding=(0,4)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3,3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        # self.convdilated = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch_, (3,3),dilation=(2,2), padding=(2,2)),  # in_ch、out_ch是通道数
        #     nn.BatchNorm2d(out_ch_),
        #     nn.PPReLU(num_parameters=1, init=0.25)
        # )
        self.conv7_7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (7, 1), dilation=(2,1),padding=(6, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 7), dilation=(1,2),padding=(0,6)),
            nn.BatchNorm2d(out_ch_),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch)
            )
        self.ReLU = nn.ReLU()
        self.conv0 = nn.Conv2d(in_ch, in_ch, (1, 1), padding=(0, 0))
        self.batchnorm0 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        c1_ch1 = self.conv1_1(x)
        c1_ch2=self.conv3_3(x)
        c1_ch3 = self.conv5_5(x)
        # c1_ch4 = self.conv7_7(x)

        c1_concat=torch.cat([c1_ch1,c1_ch2,c1_ch3],dim=1)#按维数1（列）拼接,列增加
        c1_concat=self.batchnorm0(c1_concat)
        # c1_concat += self.extra(x)
        c1_SE=SELayer(self.out_ch, reduction=16)(c1_concat)
        out=c1_SE+c1_concat
        return c1_concat

class MultiscaleResblock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleResblock,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        out_ch_=out_ch//3

        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU()
        )
        self.conv3_3=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(3,1),padding=(1,0)),#in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_,out_ch_,(1,3),padding=(0,1)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),
            # nn.Conv2d(out_ch_, out_ch_, (3, 3), padding=(1,1)),  # in_ch、out_ch是通道数
            # nn.BatchNorm2d(out_ch_),
            # nn.PPReLU(num_parameters=1, init=0.25)
        )
        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (5, 1),padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (1, 5),padding=(0,2)),
            nn.BatchNorm2d(out_ch_),
            # nn.ReLU(),

        )
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch)
            )

        self.ReLU = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        c1_ch1 = self.conv1_1(x)
        c1_ch2=self.conv3_3(x)
        c1_ch3 = self.conv5_5(x)
        # c1_ch4 = self.conv7_7(x)

        c1_concat=torch.cat([c1_ch1,c1_ch2,c1_ch3],dim=1)#按维数1（列）拼接,列增加
        # c1_concat=self.conv0(c1_concat)
        c1_concat=self.batchnorm0(c1_concat)
        out = self.extra(x) + c1_concat

        return c1_concat


class MultiscaleInception2SE(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleInception2SE,self).__init__()
        self.in_ch=in_ch
        self.out_ch=out_ch
        out_ch_=in_ch//3
        self.conv1_1=nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0))

        self.convInception2_0=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.PReLU(inplace=True),
            nn.Conv2d(out_ch_, out_ch_, (3, 3), padding=(1, 1)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.PReLU(inplace=True),
        )

        self.convInception2_1=nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.PReLU(inplace=True),
        )
        self.conv1_3=nn.Sequential(
            nn.Conv2d(out_ch_, out_ch_//2, (1, 3), padding=(0, 1)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_//2),
            nn.PReLU(inplace=True),
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(out_ch_, out_ch_//2, (3, 1), padding=(1,0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_//2),
            nn.PReLU(inplace=True),
        )

        self.PReLU = nn.PReLU(inplace=True)
        self.conv=nn.Conv2d(in_ch,out_ch, (1, 1), padding=(0, 0))

    def forward(self, x):
        ch0 = self.convInception2_0(x)
        ch0_0=self.conv1_3(ch0)
        ch0_1 = self.conv3_1(ch0)
        ch0=torch.cat([ch0_0,ch0_1],dim=1)

        ch1=self.convInception2_1(x)
        ch1_0 = self.conv1_3(ch1)
        ch1_1 = self.conv3_1(ch1)
        ch1 = torch.cat([ch1_0, ch1_1], dim=1)

        ch2=self.conv1_1(x)
        c1_concat=torch.cat([ch0,ch1,ch2],dim=1)#按维数1（列）拼接,列增加

        c1_concat += x
        c1_concat = self.PReLU(c1_concat)
        c1_concat=self.conv(c1_concat)
        # c1_concat_pool=nn.MaxPool2d(2)(c1_concat)
        c1_SE=SELayer(self.out_ch, reduction=16)(c1_concat)
        out=c1_SE+c1_concat
        # out = nn.Sigmoid()(c1_SE)  # 化成(0~1)区间
        return out
class Conv3_3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Conv3_3,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch,int(out_ch/2),(3,1),padding=(1,0)),#in_ch、out_ch是通道数
                nn.BatchNorm2d(int(out_ch/2)),
                nn.ReLU(),
                nn.Conv2d(int(out_ch/2), int(out_ch / 2), (1,3), padding=(0,1)),  # in_ch、out_ch是通道数
                nn.BatchNorm2d(int(out_ch / 2)),
                nn.ReLU()

            )
    def forward(self,x):
        return self.conv(x)


class Conv5_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv5_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, int(out_ch / 2), (5, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(int(out_ch / 2)),
            nn.ReLU(),
            nn.Conv2d(int(out_ch / 2), int(out_ch / 2), (1, 5), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(int(out_ch / 2)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, dilation=2,padding=2)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, dilation=2,padding=2)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        x:[b, ch, h, w]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add: [b, ch_in, h, w] with [b, ch_out, h, w]
        out = self.extra(x) + out

        return out
# using MultiscaleSE
class MultiscaleSENet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENet, self).__init__()
        self.conv0_intial=SingleConv(in_ch,36)
        self.downsampling0=Downsample(18,18)
        # self.conv0_0= nn.Conv2d(12, 24,(3,3),stride=2,padding=(1,1))
        # self.maxpool0_1 = nn.MaxPool2d(2)
        # # Initialize batch normalization to be used after concatenation
        # self.batch_norm = nn.BatchNorm2d(36)

        # PPReLU layer to apply after concatenating the branches



        # self.convinitial = DoubleConv(3, 36)
        # self.conv0 = MultiscaleSE(3, 36)  # 128
        self.conv1=MultiscaleSEResblock(36,72) #128
        # self.conv1_1 = MultiscaleSE(72, 72)
        self.downsampling1 = Downsample(36, 36)
        self.conv2 = MultiscaleSEResblock(72, 144) #64
        # self.conv2_1 = MultiscaleSE(144, 144)
        self.downsampling2 = Downsample(72, 72)
        self.conv3 = MultiscaleSEResblock(144, 288)  #32
        # self.conv3_1 = MultiscaleSE(288, 288)
        self.downsampling3 = Downsample(144, 144)
        self.conv4 = MultiscaleSEResblock(288, 576)  #16

        self.nonelocal4 = NONLocalBlock2D(576)

        # self.conv4_1 = MultiscaleSE(576, 576)
        self.conv4_1 = Conv3_3(576, 576)
        self.conv4_2 = Conv5_5(576, 576)

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(576, 288, 2, stride=2),
            nn.BatchNorm2d(288),
            nn.ReLU()
        )
        self.res3to5 = MultiscaleResblock(288, 288)
        self.conv6=DoubleConv(576,288)

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(288, 144, 2, stride=2),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        self.res2to6 = MultiscaleResblock(144, 144)
        self.conv8 = DoubleConv(288, 144)

        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(144, 72, 2, stride=2),
            nn.BatchNorm2d(72),
            nn.ReLU()
        )
        self.res1to7 = MultiscaleResblock(72, 72)
        self.conv10 = DoubleConv(144, 72)

        self.up11 = nn.Sequential(
            nn.ConvTranspose2d(72, 36, 2, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.res0to8 = MultiscaleResblock(36, 36)
        self.up11_32_256=nn.Sequential(
            # nn.Conv2d(288,12,1),
            # nn.BatchNorm2d(12),
            # nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=8),
            nn.Conv2d(288, 12, 1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        self.up11_64_256 = nn.Sequential(
            # nn.Conv2d(144, 12, 1),
            # nn.BatchNorm2d(12),
            # nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(144, 12, 1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        self.up11_128_256 = nn.Sequential(
            # nn.Conv2d(72, 12, 1),
            # nn.BatchNorm2d(12),
            # nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(72, 12, 1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        self.normalization = nn.BatchNorm2d(36)
        self.out_activation = nn.ReLU()
        self.conv12 = DoubleConv(72, 64)
        self.conv13 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1)
        self.maxpool=nn.MaxPool2d(2)

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        # c0_0=self.conv0_0(c0_initial)
        # c0_1 = self.maxpool0_1(c0_initial)
        # c0=torch.cat([c0_0,c0_1],dim=1)
        # c0= self.batch_norm(c0)
        # c0=self.out_activation(c0)
        c0=self.maxpool(c0_initial)

        # pool0=nn.MaxPool2d(2)(c0)
        # cinitial=self.convinitial(x)
        # c0=self.conv0(x)
        # pool0 = nn.MaxPool2d(2)(c0)
        c1=self.conv1(c0)
        pool1=self.maxpool(c1)
        c2 = self.conv2(pool1)
        pool2 = self.maxpool(c2)
        c3 = self.conv3(pool2)
        pool3 = self.maxpool(c3)
        c4 = self.conv4(pool3)
        # c4=self.nonelocal4(c4)

        up5=self.up5(c4)
        c3=self.res3to5(c3)
        merge1=torch.cat([up5,c3],dim=1)
        c6=self.conv6(merge1)

        up7=self.up7(c6)
        c2=self.res2to6(c2)
        merge2=torch.cat([up7,c2],dim=1)
        c8=self.conv8(merge2)

        up9 = self.up9(c8)
        c1=self.res1to7(c1)
        merge3 = torch.cat([up9, c1], dim=1)
        c10 = self.conv10(merge3)

        up11 = self.up11(c10)
        c0=self.res0to8(c0_initial)
        # up11_32_256=self.up11_32_256(c3)
        # up11_64_256 = self.up11_64_256(c2)
        # up11_128_256 = self.up11_128_256(c1)
        # up_addition=torch.cat([up11_32_256,up11_64_256,up11_128_256], dim=1)
        # up_addition=self.normalization(up_addition)
        # up_addition=self.out_activation(up_addition)
        # merge4 = torch.cat([up11, c0_initial,up_addition], dim=1)
        merge4 = torch.cat([up11, c0], dim=1)
        c12 = self.conv12(merge4)

        c13 = self.conv13(c12)

        out = nn.Sigmoid()(c13)

        return out

# using MultiscaleSENew
class MultiscaleSENetNew(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENetNew, self).__init__()
        self.conv0_intial=DoubleConv(in_ch,36)

        self.conv1=MultiscaleSEResblock(36,72) #128
        self.conv2 = MultiscaleSEResblock(72, 144) #64
        self.conv3 = MultiscaleSEResblock(144, 288)  #32
        self.conv4 = MultiscaleSEResblock(288, 576)  #16

        self.nonelocal4 = NONLocalBlock2D(576)

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(576, 288, 2, stride=2),
            nn.BatchNorm2d(288),
            nn.ReLU()
        )
        self.res3to5 =MultiscaleResblock(288, 288)
        self.conv6=DoubleConv(576,288)

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(288, 144, 2, stride=2),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        self.res2to6=MultiscaleResblock(144, 72)
        self.res3to6 = nn.Sequential(
            nn.ConvTranspose2d(288, 72, 2, stride=2),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 72, 1),
            nn.BatchNorm2d(72),
            nn.ReLU()
        )
        self.conv8 = DoubleConv(288, 144)

        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(144, 72, 2, stride=2),
            nn.BatchNorm2d(72),
            nn.ReLU()
        )
        self.res1to7 = MultiscaleResblock(72, 36)
        self.res2to7 = nn.Sequential(
            nn.ConvTranspose2d(144, 36, 2, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 36, 1),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.conv10 = DoubleConv(144, 72)

        self.up11 = nn.Sequential(
            nn.ConvTranspose2d(72, 36, 2, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.res0to8 = MultiscaleResblock(36, 18)
        self.res1to8 = nn.Sequential(
            nn.ConvTranspose2d(72, 18, 2, stride=2),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.Conv2d(18, 18, 1),
            nn.BatchNorm2d(18),
            nn.ReLU()
        )

        self.normalization = nn.BatchNorm2d(36)
        self.out_activation = nn.ReLU()
        self.conv12 = DoubleConv(72, 64)
        self.conv13 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1)
        self.maxpool=nn.MaxPool2d(2)

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        c0=self.maxpool(c0_initial)

        c1=self.conv1(c0)
        pool1=self.maxpool(c1)
        c2 = self.conv2(pool1)
        pool2 = self.maxpool(c2)
        c3 = self.conv3(pool2)
        pool3 = self.maxpool(c3)
        c4 = self.conv4(pool3)

        # attentionc4=self.nonelocal4(c4)

        up5=self.up5(c4)
        c5_1=self.res3to5(c3)
        merge1=torch.cat([up5,c5_1],dim=1)
        c6=self.conv6(merge1)

        up7=self.up7(c6)
        c7_1=self.res3to6(c3)
        c7_2=self.res2to6(c2)
        merge2=torch.cat([up7,c7_1,c7_2],dim=1)
        c8=self.conv8(merge2)

        up9 = self.up9(c8)
        c9_1=self.res2to7(c2)
        c9_2=self.res1to7(c1)
        merge3 = torch.cat([up9, c9_1,c9_2], dim=1)
        c10 = self.conv10(merge3)

        up11 = self.up11(c10)
        c11_1=self.res1to8(c1)
        c11_2=self.res0to8(c0_initial)
        merge4 = torch.cat([up11, c11_1,c11_2], dim=1)
        c12 = self.conv12(merge4)

        c13 = self.conv13(c12)

        out = nn.Sigmoid()(c13)

        return out

# using MultiscaleSE
class MultiscaleSENetNew2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENetNew2, self).__init__()
        self.conv0_intial=SingleConv(in_ch,36)

        self.conv1=MultiscaleSEResblock(36,72) #128
        self.conv2 = MultiscaleSEResblock(72, 144) #64
        self.conv3 = MultiscaleSEResblock(144, 288)  #32
        self.conv4 = MultiscaleSEResblock(288, 576)  #16

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(576, 288, 2, stride=2),
            nn.BatchNorm2d(288),
            nn.ReLU()
        )
        self.res3to5 = MultiscaleResblock(288, 288)
        self.conv6=DoubleConv(576,288)

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(288, 144, 2, stride=2),
            nn.BatchNorm2d(144),
            nn.ReLU()
        )
        self.res2to6 = MultiscaleResblock(144, 144)
        self.conv8 = DoubleConv(288, 144)

        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(144, 72, 2, stride=2),
            nn.BatchNorm2d(72),
            nn.ReLU()
        )
        self.res1to7 = MultiscaleResblock(72, 72)
        self.conv10 = DoubleConv(144, 72)

        self.up11 = nn.Sequential(
            nn.ConvTranspose2d(72, 36, 2, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.res0to8 = MultiscaleResblock(36, 36)

        self.normalization = nn.BatchNorm2d(36)
        self.out_activation = nn.ReLU()
        self.conv12 = DoubleConv(72, 64)
        self.conv13 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1)
        self.maxpool=nn.MaxPool2d(2)
        self.conv4to1 = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=16),
                    nn.Conv2d(576, 72, 3, padding=1),  # in_ch、out_ch是通道数
                    nn.BatchNorm2d(72),
                    nn.ReLU(),
                    nn.Conv2d(72, 1,1),
                    nn.BatchNorm2d(1),
                    nn.ReLU()
                )
        self.conv6to1 = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=8),
                    nn.Conv2d(288, 36, 3, padding=1),  # in_ch、out_ch是通道数
                    nn.BatchNorm2d(36),
                    nn.ReLU(),
                    nn.Conv2d(36, 1, 1),
                    nn.BatchNorm2d(1),
                    nn.ReLU()
                )
        self.conv8to1 = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(144, 18, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(18),
            nn.ReLU(),
            # nn.Conv2d(36, 1, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU()
                )
        self.conv10to1 = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(72, 18, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(18),
            nn.ReLU(),
            # nn.Conv2d(18, 1, 1),
            # nn.BatchNorm2d(1),
            # nn.ReLU()
                )
        self.activate=nn.ReLU()

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        c0=self.maxpool(c0_initial)

        c1=self.conv1(c0)
        pool1=self.maxpool(c1)
        c2 = self.conv2(pool1)
        pool2 = self.maxpool(c2)
        c3 = self.conv3(pool2)
        pool3 = self.maxpool(c3)
        c4 = self.conv4(pool3)

        up5=self.up5(c4)
        c3=self.res3to5(c3)
        merge1=torch.cat([up5,c3],dim=1)
        c6=self.conv6(merge1)

        up7=self.up7(c6)
        c2=self.res2to6(c2)
        merge2=torch.cat([up7,c2],dim=1)
        c8=self.conv8(merge2)

        up9 = self.up9(c8)
        c1=self.res1to7(c1)
        merge3 = torch.cat([up9, c1], dim=1)
        c10 = self.conv10(merge3)

        up11 = self.up11(c10)
        c0=self.res0to8(c0_initial)
        conv8to1 = self.conv8to1(c8)
        conv10to1 = self.conv10to1(c10)
        aa=torch.cat([conv8to1,conv10to1],dim=1)
        up11=up11+aa
        merge4 = torch.cat([up11, c0], dim=1)
        c12 = self.conv12(merge4)

        c13 = self.conv13(c12)

        out = nn.Sigmoid()(c13)
        # conv4to1= torch.abs(self.conv4to1(c4))
        # conv6to1 = torch.abs(self.conv6to1(c6))
        # conv8to1 = torch.abs(self.conv8to1(c8))
        # conv10to1 = torch.abs(self.conv10to1(c10))
        #
        # # out_anti_1=conv4to1.mul(out)
        # # out_anti_2 = conv6to1.mul(out)
        # out_anti_3 = conv8to1.mul(out)
        # out_anti_4 = conv10to1.mul(out)
        # out=(out_anti_3+out_anti_4+out)/3.0

        return out


        # return conv8to1,conv10to1,torch.pow(out,2)

# using MultiscaleSENew
class MultiscaleSENetA(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENetA, self).__init__()
        self.conv0_intial=DoubleConv(in_ch,24)

        self.conv1=MultiscaleSE_block_A(24,48) #128
        self.conv2 = MultiscaleSE_block_A(48, 96) #64
        self.conv3 = MultiscaleSE_block_A(96, 192)  #32

        # self.nonelocal4 = NONLocalBlock2D(288)

        self.up5 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.res2to4 =MultiscaleResblock(96, 96)
        self.conv4=DoubleConv(96,96)

        self.up7 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.res2to5 = nn.Sequential(
            nn.Conv2d(96,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv5 = DoubleConv(48, 48)

        self.up9 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.res1to6 = nn.Sequential(
            nn.Conv2d(48, 24,1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.conv6 = NeckConv(24, 24)


        # self.normalization = nn.BatchNorm2d(36)
        # self.out_activation = nn.ReLU()
        self.conv7 = nn.Conv2d(24, out_ch, kernel_size=1, stride=1)
        # self.maxpool=nn.MaxPool2d(2)

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        c0=F.interpolate(c0_initial,scale_factor=0.25,mode="bilinear")

        c1=self.conv1(c0)
        pool1=F.interpolate(c1,scale_factor=0.5,mode="bilinear")
        c2 = self.conv2(pool1)
        pool2 = F.interpolate(c2,scale_factor=0.5,mode="bilinear")
        c3 = self.conv3(pool2)

        # attentionc4=self.nonelocal4(c4)

        up5=self.up5(F.interpolate(c3,scale_factor=2,mode="bilinear"))
        # c5_1=self.res2to4(c2)
        # merge1=torch.cat([up5,c5_1],dim=1)
        c6=self.conv4(up5)

        up7=self.up7(F.interpolate(c6,scale_factor=2,mode="bilinear"))
        c7_1=self.res2to5(F.interpolate(c2,scale_factor=2,mode="bilinear"))
        merge2=up7+c7_1
        # merge2=torch.cat([up7,c7_1],dim=1)
        c8=self.conv5(merge2)

        up9 = self.up9(F.interpolate(c8,scale_factor=4,mode="bilinear"))
        c9_1=self.res1to6(F.interpolate(c1,scale_factor=4,mode="bilinear"))
        merge3=up9+c9_1
        # merge3 = torch.cat([up9, c9_1], dim=1)
        c10 = self.conv6(merge3)


        c11 = self.conv7(c10)

        out = nn.Sigmoid()(c11)

        return out

# using MultiscaleSENew
class MultiscaleSENetB(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENetB, self).__init__()
        self.conv0_intial=DoubleConv(in_ch,24)

        self.conv1=MultiscaleSE_block_B(24,48) #128
        self.conv2 = MultiscaleSE_block_B(48, 96) #64
        self.conv3 = MultiscaleSE_block_B(96, 192)  #32

        # self.nonelocal4 = NONLocalBlock2D(288)

        self.up5 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.res2to4 =MultiscaleResblock(96, 96)
        self.conv4=DoubleConv(96,96)

        self.up7 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.res2to5 = nn.Sequential(
            nn.Conv2d(96,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv5 = DoubleConv(48, 48)

        self.up9 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.res1to6 = nn.Sequential(
            nn.Conv2d(48, 24,1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.conv6 = NeckConv(24, 24)


        # self.normalization = nn.BatchNorm2d(36)
        # self.out_activation = nn.ReLU()
        self.conv7 = nn.Conv2d(24, out_ch, kernel_size=1, stride=1)
        # self.maxpool=nn.MaxPool2d(2)

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        c0=F.interpolate(c0_initial,scale_factor=0.25,mode="bilinear")

        c1=self.conv1(c0)
        pool1=F.interpolate(c1,scale_factor=0.5,mode="bilinear")
        c2 = self.conv2(pool1)
        pool2 = F.interpolate(c2,scale_factor=0.5,mode="bilinear")
        c3 = self.conv3(pool2)

        # attentionc4=self.nonelocal4(c4)

        up5=self.up5(F.interpolate(c3,scale_factor=2,mode="bilinear"))
        # c5_1=self.res2to4(c2)
        # merge1=torch.cat([up5,c5_1],dim=1)
        c6=self.conv4(up5)

        up7=self.up7(F.interpolate(c6,scale_factor=2,mode="bilinear"))
        c7_1=self.res2to5(F.interpolate(c2,scale_factor=2,mode="bilinear"))
        merge2=up7+c7_1
        # merge2=torch.cat([up7,c7_1],dim=1)
        c8=self.conv5(merge2)

        up9 = self.up9(F.interpolate(c8,scale_factor=4,mode="bilinear"))
        c9_1=self.res1to6(F.interpolate(c1,scale_factor=4,mode="bilinear"))
        merge3=up9+c9_1
        # merge3 = torch.cat([up9, c9_1], dim=1)
        c10 = self.conv6(merge3)


        c11 = self.conv7(c10)

        out = nn.Sigmoid()(c11)

        return out

# using MultiscaleSENew
class MultiscaleSENetC(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENetC, self).__init__()
        self.conv0_intial=DoubleConv(in_ch,24)

        self.conv1=MultiscaleSE_block_C(24,48) #128
        self.conv2 = MultiscaleSE_block_C(48, 96) #64
        self.conv3 = MultiscaleSE_block_C(96, 192)  #32

        # self.nonelocal4 = NONLocalBlock2D(288)

        self.up5 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.res2to4 =MultiscaleResblock(96, 96)
        self.conv4=DoubleConv(96,96)

        self.up7 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.res2to5 = nn.Sequential(
            nn.Conv2d(96,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv5 = DoubleConv(48, 48)

        self.up9 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.res1to6 = nn.Sequential(
            nn.Conv2d(48, 24,1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.conv6 = NeckConv(24, 24)


        # self.normalization = nn.BatchNorm2d(36)
        # self.out_activation = nn.ReLU()
        self.conv7 = nn.Conv2d(24, out_ch, kernel_size=1, stride=1)
        # self.maxpool=nn.MaxPool2d(2)

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        c0=F.interpolate(c0_initial,scale_factor=0.25,mode="bilinear")

        c1=self.conv1(c0)
        pool1=F.interpolate(c1,scale_factor=0.5,mode="bilinear")
        c2 = self.conv2(pool1)
        pool2 = F.interpolate(c2,scale_factor=0.5,mode="bilinear")
        c3 = self.conv3(pool2)

        # attentionc4=self.nonelocal4(c4)

        up5=self.up5(F.interpolate(c3,scale_factor=2,mode="bilinear"))
        # c5_1=self.res2to4(c2)
        # merge1=torch.cat([up5,c5_1],dim=1)
        c6=self.conv4(up5)

        up7=self.up7(F.interpolate(c6,scale_factor=2,mode="bilinear"))
        c7_1=self.res2to5(F.interpolate(c2,scale_factor=2,mode="bilinear"))
        merge2=up7+c7_1
        # merge2=torch.cat([up7,c7_1],dim=1)
        c8=self.conv5(merge2)

        up9 = self.up9(F.interpolate(c8,scale_factor=4,mode="bilinear"))
        c9_1=self.res1to6(F.interpolate(c1,scale_factor=4,mode="bilinear"))
        merge3=up9+c9_1
        # merge3 = torch.cat([up9, c9_1], dim=1)
        c10 = self.conv6(merge3)


        c11 = self.conv7(c10)

        out = nn.Sigmoid()(c11)

        return out


# using MultiscaleSENew
class MultiscaleSENetA_direct_skip(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MultiscaleSENetA_direct_skip, self).__init__()
        self.conv0_intial=DoubleConv(in_ch,24)

        self.conv1=MultiscaleSE_block_A(24,48) #128
        self.conv2 = MultiscaleSE_block_A(48, 96) #64
        self.conv3 = MultiscaleSE_block_A(96, 192)  #32

        # self.nonelocal4 = NONLocalBlock2D(288)

        self.up5 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.res2to4 = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.conv4=DoubleConv(96,96)

        self.up7 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.res1to5 = nn.Sequential(
            nn.Conv2d(48,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv5 = DoubleConv(48, 48)

        self.up9 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.res0to6 = nn.Sequential(
            nn.Conv2d(24, 24,1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.conv6 = NeckConv(24, 24)


        # self.normalization = nn.BatchNorm2d(36)
        # self.out_activation = nn.ReLU()
        self.conv7 = nn.Conv2d(24, out_ch, kernel_size=1, stride=1)
        # self.maxpool=nn.MaxPool2d(2)

    def forward(self,x):
        c0_initial=self.conv0_intial(x)
        c0_0=F.interpolate(c0_initial,scale_factor=0.25,mode="bilinear")

        c1=self.conv1(c0_0)
        pool1=F.interpolate(c1,scale_factor=0.5,mode="bilinear")
        c2 = self.conv2(pool1)
        pool2 = F.interpolate(c2,scale_factor=0.5,mode="bilinear")
        c3 = self.conv3(pool2)

        # attentionc4=self.nonelocal4(c4)

        up5=self.up5(F.interpolate(c3,scale_factor=2,mode="bilinear"))
        c5_1=self.res2to4(c2)
        merge1=up5+c5_1
        c6=self.conv4(merge1)

        up7=self.up7(F.interpolate(c6,scale_factor=2,mode="bilinear"))
        c7_1=self.res1to5(c1)
        merge2=up7+c7_1
        c8=self.conv5(merge2)

        up9 = self.up9(F.interpolate(c8,scale_factor=4,mode="bilinear"))
        c9_1=self.res0to6(c0_initial)
        merge3=up9+c9_1
        c10 = self.conv6(merge3)


        c11 = self.conv7(c10)

        out = nn.Sigmoid()(c11)

        return out

# https://github.com/qthequartermasterman/SEResNet18Experiments/blob/master/models/se_resnet.py
"""SE-ResNet in PyTorch
Based on preact_resnet.py
Author: Xu Ma.
Date: Apr/15/2019
"""

__all__ = ['SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNet101', 'SEResNet152']

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.se = SELayer(planes,reduction)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.se = SELayer(self.expansion*planes, reduction)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out +=shortcut
        return out


class SEResNet(nn.Module):
    def __init__(self,block,num_blocks,in_ch,out_ch,reduction=16):
        super(SEResNet, self).__init__()
        self.in_planes=24
        self.conv1 = nn.Conv2d(in_ch,24,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self._make_layer(block, 24, num_blocks[0], stride=1,reduction=reduction)
        self.layer2 = self._make_layer(block, 48, num_blocks[1], stride=1,reduction=reduction)
        self.layer3 = self._make_layer(block, 96, num_blocks[2], stride=1,reduction=reduction)
        self.layer4 = self._make_layer(block, 192, num_blocks[3], stride=1,reduction=reduction)
        self.up5 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.res2to4 = MultiscaleResblock(96, 96)
        self.conv4 = DoubleConv(96, 96)

        self.up7 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.res2to5 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv5 = DoubleConv(48, 48)

        self.up9 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.res1to6 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )

        self.conv6 = NeckConv(24, 24)
        self.conv7 = nn.Conv2d(24, out_ch, kernel_size=1, stride=1)


    #block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks,stride,reduction):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        c0_initial = self.conv1(x)
        c0_initial = self.layer1(c0_initial)
        c0 = F.interpolate(c0_initial, scale_factor=0.25, mode="bilinear")

        c1 = self.layer2(c0)
        pool1 = F.interpolate(c1, scale_factor=0.5, mode="bilinear")
        c2 = self.layer3(pool1)
        pool2 = F.interpolate(c2, scale_factor=0.5, mode="bilinear")
        c3 = self.layer4(pool2)

        # attentionc4=self.nonelocal4(c4)

        up5 = self.up5(F.interpolate(c3, scale_factor=2, mode="bilinear"))
        # c5_1=self.res2to4(c2)
        # merge1=torch.cat([up5,c5_1],dim=1)
        c6 = self.conv4(up5)

        up7 = self.up7(F.interpolate(c6, scale_factor=2, mode="bilinear"))
        c7_1 = self.res2to5(F.interpolate(c2, scale_factor=2, mode="bilinear"))
        merge2 = up7 + c7_1
        # merge2=torch.cat([up7,c7_1],dim=1)
        c8 = self.conv5(merge2)

        up9 = self.up9(F.interpolate(c8, scale_factor=4, mode="bilinear"))
        c9_1 = self.res1to6(F.interpolate(c1, scale_factor=4, mode="bilinear"))
        merge3 = up9 + c9_1
        # merge3 = torch.cat([up9, c9_1], dim=1)
        c10 = self.conv6(merge3)

        c11 = self.conv7(c10)

        out = nn.Sigmoid()(c11)

        return out


def SEResNet18(in_ch=3, out_ch=1):
    return SEResNet(SEPreActBlock, [2,2,2,2],in_ch,out_ch)


def SEResNet34(in_ch=3, out_ch=1):
    return SEResNet(SEPreActBlock, [3,4,6,3],in_ch,out_ch)


def SEResNet50(in_ch=3, out_ch=1):
    return SEResNet(SEPreActBootleneck, [3,4,6,3],in_ch,out_ch)


def SEResNet101(in_ch=3, out_ch=1):
    return SEResNet(SEPreActBootleneck, [3,4,23,3],in_ch,out_ch)


def SEResNet152(in_ch=3, out_ch=1):
    return SEResNet(SEPreActBootleneck, [3,8,36,3],in_ch,out_ch)


def test():
    net = SEResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

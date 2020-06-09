import torch.nn as nn
import torch
from models.nonelocallib.non_local_embedded_gaussian import NONLocalBlock2D
from torch.nn import functional as F

class NeckConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(NeckConv,self).__init__()
        out_ch_=out_ch//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch_, (1, 1), padding=(0, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch_, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch_),
            nn.ReLU(),
            nn.Conv2d(out_ch_, out_ch, (1, 3), dilation=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            )
    def forward(self,x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, (1, 3), dilation=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
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
            nn.BatchNorm2d(out_ch-in_ch),
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
        out_ch_=in_ch//3

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
        self.conv0 = nn.Conv2d(in_ch, out_ch, (2,2),stride=2, padding=(0, 0))
        self.batchnorm0 = nn.BatchNorm2d(in_ch)
        self.downsample=Downsample(in_ch,out_ch)

    def forward(self, x):
        c1_ch1 = self.conv1_1(x)
        c1_ch2=self.conv3_3(x)
        c1_ch3 = self.conv5_5(x)
        # c1_ch4 = self.conv7_7(x)

        c1_concat=torch.cat([c1_ch1,c1_ch2,c1_ch3],dim=1)#按维数1（列）拼接,列增加
        c1_concat=self.batchnorm0(c1_concat)
        # c1_concat=self.conv0(c1_concat)
        # c1_concat += self.extra(x)
        c1_concat=self.downsample(c1_concat)
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
            nn.ReLU()
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
            nn.ReLU(),
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
            nn.ReLU(),

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

class trangle_net(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(trangle_net,self).__init__()
        self.convinitial=NeckConv(in_ch,24)
        self.downscale4=nn.Sequential(
            nn.Conv2d(24, 24, (1,3), dilation=(1,2),padding=(0,2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(24),
            # nn.ReLU(),
            nn.Conv2d(24, 48, (3,1), dilation=(2,1), padding=(2,0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.downscale8 = MultiscaleSEResblock(48,96)
        self.downscale8_1=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, (1, 3), dilation=(1, 2), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            # nn.ReLU(),
            nn.Conv2d(48, 48, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

        self.downscale16 = MultiscaleSEResblock(96, 192)
        self.downscale16_1=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(96, 96, (1, 3), dilation=(1, 2), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(96),
            # nn.ReLU(),
            nn.Conv2d(96, 96, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.downscale16_2=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, (1, 3), dilation=(1, 2), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.attention=NONLocalBlock2D(336)
        self.upscale16=nn.Sequential(
            DoubleConv(336, 336),  # in_ch、out_ch是通道数
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(336, 96, (1, 3), dilation=(1, 2), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.upscale8 = nn.Sequential(
            DoubleConv(144, 144),  # in_ch、out_ch是通道数
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(144, 48, (1, 3), dilation=(1, 2), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.upscale4 = nn.Sequential(
            DoubleConv(48, 48),  # in_ch、out_ch是通道数
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(48, 24, (1, 3), dilation=(1, 2), padding=(0, 2)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, (3, 1), dilation=(2, 1), padding=(2, 0)),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.end = DoubleConv(48,out_ch)

    def forward(self,x):
        conv_start=self.convinitial(x)
        down4=self.downscale4(F.interpolate(conv_start,scale_factor=0.25,mode="bilinear"))
        down8=self.downscale8(down4)
        down8_1=self.downscale8_1(down4)
        down16=self.downscale16(down8)
        down16_1=self.downscale16_1(down8)
        down16_2 = self.downscale16_2(down8_1)
        down_bottle=torch.cat([down16,down16_1,down16_2],dim=1)
        # down_bottle_attention=self.attention(down_bottle)
        up8=self.upscale16(down_bottle)
        up8=up8+down8
        up8_=torch.cat([down8_1,up8],dim=1)
        up4=self.upscale8(up8_)
        up4_=up4+down4
        # up4_=torch.cat([down4,up4],dim=1)
        up1=self.upscale4(up4_)
        up1=torch.cat([conv_start,up1],dim=1)
        up1_=self.end(up1)

        out=nn.Sigmoid()(up1_)
        return out


class special(nn.Module):

    def __init__(self,in_ch,out_ch):
        super(special, self).__init__()
        self.conv=nn.Conv2d(in_ch,out_ch//2,1)

    def forward(self,x):
        conv0=self.conv




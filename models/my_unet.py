import torch.nn as nn
import torch
from models.nonelocallib.non_local_embedded_gaussian import NONLocalBlock2D

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),#in_ch、out_ch是通道数
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
    def forward(self,x):
        return self.conv(x)

class Conv3_3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Conv3_3,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch,int(out_ch/2),3,padding=1),#in_ch、out_ch是通道数
                nn.BatchNorm2d(int(out_ch/2)),
                nn.ReLU(),

            )
    def forward(self,x):
        return self.conv(x)


class Conv5_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv5_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, int(out_ch/2), 5, padding=2),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(int(out_ch/2)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNet,self).__init__()
        self.conv1 = DoubleConv(in_ch,64)
        self.pool1 = nn.MaxPool2d(2)#每次把图像尺寸缩小一半
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256,512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512,1024)
        #逆卷积
        self.up6 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv6 = DoubleConv(1024,512)
        self.up7 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv7 = DoubleConv(512,256)
        self.up8 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv8 = DoubleConv(256,128)
        self.up9 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv9 = DoubleConv(128,64)

        self.conv10 = nn.Conv2d(64,out_ch,1)


    def forward(self,x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6,c4],dim=1)#按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7,c3],dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8,c2],dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9,c1],dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)#化成(0~1)区间
        return out
class My_Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(My_Unet,self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4= DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = Conv3_3(512, 1024)
        self.conv5_2 = Conv5_5(512, 1024)
        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.nonelocal1 = NONLocalBlock2D(512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.nonelocal2 = NONLocalBlock2D(256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.nonelocal3 = NONLocalBlock2D(128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.nonelocal4 = NONLocalBlock2D(64)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # p2=self.nonelocal1(p2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        # p3 = self.nonelocal2(p3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        # p4 = self.nonelocal3(p4)
        c5_1 = self.conv5_1(p4)
        c5_2 = self.conv5_2(p4)
        c5 = torch.cat([c5_1, c5_2], dim=1)
        up_6 = self.up6(c5)
        # up_6 = self.nonelocal1(up_6)
        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        # up_7 = self.nonelocal2(up_7)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        # up_8 = self.nonelocal3(up_8)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        # up_9 = self.nonelocal4(up_9)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        # out = nn.Sigmoid()(c10)  # 化成(0~1)区间
        return c10


from torch.nn import functional as F
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

class My_Unet2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(My_Unet2,self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4= DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.res4to6 = ResBlk(512,512)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res3to7 = ResBlk(256,256)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2to8 = ResBlk(128,128)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res1to9 = ResBlk(64,64)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # p2=self.nonelocal1(p2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        # p3 = self.nonelocal2(p3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        # p4 = self.nonelocal3(p4)
        c5=self.conv5(p4)
        up_6 = self.up6(c5)
        c4_res=self.res4to6(c4)
        # up_6 = self.nonelocal1(up_6)
        merge6 = torch.cat([up_6, c4_res], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c3_res = self.res3to7(c3)
        # up_7 = self.nonelocal2(up_7)
        merge7 = torch.cat([up_7, c3_res], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c2_res = self.res2to8(c2)
        # up_8 = self.nonelocal3(up_8)
        merge8 = torch.cat([up_8, c2_res], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1_res = self.res1to9(c1)
        # up_9 = self.nonelocal4(up_9)
        merge9 = torch.cat([up_9, c1_res], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)  # 化成(0~1)区间
        return out


import torch.nn as nn
import torch
import torch.nn.functional as F

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

class CascadeNet(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(CascadeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1
        self.seg_conv1 = DoubleConv(in_ch, 64)
        self.seg_pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.seg_conv2 = DoubleConv(64, 128)
        self.seg_pool2 = nn.MaxPool2d(2)
        self.seg_conv3 = DoubleConv(128, 256)
        self.seg_pool3 = nn.MaxPool2d(2)
        self.seg_conv4 = DoubleConv(256, 512)
        self.seg_pool4 = nn.MaxPool2d(2)
        self.seg_conv5 = DoubleConv(512, 1024)
        # 逆卷积
        self.seg_up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.seg_conv6 = DoubleConv(1024, 512)
        self.seg_up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.seg_conv7 = DoubleConv(512, 256)
        self.seg_up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.seg_conv8 = DoubleConv(256, 128)
        self.seg_up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.seg_conv9 = DoubleConv(128, 64)

        self.seg_conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = F.PReLU(self.conv1(x))
        x2 = F.PReLU(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.PReLU(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.PReLU(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.PReLU(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = F.PReLU(k * x - k + self.b)

        c1 = self.seg_conv1(output)
        p1 = self.seg_pool1(c1)
        c2 = self.seg_conv2(p1)
        p2 = self.seg_pool2(c2)
        c3 = self.seg_conv3(p2)
        p3 = self.seg_pool3(c3)
        c4 = self.seg_conv4(p3)
        p4 = self.seg_pool4(c4)
        c5 = self.seg_conv5(p4)
        up_6 = self.seg_up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.seg_conv6(merge6)
        up_7 = self.seg_up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.seg_conv7(merge7)
        up_8 = self.seg_up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.seg_conv8(merge8)
        up_9 = self.seg_up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.seg_conv9(merge9)
        c10 = self.seg_conv10(c9)

        out = nn.Sigmoid()(c10)  # 化成(0~1)区间
        return out





# ! pip install --upgrade pip
# ! pip install --user paddleseg

# 系统依赖库
import paddle.nn as nn
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddle
from paddle.nn import functional as F
import random
from paddle.io import Dataset
from visualdl import LogWriter
from paddle.vision.transforms import transforms as T
import warnings
warnings.filterwarnings("ignore")
import time
from time import *

#1*1卷积（包含激活函数）
class singleconv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(singleconv_block, self).__init__()
        self.singleconv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm(ch_out),
            nn.Swish()
        )
    def forward(self, x):
        x = self.singleconv(x)
        return x
#1*1卷积（没有激活函数）
class nosingleconv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(nosingleconv_block, self).__init__()
        self.nosingleconv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm(ch_out)
        )
    def forward(self, x):
        x = self.nosingleconv(x)
        return x
#深度卷积：核3，步幅为1  不改变特征图大小
class dw31conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(dw31conv_block, self).__init__()
        self.dw31conv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=1,groups=ch_in),
            nn.BatchNorm(ch_out),
            nn.Swish()
        )

    def forward(self, x):
        x = self.dw31conv(x)
        return x
#深度卷积：核3，步幅为2   特征图大小减小一半
class dw32conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(dw32conv_block, self).__init__()
        self.dw32conv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=2,padding=1,groups=ch_in),
            nn.BatchNorm(ch_out),
            nn.Swish()
        )

    def forward(self, x):
        x = self.dw32conv(x)
        return x
#深度卷积：核5，步幅为1   不改变特征图大小
class dw51conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(dw51conv_block, self).__init__()
        self.dw51conv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=5,stride=1,padding=2,groups=ch_in),
            nn.BatchNorm(ch_out),
            nn.Swish()
        )

    def forward(self, x):
        x = self.dw51conv(x)
        return x
#深度卷积：核5，步幅为2    特征图大小减小一半
class dw52conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(dw52conv_block, self).__init__()
        self.dw52conv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=5,stride=2,padding=2,groups=ch_in),
            nn.BatchNorm(ch_out),
            nn.Swish()
        )

    def forward(self, x):
        x = self.dw52conv(x)
        return x
#SEblock
class SE_block(nn.Layer):#sequece factor=4
    def __init__(self, ch_in, ch_out):
        super(SE_block, self).__init__()
        self.SE = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1,padding=0),
            nn.Swish(),
            nn.Conv2D(ch_out,ch_in,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        self.pool=nn.AdaptiveAvgPool2D(output_size=(1,1))

    def forward(self,x):
        y=self.pool(x)
        f=self.SE(y)
        return x*f


#第一层的3*3卷积
class conv_3x3(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(ch_out),
            nn.Swish(),
            nn.Conv2D(ch_out,ch_out,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm(ch_out),
            nn.Swish()
        )
    def forward(self, x):
        x = self.conv(x)
        return x


#第二层：三个MBC
#        MBConvBlock21 n=1
class MBC_block21(nn.Layer):
    def __init__(self, channel_in=16, channel_out=32):
        super(MBC_block21, self).__init__()
        self.dw1=dw31conv_block(ch_in=channel_in,ch_out=channel_in)
        self.se=SE_block(ch_in=16,ch_out=4)
        self.conv=nosingleconv_block(ch_in=16,ch_out=channel_out)

    def forward(self,x):
        x=self.dw1(x)
        x=self.se(x)
        x=self.conv(x)
        return x

#         MBConvBlock22 n=6
class MBC_block22(nn.Layer):
    def __init__(self, channel_in=32, channel_out=48):
        super(MBC_block22, self).__init__()
        self.singleconv=singleconv_block(ch_in=32,ch_out=192)
        self.dw32conv=dw32conv_block(ch_in=192,ch_out=192)
        self.se=SE_block(ch_in=192,ch_out=8)
        self.nosingleconv=nosingleconv_block(ch_in=192,ch_out=48)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw32conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        return x
#         MBConvBlock23 n=6
class MBC_block23(nn.Layer):
    def __init__(self, channel_in=48, channel_out=48):
        super(MBC_block23, self).__init__()
        self.singleconv=singleconv_block(ch_in=48,ch_out=288)
        self.dw31conv=dw31conv_block(ch_in=288,ch_out=288)
        self.se=SE_block(ch_in=288,ch_out=12)
        self.nosingleconv=nosingleconv_block(ch_in=288,ch_out=48)
        self.dropout=nn.Dropout2D(p=0.025)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw31conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        y=self.dropout(x)
        z=x+y
        return z

#第三层：两个MBC
#         MBConvBlock31 n=6
class MBC_block31(nn.Layer):
    def __init__(self, channel_in=48, channel_out=80):
        super(MBC_block31, self).__init__()
        self.singleconv=singleconv_block(ch_in=48,ch_out=288)
        self.dw52conv=dw52conv_block(ch_in=288,ch_out=288)
        self.se=SE_block(ch_in=288,ch_out=12)
        self.nosingleconv=nosingleconv_block(ch_in=288,ch_out=80)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw52conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        return x

#         MBConvBlock32 n=6
class MBC_block32(nn.Layer):
    def __init__(self, channel_in=80, channel_out=80):
        super(MBC_block32, self).__init__()
        self.singleconv=singleconv_block(ch_in=80,ch_out=480)
        self.dw51conv=dw51conv_block(ch_in=480,ch_out=480)
        self.se=SE_block(ch_in=480,ch_out=20)
        self.nosingleconv=nosingleconv_block(ch_in=480,ch_out=80)
        self.dropout=nn.Dropout2D(p=0.05)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw51conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        y=self.dropout(x)
        z=x+y
        return z

#第四层：
#         MBConvBlock41 n=6
class MBC_block41(nn.Layer):
    def __init__(self, channel_in=80, channel_out=192):
        super(MBC_block41, self).__init__()
        self.singleconv=singleconv_block(ch_in=80,ch_out=480)
        self.dw32conv=dw32conv_block(ch_in=480,ch_out=480)
        self.se=SE_block(ch_in=480,ch_out=20)
        self.nosingleconv=nosingleconv_block(ch_in=480,ch_out=192)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw32conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        return x

#MBConvBlock42 n=6
class MBC_block42(nn.Layer):
    def __init__(self, channel_in=192, channel_out=192):
        super(MBC_block42, self).__init__()
        self.singleconv=singleconv_block(ch_in=192,ch_out=1152)
        self.dw31conv=dw31conv_block(ch_in=1152,ch_out=1152)
        self.se=SE_block(ch_in=1152,ch_out=48)
        self.nosingleconv=nosingleconv_block(ch_in=1152,ch_out=192)
        self.dropout=nn.Dropout2D(p=0.075)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw31conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        y=self.dropout(x)
        z=x+y
        return z

#第五层：
#         MBConvBlock51 n=6
class MBC_block51(nn.Layer):
    def __init__(self, channel_in=192, channel_out=320):
        super(MBC_block51, self).__init__()
        self.singleconv=singleconv_block(ch_in=192,ch_out=1152)
        self.dw51conv=dw51conv_block(ch_in=1152,ch_out=1152)
        self.se=SE_block(ch_in=1152,ch_out=48)
        self.nosingleconv=nosingleconv_block(ch_in=1152,ch_out=320)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw51conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        return x

#         MBConvBlock52 n=6
class MBC_block52(nn.Layer):
    def __init__(self, channel_in=320, channel_out=320):
        super(MBC_block52, self).__init__()
        self.singleconv=singleconv_block(ch_in=320,ch_out=1920)
        self.dw51conv=dw51conv_block(ch_in=1920,ch_out=1920)
        self.se=SE_block(ch_in=1920,ch_out=80)
        self.nosingleconv=nosingleconv_block(ch_in=1920,ch_out=320)
        self.dropout=nn.Dropout2D(p=0.1125)
    def forward(self,x):
        x=self.singleconv(x)
        x=self.dw51conv(x)
        x=self.se(x)
        x=self.nosingleconv(x)
        y=self.dropout(x)
        z=x+y
        return z


#编码器
#block1
class block1(nn.Layer):
    def __init__(self, cha_in, cha_out):
        super(block1, self).__init__()
        self.conv1=conv_3x3(ch_in=cha_in,ch_out=16)
        self.MBC21=MBC_block21(channel_in=16, channel_out=cha_out)
    def forward(self,x):
        x=self.conv1(x)
        x=self.MBC21(x)
        return x

#block2
class block2(nn.Layer):
    def __init__(self, cha_in, cha_out):
        super(block2, self).__init__()
        self.MBC22=MBC_block22(channel_in=cha_in,channel_out=cha_out)
        self.MBC23=MBC_block23(channel_in=cha_out, channel_out=cha_out)
    def forward(self,x):
        x=self.MBC22(x)
        x=self.MBC23(x)
        return x

#block3
class block3(nn.Layer):
    def __init__(self, cha_in, cha_out):
        super(block3, self).__init__()
        self.MBC31=MBC_block31(channel_in=cha_in,channel_out=cha_out)
        self.MBC32=MBC_block32(channel_in=cha_out, channel_out=cha_out)
    def forward(self,x):
        x=self.MBC31(x)
        x=self.MBC32(x)
        return x

#block4
class block4(nn.Layer):
    def __init__(self, cha_in, cha_out):
        super(block4, self).__init__()
        self.MBC41=MBC_block41(channel_in=cha_in, channel_out=192)
        self.MBC42=MBC_block42(channel_in=192, channel_out=192)
        #self.MBC43=MBC_block43(channel_in=192, channel_out=192)
        self.MBC51=MBC_block51(channel_in=192, channel_out=cha_out)
        self.MBC52=MBC_block52(channel_in=cha_out, channel_out=cha_out)
        #self.MBC53=MBC_block53(channel_in=cha_out, channel_out=cha_out)
    def forward(self,x):
        x=self.MBC41(x)
        x=self.MBC42(x)
        x=self.MBC51(x)
        x=self.MBC52(x)
        return x

#SAblock
class SA(nn.Layer):

    def __init__(self, channel, groups=8):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.cweight = self.create_parameter((1, channel // (2 * groups), 1, 1),
            default_initializer=nn.initializer.Assign(paddle.zeros((1, channel // (2 * groups), 1, 1))))
        self.cbias = self.create_parameter((1, channel // (2 * groups), 1, 1),
            default_initializer=nn.initializer.Assign(paddle.ones((1, channel // (2 * groups), 1, 1))))
        self.sweight = self.create_parameter((1, channel // (2 * groups), 1, 1),
            default_initializer=nn.initializer.Assign(paddle.zeros((1, channel // (2 * groups), 1, 1))))
        self.sbias = self.create_parameter((1, channel // (2 * groups), 1, 1),
            default_initializer=nn.initializer.Assign(paddle.ones((1, channel // (2 * groups), 1, 1))))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape((b, groups, -1, h, w))
        x = x.transpose([0, 2, 1, 3, 4])

        # flatten
        x = x.reshape((b, -1, h, w))

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape((b * self.groups, -1, h, w))
        x_0, x_1 = x.chunk(2, axis=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = paddle.concat([xn, xs], axis=1)
        out = out.reshape((b, -1, h, w))

        out = self.channel_shuffle(out, 2)
        return out

#PFF
#pff1
class PFF1block(nn.Layer):
    def __init__(self, chanel_in,chanel_out):
        super(PFF1block, self).__init__()
        self.f2 = nn.Sequential(
            nn.Conv2D(chanel_in,chanel_out,kernel_size=1, stride=1, padding=0),
            nn.UpsamplingBilinear2D(scale_factor=2)
        )
        self.sa=SA(channel=32,groups=8)

    def forward(self,w,y):#w是f1，block1的输出
        y1=self.f2(y)
        y1=self.sa(y1)
        y2=w*y1
        out=paddle.concat(x=[w,y2], axis=1)
        return out

#pff2
class PFF2block(nn.Layer):
    def __init__(self, chanel_in1,chanel_out1,chanel_in2,chanel_out2):
        super(PFF2block, self).__init__()
        self.f3 = nn.Sequential(
            nn.Conv2D(chanel_in1,chanel_in1,kernel_size=2, stride=2, padding=0,groups=chanel_in1),
            nn.Conv2D(chanel_in1,chanel_out1,kernel_size=1, stride=1, padding=0)
        )
        self.f5 = nn.Sequential(
            nn.Conv2D(chanel_in2,chanel_out2,kernel_size=1, stride=1, padding=0),
            nn.UpsamplingBilinear2D(scale_factor=2)
        )
        self.sa=SA(channel=48,groups=8)

    def forward(self,w,y,z):
        w1=self.f3(w)
        w1=self.sa(w1)
        z1=self.f5(z)
        z1=self.sa(z1)
        out1=w1*y
        out2=y*z1
        out=paddle.concat(x=[out1,out2], axis=1)
        return out
#pff3
class PFF3block(nn.Layer):
    def __init__(self, chanel_in1,chanel_out1,chanel_in2,chanel_out2):#chanel_in1=144,chanel_out1=240,chanel_in2=672,chanel_out2=240
        super(PFF3block, self).__init__()
        self.f6 = nn.Sequential(
            nn.Conv2D(chanel_in1,chanel_in1,kernel_size=2, stride=2, padding=0,groups=chanel_in1),#使用深度卷积进行下采样2
            nn.Conv2D(chanel_in1,chanel_out1,kernel_size=1, stride=1, padding=0)
        )
        self.f8 = nn.Sequential(
            nn.Conv2D(chanel_in2,chanel_out2,kernel_size=1, stride=1, padding=0),
            nn.UpsamplingBilinear2D(scale_factor=2)
        )
        self.sa=SA(channel=80,groups=8)

    def forward(self,w,y,z):
        w1=self.f6(w)
        w1=self.sa(w1)
        z1=self.f8(z)
        z1=self.sa(z1)
        out1=w1*y
        out2=y*z1
        out=paddle.concat(x=[out1,out2], axis=1)
        return out
#pff4
class PFF4block(nn.Layer):
    def __init__(self, chanel_in,chanel_out):#输入通道数=240，输出通道数=672
        super(PFF4block, self).__init__()
        self.f9 = nn.Sequential(
            nn.Conv2D(chanel_in,chanel_in,kernel_size=2, stride=2, padding=0,groups=chanel_in),
            nn.Conv2D(chanel_in,chanel_out,kernel_size=1, stride=1, padding=0)
        )
        self.sa=SA(channel=320,groups=8)

    def forward(self,y,w):
        y1=self.f9(y)
        y1=self.sa(y1)
        out1=w*y1
        out=paddle.concat(x=[w,out1], axis=1)
        return out
#MAblock  通道压缩比例r取的是4,不知道该取几
class MA_block(nn.Layer):
    def __init__(self, ch_in, ch_out,ch_out1,height,weight):#ch_out=4*ch_out1,ch_in=ch_out
        super(MA_block, self).__init__()
        self.first_branch = nn.Sequential(
            nn.Conv2D(ch_in,ch_out1,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm(ch_out1),
            nn.ReLU(),
            nn.Conv2D(ch_out1,ch_out,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        self.second_branch= nn.Sequential(
            nn.Conv2D(ch_in,ch_out1,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm(ch_out1),
            nn.ReLU(),
            nn.Conv2D(ch_out1,ch_out,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        self.third_branch= nn.Sequential(
            nn.Conv2D(ch_in,ch_out1,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm(ch_out1),
            nn.ReLU(),
            nn.Conv2D(ch_out1,ch_out,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        self.pool1=nn.AdaptiveAvgPool2D(output_size=(1,1))
        self.pool2=nn.AvgPool2D(kernel_size=(height,1))
        self.pool3=nn.AvgPool2D(kernel_size=(1,weight))

    def forward(self,x):
        y=self.pool1(x)
        y=self.first_branch(y)
        y=paddle.multiply(x, y)
        w=self.pool2(x)
        w=self.second_branch(w)
        w=paddle.multiply(x, w)
        z=self.pool3(x)
        z=self.third_branch(z)
        z=paddle.multiply(x, z)
        out=y+w+z
        return out


# 基础模块+主网络
class conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),  # BN层用不用？源码没用
            nn.LeakyReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),  # BN层用不用？源码没用
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2D(scale_factor=2),
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_1x1(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_1(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
# 主网络
class PMC_Net(nn.Layer):
    def __init__(self, chan_in=3, chan_out=2):
        super(PMC_Net, self).__init__()
        self.block1 = conv_block(ch_in=chan_in, ch_out=16)
        self.block2 = block1(cha_in=16, cha_out=32)
        self.block3 = block2(cha_in=32, cha_out=48)
        self.block4 = block3(cha_in=48, cha_out=80)
        self.block5 = block4(cha_in=80, cha_out=320)

        self.PFF1 = PFF1block(chanel_in=48, chanel_out=32)
        self.PFF2 = PFF2block(chanel_in1=32, chanel_out1=48, chanel_in2=80, chanel_out2=48)
        self.PFF3 = PFF3block(chanel_in1=48, chanel_out1=80, chanel_in2=320, chanel_out2=80)
        self.PFF4 = PFF4block(chanel_in=80, chanel_out=320)
        self.MA1 = MA_block(ch_in=64, ch_out=64, ch_out1=16, height=128, weight=128)
        self.MA2 = MA_block(ch_in=96, ch_out=96, ch_out1=24, height=64, weight=64)
        self.MA3 = MA_block(ch_in=160, ch_out=160, ch_out1=40, height=32, weight=32)
        self.MA4 = MA_block(ch_in=640, ch_out=640, ch_out1=140, height=16, weight=16)

        self.conv4 = conv_block(ch_in=640, ch_out=320)
        self.conv3 = conv_block(ch_in=160, ch_out=80)
        self.conv2 = conv_block(ch_in=96, ch_out=48)
        self.conv1 = conv_block(ch_in=64, ch_out=32)

        self.up4 = up_conv(ch_in=320, ch_out=80)
        # self.up4=up_conv1(ch_in=320,ch_out1=192,ch_out2=80)
        self.up3 = up_conv(ch_in=80, ch_out=48)
        self.up2 = up_conv(ch_in=48, ch_out=32)
        self.up1 = up_conv(ch_in=32, ch_out=16)
        self.conv3x31 = conv_block(ch_in=640, ch_out=320)
        self.conv3x32 = conv_block(ch_in=160, ch_out=80)
        self.conv3x33 = conv_block(ch_in=96, ch_out=48)
        self.conv3x34 = conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2D(16, chan_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.block1(x)  # 16 544*384
        x1 = self.block2(x)  # 32 272*192
        x2 = self.block3(x1)  # 48 136*96
        x3 = self.block4(x2)  # 80 68*48
        x4 = self.block5(x3)  # 320 34*24

        p1 = self.PFF1(w=x1, y=x2)  # 64 272*192
        p2 = self.PFF2(w=x1, y=x2, z=x3)  # 96 136*96
        p3 = self.PFF3(w=x2, y=x3, z=x4)  # 160 68*48
        p4 = self.PFF4(y=x3, w=x4)  # 640 34*24

        m1 = self.MA1(p1)  # 64 272*192
        m2 = self.MA2(p2)  # 96 136*96
        m3 = self.MA3(p3)  # 160 68*48
        m4 = self.MA4(p4)  # 640 34*24

        m1 = self.conv1(m1)  # 32 272*192
        m2 = self.conv2(m2)  # 48 136*96
        m3 = self.conv3(m3)  # 80 68*48
        m4 = self.conv4(m4)  # 320 34*24

        d4 = self.up4(m4)  # 80 68*48
        d4 = paddle.concat(x=[d4, m3], axis=1)  # 160 68*48
        d4 = self.conv3x32(d4)  # 80 68*48
        d3 = self.up3(d4)  # 48  136*96
        d3 = paddle.concat(x=[d3, m2], axis=1)  # 96  136*96
        d3 = self.conv3x33(d3)  # 48  136*96
        d2 = self.up2(d3)  # 32  272*192
        d2 = paddle.concat(x=[d2, m1], axis=1)  # 64  272*192
        d2 = self.conv3x34(d2)  # 32  272*192
        d1 = self.up1(d2)  # 16   544*384
        # d1 = paddle.concat(x=[d1, x], axis=1)
        d1 = self.Conv_1x1(d1)

        return d1

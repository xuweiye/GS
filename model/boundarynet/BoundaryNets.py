import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch import sigmoid
from torch.nn import GroupNorm
from .resnet_model import *
from .sync_batchnorm import SynchronizedBatchNorm2d
from .MultiScaleFusion import MultiScaleFusion

class DiffNet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(DiffNet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        # self.bn1 = SynchronizedBatchNorm2d(64)
        self.gn1 = GroupNorm(2,64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        # self.bn2 = SynchronizedBatchNorm2d(64)
        self.gn2 = GroupNorm(2,64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        # self.bn3 = SynchronizedBatchNorm2d(64)
        self.gn3 = GroupNorm(2,64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        # self.bn4 = SynchronizedBatchNorm2d(64)
        self.gn4 = GroupNorm(2, 64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)


        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        # self.bn5 = SynchronizedBatchNorm2d(64)
        self.gn5 = GroupNorm(2, 64)
        self.relu5 = nn.ReLU(inplace=True)


        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        # self.bn_d4 = SynchronizedBatchNorm2d(64)
        self.gn_d4 = GroupNorm(2, 64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        # self.bn_d3 = SynchronizedBatchNorm2d(64)
        self.gn_d3 = GroupNorm(2, 64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        # self.bn_d2 = SynchronizedBatchNorm2d(64)
        self.gn_d2 = GroupNorm(2, 64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        # self.bn_d1 = SynchronizedBatchNorm2d(64)
        self.gn_d1 = GroupNorm(2, 64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        # hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx1 = self.relu1(self.gn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        # hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx2 = self.relu2(self.gn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        # hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx3 = self.relu3(self.gn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        # hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx4 = self.relu4(self.gn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        # hx5 = self.relu5(self.bn5(self.conv5(hx)))
        hx5 = self.relu5(self.gn5(self.conv5(hx)))
        hx = self.upscore2(hx5)

        # d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        d4 = self.relu_d4(self.gn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        # d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        d3 = self.relu_d3(self.gn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        # d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        d2 = self.relu_d2(self.gn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        # d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))
        d1 = self.relu_d1(self.gn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))
        residual = self.conv_d0(d1)

        return x + residual

class BoundaryNets(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(BoundaryNets,self).__init__()

        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        # self.inbn = SynchronizedBatchNorm2d(64)
        self.ingn = GroupNorm(2,64)
        self.inrelu = nn.ReLU(inplace=True)

        self.encoder1 = resnet.layer1 #384
        self.encoder2 = resnet.layer2 #192
        self.encoder3 = resnet.layer3 #96
        self.encoder4 = resnet.layer4 #48

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        # ------------multi scale fusion---------------
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #24

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.resb6_1 = BasicBlock(512,512,dilation=2)
        self.resb6_2 = BasicBlock(512,512,dilation=2)
        self.resb6_3 = BasicBlock(512,512,dilation=2) #12
        # self.mulfu = MultiScaleFusion(dim_in=512,
        #                  dim_out=512,
        #                  rate=1,
        #                  bn_mom=0.0003)
        self.mulfu = MultiScaleFusion(dim_in=512,
                         dim_out=512,
                         rate=1)
        

        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 24
        # self.bn6d_1 = SynchronizedBatchNorm2d(512)
        self.gn6d_1 = GroupNorm(16,512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        # self.bn6d_m = SynchronizedBatchNorm2d(512)
        self.gn6d_m = GroupNorm(16,512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        # self.bn6d_2 = SynchronizedBatchNorm2d(512)
        self.gn6d_2 = GroupNorm(16,512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 24
        # self.bn5d_1 = SynchronizedBatchNorm2d(512)
        self.gn5d_1 = GroupNorm(16,512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        # self.bn5d_m = SynchronizedBatchNorm2d(512)
        self.gn5d_m = GroupNorm(16,512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        # self.bn5d_2 = SynchronizedBatchNorm2d(512)
        self.gn5d_2 = GroupNorm(16,512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------
        self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1) # 48
        # self.bn4d_1 = SynchronizedBatchNorm2d(512)
        self.gn4d_1 = GroupNorm(16,512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        # self.bn4d_m = SynchronizedBatchNorm2d(512)
        self.gn4d_m = GroupNorm(16,512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        # self.bn4d_2 = SynchronizedBatchNorm2d(256)
        self.gn4d_2 = GroupNorm(8,256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv2d(512,256,3,padding=1) # 96
        # self.bn3d_1 = SynchronizedBatchNorm2d(256)
        self.gn3d_1 = GroupNorm(8,256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        # self.bn3d_m = SynchronizedBatchNorm2d(256)
        self.gn3d_m = GroupNorm(8,256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        # self.bn3d_2 = SynchronizedBatchNorm2d(128)
        self.gn3d_2 = GroupNorm(4,128)
        self.relu3d_2 = nn.ReLU(inplace=True)


        self.conv2d_1 = nn.Conv2d(256,128,3,padding=1) # 192
        # self.bn2d_1 = SynchronizedBatchNorm2d(128)
        self.gn2d_1 = GroupNorm(4,128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        # self.bn2d_m = SynchronizedBatchNorm2d(128)
        self.gn2d_m = GroupNorm(4,128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        # self.bn2d_2 = SynchronizedBatchNorm2d(64)
        self.gn2d_2 = GroupNorm(2,64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 384
        # self.bn1d_1 = SynchronizedBatchNorm2d(64)
        self.gn1d_1 = GroupNorm(2,64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        # self.bn1d_m = SynchronizedBatchNorm2d(64)
        self.gn1d_m = GroupNorm(2,64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        # self.bn1d_2 = SynchronizedBatchNorm2d(64)
        self.gn1d_2 = GroupNorm(2,64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)

        ## -------------Refine Module-------------
        self.diffnet = DiffNet(1,64)


    def forward(self,x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)#CB1
        # hx = self.inbn(hx)
        hx = self.ingn(hx)
        hx = self.inrelu(hx)
        #basic blocsk 的数量为[3,4,6,3]
        h1 = self.encoder1(hx) #384
        h2 = self.encoder2(h1) #192
        h3 = self.encoder3(h2) #96
        h4 = self.encoder4(h3) #48
        
        ## -------------Bridge-------------
        hx = self.pool4(h4) #24

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) #12

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        
        
        hbg = self.mulfu(h6)

        ## -------------Decoder------------
        # hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        # hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        # hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))
        hx = self.relu6d_1(self.gn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.gn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.gn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 12 -> 24
      
        # hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        # hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        # hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))
        hx = self.relu5d_1(self.gn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.gn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.gn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 24 -> 48

        # hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        # hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        # hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))
        hx = self.relu4d_1(self.gn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hx = self.relu4d_m(self.gn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.gn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 48 -> 96

        # hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        # hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        # hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))
        hx = self.relu3d_1(self.gn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx = self.relu3d_m(self.gn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.gn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 96 -> 192

        # hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        # hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        # hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))
        hx = self.relu2d_1(self.gn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx = self.relu2d_m(self.gn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.gn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 192 -> 384

        # hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        # hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        # hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))
        hx = self.relu1d_1(self.gn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.gn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.gn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) 

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6) 

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) 

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) 

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) 

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) 

        d1 = self.outconv1(hd1) 

        ## -------------difference boundary net-------------
        dout = self.diffnet(d1) # 384
        return F.sigmoid(dout), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(db)

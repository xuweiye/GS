import torch.nn as nn
import torch.nn.functional as F

class Base_Decoder(nn.Module):
    def __init__(self,num_classes):
        super(Base_Decoder, self).__init__()
        self.conv_d4 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.bn_d4 = nn.BatchNorm2d(512)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn_d3 = nn.BatchNorm2d(256)
        self.relu_d3 = nn.ReLU(inplace=True)
        self.conv_d2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn_d2 = nn.BatchNorm2d(128)
        self.relu_d2 = nn.ReLU(inplace=True)

        # self.conv_d1 = nn.Conv2d(128, 64, 3, 1, 0)
        # self.bn_d1 = nn.BatchNorm2d(64)
        # self.relu_d1 = nn.ReLU(inplace=True)
        #
        # self.conv_d0 = nn.Conv2d(64, 32, 3, 1, 0)
        # self.bn_d0 = nn.BatchNorm2d(32)
        # self.relu_d0 = nn.ReLU(inplace=True)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.conv_f = nn.Conv2d(128,1,1, 1, 0)

    def forward(self,x):
        x = self.relu_d4(self.bn_d4(self.conv_d4(x)))
        x = self.upscore2(x)
        x = self.relu_d3(self.bn_d3(self.conv_d3(x)))
        x = self.upscore2(x)
        x = self.relu_d2(self.bn_d2(self.conv_d2(x)))
        x = self.upscore2(x)
        x = self.conv_f(x)
        x = self.upscore4(x)
        # #56->112
        # x = self.relu_d1(self.bn_d1(self.conv_d1(x)))
        # x = self.upscore2(x)
        # # 112->224
        # x = self.relu_d0(self.bn_d0(self.conv_d0(x)))
        # x = self.upscore2(x)
        return F.sigmoid(x)






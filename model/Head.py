import torch.nn as nn
import torch





class BiMFFHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=64, norm_cfg=None):
        super(BiMFFHead, self).__init__()
        #out_size = (in_size - 1) * S + K - 2*P + output_padding
        # self.head2 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 8, stride=4, padding=2, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        # self.head3 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
        #                            nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 8, stride=4, padding=2, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        # self.head4 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU6(),
        #                            nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        # self.head5 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU6(),
        #                            nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 32, stride=16, padding=8, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())

        self.head2 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head3 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
            nn.ReLU())
        self.head4 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU())
        self.head5 = nn.Sequential(
            nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU())
        self.upscore5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, d2,d3,d4,d5):
        # B 64 224 224
        head2 = self.upscore2(self.head2(d2))
        head3 = self.upscore3(self.head3(d3))
        head4 = self.upscore4(self.head4(d4))
        head5 = self.upscore5(self.head2(d5))

        return torch.cat([head2, head3, head4, head5], dim=1), head2, head3, head4, head5




class CA_BIMFFHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, mla_channels=256, mlahead_channels=64,num_classes=2,
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(CA_BIMFFHead, self).__init__(**kwargs)
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_classes

        self.mlahead = BiMFFHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.conv2 = nn.Sequential(nn.Conv2d(mlahead_channels,1,3,padding=1),nn.BatchNorm2d(1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(mlahead_channels,1,3,padding=1),nn.BatchNorm2d(1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(mlahead_channels,1,3,padding=1),nn.BatchNorm2d(1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(mlahead_channels,1,3,padding=1),nn.BatchNorm2d(1), nn.ReLU())
        self.global_features = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.conv_all = nn.Conv2d(self.mlahead_channels, 1, 1)

    def forward(self, inputs):
        x,d2,d3,d4,d5 = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        d2 = self.conv2(d2)
        d3 = self.conv2(d3)
        d4 = self.conv2(d4)
        d5 = self.conv2(d5)
        x = self.global_features(x)
        x = self.conv_all(x)
        # return edge, x
        return x, d2, d3, d4, d5
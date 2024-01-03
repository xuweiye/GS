import torch
import torch.nn as nn

class RS_Net(nn.Module):
    def __init__(self, n_bands, n_cls):
        super(RS_Net, self).__init__()

        # -----------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(nn.Conv2d(n_bands, 32, kernel_size=3, padding=1),nn.ReLU(),nn.BatchNorm2d(32))
        self.conv1_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(),nn.BatchNorm2d(32))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # -----------------------------------------------------------------------
        self.conv2_1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.conv2_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # -----------------------------------------------------------------------
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # -----------------------------------------------------------------------
        self.conv4_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.conv4_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # -----------------------------------------------------------------------
        self.conv5_1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        self.conv5_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512))

        # -----------------------------------------------------------------------
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6_1 = nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),nn.Dropout(0.5),nn.ReLU())
        self.conv6_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Dropout(0.5), nn.ReLU())

        # -----------------------------------------------------------------------
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7_1 = nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),nn.Dropout(0.5),nn.ReLU())
        self.conv7_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.Dropout(0.5), nn.ReLU())


        # -----------------------------------------------------------------------
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8_1 = nn.Sequential(nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),nn.Dropout(0.5),nn.ReLU())
        self.conv8_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.Dropout(0.5), nn.ReLU())


        # -----------------------------------------------------------------------
        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv9_1 = nn.Sequential(nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),nn.Dropout(0.5),nn.ReLU())
        self.conv9_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.Dropout(0.5), nn.ReLU())



        # -----------------------------------------------------------------------
        self.conv10 = nn.Conv2d(32, n_cls, kernel_size=1)

    def forward(self, x):
        # -----------------------------------------------------------------------
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)

        # -----------------------------------------------------------------------
        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool2(conv2)

        # -----------------------------------------------------------------------
        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool3(conv3)

        # -----------------------------------------------------------------------
        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.pool4(conv4)

        # -----------------------------------------------------------------------
        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)

        # -----------------------------------------------------------------------
        up6 = torch.cat([self.up6(conv5), conv4], dim=1)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)

        # -----------------------------------------------------------------------
        up7 = torch.cat([self.up7(conv6), conv3], dim=1)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)

        # -----------------------------------------------------------------------
        up8 = torch.cat([self.up8(conv7), conv2], dim=1)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)

        # -----------------------------------------------------------------------
        up9 = torch.cat([self.up9(conv8), conv1], dim=1)
        conv9 = self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)

        # -----------------------------------------------------------------------
        conv10 = self.conv10(conv9)

        return torch.sigmoid(conv10)
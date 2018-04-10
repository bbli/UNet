import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from utils import *
# from functools import partial
from Data import *

import os
import matplotlib
# checks environmental variables
if (("DISPLAY" not in os.environ) or
    (os.environ["DISPLAY"] == "")):
        matplotlib.use('Agg')
else:
        matplotlib.use('Qt5Agg')
from skimage import io
from skimage import img_as_float



class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # note weights are being initalized randomly at the moment
        self.conv1a = nn.Conv2d(1,64,3)
        self.conv1b = nn.Conv2d(64,64,3)

        self.conv2a = nn.Conv2d(64,128,3)
        self.conv2b = nn.Conv2d(128,128,3)

        self.conv3a = nn.Conv2d(128,256,3)
        self.conv3b = nn.Conv2d(256,256,3)

        self.conv4a = nn.Conv2d(256,512,3)
        self.conv4b = nn.Conv2d(512,512,3)

        self.conv5a = nn.Conv2d(512,1024,3)
        self.conv5b = nn.Conv2d(1024,1024,3)

        self.maxpool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(1024,10,3)
        # self.encode1 = nn.Sequential([nn.Conv2d(1,64,3),nn.ReLU(inplace=True)])
        # self.encode2 = nn.Sequential([nn.Conv2d(64,128,3),nn.ReLU(inplace=True)])

    def forward(self,x):
        x = F.relu(self.conv1a(x))
        z1 = F.relu(self.conv1b(x))

        x = self.maxpool(z1)
        x = F.relu(self.conv2a(x))
        z2 = F.relu(self.conv2b(x))

        x = self.maxpool(z2)
        x = F.relu(self.conv3a(x))
        z3 = F.relu(self.conv3b(x))

        x = self.maxpool(z3)
        x = F.relu(self.conv4a(x))
        z4 = F.relu(self.conv4b(x))

        x = self.maxpool(z4)
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))

        out = self.up1(x)
        return x,out

if __name__ == '__main__':
    # input_data = img_as_float(io.imread('/data/bbli/gryllus_disk_images/cmp_1_1_T0000.tif')) :

    # x = Variable(torch.from_numpy(input_data).float()).cuda()
    # x = x.view(1,1,2001,2001)

    path = '/data/bbli/gryllus_disk_images/'

    factor = 4
    downscale = partial(downscale_local_mean, 
                factors=(factor,factor))
    transforms = Compose([rescale, downscale, toTorch ])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])

    dataset = ParhyaleDataset(path,transforms)
    #############################################
    img = dataset[0]
    img = tensor_format(img)

    model = UNet().cuda()
    z = model(img)

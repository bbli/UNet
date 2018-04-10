import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from utils import *
# from functools import partial
import Data as D

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
        self.conv1 = nn.Conv2d(1,64,3)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64,128,3)
        self.conv3 = nn.Conv2d(128,256,3)
        self.conv4 = nn.Conv2d(256,512,3)

        self.encode1 = nn.Sequential([nn.Conv2d(1,64,3),nn.ReLU(inplace=True)])
        self.encode2 = nn.Sequential([nn.Conv2d(64,128,3),nn.ReLU(inplace=True)])
    def forward(self,x):
        # input dimensions: torch.Size([1, 1, 2001, 2001])
        print(x.shape)
        z = F.relu(self.conv1(x))
        print(z.shape)
        # now has dimensions: torch.Size([1, 64, 1994, 1994])
        z = self.maxpool(z)
        print(z.shape)
        # now has dimensions torch.Size([1,64,997,997])

        z = F.relu(self.conv2(z))
        print(z.shape)
        # now has dimensions torch.Size([1,128,990,990])
        z = self.maxpool(z)
        print(z.shape)

        z = F.relu(self.conv3(z))
        print(z.shape)
        z = self.maxpool(z)
        print(z.shape)
        
        z = F.relu(self.conv4(z))
        print(z.shape)
        z = self.maxpool(z)
        print(z.shape)
        return z

if __name__ == '__main__':
    # input_data = img_as_float(io.imread('/data/bbli/gryllus_disk_images/cmp_1_1_T0000.tif')) :

    # x = Variable(torch.from_numpy(input_data).float()).cuda()
    # x = x.view(1,1,2001,2001)

    path = '/data/bbli/gryllus_disk_images/'

    original_size=2001
    factor = 3
    #make sure this divides evenly
    down_size= original_size//factor
    downscale = partial(downscale_local_mean, 
                factors=(factor,factor))
    transforms = Compose([downscale, toTorch(down_size),F.normalize ])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])

    dataset = ParhyaleDataset(path,transforms)

    img = dataset[0]
    img = tensor_format(img)

    model = UNet().cuda()
    z = model(img)

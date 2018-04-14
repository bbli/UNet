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
class Convolve(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size)
    def forward(self,x):
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # note weights are being initalized randomly at the moment

        self.encode1 = Convolve(1,64,3)
        self.encode2 = Convolve(64,128,3)
        self.encode3 = Convolve(128,256,3)
        self.encode4 = Convolve(256,512,3)

        self.center = Convolve(512,1024,3)

        self.decode4 = Convolve(1024,512,3)
        self.decode3 = Convolve(512,256,3)
        self.decode2 = Convolve(256,128,3)
        self.decode1 = Convolve(128,64,3)

        self.convdown1a = nn.Conv2d(1,64,3)
        self.convdown1b = nn.Conv2d(64,64,3)

        self.convdown2a = nn.Conv2d(64,128,3)
        self.convdown2b = nn.Conv2d(128,128,3)

        self.convdown3a = nn.Conv2d(128,256,3)
        self.convdown3b = nn.Conv2d(256,256,3)

        self.convdown4a = nn.Conv2d(256,512,3)
        self.convdown4b = nn.Conv2d(512,512,3)

        self.convdown5a = nn.Conv2d(512,1024,3)
        self.convdown5b = nn.Conv2d(1024,1024,3)

        self.maxpool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2)
        self.up2 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2)
        # self.encode1 = nn.Sequential([nn.Conv2d(1,64,3),nn.ReLU(inplace=True)])
        # self.encode2 = nn.Sequential([nn.Conv2d(64,128,3),nn.ReLU(inplace=True)])

        self.convup5a = nn.Conv2d(1024,512,3)
        self.convup5b = nn.Conv2d(512,512,3)


    def forward(self,x):

        # x = self.convup5a(f4)
        # x = self.convup5b(x)
        # u3 = self.up2(x)

        d1= self.encode1(x)

        d2= self.maxpool(d1)
        d2= self.encode2(d2)

        d3 = self.maxpool(d2)
        d3 = self.encode3(d3)

        d4 = self.maxpool(d3)
        d4 = self.encode4(d4) 

        c = self.maxpool(d4)
        c = self.center(c)

        # u4 = self.up4(c)

        return c

def crop_and_concat(upsampled, bypass):
    ## may run to troubles if c is not even. 
    ## So, let us make sure that error is thrown
    c = (bypass.size()[2] - upsampled.size()[2]) // 2
    assert c%2 ==0, "This cropping method will not return an image that has the same dimensions as the upsampled one"
    bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision
from tensorboardX import SummaryWriter
from utils import *
# from functools import partial
from Data import *
import ipdb
import math

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



class Convolve(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,string):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size)
        self.string = string
    def forward(self,x):
        print("{} size before convolve: {}".format(self.string, x.shape))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        print("{} size after convolve: {}".format(self.string, x.shape))
        return x

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=2)
    def forward(self,x):
        x = self.up(x)
        print(x.shape)
        return x

# class LossFunction(Function):
    # def __init__(self):

    # def forward(self,x):

    # def backward(self,x):
            

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # note weights are being initalized randomly at the moment
        self.maxpool = nn.MaxPool2d(2)

        self.encode1 = Convolve(1,64,3,'d1')
        self.encode2 = Convolve(64,128,3,'d2')
        self.encode3 = Convolve(128,256,3,'d3')
        self.encode4 = Convolve(256,512,3,'d4')

        self.center = Convolve(512,1024,3,'c')

        self.decode4 = Convolve(1024,512,3,'u4')
        self.decode3 = Convolve(512,256,3,'u3')
        self.decode2 = Convolve(256,128,3,'u2')
        self.decode1 = Convolve(128,64,3,'u1')


        self.up4 = UpSample(1024,512,3)
        self.up3 = UpSample(512,256,3)
        self.up2 = UpSample(256,128,3)
        self.up1 = UpSample(128,64,3)

        # output channel is 1 b/c we have grayscale
        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):
        d1= self.encode1(x)

        d2= self.maxpool(d1)
        d2= self.encode2(d2)

        d3 = self.maxpool(d2)
        d3 = self.encode3(d3)

        d4 = self.maxpool(d3)
        d4 = self.encode4(d4) 

        c = self.maxpool(d4)
        c = self.center(c)

        u4 = self.up4(c)
        u4 = crop_and_concat(u4,d4)
        u4 = self.decode4(u4)

        u3 = self.up3(u4)
        # assert 1==0, "Debug from here"
        # ipdb.set_trace()
        u3 = crop_and_concat(u3,d3)
        u3 = self.decode3(u3)

        u2 = self.up2(u3)
        u2 = crop_and_concat(u2,d2)
        u2 = self.decode2(u2)

        u1 = self.up1(u2)
        u1 = crop_and_concat(u1,d1)
        u1 = self.decode1(u1)

        u1 = self.final(u1)
        print(u1.shape)
        return u1

def crop_and_concat(upsampled, bypass):
    ## may run to troubles if c is not even. 
    ## So, let us make sure that error is throw
    size = (bypass.size()[2] - upsampled.size()[2])
    if size%2 == 0:
        # print("c is even")
        c = size//2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    else:
        # print("c is odd")
        c = size//2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
        bypass = F.pad(bypass, (-1,0, -1,0))

    return torch.cat((upsampled, bypass), 1)

def weightInitialization(m):
    print("Name",m.__class__.__name__)
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.normal_(0, math.sqrt(2. / n))
        print("Changed!")
        global change_count 
        change_count +=1

if __name__ == '__main__':
    change_count =0
    # input_data = img_as_float(io.imread('/data/bbli/gryllus_disk_images/cmp_1_1_T0000.tif')) :

    # x = Variable(torch.from_numpy(input_data).float()).cuda()
    # x = x.view(1,1,2001,2001)

    path = '/data/bbli/gryllus_disk_images/'

    factor = 4
    downscale = partial(downscale_local_mean, 
                factors=(factor,factor))
    transforms = Compose([rescale, downscale, toTorch])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])

    dataset = ParhyaleDataset(path,transforms)
    #############################################
    img = dataset[0]
    img = tensor_format(img)

    model = UNet().cuda()
    model.apply(weightInitialization)
    z = model(img)

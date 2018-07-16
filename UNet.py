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
    def __init__(self,in_channels,out_channels,kernel_size,string,show=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.string = string
        self.show = show
    def forward(self,x):
        if self.show:
            print("{} size before convolve: {}".format(self.string, x.shape))

        # x = self.batch1(self.conv1(x))
        x = self.conv1(x)
        x = F.relu(x)
        
        # x = self.batch2(self.conv2(x))
        x = self.conv2(x)
        x = F.relu(x)

        if self.show:
            print("{} size after convolve: {}".format(self.string, x.shape))
        return x

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,show=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=2)
        self.show = show
        # self.up = nn.Upsample(scale_factor=2)
    def forward(self,x):
        x = self.up(x)
        if self.show:
             print(x.shape)
        return x
            
class UNet(nn.Module):
    def __init__(self,kernel_size,feature_maps,show=False):
        super().__init__()
        # note weights are being initalized randomly at the moment
        self.kernel_size=kernel_size
        self.feature= feature_maps
        self.maxpool = nn.MaxPool2d(2)

        self.encode1 = Convolve(1,self.feature,self.kernel_size,'d1',show)
        self.encode2 = Convolve(self.feature,self.feature*2,self.kernel_size,'d2',show)
        self.encode3 = Convolve(self.feature*2,self.feature*4,self.kernel_size,'d3',show)

        self.center = Convolve(self.feature*4,self.feature*8,self.kernel_size,'c',show)

        self.decode3 = Convolve(self.feature*8,self.feature*4,self.kernel_size,'u3',show)
        self.decode2 = Convolve(self.feature*4,self.feature*2,self.kernel_size,'u2',show)
        self.decode1 = Convolve(self.feature*2,self.feature,self.kernel_size,'u1',show)


        self.up3 = UpSample(self.feature*8,self.feature*4,self.kernel_size)
        self.up2 = UpSample(self.feature*4,self.feature*2,self.kernel_size)
        self.up1 = UpSample(self.feature*2,self.feature,self.kernel_size)

        self.final = nn.Conv2d(self.feature,2,1)

    def forward(self,x):
        d1= self.encode1(x)

        d2= self.maxpool(d1)
        d2= self.encode2(d2)

        d3 = self.maxpool(d2)
        d3 = self.encode3(d3)

        c = self.maxpool(d3)
        c = self.center(c)

        u3 = self.up3(c)
        u3 = crop_and_concat(u3,d3)
        u3 = self.decode3(u3)

        u2 = self.up2(u3)
        u2 = crop_and_concat(u2,d2)
        u2 = self.decode2(u2)

        u1 = self.up1(u2)
        u1 = crop_and_concat(u1,d1)
        u1 = self.decode1(u1)

        u1 = self.final(u1)
        return F.softmax(u1,dim=1)

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
    # print("Name",m.__class__.__name__)
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.normal_(0, math.sqrt(2. / n))
        # print("Changed!")
        # global change_count 
        # change_count +=1

if __name__ == '__main__':
    # img,label = next(iter(train_loader))
    import numpy as np
    lookup_table = np.zeros(20,dtype='int16')
    lookup_table[3]=45
    lookup_table[4]=62
    lookup_table[5]=80
    lookup_table[6]=100
    lookup_table[7]=120
    lookup_table[8]=137
    lookup_table[9]=155
    kernel_size = 8
    feature_maps = 32
    print("Kernel Size", kernel_size)
    print("Initial Feature Maps",feature_maps)
    size = 400+2*int(lookup_table[kernel_size])
    img = torch.Tensor(1,1,size,size)
    img = tensor_format(img)
    # label = tensor_format(label)

    # model = FourLayerUNet(kernel_size,feature_maps,show=True).cuda(1)
    model = UNet(kernel_size,feature_maps,show=True).cuda(1)
    model.apply(weightInitialization)

    z = model(img)
    print("Dimension of output of Unet: "+str(z.shape))
    # z,label = crop(z,label)
    # print("Accuracy", score(z,label))
    

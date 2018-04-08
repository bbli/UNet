import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter

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



class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # note weights are being initalized randomly at the moment
        self.conv1 = nn.Conv2d(1,64,8)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64,128,8)
        self.conv3 = nn.Conv2d(128,256,8)
        self.conv4 = nn.Conv2d(256,512,8)
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
    input_data = img_as_float(io.imread('/data/bbli/gryllus_disk_images/cmp_1_1_T0000.tif')) :

    x = Variable(torch.from_numpy(input_data).float()).cuda()
    x = x.view(1,1,2001,2001)
    model = UNet().cuda()
    z = model(x)


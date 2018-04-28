import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch


from skimage import io
from skimage import img_as_float

# path = '/data/bbli/gryllus_disk_images/'
# # pic = os.listdir(path)

# # # image = img_as_float(io.imread(path+pic[0])) 
# # for i in pic:
    # # # image =io.imread(path+i) 
    # # image =img_as_float( io.imread(path+i) )
    # # print(image.mean())
    # # print(image.max())
# class ParhyaleDataset(Dataset):
    # def __init__(self,path,transform=None):
        # self.images = stackImages(readImages(path))
        # # self.images = readImages(path)

# def readImages(path):
    # image_list = []
    # pictures = os.listdir(path)    
    # for string in pictures:
        # image = io.imread(path+string)
        # image_list.append(image)
    # return image_list

# def stackImages(image_list):
    # number = len(image_list)
    # size, _ = image_list[0].shape
    # batch_array = np.zeros((number,size,size))
    # for i,pic in enumerate(image_list):
        # batch_array[i] = pic
    # return batch_array

# dataset = ParhyaleDataset(path)

## Testing if Tranpose Convolution is the same as Regular Convolution but with max padding
# dummy = Variable(torch.Tensor(1,1,5,5))
# Up = nn.ConvTranspose2d(1,4,3,1,0)
# Up2 = nn.Conv2d(1,4,3,1,2)
# z = Up(dummy)
# print(z.shape)
# z2 = Up(dummy)
# print(z2.shape)
# print((z-z2).sum())

## Testing if Transpose Convolution with stride =2 is the same as UpScale followed 2by2 kernel

dummy = Variable(torch.Tensor(1,1,5,5))
Up = nn.ConvTranspose2d(1,4,2,1,0)
z = Up(dummy)
print(z.shape)
Up3 = nn.Upsample(scale_factor=2, mode = 'bilinear')
z3 = Up3(dummy)
print(z3.shape)

# I feel like Upsample is the correct way to go, since it should mirror the Max Pooling

##TEsting Standarize class
test_images = np.array([[[0,1],[1,1]],[[2,3],[3,3]]])
test_img = np.array([[1,2],[2,2]])




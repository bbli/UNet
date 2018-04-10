import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch


from skimage import io
from skimage import img_as_float

path = '/data/bbli/gryllus_disk_images/'
# pic = os.listdir(path)

# # image = img_as_float(io.imread(path+pic[0])) 
# for i in pic:
    # # image =io.imread(path+i) 
    # image =img_as_float( io.imread(path+i) )
    # print(image.mean())
    # print(image.max())
class ParhyaleDataset(Dataset):
    def __init__(self,path,transform=None):
        self.images = stackImages(readImages(path))
        # self.images = readImages(path)

def readImages(path):
    image_list = []
    pictures = os.listdir(path)    
    for string in pictures:
        image = io.imread(path+string)
        image_list.append(image)
    return image_list

def stackImages(image_list):
    number = len(image_list)
    size, _ = image_list[0].shape
    batch_array = np.zeros((number,size,size))
    for i,pic in enumerate(image_list):
        batch_array[i] = pic
    return batch_array

dataset = ParhyaleDataset(path)

dummy = Variable(torch.Tensor(1,1,4,4))
Up = nn.ConvTranspose2d(1,4,3)
z = Up(dummy)

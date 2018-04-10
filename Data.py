import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
from skimage import img_as_float
import os
import numpy as np
from skimage.transform import downscale_local_mean
# from sklearn.preprocessing import StandardScaler
from functools import partial

from utils import *
    
##########################################################

class ParhyaleDataset(Dataset):
    def __init__(self,path,transform=None):
        self.images = stackImages(readImages(path))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image = self.images[index]
        if self.transform:
            return self.transform(image)
        else:
            return image

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

######################################################

def toTorch(image):
    down_size,_ = image.shape
    new_image = torch.from_numpy(image).float()
    new_image = new_image.view(1,1,down_size,down_size)
    return new_image

def rescale(image):
    return image/300

# class Standarize():
    # def __init__(self):
        # self.scalar = StandardScaler()
    # def __call__(self, image):
        # self.scalar.fit(image)
        # return self.scalar.transform(image)

##########################################################
if __name__=='__main__':
    path = '/data/bbli/gryllus_disk_images/'

    factor = 4
    downscale = partial(downscale_local_mean, 
                factors=(factor,factor))
    transforms = Compose([rescale, downscale, toTorch ])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])

    dataset = ParhyaleDataset(path,transforms)

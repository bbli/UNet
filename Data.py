import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
from skimage import img_as_float
import os
import numpy as np
from skimage.transform import downscale_local_mean
from sklearn.preprocessing import StandardScaler
from functools import partial
import ipdb

from utils import *
    
##########################################################
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

def downsize(images,factor):
    new_tensor = downscale_local_mean(images[0],factors=(factor,factor))
    new_tensor = new_tensor.reshape(1,*new_tensor.shape)
    for image in images[1:]:
        new_image = downscale_local_mean(image,factors=(factor,factor))
        new_image = new_image.reshape(1,*new_image.shape)
        # ipdb.set_trace()
        new_tensor = np.concatenate((new_tensor,new_image),axis=0)
    return new_tensor
class ParhyaleDataset(Dataset):
    def __init__(self,path,factor=None,transform=None):
        self.transform = transform
        self.images = stackImages(readImages(path))
        if factor:
            self.images = downsize(self.images,factor)
        print(np.mean(self.images[0]))
    def fit(self,scalers):
        for scaler in scalers:
            scaler.fit(self.images)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image = self.images[index]
        if self.transform:
            return self.transform(image)
        else:
            return image
######################################################

def toTorch(image):
    down_size,_ = image.shape
    new_image = torch.from_numpy(image).float()
    new_image = new_image.view(1,down_size,down_size)
    return new_image

## wrapper so StandardScaler will work inside a Pytorch Compose
#images is a 3D Tensor
# class Standarize():
    # def __init__(self):
        # self.scaler = StandardScaler(with_std=False)
    # def __call__(self, image):
        # ## reshape
        # shape = image.shape[-1]
        # image = image.reshape(1,shape*shape)
        # image=self.scaler.transform(image)
        # return image.reshape(shape,shape)
    # def fit(self,images):
        # ## reshape
        # length = len(images)
        # images = images.reshape(length,-1)
        # self.scaler.fit(images)
class Standarize(StandardScaler):
    def __init__(self,with_std=False):
        ## No need to pass self b/c this is call time
        super().__init__(with_std=False)
    def __call__(self,image):
        ## rescale
        shape = image.shape[-1]
        image = image.reshape(1,shape*shape)
        image=self.transform(image)
        return image.reshape(shape,shape)
    def fit(self,images):
        ## reshape
        length = len(images)
        images = images.reshape(length,-1)
        super().fit(images)

##########################################################
if __name__=='__main__':
    path = '/data/bbli/gryllus_disk_images/'

    center = Standarize()
    transforms = Compose([center,toTorch ])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])

    dataset = ParhyaleDataset(path,factor=4,transform=transforms)
    dataset.fit([center])

    train_set = DataLoader(dataset,shuffle=True)
    count =0
    for i,_ in enumerate(dataset):
        a = np.mean(dataset[i].numpy())
        print(a)
        count += a 

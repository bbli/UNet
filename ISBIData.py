import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
import numpy as np
import ipdb

from DataUtils import *

##########################################################
def rescale(image):
    return image/255

class ISBIDataset(Dataset):
    def __init__(self,path,transform = None,factor= None):
        self.images = io.imread(path+'train-volume.tif')
        self.labels = io.imread(path+'train-labels.tif')
        self.images, self.labels  = rescale(self.images), rescale(self.labels)
        self.transform = transform
        if factor:
            self.images = downsize(self.images,factor)
            self.labels = downsize(self.labels,factor)
        ## To test if num_images is reason for better accuracy
        # self.images = self.images[0:10]
        # self.labels = self.labels[0:10]
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            return imageToTorch(self.transform(image)),labelToTorch(label)
        else:
            return imageToTorch(image), labelToTorch(label)
    def fit(self,scalers):
        for scaler in scalers:
            scaler.fit(self.images)
##########################################################

if __name__ == '__main__':
    path = '/home/bbli/ML_Code/UNet/Data/'
    center = Standarize()
    pad_size = 88 ##assuming kernel =3 and padder applies this to both sides(which it does)
    pad = Padder(pad_size)
    transforms = Compose([center,pad])

    dataset = ISBIDataset(path,transforms)
    dataset.fit([center])
    checkTrainSetMean(dataset)
    train_loader = DataLoader(dataset,shuffle=True)

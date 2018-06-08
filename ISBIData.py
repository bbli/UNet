import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
import numpy as np
import ipdb

from utils import *

##########################################################
def label_format(img):
    img = rescale(img)
    img = img.astype(int)
    # ipdb.set_trace()
    # In theory, it shouldn't matter whether fg gets mapped to 0 or 1
    img = 1*np.logical_not(img) 
    
    img = torch.from_numpy(img).long()
    img = img.view(1,*img.shape)
    return img

def image_format(img):
    img = rescale(img)
    img = toTorch(img)
    return img

def toTorch(image):
    down_size,_ = image.shape
    new_image = torch.from_numpy(image).float()
    new_image = new_image.view(1,1,down_size,down_size)
    return new_image

def rescale(image):
    return image/255

class ISBIDataset(Dataset):
    def __init__(self,path,transform = None):
        self.images = io.imread(path+'train-volume.tif')
        self.labels = io.imread(path+'train-labels.tif')
        self.transform = transform
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
    pad_size = 160
    pad = Padder(pad_size)
    transforms = Compose([center,pad])

    dataset = ISBIDataset(path,transforms)
    dataset.fit([center])
    checkTrainSetMean(dataset)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
import numpy as np
import ipdb

class ISBIDataset(Dataset):
    def __init__(self,path):
        self.train = io.imread(path+'train-volume.tif')
        self.label = io.imread(path+'train-labels.tif')
    def __len__(self):
        return len(self.train)
    def __getitem__(self,index):
        image = self.train[index]
        label_image = self.label[index]
        return image_format(image),label_format(label_image)
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
##########################################################

if __name__ == '__main__':
    path = '/home/bbli/ML_Code/UNet/'
    dataset = ISBIDataset(path)

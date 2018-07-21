import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from skimage import io
from skimage import img_as_float
import os
import numpy as np
from skimage.transform import downscale_local_mean
import ipdb
from DataUtils import *

from utils import *
from UNet import *
    
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


def fixLabeling(labels):
    labels[labels==0] = 1
    return labels-1

class ParhyaleDataset(Dataset):
    def __init__(self,image_path,label_path,factor=5,transform=None):
        self.transform = transform
        self.images = stackImages(readImages(image_path))
        self.labels = stackImages(readImages(label_path))
        self.labels = fixLabeling(self.labels)
        if factor:
            self.images = downsize(self.images,factor)
            self.labels = downsize(self.labels,factor)
        print("Mean pixel value-before transforms: ", np.mean(self.images[0]))
        print("Percentage of cells in first image: ", np.mean(self.labels[0]))
        # printVariance(self.images)
    def fit(self,scalers):
        for scaler in scalers:
            scaler.fit(self.images)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            return imageToTorch(self.transform(image)),labelToTorch(label)
        else:
            return imageToTorch(image), labelToTorch(label)

if __name__=='__main__':
    ################### **Creating Dataset** #########################
    train_images_path = '/data/bbli/gryllus_disk_images/train/images/'
    train_labels_path = '/data/bbli/gryllus_disk_images/train/labels/'
    test_images_path = '/data/bbli/gryllus_disk_images/val/images/'
    test_labels_path = '/data/bbli/gryllus_disk_images/val/labels/'


    center = Standarize()
    pad_size = 160
    pad = Padder(pad_size)
    transforms = Compose([center,pad])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])
    ##########################################################
    train_dataset = ParhyaleDataset(train_images_path,train_labels_path,transform=transforms)
    train_dataset.fit([center])
    checkTrainSetMean(train_dataset)

    test_dataset = ParhyaleDataset(test_images_path,test_labels_path,transform=transforms)
    ################### **Export Variables** #########################
    train_loader = DataLoader(train_dataset,shuffle=True)
    test_loader = DataLoader(test_dataset,shuffle=True)


    img,label = next(iter(train_loader))
    print("Pad size: ",pad_size)
    # size = 700
    # img = torch.Tensor(1,1,size,size)
    # make into pytorch cuda variables

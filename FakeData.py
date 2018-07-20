import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np
from skimage.transform import downscale_local_mean
from sklearn.preprocessing import StandardScaler
import skimage.util as util
import ipdb

from utils import *
from DataUtils import *
from UNet import *
    
##########################################################

class FakeDataset(Dataset):
    def __init__(self,image_path,label_path,factor=None,transform=None):
        self.transform = transform
        self.images = np.load(image_path)
        ## Getting one image for testing purposes
        # if num_pic:
            # self.images = self.images[0:num_pic]
            # printVariance(self.images)
            # self.images = self.images.reshape(1,*self.images.shape)
        self.labels = np.load(label_path)
        if factor:
            self.images = downsize(self.images,factor)
        # print(np.mean(self.images[0]))
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



##########################################################
if __name__=='__main__':
    train_images_path = '/home/bbli/ML_Code/UNet/Data/fake/train_images.npy'
    train_labels_path = '/home/bbli/ML_Code/UNet/Data/fake/train_labels.npy'

    #### Defining transform and Dataset class######
    center1 = Standarize1()
    pad1 = Padder(100)
    transforms1 = Compose([center1,pad1])

    train_dataset1 = FakeDataset(train_images_path,train_labels_path,transform=transforms1)
    train_dataset1.fit([center1])

    center = Standarize()
    pad = Padder(100)
    transforms = Compose([center,pad])

    train_dataset = FakeDataset(train_images_path,train_labels_path,transform=transforms)
    train_dataset.fit([center])

    center_image1 = train_dataset1[0][0].numpy()
    center_image = train_dataset[0][0].numpy()

    checkTrainSetMean(train_dataset)
    ##########################################################

    train_set = DataLoader(train_dataset,shuffle=False)

    img,label = next(iter(train_set))
    # size = 700
    # img = torch.Tensor(1,1,size,size)
    # make into pytorch cuda variables
    img = tensor_format(img)
    label = tensor_format(label)

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
import skimage.util as util
import ipdb

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

def downsize(images,factor):
    new_tensor = downscale_local_mean(images[0],factors=(factor,factor))
    new_tensor = new_tensor.reshape(1,*new_tensor.shape)
    for image in images[1:]:
        new_image = downscale_local_mean(image,factors=(factor,factor))
        new_image = new_image.reshape(1,*new_image.shape)
        # ipdb.set_trace()
        new_tensor = np.concatenate((new_tensor,new_image),axis=0)
    return new_tensor

def fixLabeling(labels):
    labels[labels==0] = 1
    return labels-1

class ParhyaleDataset(Dataset):
    def __init__(self,image_path,label_path,factor=4,transform=None):
        self.transform = transform
        self.images = stackImages(readImages(image_path))
        self.labels = stackImages(readImages(label_path))
        self.labels = fixLabeling(self.labels)
        if factor:
            self.images = downsize(self.images,factor)
            self.labels = downsize(self.labels,factor)
        print("Mean pixel value of first image: ", np.mean(self.images[0]))
        print("Percentage of cells in first image: ", np.mean(self.labels[0]))
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
######################################################
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
        ## reshape so we can average images across samples for
        ## each spatial location
        length = len(images)
        images = images.reshape(length,-1)
        super().fit(images)

def Padder(factor):
    def f(image):
        return util.pad(image,factor,mode='constant',constant_values=0) 
    return f

##########################################################
if __name__=='__main__':
    train_images_path = '/data/bbli/gryllus_disk_images/train/images/'
    train_labels_path = '/data/bbli/gryllus_disk_images/train/labels/'

    #### Defining transform and Dataset class######
    center = Standarize()
    pad = Padder(100)
    transforms = Compose([center,pad])

    train_dataset = ParhyaleDataset(train_images_path,train_labels_path,factor=4,transform=transforms)
    train_dataset.fit([center])
    checkTrainSetMean(train_dataset)
    ##########################################################

    train_set = DataLoader(train_dataset,shuffle=True)

    img,label = next(iter(train_set))
    # size = 700
    # img = torch.Tensor(1,1,size,size)
    # make into pytorch cuda variables
    img = tensor_format(img)
    label = tensor_format(label)

    model = UNet().cuda()
    model.apply(weightInitialization)

    z = model(img)
    print("Dimension of output of Unet: "+str(z.shape))
    z,label = crop(z,label)
    print("Accuracy", score(z,label))

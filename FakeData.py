import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np
from skimage.transform import downscale_local_mean
from sklearn.preprocessing import StandardScaler
import skimage.util as util
import ipdb

from utils import *
from UNet import *
    
##########################################################
def downsize(images,factor):
    new_tensor = downscale_local_mean(images[0],factors=(factor,factor))
    new_tensor = new_tensor.reshape(1,*new_tensor.shape)
    for image in images[1:]:
        new_image = downscale_local_mean(image,factors=(factor,factor))
        new_image = new_image.reshape(1,*new_image.shape)
        # ipdb.set_trace()
        new_tensor = np.concatenate((new_tensor,new_image),axis=0)
    return new_tensor

class FakeDataset(Dataset):
    def __init__(self,image_path,label_path,factor=None,transform=None):
        self.transform = transform
        self.images = np.load(image_path)
        ## Getting one image for testing purposes
        # self.images = self.images[0]
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
######################################################

def imageToTorch(image):
    new_image = torch.from_numpy(image).float()
    ## extra dimension for the feature channel
    new_image = new_image.view(1,*image.shape)
    return new_image

def labelToTorch(image):
    return torch.from_numpy(image).long()

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
    def __init__(self):
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

def Padder(factor):
    def f(image):
        return util.pad(image,factor,mode='constant',constant_values=0) 
    return f


##########################################################
if __name__=='__main__':
    train_images_path = '/home/bbli/ML_Code/UNet/Data/fake/train_images.npy'
    train_labels_path = '/home/bbli/ML_Code/UNet/Data/fake/train_labels.npy'

    #### Defining transform and Dataset class######
    center = Standarize()
    pad = Padder(100)
    transforms = Compose([center,pad])

    train_dataset = FakeDataset(train_images_path,train_labels_path,transform=transforms)
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

    model = UNet().cuda(1)
    model.apply(weightInitialization)

    z = model(img)
    print(z.shape)
    print(score(z,label))

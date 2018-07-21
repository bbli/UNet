import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np
from skimage.transform import downscale_local_mean
from sklearn.preprocessing import StandardScaler
import skimage.util as util
import ipdb 

def downsize(images,factor):
    new_tensor = downscale_local_mean(images[0],factors=(factor,factor))
    new_tensor = new_tensor.reshape(1,*new_tensor.shape)
    for image in images[1:]:
        new_image = downscale_local_mean(image,factors=(factor,factor))
        new_image = new_image.reshape(1,*new_image.shape)
        # ipdb.set_trace()
        new_tensor = np.concatenate((new_tensor,new_image),axis=0)
    return new_tensor

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
class Standarize1(StandardScaler):
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

class Standarize():
    def fit(self,image_stack):
        self.mean_values=image_stack.mean(axis=0)
        self.std = image_stack.std(axis=0)+1e-8
    def __call__(self,image):
       return (image-self.mean_values)/self.std
       # return image-self.mean_values

def Padder(factor):
    def f(image):
        return util.pad(image,factor,mode='constant',constant_values=0) 
    return f

def imageToTorch(image):
    new_image = torch.from_numpy(image).float()
    ## extra dimension for the feature channel
    new_image = new_image.view(1,*image.shape)
    return new_image

def labelToTorch(image):
    return torch.from_numpy(image).long()

def printVariance(numpy_stack):
    # if len(images)>1:
    std = numpy_stack.std(axis=0)
    print("{} Images -> {} Standard Deviation".format(len(numpy_stack),std.mean()))
    # else:
        # print("Not enough images to take standard deviation")

def checkTrainSetMean(train_dataset):
    mean =0
    ## final mean should be 0 since each pixel location has been normalized to 0 mean, and we are adding them all up as random variables
    # numbers are -0.003, 
    for i,_ in enumerate(train_dataset):
        a = np.mean(train_dataset[i][0].numpy())
        mean += a 
    print("Mean pixel value-after transforms: {}".format(mean))

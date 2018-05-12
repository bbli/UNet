## Debug Functions
import ipdb
from inspect import getsource
from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch

def code(function):
    print(getsource(function))

## Training Functions
def tensor_format(tensor):
        new_tensor = Variable(tensor,requires_grad=False).cuda()
        return new_tensor

import seaborn as sns
def hist(image):
    sns.distplot(image)
    plt.show()

##########################################################
def score(outputs, labels):
    '''
    Input: outputs is a 4D Pytorch Variable, labels is a 3D Pytorch variable. 
    '''
    assert outputs.shape[-1] == labels.shape[-1], "Tensors should have the same spatial dimension"
    outputs,labels = reduceTo2D(outputs,labels)

    boo = (outputs== labels)
    return boo.mean()

def reduceTo2D(outputs,labels):
    outputs = outputs.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    ## reduce down to 3D tensor by argmaxing the feature channel
    outputs = np.argmax(outputs,axis=1)
    ## reduce down to an image
    labels = labels[0]
    outputs = outputs[0]
    return outputs,labels

def crop(outputs,labels):
    '''
    Will figure out which one is larger and then crop accordingly
    Input: outputs is a 4D Pytorch Variable, labels is a 3D Pytorch variable. 
    '''
    x = outputs.shape[-1]
    y = labels.shape[-1]
    diff = x-y
    index = abs(diff)
    if diff<0:
        return outputs, labels[0,index:,index:]
    elif diff>0:
        return outputs[:,:,index:,index:], labels
    else:
        return outputs,labels
##########################################################

def checkTrainSetMean(train_dataset):
    mean =0
    ## final mean should be 0 since each pixel location has been normalized to 0 mean, and we are adding them all up as random variables
    # numbers are -0.003, 
    for i,_ in enumerate(train_dataset):
        a = np.mean(train_dataset[i][0].numpy())
        mean += a 
    print("Mean pixel value: {}".format(mean))


def getWeightMap(dataloader):
    total_mean =0
    for img,label in dataloader:
        label = label.numpy()
        total_mean += label.mean()
        print("Label percentage: ",label.mean())
    total_mean = total_mean/len(dataloader)
    return np.array([total_mean,1-total_mean])

def showComparsion(output,label):
    '''
    Input: output is a 4D Pytorch Variable, label is a 3D Pytorch Variable
    '''
    output, label = reduceTo2D(output,label)
    fig = plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow(output)
    plt.title("Prediction")
    plt.subplot(1,2,2)
    plt.imshow(label)
    plt.title("Label")
    plt.show()
    # print("Call plt.show() to see prediction")

def reduceTo2D(outputs,labels):
    outputs = outputs.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    ## reduce down to 3D tensor by argmaxing the feature channel
    outputs = np.argmax(outputs,axis=1)
    ## reduce down to an image
    labels = labels[0]
    outputs = outputs[0]
    return outputs,labels

def cyclic(period):
    def f(episode):
        modulus = episode % period
        return 1/(1+0.05*modulus)
    return f

def imageToTorch(image):
    new_image = torch.from_numpy(image).float()
    ## extra dimension for the feature channel
    new_image = new_image.view(1,*image.shape)
    return new_image

def labelToTorch(image):
    return torch.from_numpy(image).long()


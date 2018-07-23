## Debug Functions
import ipdb
from inspect import getsource
from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from numpy import cos,pi
from sklearn.preprocessing import StandardScaler
import skimage.util as util
import torch.nn.functional as F

def code(function):
    print(getsource(function))

## Training Functions
def tensor_format(tensor):
        new_tensor = Variable(tensor,requires_grad=False).cuda(1)
        return new_tensor

import seaborn as sns
def hist(image):
    sns.distplot(image)
    plt.show()

def getFrequencyOfDeadNeurons(torch_variable):
    numpy_matrix = torch_variable.cpu().data.numpy()
    negative_values = (numpy_matrix<0).sum()
    total_number_of_values = getProductofTuple(numpy_matrix.shape)
    return negative_values/total_number_of_values

def getProductofTuple(matrix_shape):
    prod=1
    for i in matrix_shape:
        prod = prod*i
    return prod
##########################################################
def score(score_variable, labels):
    '''
    Input: outputs is a 4D Pytorch Variable, labels is a 3D Pytorch variable. 
    '''
    assert score_variable.shape[-1] == labels.shape[-1], "Tensors should have the same spatial dimension"
    cell_prob = getPrediction(score_variable)
    labels = reduceLabelTo2D(labels)
    boo = (cell_prob== labels)
    return boo.mean()

def getPrediction(score_variable):
    cell_prob = F.softmax(score_variable,dim=1)
    cell_prob = cell_prob.cpu().data.numpy()
    pred = np.argmax(cell_prob,axis=1)
    return pred[0]

def getSigmoidPred(score_variable):
    cell_prob = getSigmoidProb(score_variable) 
    cell_diff = cell_prob - 0.5 
    return np.heaviside(cell_diff,1)

def reduceLabelTo2D(labels):
    labels = labels.cpu().data.numpy()
    return labels[0]
##########################################################
def sigmoidScore(score_variable,label):
    # cell_prob = getSigmoidProb(score_variable)
    label = reduceLabelTo2D(label)
    # diff = np.abs(cell_prob - label)
    # boo = diff<0.5
    cell_pred = getSigmoidPred(score_variable)
    boo = (cell_pred==label)
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

def getCellProb(score_variable):
    ##create softmax probs
    cell_prob = F.softmax(score_variable,dim=1)
    cell_prob = cell_prob.cpu().data.numpy()
    ## get foreground probs
    cell_prob = cell_prob[0,1,:,:]
    return cell_prob

def getSigmoidProb(score_variable):
    '''
    returns a 2d numpy matrix of the probabilities for each pixel
    '''
    cell_prob = F.sigmoid(score_variable)
    cell_prob = cell_prob.cpu().data.numpy()
    return cell_prob[0,0]

def logImage(numpy_array):
    '''
    Converts numpy image into a 3D Torch Tensor
    '''
    numpy_array = numpy_array.reshape(1,*numpy_array.shape)
    numpy_array = torch.from_numpy(numpy_array)
    return numpy_array

def logInitialCellProb(torch_tensor,count,w,dict_of_images):
    if count==1:
        cell_prob = getCellProb(torch_tensor)
        w.add_image("Initial Cell Probability",logImage(cell_prob),count)
        dict_of_images["Initial Cell Prob"] = cell_prob
        print("Initial Cell Prob Mean:",cell_prob.mean())
        # print("Sample of Initial Cell Probabilties")
        # print(cell_prob[100:105,100:105])

def logInitialSigmoidProb(torch_tensor,count,w,dict_of_images):
    if count==1:
        cell_prob = getSigmoidProb(torch_tensor)
        w.add_image("Initial Cell Probability",logImage(cell_prob),count)
        dict_of_images["Initial Cell Prob"] = cell_prob
        print("Initial Cell Prob Mean:",cell_prob.mean())
        # print("Sample of Initial Cell Probabilties")
        # print(cell_prob[100:105,100:105])

def logFinalCellProb(score_variable,w,dict_of_images):
    final_cell_prob = getCellProb(score_variable)
    w.add_image("Final Cell Probability",logImage(final_cell_prob),1)
    dict_of_images["Final Cell Prob"] = final_cell_prob
    print("Final Cell Prob Mean:",final_cell_prob.mean())
    print("Sample of Final Cell Probabilties")
    print(final_cell_prob[100:105,100:105])

def logFinalSigmoidProb(score_variable,w,dict_of_images):
    final_cell_prob = getSigmoidProb(score_variable)
    w.add_image("Final Cell Probability",logImage(final_cell_prob),1)
    dict_of_images["Final Cell Prob"] = final_cell_prob
    print("Final Cell Prob Mean:",final_cell_prob.mean())
    print("Sample of Final Cell Probabilties")
    print(final_cell_prob[100:105,100:105])

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
        raise Exception("output size should be greater than label size")
    elif diff>0:
        return outputs[:,:,index:,index:], labels
    else:
        return outputs,labels
##########################################################



def getWeightMap(dataloader):
    total_mean =0
    for img,label in dataloader:
        label = label.numpy()
        total_mean += label.mean()
        # print("Label percentage: ",label.mean())
    total_mean = total_mean/len(dataloader)
    return np.array([total_mean,1-total_mean])

def showComparsion(output,label):
    '''
    Input: output is a 4D Pytorch Variable, label is a 3D Pytorch Variable
    Note this function may not work anymore because output is now a score variable.
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

def cosine(period):
    def f(episode):
        modulus = episode % period
        return 0.5*(1.1+cos(pi*modulus/period))
    return f

def decay_cosine(period):
    def f(episode):
        modulus = episode % period
        cycles = episode//period
        return 0.5*(1.1+cos(pi*modulus/period))/(cycles+1)
    return f

def doubler(time_steps):
    def f(episode):
        ratio = episode // time_steps
        return 2**ratio
    return f

def changeForBCELoss(output,label):
    return output[0],label.float()

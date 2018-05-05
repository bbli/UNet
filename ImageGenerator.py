import numpy as np
import math
import ipdb
from skimage import draw
from skimage import io
from skimage.filters import gaussian
from skimage.util import random_noise

##########################################################
##########################
def PointDistance(point,list_point):
    return math.sqrt(np.sum((point-list_point)**2))

def ValidPoint(point,current_list,radius):
    # point is a 2D numpy array
    # current_list is a list of 2D numpy arrays
    output = True
    for list_point in current_list:
        if PointDistance(point,list_point)<2*radius:
            output = False
    return output

def CreateCenterCoordinates(current_list,radius,length):
    while True:
        point = np.array([np.random.uniform(radius,length-radius),np.random.uniform(radius,length-radius)])
        if ValidPoint(point,current_list,radius):
            current_list.append(point)
            return point
##########################

def RandomImage(num,radius,length=500):
    image = np.zeros((length,length))
    list_of_centers=[]
    for _ in range(num):
        x,y = CreateCenterCoordinates(list_of_centers,radius,length)
        rr, cc = draw.circle(x,y,radius,shape=image.shape)
        image[rr,cc] = 1
    return image
##########################################################



if __name__ == '__main__':
    test_point = np.array([1,2])
    test_point2 = np.array([2,4])
    print(PointDistance(test_point,test_point2))
    
    num_train = 11
    num_test =1
    radius = 15
    train_labels = RandomImage(10,radius)
    train_labels = train_labels.reshape(1,*train_labels.shape)
    for _ in range(num_train-1):
        random_num = np.random.randint(1,20)
        img = RandomImage(random_num,radius)
        img= img.reshape(1,*img.shape)
        train_labels = np.concatenate((train_labels, img),axis=0)

    test_labels = RandomImage(10,radius)
    test_labels = test_labels.reshape(1,*test_labels.shape)

##########################################################
    for i,image in enumerate(train_labels):
        new_image = random_noise(image)
        new_image = gaussian(new_image,sigma=1)
        new_image = new_image.reshape(1,*new_image.shape)
        if i==0:
            train_images = new_image
        else:
            train_images = np.concatenate((train_images,new_image),axis=0)
    for i,image in enumerate(test_labels):
        new_image = random_noise(image)
        new_image = gaussian(new_image,sigma=1)
        new_image = new_image.reshape(1,*new_image.shape)
        if i==0:
            test_images = new_image
        else:
            test_images = np.concatenate((train_images,new_image),axis=0)

    
    np.save('fake/train_labels',train_labels)
    np.save('fake/test_labels',test_labels)
    np.save('fake/train_images',train_images)
    np.save('fake/test_images',test_images)

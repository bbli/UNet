import numpy as np
import math

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
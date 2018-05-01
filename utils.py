## Debug Functions
from inspect import getsource
from sys import getsizeof

def code(function):
    print(getsource(function))

## Training Functions
from torch.autograd import Variable
def tensor_format(tensor):
        new_tensor = Variable(tensor,requires_grad=False).cuda()
        return new_tensor

import seaborn as sns
def hist(image):
    sns.distplot(image)
    plt.show()

def score(outputs, labels):
    '''
    Assumptions: outputs and labels are Pytorch variables
    '''
    pred = argMax(outputs)
    truth = (pred == labels.data)
    return truth.sum()/len(truth)


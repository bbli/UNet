## Debug Functions
from inspect import getsource
from sys import getsizeof

def code(function):
    print(getsource(function))

## Training Functions
def tensor_format(tensor):
        new_tensor = Variable(tensor,requires_grad=False).cuda()
        return new_tensor

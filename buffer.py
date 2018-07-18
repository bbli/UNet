import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *
import numpy as np


## Testing if CrossEntropy needs probs or scores -> Turns out to be scores
criterion = nn.CrossEntropyLoss()

# input_data = torch.ones((3,2)).float()
# input_data = Variable(input_data,requires_grad=False)

# target = np.ones(3)
# target = torch.from_numpy(target).long()
# target = Variable(target,requires_grad = False)

# loss = criterion(input_data,target)

## Testing If CrossEntropy knows which dimension to take softmax-> guess so

input_data = torch.ones((1,2,2,2))
input_data[0,0,:,:] = 2
input_data = Variable(input_data,requires_grad=False)

target = torch.ones((1,2,2)).long()
target = Variable(target,requires_grad=False)

loss = criterion(input_data, target)


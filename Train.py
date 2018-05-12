from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb

from utils import *
from Data import *
from UNet import *

kernel_size=6
feature_maps=16

net = UNet(kernel_size,feature_maps).cuda()
net.apply(weightInitialization)
net.train()

learn_rate = 1e-2
momentum_rate = 0.8
cyclic_rate = 25
epochs = 50
alpha = 0.06
weight_map = getWeightMap(train_loader)
print("Weight Map: ", weight_map)
# weight_map = np.array([alpha,1-alpha])
training_parameters = "Learning Rate: {} \n Momentum: {} \n Cycle Length: {} \n Number of epochs: {}\n Weight Map: {}".format(learn_rate,momentum_rate,cyclic_rate, epochs, weight_map)
model_parameters = "Kernel Size: {} Initial Feature Maps: {}".format(kernel_size,feature_maps)

w = SummaryWriter()
w.add_text('Training Parameters',training_parameters)
w.add_text('Model Parameters',model_parameters)

# ipdb.set_trace()
# weight_map = np.array([0.01,0.99])

weight_map = tensor_format(torch.FloatTensor(weight_map))
criterion = nn.CrossEntropyLoss(weight=weight_map)


optimizer = optim.SGD(net.parameters(),lr = learn_rate,momentum=momentum_rate)
scheduler = LambdaLR(optimizer,lr_lambda=cosine(cyclic_rate))

count =0
for epoch in range(epochs):
    for idx,(img,label) in enumerate(train_loader):
        count += 1
        ################### **Training the Network** #########################
        img, label = tensor_format(img), tensor_format(label)
        # learn_rate = next(lr_generator)

        ########## Create NEW one after every ITERATION for WEIGHT MAP
        # optimizer = optim.SGD(net.parameters(),lr=learn_rate,momentum=0.90, nesterov=True,weight_decay=1e-4)
        optimizer.zero_grad()
        # printModel(net,optimizer)
        # ipdb.set_trace()
        # before_weights = weightMag(net)
        #########################
        output = net(img)
        output, label = crop(output,label)
        acc = score(output,label)
        # print(acc)
        w.add_scalar(' Accuracy', float(acc),count)
        # print("Accuracy: {}".format(acc))
        #########################
        loss = criterion(output, label)

        w.add_scalar('Loss', loss.data[0],count)
        # print("Loss value: {}".format(loss))
        #########################
        loss.backward()
        optimizer.step()
        scheduler.step()
        ################################################################
        # after_weights =weightMag(net)
        # relDiff_list = relDiff(before_weights,after_weights)
        # relDiff_dict = listToDict(relDiff_list)
        # w.add_scalars('LayerChanges',relDiff_dict,count)


net.eval()
for img, label in test_loader:
    img, label = tensor_format(img), tensor_format(label)
    output = net(img)
    output, label = crop(output,label)
    test_score = score(output,label)
    print("Test score: {}".format(test_score))
    output, label = reduceTo2D(output,label)
w.add_text("Test score","Test score: "+str(test_score))
w.close()

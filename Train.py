from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import torch.nn as nn
import ipdb

import utils
from FakeData import *
from UNet import *

################### **Creating Dataset** #########################
train_path = '/home/bbli/ML_Code/UNet/fake/train_images.npy'
test_path = '/home/bbli/ML_Code/UNet/fake/train_images.npy'

center = Standarize()
transforms = Compose([center,toTorch ])
# transforms = Compose ([ToTensor(),Standarize(0,1)])

train_dataset = FakeDataset(train_path,transform=transforms)
train_dataset.fit([center])

train_loader = DataLoader(train_dataset,shuffle=True)
##########################################################

criterion = nn.CrossEntropyLoss()
net = UNet().cuda()
net.apply(weightInitialization)
net.train()
epochs = 5
count =0
for epoch in range(epochs):
    for idx,(img,label) in enumerate(train_loader):
        count += 1
        ################### **Formatting Data** #########################
        img, label = tensor_format(img), tensor_format(label)
        label = label.long()
        ################### **Training the Network** #########################
        # learn_rate = next(lr_generator)

        ########## Create NEW one after every ITERATION for WEIGHT MAP
        # optimizer = optim.SGD(net.parameters(),lr=learn_rate,momentum=0.90, nesterov=True,weight_decay=1e-4)
        optimizer = optim.Adam(net.parameters(),lr = 0.01)
        optimizer.zero_grad()
        # printModel(net,optimizer)
        # ipdb.set_trace()
        # before_weights = weightMag(net)
        #########################
        output = net(img)

        acc = score(output,label)
        w.add_scalar('Accuracy', acc,count)
        # print("Accuracy: {}".format(acc))
        #########################
        loss = criterion(output, label)

        w.add_scalar('Loss', loss.data[0],count)
        # print("Loss value: {}".format(loss))
        #########################
        loss.backward()
        optimizer.step()
        ################################################################
        # after_weights =weightMag(net)
        # relDiff_list = relDiff(before_weights,after_weights)
        # relDiff_dict = listToDict(relDiff_list)
        # w.add_scalars('LayerChanges',relDiff_dict,count)

w.close()


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb

from utils import *
from FakeData import *
from UNet import *

################### **Creating Dataset** #########################
train_images_path = '/home/bbli/ML_Code/UNet/Data/fake/train_images.npy'
train_labels_path = '/home/bbli/ML_Code/UNet/Data/fake/train_labels.npy'
test_images_path = '/home/bbli/ML_Code/UNet/Data/fake/test_images.npy'
test_labels_path = '/home/bbli/ML_Code/UNet/Data/fake/test_labels.npy'

center = Standarize()
pad = Padder(100)
transforms = Compose([center,pad])
# transforms = Compose ([ToTensor(),Standarize(0,1)])
##########################################################
train_dataset = FakeDataset(train_images_path,train_labels_path,transform=transforms)
train_dataset.fit([center])
checkTrainSetMean(train_dataset)

train_loader = DataLoader(train_dataset,shuffle=True)
##########################################################
test_dataset = FakeDataset(test_images_path,test_labels_path,transform=transforms)
test_loader = DataLoader(test_dataset,shuffle=True)
##########################################################

net = UNet().cuda()
net.apply(weightInitialization)
net.train()

learn_rate = 1e-2
momentum_rate = 0.8
cyclic_rate = 25
epochs = 50
weight_map = getWeightMap(train_loader)
training_parameters = "Learning Rate: {} \n Momentum: {} \n Cycle Length: {} \n Number of epochs: {}\n Weight Map: {}".format(learn_rate,momentum_rate,cyclic_rate, epochs, weight_map)

w = SummaryWriter()
w.add_text('Training Parameters',training_parameters)

# ipdb.set_trace()
# weight_map = np.array([0.01,0.99])

alpha = 0.5
weight_map = np.array([alpha,1-alpha])
weight_map = tensor_format(torch.FloatTensor(weight_map))
criterion = nn.CrossEntropyLoss(weight=weight_map)


optimizer = optim.SGD(net.parameters(),lr = learn_rate,momentum=momentum_rate)
scheduler = LambdaLR(optimizer,lr_lambda=cyclic(cyclic_rate))

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

w.close()

net.eval()
for img, label in test_loader:
    img, label = tensor_format(img), tensor_format(label)
    output = net(img)
    output, label = crop(output,label)
    print("Test score: {}".format(score(output,label)))
    showComparsion(output,label)

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
def learningRateFinder(net,max_learn_rate):
    weight_map = getWeightMap(train_loader)
    print("Weight Map: ", weight_map)
    weight_map = tensor_format(torch.FloatTensor(weight_map))

    criterion = nn.CrossEntropyLoss(weight=weight_map)
    optimizer = optim.SGD(net.parameters(),lr=1e-6,momentum=0.8)
    scheduler = LambdaLR(optimizer, lr_lambda = doubler(5))

    group = next(iter(optimizer.param_groups))
    loss_list = []
    lr_list = []

    while group['lr']<max_learn_rate:
        for idx,(img,label) in enumerate(train_loader):
            img,label = tensor_format(img), tensor_format(label)

            output = net(img)
            output, label = crop(output,label)

            loss = criterion(output,label)
            loss_list.append(loss.data[0])
            lr_list.append(group['lr'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


    plt.subplot(1,2,1)
    plt.plot(loss_list)
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(lr_list)
    plt.title("Learning Rate")
    plt.show()

if __name__ == '__main__':
    kernel_size=6
    feature_maps=16

    net = UNet(kernel_size,feature_maps).cuda()
    net.apply(weightInitialization)
    net.train()
    learningRateFinder(net,1)

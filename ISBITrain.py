from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb
import scipy.misc

from utils import *
from ISBIData import *
from UNet import *
def trainModel(lr,train_loader,w):

    learn_rate = lr
    momentum_rate = 0.8
    cyclic_rate = 80
    epochs = 62
    kernel_size = 3
    feature_maps = 32

    net = ThreeLayerUNet(kernel_size,feature_maps).cuda(1)
    net.apply(weightInitialization)
    net.train()

    # alpha = 0.06
    # weight_map = np.array([alpha,1-alpha])
    weight_map = getWeightMap(train_loader)
    # print("Weight Map: ", weight_map)
    training_parameters = "SGD Learning Rate: {} \n Momentum: {} \n Cycle Length: {} \n Number of epochs: {}\n Weight Map: {}".format(learn_rate,momentum_rate,cyclic_rate, epochs, weight_map)
    model_parameters = "Kernel Size: {} Initial Feature Maps: {}".format(kernel_size,feature_maps)

    w.add_text('Training Parameters',training_parameters)
    w.add_text('Model Parameters',model_parameters)

    weight_map = tensor_format(torch.FloatTensor(weight_map))
    criterion = nn.CrossEntropyLoss(weight=weight_map)


    optimizer = optim.SGD(net.parameters(),lr = learn_rate,momentum=momentum_rate)
    # optimizer = optim.Adam(net.parameters(),lr = 0.01,betas=(0.9,learn_rate))
    scheduler = LambdaLR(optimizer,lr_lambda=cosine(cyclic_rate))

    count =0
    for epoch in range(epochs):
        for idx,(img,label) in enumerate(train_loader):
            count += 1
            ################### **Feed Foward** #########################
            img, label = tensor_format(img), tensor_format(label)

            output = net(img)
            output, label = crop(output,label)
            loss = criterion(output, label)

            ################### **Logging** #########################
            w.add_scalar('Loss', loss.data[0],count)
            # print("Loss value: {}".format(loss))

            acc = score(output,label)
            w.add_scalar('Accuracy', float(acc),count)
            # print("Accuracy: {}".format(acc))
            ################### **Update Back** #########################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net


learning_parameters = [1.2e-2,1e-2,8e-3]
# beta_parameters = [0.96,0.99,0.995,0.999]
run_count = 0

for lr in learning_parameters:
    w = SummaryWriter()

    run_count += 1
    print("Run :",run_count)
    print("Learning Rate: {}".format(lr))

    model = trainModel(lr,train_loader,w)

    w.close()

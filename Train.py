from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb
import scipy.misc
import os

from utils import *
from Data import readImages,stackImages,downsize,fixLabeling,ParhyaleDataset,Standarize,Padder,test_loader
from UNet import *


def dataCreator(ks):
    lookup_table = np.zeros(20,dtype='int16')
    ## 3 Layers
    lookup_table[3]=45
    lookup_table[4]=62
    lookup_table[5]=80
    lookup_table[6]=100
    lookup_table[7]=120
    lookup_table[8]=137
    lookup_table[9]=156

    ## 4 Layers
    # lookup_table[3]=45
    # lookup_table[4]=62
    # lookup_table[5]=80
    # lookup_table[6]=100
    # lookup_table[7]=120
    # lookup_table[8]=135
    # lookup_table[9]=155
    ################### **Creating Dataset** #########################
    train_images_path = '/data/bbli/gryllus_disk_images/train/images/'
    train_labels_path = '/data/bbli/gryllus_disk_images/train/labels/'
    test_images_path = '/data/bbli/gryllus_disk_images/val/images/'
    test_labels_path = '/data/bbli/gryllus_disk_images/val/labels/'


    center = Standarize()
    pad_size = lookup_table[ks]
    # assert pad_size != 0, "You have not initialized the padding for this kernel size"
    print("Pad size: ",pad_size)
    pad = Padder(pad_size)
    transforms = Compose([center,pad])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])
    ##########################################################
    train_dataset = ParhyaleDataset(train_images_path,train_labels_path,transform=transforms)
    train_dataset.fit([center])
    checkTrainSetMean(train_dataset)

    test_dataset = ParhyaleDataset(test_images_path,test_labels_path,transform=transforms)
    ################### **Export Variables** #########################
    train_loader = DataLoader(train_dataset,shuffle=True)
    test_loader = DataLoader(test_dataset,shuffle=True)
    return train_loader,test_loader

def trainModel(ks,fm,lr,train_loader,w):

    kernel_size = ks
    feature_maps = fm
    learn_rate = lr
    momentum_rate = 0.8
    cyclic_rate = 31
    epochs = 60

    net = UNet(kernel_size,feature_maps).cuda(1)
    # net.apply(weightInitialization)
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
    # optimizer2 = optim.Adam(net.parameters(),lr = 0.5*learn_rate)
    scheduler = LambdaLR(optimizer,lr_lambda=cosine(cyclic_rate))

    count =0
    for epoch in range(epochs):
        for idx,(img,label) in enumerate(train_loader):
            count += 1
            ################### **Feed Foward** #########################
            img, label = tensor_format(img), tensor_format(label)

            output = net(img)
            output, label = crop(output,label)
            logInitialCellProb(output,count,w,g_dict_of_images)
            loss = criterion(output, label)

            ################### **Logging** #########################
            w.add_scalar('Loss', loss.data[0],count)
            # print("Loss value: {}".format(loss))

            acc = score(output,label)
            w.add_scalar('Accuracy', float(acc),count)
            # print("Accuracy: {}".format(acc))
            ################### **Update Back** #########################
            # if epoch<45:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # else:
                # optimizer2.zero_grad()
                # loss.backward()
                # optimizer2.step()
    return net

def testModel(net,test_loader,w):
    net.eval()
    for i,(img,label) in enumerate(test_loader):
        img, label = tensor_format(img), tensor_format(label)
        output = net(img)
        output, label = crop(output,label)
        test_score = score(output,label)

        ################### **Prep for Logging** #########################
        logFinalCellProb(output,w,g_dict_of_images)
        seg = getPrediction(output)
        label = reduceLabelTo2D(label)
        ################### **Logging** #########################
        # w.add_image("Input",img[0],i)
        ## These two will be LongTensors, which should be ok since values are either 1 or 0
        w.add_image("Segmentation",logImage(seg),i)
        w.add_image("Label",logImage(label),i)
        w.add_text("Test score","Test score: "+str(test_score))
        # scipy.misc.imsave("pics/"+str(label)+".tiff",output)
    return test_score


feature_maps=32
ks = 8
lr = 3e-2
os.chdir('one_train_image')
# os.chdir('debug')

g_dict_of_images={}
train_loader,test_loader = dataCreator(ks)
w = SummaryWriter()
w.add_text("Thoughts","UNet doesn't take softmax anymore, so in theory, things should work now, because we are not being pushed to 50 percent now")
print("Kernel Size: {} Learning Rate: {}".format(ks,lr))
model = trainModel(ks,feature_maps,lr,train_loader,w)
test_score = testModel(model,test_loader,w)
print("Test score: ",str(test_score))

w.close()
# torch.save(best_model.state_dict(),'best_model.pt')
final_cell_prob = g_dict_of_images['Final Cell Prob']
print(final_cell_prob.mean())
print(final_cell_prob[100:105,100:105])
# cell_prob= g_dict_of_images['Inital Cell Prob']
# seg = cell_prob*(cell_prob>0.5)
# import matplotlib.pyplot as plt
# plt.imshow(seg,cmap='gray')
# plt.grid(b=False)

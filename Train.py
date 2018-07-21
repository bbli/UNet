from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb
import scipy.misc
import os
# from FakeData import *
from Data import *
from DataUtils import *

from utils import *
# from Data import readImages,stackImages,downsize,fixLabeling,ParhyaleDataset,Standarize,Padder,test_loader
from UNet import *

def initialNetGenerator(ks,fm,train_loader):
    good_net = False
    while not good_net:
        net = UNet(ks,fm).cuda(1)
        net.apply(weightInitialization)
        net.train()
        list_of_booleans=[]
        for img,label in train_loader:
            img, label = tensor_format(img), tensor_format(label)
            output = net(img)
            output, label = crop(output,label)
            cell_prob_mean = getCellProb(output).mean()
            diff = abs(cell_prob_mean-0.5)
            list_of_booleans.append(diff>0.2)
        outlier = any(list_of_booleans)
        if outlier:
            pass
        else:
            # print("One Initial Cell Probability: ",cell_prob_mean)
            return net

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
    # train_images_path = '/home/bbli/ML_Code/UNet/Data/fake1/train_images.npy'
    # train_labels_path = '/home/bbli/ML_Code/UNet/Data/fake1/train_labels.npy'
    # test_images_path = '/home/bbli/ML_Code/UNet/Data/fake1/test_images.npy'
    # test_labels_path = '/home/bbli/ML_Code/UNet/Data/fake1/test_labels.npy'


    center = Standarize()
    pad_size = lookup_table[ks]
    # assert pad_size != 0, "You have not initialized the padding for this kernel size"
    # print("Pad size: ",pad_size)
    pad = Padder(pad_size)
    transforms = Compose([center,pad])
    # transforms = Compose ([ToTensor(),Standarize(0,1)])
    ##########################################################
    # print("Hi")
    train_dataset = ParhyaleDataset(train_images_path,train_labels_path,transform=transforms)
    # train_dataset = FakeDataset(train_images_path,train_labels_path,transform=transforms)
    train_dataset.fit([center])
    checkTrainSetMean(train_dataset)

    test_dataset = ParhyaleDataset(test_images_path,test_labels_path,transform=transforms)
    # test_dataset = FakeDataset(test_images_path,test_labels_path,transform=transforms)
    ################### **Export Variables** #########################
    train_loader = DataLoader(train_dataset,shuffle=True)
    test_loader = DataLoader(test_dataset,shuffle=True)
    return train_loader,test_loader

def trainModel(ks,fm,lr,train_loader,w):

    kernel_size = ks
    feature_maps = fm
    learn_rate = lr
    momentum_rate = 0.75
    cyclic_rate = 120
    epochs = 60

    net = initialNetGenerator(kernel_size,feature_maps,train_loader)
    ipdb.set_trace()

    alpha = 0.4
    # weight_map = np.array([alpha,1-alpha])
    weight_map = getWeightMap(train_loader)
    # print("Weight Map: ", weight_map)
    training_parameters = "SGD Learning Rate: {} \n Momentum: {} \n Cycle Length: {} \n Number of epochs: {}\n Weight Map: {}".format(learn_rate,momentum_rate,cyclic_rate, epochs, weight_map)
    model_parameters = "Kernel Size: {} Initial Feature Maps: {}".format(kernel_size,feature_maps)

    w.add_text('Training Parameters',training_parameters)
    w.add_text('Model Parameters',model_parameters)

    weight_map = tensor_format(torch.FloatTensor(weight_map))
    criterion = nn.CrossEntropyLoss(weight=weight_map)
    # criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(net.parameters(),lr = learn_rate,momentum=momentum_rate)
    optimizer2 = optim.SGD(net.parameters(),lr = 0.2*learn_rate,momentum=0.9*momentum_rate)
    scheduler = LambdaLR(optimizer,lr_lambda=cosine(cyclic_rate))
    scheduler2 = LambdaLR(optimizer,lr_lambda=cosine(cyclic_rate))

    count =0
    for epoch in range(epochs):
        for img,label in train_loader:
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
            w.add_scalar('Percentage of Dead Neurons',net.final_conv_dead_neurons,count)
            # print("Accuracy: {}".format(acc))
            ################### **Update Back** #########################
            if epoch<40:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
                scheduler2.step()
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


fm =32
# ks = 6
# lr = 1e-3
# os.chdir('level_out_loss/initial_cell_prob')
# os.chdir('level_out_loss/learn_rate')
# os.chdir('level_out_loss/fake1')

os.chdir('level_out_loss/hyper')
# os.chdir('level_out_loss/num_pic')
# os.chdir('level_out_loss/normalization')
# os.chdir('debug')
count = 0
dict_of_image_dicts ={}
kernel_list = [3,5,8]
learn_rate_list = [3e-4,1e-3,4e-3,8e-3]
for ks in kernel_list:
    for lr in learn_rate_list:
        count += 1
        print("Run:",count)
        g_dict_of_images={}

        train_loader,test_loader = dataCreator(ks)
        w = SummaryWriter()
        # w.add_text("Thoughts","Shit forgot about equating the kernel size")
        # print("Kernel Size: {} Learning Rate: {}".format(ks,lr))
        model = trainModel(ks,fm,lr,train_loader,w)
        test_score = testModel(model,test_loader,w)
        print("Test score: ",str(test_score))
        w.close()

        string = "ks_"+str(ks)+"lr_"+str(lr)
        dict_of_image_dicts[string]=g_dict_of_images

# torch.save(best_model.state_dict(),'best_model.pt')
# print("Log post train thoughts:")
# cell_prob= g_dict_of_images['Inital Cell Prob']
# seg = cell_prob*(cell_prob>0.5)
# import matplotlib.pyplot as plt
# plt.imshow(seg,cmap='gray')
# plt.grid(b=False)

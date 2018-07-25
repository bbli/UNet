from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb
import scipy.misc
import os
from DataUtils import *
# from FakeData import *
# from Data import *
from ISBIData import *

from utils import *
# from Data import readImages,stackImages,downsize,fixLabeling,ParhyaleDataset,Standarize,Padder,test_loader
from UNet import *
class DiceLoss(nn.Module):
    def __init__(self,smooth_factor=1):
        super().__init__()
        self.smooth_factor = smooth_factor
    def forward(self,scores_matrix,targets_matrix):
        '''
        Make sure targets_matrix has type float
        we assume both torch matrices are 3D
        '''
        # assert scores_matrix.shape == targets_matrix.shape, "scores and targets not the same shape"
        # num_pixels = getProductofTuple(scores_matrix.shape)
        # smooth = num_pixels*self.smooth_factor

        preds_matrix = F.sigmoid(scores_matrix) 
        ## /2 to account for 2* in next line
        pic_intersection = preds_matrix*targets_matrix+self.smooth_factor/2
        pic_union = preds_matrix+targets_matrix+self.smooth_factor

        overlap = (2*pic_intersection.sum())/(pic_union.sum())
        return 1-overlap
        
class UnBiasedDiceLoss(nn.Module):
    def __init__(self,smooth_factor=0.5,fg_weight=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.fg_weight = fg_weight
    def forward(self,scores_matrix,targets_matrix):
        '''
        Make sure targets_matrix has type float
        we assume both torch matrices are 3D
        '''
        preds_matrix = F.sigmoid(scores_matrix) 
        ## /2 to account for 2* in next line
        pic_intersection = preds_matrix*targets_matrix+self.smooth_factor/2
        pic_union = preds_matrix+targets_matrix+self.smooth_factor
        fg_overlap = (2*pic_intersection.sum())/(pic_union.sum())

        bg_preds_matrix = 1-preds_matrix
        bg_targets = 1-targets_matrix
        ## /2 to account for 2* in next line
        bg_pic_intersection = bg_preds_matrix*bg_targets+self.smooth_factor/2
        bg_pic_union = bg_preds_matrix+bg_targets+self.smooth_factor
        bg_overlap = (2*bg_pic_intersection.sum())/(bg_pic_union.sum())
        return 2-self.fg_weight*fg_overlap-bg_overlap
        # return 1-bg_overlap

# class BinaryCrossEntropy(nn.Module):
    # def __init__(self):
        # super().__init__()
    # def forward(self,scores_matrix,targets_matrix):

def initialNetGenerator(ks,fm,train_loader):
    good_net = False
    while not good_net:
        net = UNet(ks,fm,show=False).cuda(1)
        net.apply(weightInitialization)
        net.train()
        list_of_booleans=[]
        for img,label in train_loader:
            img, label = tensor_format(img), tensor_format(label)
            output = net(img)
            output, label = crop(output,label)
            cell_prob_mean = getCellProb(output).mean()
            # cell_prob_mean = getSigmoidProb(output).mean()
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
    ## 2 Layers, Image Size 401
    # lookup_table[3]=20
    # lookup_table[4]=30
    # lookup_table[5]=38
    # lookup_table[6]=47
    # lookup_table[7]=55
    # lookup_table[8]=65
    # lookup_table[9]=75
    ## 3 Layers, Image Size 401
    lookup_table[3]=45
    lookup_table[4]=62
    lookup_table[5]=80
    lookup_table[6]=100
    lookup_table[7]=120
    lookup_table[8]=137
    lookup_table[9]=156

    ## 4 Layers, Image Size 401
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
    path = '/home/bbli/ML_Code/UNet/Data/'


    # center = Standarize()
    center1 = Standarize()
    pad_size = lookup_table[ks]
    assert pad_size != 0, "You have not initialized the padding for this kernel size"
    # print("Pad size: ",pad_size)
    pad = Padder(pad_size)
    # transforms = Compose([center,pad])
    transforms1 = Compose([center1,pad])
    ##########################################################
    # print("Hi")
    # train_dataset = ParhyaleDataset(train_images_path,train_labels_path,transform=transforms)
    train_dataset1 = ISBIDataset(path,transforms1,factor=2)
    # train_dataset = FakeDataset(train_images_path,train_labels_path,transform=transforms)

    # train_dataset.fit([center])
    train_dataset1.fit([center1])
    # checkTrainSetMean(train_dataset)
    checkTrainSetMean(train_dataset1)

    # test_dataset = ParhyaleDataset(test_images_path,test_labels_path,transform=transforms)
    # test_dataset = FakeDataset(test_images_path,test_labels_path,transform=transforms)
    ################### **Export Variables** #########################
    # train_loader = DataLoader(train_dataset,shuffle=True)
    train_loader1 = DataLoader(train_dataset1,shuffle=True)
    # test_loader = DataLoader(test_dataset,shuffle=True)
    # return train_loader,test_loader
    return train_loader1

## Include Smoothing factor
def trainModel(lr,ks,fm,train_loader,w):

    kernel_size = ks
    feature_maps = fm
    learn_rate = lr
    momentum_rate = 0.75
    # momentum_rate = m
    cyclic_rate = 120
    # total_num_iterations = 80
    epochs = 40

    net = initialNetGenerator(kernel_size,feature_maps,train_loader)

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
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = UnBiasedDiceLoss(fg_weight=fg)
    # criterion = UnBiasedDiceLoss()
    # criterion1 = DiceLoss()
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
            # logInitialSigmoidProb(output,count,w,g_dict_of_images)
            ## also works for Dice Loss
            loss = criterion(output,label)
            # loss = criterion(*changeForBCEAndDiceLoss(output,label))
            # loss1 = criterion1(*changeForBCEAndDiceLoss(output,label))

            ################### **Logging** #########################
            # w.add_scalar('UnBiasedDiceLossLoss', loss.data[0],count)
            # w.add_scalar('Overlap',getSoftOverLap(output,label),count)
            w.add_scalar('Loss', loss.data[0],count)
            # print("Loss value: {}".format(loss))

            acc = score(output,label)
            # acc = sigmoidScore(output,label)
            w.add_scalar('Accuracy', float(acc),count)
            w.add_scalar('Percentage of Dead Neurons',net.final_conv_dead_neurons,count)
            # print("Accuracy: {}".format(acc))
            ################### **Update Back** #########################
            # if epoch<34:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # else:
                # optimizer2.zero_grad()
                # loss.backward()
                # optimizer2.step()
                # scheduler2.step()
    return net

def testModel(net,test_loader,w):
    net.eval()
    for i,(img,label) in enumerate(test_loader):
        img, label = tensor_format(img), tensor_format(label)
        output = net(img)
        output, label = crop(output,label)
        test_score = score(output,label)
        # test_score = sigmoidScore(output,label)

        ################### **Prep for Logging** #########################
        logFinalCellProb(output,w,g_dict_of_images)
        # logFinalSigmoidProb(output,w,g_dict_of_images)
        seg = getPrediction(output)
        # seg = getSigmoidPred(output)
        label = reduceLabelTo2D(label)
        ################### **Logging** #########################
        # w.add_image("Input",img[0],i)
        ## These two will be LongTensors, which should be ok since values are either 1 or 0
        w.add_image("Segmentation",logImage(seg),i)
        w.add_image("Label",logImage(label),i)
        w.add_text("Test score","Test score: "+str(test_score))
        # scipy.misc.imsave("pics/"+str(label)+".tiff",output)
        break
    return test_score

fm =32
ks = 3
lr = 8e-3
# os.chdir('level_out_loss/learn_rate')
# os.chdir('level_out_loss/fake1')
# os.chdir('level_out_loss/num_pic')
# os.chdir('level_out_loss/normalization')
# os.chdir('binary_loss')
# os.chdir('dice_loss/smooth_hyper')
os.chdir('final')

# os.chdir('two_layer')
# os.chdir('debug')
# os.chdir('debug_dice_loss')
count = 0
dict_of_image_dicts ={}
# learn_rate_list = [2e-2,8e-3,2e-3]
# smooth_factor_list = [0.3,1,2]
# fg_weight_list = [0.5,1,1.5]
# kernel_list = [3,5,8]
# fm_list = [32,16,8]
g_dict_of_images={}
# for lr in learn_rate_list:
# for fg in fg_weight_list:
for _ in range(4):
    count += 1
    print("Run:",count)
    # train_loader,test_loader = dataCreator(ks)
    train_loader = dataCreator(ks)
    w = SummaryWriter()
    w.add_text("Thoughts","Now actually with cross entropy since I need to change UNet back to 2 feature maps")
    # print("Smoothing Factor: {} Learning Rate: {}".format(s,lr))
    # print("FG Factor: {} Learning Rate: {}".format(fg,lr))
    # print("Kernel Size: {} Learning Rate: {}".format(ks,lr))
    # print("Feature Maps: {} Learning Rate: {}".format(fm,lr))
    model = trainModel(lr,ks,fm,train_loader,w)
    # test_score = testModel(model,test_loader,w)
    test_score = testModel(model,train_loader,w)
    # print("Test score: ",str(test_score))
    w.close()

    # string = "ks_"+str(ks)+"lr_"+str(lr)
    # dict_of_image_dicts[string]=g_dict_of_images
###### Thoughts: Trying SGD again with very low learning rate

# torch.save(best_model.state_dict(),'best_model.pt')
# print("Log post train thoughts:")
# cell_prob= g_dict_of_images['Inital Cell Prob']
# seg = cell_prob*(cell_prob>0.5)
# import matplotlib.pyplot as plt
# plt.imshow(seg,cmap='gray')
# plt.grid(b=False)

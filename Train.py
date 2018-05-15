from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import ipdb

from utils import *
from Data import readImages,stackImages,downsize,fixLabeling,ParhyaleDataset,Standarize,Padder,test_loader
from UNet import *


def dataCreator(ks):
    lookup_table = np.zeros(20,dtype='int16')
    lookup_table[6]=100
    lookup_table[7]=120
    lookup_table[8]=135
    lookup_table[9]=155
    ################### **Creating Dataset** #########################
    train_images_path = '/data/bbli/gryllus_disk_images/train/images/'
    train_labels_path = '/data/bbli/gryllus_disk_images/train/labels/'
    test_images_path = '/data/bbli/gryllus_disk_images/val/images/'
    test_labels_path = '/data/bbli/gryllus_disk_images/val/labels/'


    center = Standarize()
    pad_size = lookup_table[ks]
    assert pad_size != 0, "You have not initialized the padding for this kernel size"
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
    cyclic_rate = 80
    epochs = 62

    net = UNet(kernel_size,feature_maps).cuda(1)
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

def testModel(net,test_loader,w):
    net.eval()
    for i,(img,label) in enumerate(test_loader):
        img, label = tensor_format(img), tensor_format(label)
        output = net(img)
        output, label = crop(output,label)
        test_score = score(output,label)
        output, label = reduceTo2D(output,label)
        
        ################### **Logging** #########################
        w.add_image("Input",img[0],i)
        ## These two will be LongTensors, which should be ok since values are either 1 or 0
        w.add_image("Prediction",logImage(output),i)
        w.add_image("Label",logImage(label),i)
        w.add_text("Test score","Test score: "+str(test_score))
    return test_score

def logImage(numpy_array):
    '''
    Converts numpy image into a 3D Torch Tensor
    '''
    numpy_array = numpy_array.reshape(1,*numpy_array.shape)
    numpy_array = torch.from_numpy(numpy_array)
    return numpy_array



kernel_size_parameters = [7,8,9]
feature_maps=32
learning_parameters = [1.2e-2,1e-2,8e-3]
# beta_parameters = [0.96,0.99,0.995,0.999]
run_count = 0
models_list =[]
test_score_list = []
best_percentage =0

for i,ks in enumerate(kernel_size_parameters):
    train_loader,test_loader = dataCreator(ks)
    for j,lr in enumerate(learning_parameters):
        w = SummaryWriter()

        run_count += 1
        print("Run :",run_count)
        print("Kernel Size: {} Learning Rate: {}".format(ks,lr))

        model = trainModel(ks,feature_maps,lr,train_loader,w)
        models_list.append(model)

        test_score = testModel(model,test_loader,w)
        test_score_list.append(test_score)
        print("Test score: ",str(test_score))

        if test_score>best_percentage:
            best_model = model
            best_percentage = test_score

        w.close()
torch.save(best_model.state_dict(),'best_model.pt')

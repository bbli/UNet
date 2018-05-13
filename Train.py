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

def trainModel(ks,fm,lr,w):
    kernel_size = ks
    feature_maps = fm
    learn_rate = lr
    momentum_rate = 0.8
    cyclic_rate = 25
    epochs = 50

    net = UNet(kernel_size,feature_maps).cuda()
    net.apply(weightInitialization)
    net.train()

    # alpha = 0.06
    # weight_map = np.array([alpha,1-alpha])
    weight_map = getWeightMap(train_loader)
    print("Weight Map: ", weight_map)
    training_parameters = "Learning Rate: {} \n Momentum: {} \n Cycle Length: {} \n Number of epochs: {}\n Weight Map: {}".format(learn_rate,momentum_rate,cyclic_rate, epochs, weight_map)
    model_parameters = "Kernel Size: {} Initial Feature Maps: {}".format(kernel_size,feature_maps)

    w.add_text('Training Parameters',training_parameters)
    w.add_text('Model Parameters',model_parameters)

    weight_map = tensor_format(torch.FloatTensor(weight_map))
    criterion = nn.CrossEntropyLoss(weight=weight_map)


    optimizer = optim.SGD(net.parameters(),lr = learn_rate,momentum=momentum_rate)
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

def testModel(net,w):
    net.eval()
    for img, label in test_loader:
        img, label = tensor_format(img), tensor_format(label)
        output = net(img)
        output, label = crop(output,label)
        test_score = score(output,label)
        print("Test score: {}".format(test_score))
        output, label = reduceTo2D(output,label)
    w.add_text("Test score","Test score: "+str(test_score))
    return test_score

kernel_size_parameters = [4,5,6,7,8]
feature_maps=16
learning_parameters = [8e-3,4e-3,1e-3,3e-4]
run_count = 0
models_list =[]
test_score_list = []
best_percentage =0

for i,ks in enumerate(kernel_size_parameters):
    for j,lr in enumerate(learning_parameters):
        w = SummaryWriter()

        run_count += 1
        model = trainModel(ks,feature_maps,lr,w)
        models_list.append(model,w)

        test_score = testModel(model)
        test_score_list.append(test_score)

        if test_score>best_percentage:
            best_model = model
            best_percentage = test_score

        w.close()
torch.save(best_model.state_dict(),'best_model.pt')

criterion = nn.CrossEntropyLoss()


for episode in range(num_episodes):
    ################### **Getting and formatting the human trained data** #########################
    observations, actions = dataCollector(env)
    observations, actions = np.array(observations), np.array(actions)
    # pdb.set_trace()
    # observations,actions = balanceDataset(observations,actions)
    # scalar = Standarize()
    # observations = scalar(observations)
    # print("Observation mean: {}".format(observations.mean()))
    # print("Observation std: {}".format(observations.std()))
    observations, actions = torch.DoubleTensor(observations), torch.LongTensor(actions)
    observations, actions = tensor_format(observations), tensor_format(actions)

    ################### **Training the Network** #########################
    learn_rate = next(lr_generator)
    optimizer = optim.SGD(net.parameters(),lr=learn_rate,momentum=0.90, nesterov=True,weight_decay=1e-4)
    printModel(net,optimizer)

    train_iterations=3
    for i in range(train_iterations):
        count += 1
        # ipdb.set_trace()
        before_weights = weightMag(net)
        ################################################################
        optimizer.zero_grad()
        #########################
        outputs = net(observations)

        acc = score(outputs,actions)
        w.add_scalar('Accuracy', acc,count)
        print("Accuracy: {}".format(acc))
        #########################
        loss = criterion(outputs, actions)

        w.add_scalar('Loss', loss.data[0],count)
        print("Loss value: {}".format(loss))
        #########################
        loss.backward()
        optimizer.step()
        ################################################################
        after_weights =weightMag(net)
        relDiff_list = relDiff(before_weights,after_weights)
        relDiff_dict = listToDict(relDiff_list)
        w.add_scalars('LayerChanges',relDiff_dict,count)

        print("Network updated!")

w.close()


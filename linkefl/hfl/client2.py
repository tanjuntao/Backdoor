import math

import torch
from torch import nn
from torchvision import datasets, transforms

from linkefl.hfl.common.data_io import MyData_image,MyData_tabular
from linkefl.hfl.core.Nets import LogReg
from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.Nets import Nets
from linkefl.hfl.core.Client import Client


if __name__ == "__main__":

    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    HOST = "127.0.0.1"
    PORT = 23706
    world_size = 2
    partyid = 2
    role = "client"
    client_messenger = messenger(
        HOST,
        PORT,
        role="client",
        partyid=partyid,
        world_size=world_size,
    )

    model_name = "HFLNN"
    #["mnist", "cifar10", "fashion_mnist"]
    # dataset_name = "cifar10"
    dataset_name = "mnist"
    # data_name = "fashion_mnist"

    data_path = "../../../LinkeFL/linkefl/hfl/data"
    Testset = MyData_image(dataset_name, data_path=data_path, train=False)
    Trainset = Testset
    # Trainset = MyData_image(data_name, data_path=data_path, train=True)

    aggregator = "FedAvg"
    # aggregator = 'FedAvg_seq'
    # aggregator = 'PersonalizedFed'
    # aggregator = 'Scaffold'

    # 神经网络模型模型
    net_name = 'CNN'
    # net_name = "LeNet"
    # net_name = "ResNet18"
    num_classes = 10
    num_channels = 1
    model = Nets(net_name, num_classes,dataset_name, num_channels)

    epoch = 1
    learningrate = 0.01
    batch_size = 64
    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.5)

    client1 = Client(
        messenger=client_messenger,
        world_size=world_size,
        partyid=partyid,
        model=model,
        optimizer=optimizer,
        aggregator=aggregator,
        lossfunction=lossfunction,
        device=device,
        epoch=epoch,
        batch_size=batch_size,
        model_path="./models",
        model_name=model_name,
    )


    print("Client training...")
    model_parameters = client1.fit(Trainset, Testset)
    print("Client training done.")

    test_accuracy, test_loss = client1.score(Testset)

    Client.online_inference(Testset,model_name=model_name,loss_fn=lossfunction,device=device)

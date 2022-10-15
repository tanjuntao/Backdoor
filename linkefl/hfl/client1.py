import math

import torch
from torch import nn
from torchvision import datasets, transforms

from linkefl.hfl.customed_optimizer import ScaffoldOptimizer
from linkefl.hfl.hfl import Client
from linkefl.hfl.utils import Partition, Nets, ResNet18


def setClient():
    if aggregator in {'FedAvg', 'FedAvg_seq'}:
        server = Client(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        optimizer=optimizer,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch)

    elif aggregator == 'FedProx':
        server = Client(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        optimizer=optimizer,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        mu=mu)

    elif aggregator == 'Scaffold':
        server = Client(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        optimizer=optimizer,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        E=E,
                        lr=learningrate)

    elif aggregator == 'PersonalizedFed':
        server = Client(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        optimizer=optimizer,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        kp=kp)

    elif aggregator == 'FedDP':
        server = Client(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        optimizer=optimizer,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        lr=learningrate,
                        dp_mechanism=dp_mechanism,
                        dp_delta=dp_delta,
                        dp_epsilon=dp_epsilon,
                        dp_clip=dp_clip)

    else:
        raise Exception("Invalid aggregation rule")
    return server


if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 设置相关参数
    HOST = '127.0.0.1'
    PORT = 23705
    world_size = 2
    partyid = 1

    learningrate = 0.01
    epoch = 20
    iter = 1
    model_name = 'SimpleCNN'
    num_classes = 10
    num_channels = 1
    model = Nets(model_name, num_classes, num_channels)
    model.to(device)
    # aggregator = 'FedAvg'
    # aggregator = 'FedAvg_seq'
    optimizer = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.5)
    lossfunction = nn.CrossEntropyLoss()

    # FedProx
    aggregator = 'FedProx'
    mu = 0.02

    # Scaffold
    aggregator = 'Scaffold'
    E = 30
    optimizer = ScaffoldOptimizer(model.parameters(), lr=learningrate, weight_decay=1e-4)

    # PersonalizedFed
    aggregator = 'PersonalizedFed'
    kp = 0  # rate of personalized lyaer

    # Differential Privacy Based Federated Learning
    aggregator = 'FedDP'
    dp_mechanism = 'Laplace'
    dp_clip = 10
    dp_epsilon = 100/math.sqrt(epoch)
    dp_delta = 1e-5

    # 加载训练数据
    print("Loading dataset...")
    dataset = 'data/test/data_of_client1_100'
    Trainset = torch.load(dataset)
    print("Done.")

    client = setClient()

    #
    print("Client training...")
    model_parameters = client.train(Trainset)
    print("Client training done.")

    # 加载测试数据

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)

    test_accuracy, test_loss = client.test(dataset_test)

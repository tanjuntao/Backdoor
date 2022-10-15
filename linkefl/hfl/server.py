import torch
from torch import nn
from torchvision import datasets, transforms

from linkefl.hfl.hfl import Server
from linkefl.hfl.utils import Nets


def setServer():
    if aggregator in {'FedAvg', 'FedAvg_seq', "FedDP"}:
        server = Server(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch)
    elif aggregator == 'FedProx':
        server = Server(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        mu=mu)

    elif aggregator == 'Scaffold':
        server = Server(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        E=E)

    elif aggregator == 'PersonalizedFed':
        server = Server(HOST=HOST,
                        PORT=PORT,
                        world_size=world_size,
                        partyid=partyid,
                        model=model,
                        aggregator=aggregator,
                        lossfunction=lossfunction,
                        device=device,
                        epoch=epoch,
                        kp=kp)
    else:
        raise Exception("Invalid aggregation rule")
    return server


if __name__ == '__main__':
    # 设置相关参数
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    HOST = '127.0.0.1'
    PORT = 23705
    world_size = 2
    partyid = 0

    epoch = 20
    # aggregator = 'FedAvg'
    aggregator = 'FedAvg_seq'
    model_name = 'SimpleCNN'
    num_classes = 10
    num_channels = 1
    model = Nets(model_name, num_classes, num_channels)
    model.to(device)

    learningrate = 0.01
    lossfunction = nn.CrossEntropyLoss()
    role = 'server'

    # FedProx
    aggregator = 'FedProx'
    mu = 0.02

    # Scaffold
    aggregator = 'Scaffold'
    E = 30

    # PersonalizedFed
    aggregator = 'PersonalizedFed'
    kp = 0  # rate of personalized lyaer

    # Differential Privacy Based Federated Learning
    aggregator = 'FedDP'

    server = setServer()

    print(" Server training...")
    model = server.train()
    print("Server training done.")

    # 加载测试数据

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)

    test_accuracy, test_loss = server.test(dataset_test)

import math

import torch
from torch import nn

from linkefl.hfl.common.data_io import MyData_image
from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.hfl import Client
from linkefl.hfl.core.Nets import Nets


def setClient():
    if aggregator in {"FedAvg", "FedAvg_seq"}:
        server = Client(
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

    elif aggregator == "FedProx":
        server = Client(
            messenger=client_messenger,
            world_size=world_size,
            partyid=partyid,
            model=model,
            optimizer=optimizer,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            mu=mu,
        )

    elif aggregator == "Scaffold":
        server = Client(
            messenger=client_messenger,
            world_size=world_size,
            partyid=partyid,
            model=model,
            optimizer=optimizer,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            E=E,
            lr=learningrate,
        )

    elif aggregator == "PersonalizedFed":
        server = Client(
            messenger=client_messenger,
            world_size=world_size,
            partyid=partyid,
            model=model,
            optimizer=optimizer,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            kp=kp,
        )

    elif aggregator == "FedDP":
        server = Client(
            messenger=client_messenger,
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
            dp_clip=dp_clip,
        )

    else:
        raise Exception("Invalid aggregation rule")
    return server


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    # 设置相关参数
    HOST = "127.0.0.1"
    PORT = 23706
    world_size = 2
    partyid = 2

    client_messenger = messenger(
        HOST,
        PORT,
        role="client",
        partyid=partyid,
        world_size=world_size,
    )

    # data_name = "CIFAR10"
    data_name = "MNIST"
    data_path = "../../../LinkeFL/linkefl/hfl/data"
    epoch = 10
    aggregator = "FedAvg"
    learningrate = 0.1
    epoch = 5
    iter = 1
    batch_size = 64


    # 神经网络模型模型
    # model_name = 'LeNet'
    model_name = 'CNN'
    # model_name = "ResNet18"
    num_classes = 10
    num_channels = 1
    model = Nets(model_name, num_classes, num_channels)

    model.to(device)

    # aggregator = 'FedAvg_seq'
    optimizer = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.5)
    lossfunction = nn.CrossEntropyLoss()

    # # FedProx
    # aggregator = 'FedProx'
    mu = 0.02
    #
    # # Scaffold
    # aggregator = 'Scaffold'
    E = 30
    # optimizer = ScaffoldOptimizer(
    #     model.parameters(),
    #     lr=learningrate,
    #     weight_decay=1e-4
    # )
    #
    # # PersonalizedFed
    # aggregator = 'PersonalizedFed'
    kp = 0  # rate of personalized lyaer
    #
    # Differential Privacy Based Federated Learning
    # aggregator = 'FedDP'
    dp_mechanism = "Laplace"
    dp_clip = 10
    dp_epsilon = 100 / math.sqrt(epoch)
    dp_delta = 1e-5

    print("Loading dataset...")

    Testset = MyData_image(data_name,data_path=data_path,train=False)
    Trainset = Testset

    print("Done.")

    client = setClient()

    print("Client training...")
    model_parameters = client.train(Trainset, Testset)
    print("Client training done.")

    test_accuracy, test_loss = client.test(Testset)

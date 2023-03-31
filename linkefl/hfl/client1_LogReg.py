import math
import torch
from torch import nn

from linkefl.hfl.common.data_io import MyData
from linkefl.hfl.core.Nets import LinReg
from linkefl.hfl.utils.lossfunction import MSEloss
from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.Client import Client
from linkefl.hfl.core.Nets import LogReg


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    # 设置相关参数
    HOST = "127.0.0.1"
    PORT = 23705
    world_size = 2
    partyid = 1

    client_messenger = messenger(
        HOST,
        PORT,
        role="client",
        partyid=partyid,
        world_size=world_size,
    )


    dataset_name = "digits"
    epoch = 1
    learningrate = 0.01
    iter = 5
    batch_size = 64

    # 线性回归模型
    model_name = "HFLLogReg"
    in_features = 64
    num_classes = 2
    model = LogReg(in_features, num_classes)

    model.to(device)
    aggregator = "FedAvg"
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

    Trainset = MyData(
        name=dataset_name,
        root="../../data",
        train=True,
        download=True,
    )
    Testset = MyData(
        name=dataset_name,
        root="../../data",
        train=False,
        download=True,
    )
    print("Done.")

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

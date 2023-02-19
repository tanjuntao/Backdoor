import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from linkefl.hfl.hfl import Server,inference_hfl
from linkefl.hfl.mydata import myData
from linkefl.hfl.utils.Nets import LogReg, Nets
from linkefl.common.factory import crypto_factory, logger_factory, messenger_factory


def setServer():
    if aggregator in {"FedAvg", "FedAvg_seq", "FedDP"}:
        server = Server(
            HOST=HOST,
            PORT=PORT,
            world_size=world_size,
            partyid=partyid,
            model=model,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            logger=logger_factory("active_party"),
            model_path="./models",
            model_name=model_name,
        )

    elif aggregator == "FedProx":
        server = Server(
            HOST=HOST,
            PORT=PORT,
            world_size=world_size,
            partyid=partyid,
            model=model,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            mu=mu,
        )

    elif aggregator == "Scaffold":
        server = Server(
            HOST=HOST,
            PORT=PORT,
            world_size=world_size,
            partyid=partyid,
            model=model,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            E=E,
        )

    elif aggregator == "PersonalizedFed":
        server = Server(
            HOST=HOST,
            PORT=PORT,
            world_size=world_size,
            partyid=partyid,
            model=model,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            kp=kp,
        )
    else:
        raise Exception("Invalid aggregation rule")
    return server


if __name__ == "__main__":
    # 设置相关参数
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    HOST = "127.0.0.1"
    PORT = 23705
    world_size = 2
    partyid = 0
    model_name = "LogReg"

    dataset_name = "digits"
    # dataset_name = "mnist"
    epoch = 5
    aggregator = "FedAvg"
    # aggregator = 'FedAvg_seq'

    # 逻辑回归模型
    model_name = "LogisticRegression_server"
    in_features = 64
    num_classes = 2
    model = LogReg(in_features, num_classes)

    model.to(device)

    learningrate = 0.01
    lossfunction = nn.CrossEntropyLoss()
    role = "server"

    _logger = logger_factory(role="active_party")
    # # FedProx
    # aggregator = 'FedProx'
    mu = 0.02
    #
    # # Scaffold
    # aggregator = 'Scaffold'
    E = 30
    #
    # # PersonalizedFed
    # aggregator = 'PersonalizedFed'
    kp = 0  # rate of personalized lyaer
    #
    # Differential Privacy Based Federated Learning
    # aggregator = 'FedDP'

    server = setServer()

    # 加载测试数据

    Testset = myData(
        name=dataset_name,
        root="../../data",
        train=False,
        download=True,
    )

    print(len(Testset))
    print(" Server training...")
    model = server.train(Testset)
    print("Server training done.")
    test_accuracy, test_loss = server.test(Testset)

    results = inference_hfl(Testset,model_arch=LogReg(in_features, num_classes),
                        model_name=model_name,loss_fn=lossfunction,device=device)


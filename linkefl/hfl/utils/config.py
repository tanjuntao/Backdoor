import torch
from torch import nn

from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.Nets import Nets
from linkefl.common.factory import logger_factory
from linkefl.hfl.common.data_io import MyData_image


if __name__ == "__main__":
    # 设置相关参数
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    HOST = "127.0.0.1"
    PORT = [23705,23706]
    world_size = 2
    partyid = 0

    server_messenger = messenger(
        HOST,
        PORT,
        role="server",
        partyid=partyid,
        world_size=world_size,
    )

    data_name = "CIFAR10"
    # data_name = "MNIST"
    data_path = "../../../LinkeFL/linkefl/hfl/data"
    Testset = MyData_image(data_name,data_path=data_path,train=False)

    epoch = 5
    aggregator = "FedAvg"


    # 神经网络模型模型
    model_name = 'CNN'
    num_classes = 10
    num_channels = 3
    model = Nets(model_name, num_classes, num_channels)

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
    aggregator = 'PersonalizedFed'
    kp = 0  # rate of personalized lyaer
    #
    # Differential Privacy Based Federated Learning
    # aggregator = 'FedDP'

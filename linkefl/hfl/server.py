import torch
from torch import nn

from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.Nets import Nets
from linkefl.common.factory import logger_factory
from linkefl.hfl.common.data_io import MyData_image
from linkefl.hfl.core.Server import Server

if __name__ == "__main__":

    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    HOST = "127.0.0.1"
    PORT = [23705,23706]
    world_size = 1
    partyid = 0

    server_messenger = messenger(
        HOST,
        PORT,
        role="server",
        partyid=partyid,
        world_size=world_size,
    )

    model_name = "HFLNN"
    #["mnist", "cifar10", "fashion_mnist"]
    # dataset_name = "cifar10"
    dataset_name = "mnist"
    # data_name = "fashion_mnist"
    model_dir = "./models"
    pics_dir = "./pictures"
    data_path = "../../../LinkeFL/linkefl/hfl/data"
    Testset = MyData_image(dataset_name,data_path=data_path,train=False)

    aggregator = "FedAvg"
    # aggregator = 'FedAvg_seq'
    # aggregator = 'PersonalizedFed'
    # aggregator = 'Scaffold'

    # 神经网络模型模型
    net_name = 'CNN'
    # model_name = "LeNet"
    # net_name = "ResNet18"
    num_classes = 10
    num_channels = 1

    epoch = 2
    learningrate = 0.1
    lossfunction = nn.CrossEntropyLoss()
    role = "server"


    _logger = logger_factory(role="active_party")

    model = Nets(net_name, num_classes, dataset_name,num_channels)

    model.to(device)
    server = Server(
            messenger=server_messenger,
            world_size=world_size,
            partyid=partyid,
            model=model,
            aggregator=aggregator,
            lossfunction=lossfunction,
            device=device,
            epoch=epoch,
            logger=logger_factory("active_party"),
            model_dir="./models",
            model_name=model_name,
            saving_model=False,
            task= "multi",
        )

    print(" Server training...")
    model = server.fit(Testset)
    print("Server training done.")
    test_accuracy, test_loss = server.score(Testset)

    results = Server.online_inference(Testset,model_name=model_name,model_dir=model_dir,loss_fn=lossfunction,device=device)




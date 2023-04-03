import torch

from linkefl.hfl.common.data_io import MyData
from linkefl.hfl.core.Nets import LinReg
from linkefl.common.factory import logger_factory
from linkefl.hfl.utils.lossfunction import MSEloss
from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.Server import Server



if __name__ == "__main__":
    # 设置相关参数
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

    model_dir = "./models"
    dataset_name = "diabetes"
    # dataset_name = "mnist"
    epoch = 10
    aggregator = "FedAvg"
    # aggregator = 'FedAvg_seq'

    # 逻辑回归模型
    model_name = "HFLLinReg"
    in_features = 10
    model = LinReg(in_features)

    model.to(device)

    learningrate = 0.01
    lossfunction = MSEloss

    model.to(device)

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


    # 加载测试数据

    Testset = MyData(
        name=dataset_name,
        root="../../data",
        train=False,
        download=True,
    )

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
        )

    print(" Server training...")
    model = server.fit(Testset)
    print("Server training done.")
    test_accuracy, test_loss = server.score(Testset)

    result = Server.online_inference(Testset,model_name=model_name,model_path=model_dir,loss_fn=lossfunction,device=device)

    print(result)

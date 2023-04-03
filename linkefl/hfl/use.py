import torch
from torch import nn

from linkefl.hfl.common.socket_hfl import messenger
from linkefl.hfl.core.Nets import Nets
from linkefl.common.factory import logger_factory
from linkefl.hfl.common.data_io import MyData_image
from linkefl.hfl.core.Server import Server


if __name__ == "__main__":

    dataset_name = "mnist"
    data_path = "../../../LinkeFL/linkefl/hfl/data"
    Testset = MyData_image(dataset_name,data_path=data_path,train=False)
    print(len(Testset))
    train,test = torch.utils.data.random_split(Testset,[8000,2000])



import datetime
import os
import pathlib
from math import ceil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from linkefl.base.component_base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.hfl.core.inference import inference_hfl
from linkefl.hfl.core.training_method import Train_client, Train_server
from linkefl.modelio import TorchModelIO


class Server(BaseModelComponent):
    def __init__(
        self,
        messenger,
        world_size,
        partyid,
        model,
        logger,
        aggregator="FedAvg",
        lossfunction=F.nll_loss,
        device=torch.device("cpu"),
        epoch=10,
        mu=0.01,
        E=30,
        kp=0.1,
        batch_size=64,
        BUFSIZ=1024000000,
        saving_model=True,
        model_dir="./models",
        model_name=None,
        algo_name="",
    ):
        """
        HOST:联邦学习server的ip
        PORT:端口号
        world_size:client的数量
        partyid:当前的id，id为0是server
        net:神经网络模型
        epoch:总训练的迭代次数
        device:训练选择的设备
        lossfunction:损失函数
        BUFSIZ:数据传输的buffer_size
        batch_size:神经网络训练的batch_size
        iter:每个client的内循环
        """
        self.messenger = messenger
        self.party = partyid
        self.world_size = world_size
        self.partyid = partyid
        self.model = model
        self.aggregator = aggregator
        self.lossfunction = lossfunction
        self.device = device
        self.epoch = epoch
        self.BUFSIZ = BUFSIZ
        self.batch_size = batch_size
        self.iter = iter
        self.kp = kp
        self.logger = logger
        self.saving_model = saving_model

        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.ACTIVE_NAME,
                        algo_name=algo_name,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            self.pics_dir = self.model_dir
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def _init_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def fit(self, validset, trainset="", role="server"):
        # server

        if self.aggregator == "FedAvg":
            self.model = Train_server.train_basic(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                validset,
                self.lossfunction,
                self.logger,
                self.model_dir,
                self.model_name,
            )
        elif self.aggregator == "FedAvg_seq":
            self.model = Train_server.train_FedAvg_seq(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                validset,
                self.lossfunction,
                self.logger,
                self.model_dir,
                self.model_name,
            )
        elif self.aggregator == "FedProx":
            self.model = Train_server.train_basic(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                validset,
                self.lossfunction,
                self.logger,
                self.model_dir,
                self.model_name,
            )
        elif self.aggregator == "Scaffold":
            self.model = Train_server.train_Scaffold(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                validset,
                self.lossfunction,
                self.logger,
                self.model_dir,
                self.model_name,
            )
        elif self.aggregator == "PersonalizedFed":
            self.model = Train_server.train_PersonalizedFed(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                self.kp,
                validset,
                self.lossfunction,
                self.logger,
                self.model_dir,
                self.model_name,
            )
        elif self.aggregator == "FedDP":
            self.model = Train_server.train_basic(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                validset,
                self.lossfunction,
                self.logger,
                self.model_dir,
                self.model_name,
            )

        self.messenger.close()

    def score(self, testset, role="server"):
        test_loss = 0
        correct = 0
        test_set = self._init_dataloader(testset)
        self.model.eval()
        num_batches = ceil(len(test_set.dataset) / float(self.batch_size))

        for idx, (data, target) in enumerate(test_set):
            data, target = data.to(self.device), target.to(self.device).to(torch.long)
            log_probs = self.model(data)
            # test_loss += self.lossfunction(log_probs, target, reduction='sum').item()
            test_loss += self.lossfunction(log_probs, target).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= num_batches
        accuracy = 100.00 * correct / len(test_set.dataset)
        print(
            "\nTest set:\nAverage loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, len(test_set.dataset), accuracy
            )
        )

        return accuracy, test_loss

    @staticmethod
    def online_inference(
        dataset,
        model_name,
        model_dir="./models",
        loss_fn=None,
        infer_step=64,
        device=torch.device("cpu"),
        optimizer_arch=None,
        role="server",
    ):

        scores = inference_hfl(
            dataset=dataset,
            model_name=model_name,
            model_dir=model_dir,
            loss_fn=loss_fn,
            device=device,
        )
        return scores

    def get_model_params(self):
        return self.model.state_dict()

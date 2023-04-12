import datetime
import os
import pathlib
from math import ceil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from linkefl.base.component_base import BaseModelComponent
from linkefl.common.const import Const
from linkefl.hfl.core.customed_optimizer import ScaffoldOptimizer
from linkefl.hfl.core.inference import inference_hfl
from linkefl.hfl.core.training_method import Train_client, Train_server
from linkefl.modelio import TorchModelIO


class Client(BaseModelComponent):
    def __init__(
        self,
        messenger,
        world_size,
        partyid,
        model,
        optimizer,
        task,
        aggregator="FedAvg",
        lossfunction=F.nll_loss,
        device=torch.device("cpu"),
        epoch=10,
        mu=0.01,
        E=30,
        lr=0.01,
        kp=0.1,
        BUFSIZ=1024000000,
        batch_size=64,
        iter=1,
        dp_mechanism="Laplace",
        dp_clip=10,
        dp_epsilon=1,
        dp_delta=1e-5,
        saving_model=True,
        model_dir="./models",
        model_name=None,
        algo_name="",
    ):
        """
        messenger: 通信组建
        world_size:client的数量
        partyid:当前的id，id为0是server
        net:神经网络模型
        optimizer:神经网络训练优化器
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
        self.optimizer = optimizer
        self.aggregator = aggregator
        self.lossfunction = lossfunction
        self.device = device
        self.epoch = epoch
        self.BUFSIZ = BUFSIZ
        self.batch_size = batch_size
        self.iter = iter
        self.saving_model = saving_model
        self.task = task
        self.model_dir = model_dir
        self.model_name = model_name
        self.pics_dir = self.model_dir
        if self.saving_model:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if model_dir is None:
                default_dir = "models"
                model_dir = os.path.join(default_dir, self.create_time)
            if model_name is None:
                model_name = (
                    "{time}-{role}-{algo_name}".format(
                        time=self.create_time,
                        role=Const.PASSIVE_NAME,
                        algo_name=algo_name,
                    )
                    + ".model"
                )
            self.model_dir = model_dir
            self.model_name = model_name
            self.pics_dir = self.model_dir
            if not os.path.exists(self.model_dir):
                pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # FedProx
        self.mu = mu

        # Scaffold
        self.E = E
        self.lr = lr

        # PersonalizedFed
        self.kp = kp

        # Differential Privacy Based Federated Learning
        self.dp_mechanism = dp_mechanism
        self.dp_clip = dp_clip
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

    def _init_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def get_aggregator(self):
        if self.aggregator == "FedAvg":
            return Train_client.train_basic
        elif self.aggregator == "FedAvg_seq":
            return Train_client.train_basic
        elif self.aggregator == "FedProx":
            return Train_client.train_FedProx
        elif self.aggregator == "Scaffold":
            return Train_client.train_Scaffold
        elif self.aggregator == "PersonalizedFed":
            return Train_client.train_PersonalizedFed
        else:
            raise Exception("Invalid aggregation rule")

    def fit(self, trainset, validset, role="client"):
        train_set = self._init_dataloader(trainset)
        optimizer = self.optimizer
        lf = self.lossfunction
        # model = self.model
        num_batches = ceil(len(train_set.dataset) / float(self.batch_size))
        # train_method = self.get_aggregator()

        if self.aggregator == "FedAvg":
            self.model = Train_client.train_basic(
                self.messenger,
                self.partyid,
                self.epoch,
                train_set,
                self.model,
                optimizer,
                lf,
                self.iter,
                self.device,
                num_batches,
                validset,
                self.model_dir,
                self.model_name,
                self.task,
                self.saving_model,
                self.pics_dir,
            )

        elif self.aggregator == "FedAvg_seq":
            self.model = Train_client.train_basic(
                self.messenger,
                self.partyid,
                self.epoch,
                train_set,
                self.model,
                optimizer,
                lf,
                self.iter,
                self.device,
                num_batches,
                validset,
                self.model_dir,
                self.model_name,
                self.task,
                self.saving_model,
                self.pics_dir,
            )

        elif self.aggregator == "FedProx":
            self.model = Train_client.train_FedProx(
                self.messenger,
                self.partyid,
                self.epoch,
                train_set,
                self.model,
                optimizer,
                lf,
                self.iter,
                self.device,
                num_batches,
                self.mu,
                validset,
                self.model_dir,
                self.model_name,
                self.task,
                self.saving_model,
                self.pics_dir,
            )

        elif self.aggregator == "Scaffold":
            optimizer = ScaffoldOptimizer(
                self.model.parameters(), lr=self.lr, weight_decay=1e-4
            )
            self.model = Train_client.train_Scaffold(
                self.messenger,
                self.partyid,
                self.epoch,
                train_set,
                self.model,
                optimizer,
                lf,
                self.iter,
                self.device,
                num_batches,
                validset,
                self.model_dir,
                self.model_name,
                self.task,
                self.saving_model,
                self.pics_dir,
                self.E,
                self.lr,
            )

        elif self.aggregator == "PersonalizedFed":
            self.model = Train_client.train_PersonalizedFed(
                self.messenger,
                self.partyid,
                self.epoch,
                train_set,
                self.model,
                optimizer,
                lf,
                self.iter,
                self.device,
                num_batches,
                self.kp,
                validset,
                self.model_dir,
                self.model_name,
                self.task,
                self.saving_model,
                self.pics_dir,
            )

        elif self.aggregator == "FedDP":
            self.model = Train_client.train_FedDP(
                self.messenger,
                self.partyid,
                self.epoch,
                train_set,
                self.model,
                optimizer,
                lf,
                self.iter,
                self.device,
                num_batches,
                self.lr,
                self.dp_mechanism,
                self.dp_clip,
                self.dp_epsilon,
                self.dp_delta,
                validset,
                self.model_dir,
                self.model_name,
                self.task,
                self.saving_model,
                self.pics_dir,
            )

    def score(self, testset, role="client"):
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
        role="client",
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

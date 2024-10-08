from math import ceil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from linkefl.hfl.core.training_method import Train_client, Train_server
from linkefl.modelio import TorchModelIO
from linkefl.base.component_base import BaseModelComponent


class Server:
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
            model_path="./models",
            model_name=None,
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
        self.model_path = model_path
        self.model_name = model_name

    def _init_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def train(self, testset):
        # server

        if self.aggregator == "FedAvg":
            self.model = Train_server.train_basic(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                testset,
                self.lossfunction,
                self.logger,
                self.model_path,
                self.model_name,
            )
        elif self.aggregator == "FedAvg_seq":
            self.model = Train_server.train_FedAvg_seq(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                testset,
                self.lossfunction,
            )
        elif self.aggregator == "FedProx":
            self.model = Train_server.train_basic(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                testset,
                self.lossfunction,
            )
        elif self.aggregator == "Scaffold":
            self.model = Train_server.train_Scaffold(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                testset,
                self.lossfunction,
            )
        elif self.aggregator == "PersonalizedFed":
            self.model = Train_server.train_PersonalizedFed(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                self.kp,
                testset,
                self.lossfunction,
            )
        elif self.aggregator == "FedDP":
            self.model = Train_server.train_basic(
                self.epoch,
                self.world_size,
                self.messenger,
                self.model,
                self.device,
                testset,
                self.lossfunction,
            )

        self.messenger.close()

    def test(self, testset):
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

    def get_model_params(self):
        return self.model.state_dict()


class Client:
    def __init__(
            self,
            messenger,
            world_size,
            partyid,
            model,
            optimizer,
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
            iter=5,
            dp_mechanism="Laplace",
            dp_clip=10,
            dp_epsilon=1,
            dp_delta=1e-5,
            model_path="./models",
            model_name=None,
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
        self.model_path = model_path
        self.model_name = model_name

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

    def train(self, trainset, testset):
        role = "client"

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
                testset,
                self.model_path,
                self.model_name,
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
                testset,
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
                testset,
            )

        elif self.aggregator == "Scaffold":
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
                self.E,
                self.lr,
                testset,
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
                testset,
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
                testset,
            )

    def test(self, testset):
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

    def get_model_params(self):
        return self.model.state_dict()


def inference_hfl(
        dataset,
        model_arch,
        model_name,
        model_path="./models",
        loss_fn=None,
        infer_step=64,
        device=torch.device("cpu"),
        optimizer_arch=None,
):
    # 加载模型
    model = TorchModelIO.load(model_dir=model_path, model_name=model_name)["model"]

    # 加载数据
    dataloader = DataLoader(dataset, batch_size=infer_step, shuffle=False)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0
    # labels, probs = np.array([]), np.array([])  # used for computing AUC score
    preds = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device).to(torch.long)
            log_probs = model(data)
            # test_loss += self.lossfunction(log_probs, target, reduction='sum').item()
            test_loss += loss_fn(log_probs, target).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            preds.extend(y_pred.numpy().tolist())

    test_loss /= num_batches
    acc = correct / len(dataloader.dataset)
    scores = {"acc": acc, "auc": 0, "loss": test_loss, "preds": preds}

    return scores


def inference_hfl_linear(
        dataset,
        model_arch,
        model_name,
        model_path="./models",
        loss_fn=None,
        infer_step=64,
        device=torch.device("cpu"),
        optimizer_arch=None,
):
    # 加载模型
    model = TorchModelIO.load(model_dir=model_path, model_name=model_name)["model"]

    # 加载数据
    dataloader = DataLoader(dataset, batch_size=infer_step, shuffle=False)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    preds = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device).to(torch.long)
            log_probs = model(data)
            # test_loss += self.lossfunction(log_probs, target, reduction='sum').item()
            test_loss += loss_fn(log_probs, target).item()
            y_pred = log_probs.data
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            preds.extend(y_pred.numpy().tolist())
    # print(probs)
    test_loss /= num_batches
    # acc = correct / len(dataloader.dataset)
    scores = {"acc": 0, "auc": 0, "loss": test_loss, "preds": preds}

    return scores

import copy

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from linkefl.hfl.utils.dp_mechanism import gradient_clip


class Aggregator_server:
    @staticmethod
    def FedAvg(recDatas):
        num_local_dataset = [data["num_dataset"] for data in recDatas]
        sum = 0

        for i in range(len(num_local_dataset)):
            sum += num_local_dataset[i]
        w = [i / sum for i in num_local_dataset]
        nets = [data["net"] for data in recDatas]
        net = {}
        for i in range(len(nets[0].keys())):
            keys = [list(net.keys())[i] for net in nets]
            net[keys[0]] = np.sum(
                [np.array(nets[j][keys[0]]) * w[j] for j in range(len(nets))], axis=0
            )
            net[keys[0]] = net[keys[0]].tolist()

        return net

    @staticmethod
    def FedAvg_seq(recDatas):
        """
        :param nets: client的网络参数列表。list of dictionary，维度不限，key不限，可适用于任意多个网络和任意维度的网络。
        :return: 合并后的网络参数，dictionary
        """
        """
        应对不同机器的网络参数的key 名字不同的情况：默认网络的 key不同时顺序仍然形同，可一一对应。
        """
        nets = [data["net"] for data in recDatas]
        net = {}
        for i in range(len(nets[0].keys())):
            keys = [list(net.keys())[i] for net in nets]
            net[keys[0]] = np.mean(
                [np.array(nets[j][keys[0]]) for j in range(len(nets))], axis=0
            )
            net[keys[0]] = net[keys[0]].tolist()
        return net

    @staticmethod
    def FedProx(recDatas):
        nets = [data["net"] for data in recDatas]
        net = {}
        for i in range(len(nets[0].keys())):
            keys = [list(net.keys())[i] for net in nets]
            net[keys[0]] = np.mean(
                [np.array(nets[j][keys[0]]) for j in range(len(nets))], axis=0
            )
            net[keys[0]] = net[keys[0]].tolist()

        return net

    @staticmethod
    def Scaffold(recDatas):
        nets = [data["net"] for data in recDatas]
        net = {}
        for i in range(len(nets[0].keys())):
            keys = [list(net.keys())[i] for net in nets]
            net[keys[0]] = np.mean(
                [np.array(nets[j][keys[0]]) for j in range(len(nets))], axis=0
            )
            net[keys[0]] = net[keys[0]].tolist()

        return net

    @staticmethod
    def PersonalizedFed(recDatas, kp):
        num_local_dataset = [data["num_dataset"] for data in recDatas]
        sum = 0

        for i in range(len(num_local_dataset)):
            sum += num_local_dataset[i]
        w = [i / sum for i in num_local_dataset]
        nets = [data["net"] for data in recDatas]
        net = {}
        total = len(nets[0].keys())
        n = 0
        for i in range(len(nets[0].keys())):
            keys = [list(net.keys())[i] for net in nets]
            net[keys[0]] = np.sum(
                [np.array(nets[j][keys[0]]) * w[j] for j in range(len(nets))], axis=0
            )
            net[keys[0]] = net[keys[0]].tolist()
            n += 1
            if n > int(total * (1 - kp)):
                net[keys[0]] = np.zeros(np.array(nets[0][keys[0]]).shape)
                net[keys[0]] = net[keys[0]].tolist()
        return net


class Aggregator_client:
    @staticmethod
    def FedAvg(train_set, model, optimizer, lf, iter, device):
        model.train()
        for i in range(iter):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target).to(torch.long)
                optimizer.zero_grad()
                output = model(data)
                loss = lf(output, target)
                # print("data:",data)
                print("target:",target)
                print("output:",output)
                print("loss:",loss)
                print("loss function:",lf)
                print(model.state_dict())
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
        return model.state_dict(), epoch_loss

    @staticmethod
    def FedAvg_seq(train_set, model, optimizer, lf, iter, device):
        model.train()
        for i in range(iter):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = lf(output, target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

        return model.state_dict(), epoch_loss

    @staticmethod
    def FedProx(train_set, model, optimizer, lf, iter, device, mu=0.01):
        global_model = copy.deepcopy(model)
        model.train()
        for i in range(iter):
            epoch_loss = 0.0
            for data, target in train_set:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = lf(output, target) + (mu / 2) * proximal_term
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

        return model.state_dict(), epoch_loss

    @staticmethod
    def Scaffold(
        train_set, model, optimizer, lf, iter, device, server_control, client_control
    ):
        lr_step = StepLR(optimizer, step_size=10, gamma=0.1)
        model.train()
        for i in range(iter):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = lf(output, target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step(server_control, client_control)
            lr_step.step()

        return model, epoch_loss

    @staticmethod
    def PersonalizedFed(train_set, model, optimizer, lf, iter, device):
        model.train()
        for i in range(iter):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = lf(output, target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

        return model.state_dict(), epoch_loss

    @staticmethod
    def FedDP(train_set, model, optimizer, lf, iter, device, dp_mechanism, dp_clip):
        model.train()
        for i in range(iter):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = lf(output, target)
                epoch_loss += loss.item()
                loss.backward()

                # gradient clip
                model = gradient_clip(model, dp_mechanism, dp_clip)

                optimizer.step()

        return model, epoch_loss

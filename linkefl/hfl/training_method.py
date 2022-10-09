import collections
import copy
import torch
from aggregator import Aggregator_server, Aggregator_client
from dp_mechanism import add_dp


# tensor to list
def modelpara_to_list(para):
    para = dict(para)
    for key in para:
        # para[key] = torch.tensor(para[key])
        para[key] = para[key].cpu().numpy().tolist()
    return para


# list to tensor
def list_to_tensor(data):
    for key in data:
        data[key] = torch.tensor(data[key])
    return data


class Train_server:

    @staticmethod
    def train_basic(epoch, world_size, server, model, device):
        aggregator = Aggregator_server.FedAvg
        for j in range(epoch):

            recDatas = []
            for i in range(world_size):
                recData = server.rec(id=i + 1)
                recDatas.append(recData)

            # 聚合参数
            if len(recDatas) > 1:
                new_net = aggregator(recDatas)
            else:
                new_net = recDatas[0]["net"]

            # 向 client返回参数
            for i in range(world_size):
                server.send(new_net, id=i + 1)

            # 格式转换 list to tensor
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)

            # 加载参数到网络
            model.load_state_dict(new_net)

        return model

    @staticmethod
    def train_FedAvg_seq(epoch, world_size, server, model, device):
        aggregator = Aggregator_server.FedAvg_seq
        for j in range(epoch):

            recDatas = []
            for i in range(world_size):
                recData = server.rec(id=i + 1)
                recDatas.append(recData)

            # 聚合参数
            if len(recDatas) > 1:
                new_net = aggregator(recDatas)
            else:
                new_net = recDatas[0]["net"]

            # 向 client返回参数
            for i in range(world_size):
                server.send(new_net, id=i + 1)

            # 格式转换 list to tensor
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)

            # 加载参数到网络
            model.load_state_dict(new_net)

        return model

    @staticmethod
    def train_Scaffold(epoch, world_size, server, model, device, E=30):

        server_control = {}
        server_delta_control = {}
        server_delta_y = {}
        for k, v in model.named_parameters():
            server_control[k] = torch.zeros_like(v.data).numpy().tolist()
            server_delta_control[k] = torch.zeros_like(v.data).numpy().tolist()
            server_delta_y[k] = torch.zeros_like(v.data).numpy().tolist()

        # 构造初始化模型参数和设置
        data = {}
        data["net"] = modelpara_to_list(model.state_dict())
        data["control"] = server_control
        data["delta_control"] = server_delta_control
        data["delta_y"] = server_delta_y

        # 向 client返回参数
        for i in range(world_size):
            server.send(data, id=i + 1)

        aggregator = Aggregator_server.FedAvg

        for j in range(epoch):

            recDatas = []
            for i in range(world_size):
                recData = server.rec(id=i + 1)
                recDatas.append(recData)

            # 聚合参数
            if len(recDatas) > 1:
                new_net = aggregator(recDatas)
            else:
                new_net = recDatas[0]["net"]

            # list to tensor
            for i in range(len(recDatas)):
                recDatas[i]["control"] = list_to_tensor(recDatas[i]["control"])
                recDatas[i]["delta_control"] = list_to_tensor(recDatas[i]["delta_control"])

            # 更新control
            x = {}
            c = {}
            # init
            for k, v in model.named_parameters():
                x[k] = torch.zeros_like(v.data)
                c[k] = torch.zeros_like(v.data)

            for i in range(world_size):
                for k, v in model.named_parameters():
                    x[k] += (recDatas[i]["control"][k]) / world_size  # averaging
                    c[k] += (recDatas[i]["delta_control"][k]) / world_size  # averaging

            # update x and c
            for k, v in model.named_parameters():
                v.data += x[k].data  # lr=1
                server_control[k] += c[k].data.cpu().numpy()
                server_control[k] = server_control[k].tolist()

            data = {}
            data["net"] = new_net
            data["control"] = (server_control)
            data["delta_control"] = (server_delta_control)
            data["delta_y"] = (server_delta_y)

            # 向 client返回参数
            for i in range(world_size):
                server.send(data, id=i + 1)

            # 格式转换 list to tensor
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)

            # 加载参数到网络
            model.load_state_dict(new_net)

        return model

    @staticmethod
    def train_PersonalizedFed(epoch, world_size, server, model, device, kp):

        aggregator = Aggregator_server.PersonalizedFed
        for j in range(epoch):

            recDatas = []
            for i in range(world_size):
                recData = server.rec(id=i + 1)
                recDatas.append(recData)

            # 聚合参数
            if len(recDatas) > 1:
                new_net = aggregator(recDatas, kp)
            else:
                new_net = recDatas[0]["net"]

            # 向 client返回参数
            for i in range(world_size):
                server.send(new_net, id=i + 1)

            # 格式转换 list to tensor
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)

            # 加载参数到网络
            model.load_state_dict(new_net)

        return model


class Train_client:

    @staticmethod
    def train_basic(client, partyid, epoch, train_set, model, optimizer, lf, iter, device, num_batches):
        model.train()
        aggregator = Aggregator_client.FedAvg
        for epoch in range(epoch):
            # 模型训练

            client_net, epoch_loss = aggregator(train_set, model, optimizer, lf, iter, device)

            print('partyid: ', partyid, ', epoch ', epoch, ': ', epoch_loss / num_batches)

            # tensor to list
            client_net = dict(client_net)
            for key in client_net:
                client_net[key] = client_net[key].cpu().numpy().tolist()

            # 拼接传输的数据内容
            data = {}
            data["net"] = client_net
            data["partyid"] = partyid
            data["num_dataset"] = len(train_set.dataset)

            client.send(data)
            new_net = client.rec()

            # 加载新的网络参数
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)
            model.load_state_dict(new_net)

        return model

    @staticmethod
    def train_FedProx(client, partyid, epoch, train_set, model, optimizer, lf, iter, device, num_batches, mu):
        model.train()
        aggregator = Aggregator_client.FedProx
        for epoch in range(epoch):
            # 模型训练

            client_net, epoch_loss = aggregator(train_set, model, optimizer, lf, iter, device, mu)

            print('partyid: ', partyid, ', epoch ', epoch, ': ', epoch_loss / num_batches)

            # tensor to list
            client_net = dict(client_net)
            for key in client_net:
                client_net[key] = client_net[key].cpu().numpy().tolist()

            # 拼接传输的数据内容
            data = {}
            data["net"] = client_net
            data["partyid"] = partyid
            data["num_dataset"] = len(train_set.dataset)

            client.send(data)
            new_net = client.rec()

            # 加载新的网络参数
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)
            model.load_state_dict(new_net)

        return model

    @staticmethod
    def train_Scaffold(client, partyid, epoch, train_set, model, optimizer, lf, iter, device, num_batches, E=30,
                       lr=0.01):

        model.train()
        aggregator = Aggregator_client.Scaffold

        #
        data = client.rec()

        # 加载初始化模型参数
        net = data["net"]
        for key in net:
            net[key] = torch.tensor((net[key])).to(device)
        global_net = collections.OrderedDict(net)
        model.load_state_dict(global_net)

        global_model = copy.deepcopy(model)

        server_control = list_to_tensor(data["control"])
        server_delta_control = list_to_tensor(data["delta_control"])
        server_delta_y = list_to_tensor(data["delta_y"])

        client_control = copy.deepcopy(server_control)
        client_delta_control = copy.deepcopy(server_delta_control)
        client_delta_y = copy.deepcopy(server_delta_y)

        for epoch in range(epoch):
            # 模型训练

            model, epoch_loss = aggregator(train_set, model, optimizer, lf, iter, device,
                                           server_control, server_delta_y)

            print('partyid: ', partyid, ', epoch ', epoch, ': ', epoch_loss / num_batches)

            client_net = model.state_dict()

            # 更新control
            temp = {}
            for k, v in model.named_parameters():
                temp[k] = v.data.clone()

            for k, v in global_model.named_parameters():
                local_steps = E * len(train_set.dataset)
                client_control[k] = client_control[k] - server_control[k] + (v.data - temp[k]) / (local_steps * lr)
                client_delta_y[k] = temp[k] - v.data
                client_delta_control[k] = - server_control[k] + (v.data - temp[k]) / (local_steps * lr)

            # tensor to list
            client_net = dict(client_net)
            for key in client_net:
                client_net[key] = client_net[key].cpu().numpy().tolist()

            # 拼接传输的数据内容
            data = {}
            data["net"] = client_net
            data["partyid"] = partyid
            data["num_dataset"] = len(train_set.dataset)
            data["control"] = modelpara_to_list(client_control)
            data["delta_control"] = modelpara_to_list(client_delta_control)
            data["delta_y"] = modelpara_to_list(client_delta_y)

            client.send(data)

            data = client.rec()
            global_net = data["net"]
            server_control = list_to_tensor(data["control"])

            # for key in server_control:
            #     print(server_control[key].size())

            server_delta_control = list_to_tensor(data["delta_control"])
            server_delta_y = list_to_tensor(data["delta_y"])

            # 加载新的网络参数
            for key in global_net:
                global_net[key] = torch.tensor((global_net[key])).to(device)
            global_net = collections.OrderedDict(global_net)
            model.load_state_dict(global_net)

        return model

    @staticmethod
    def train_PersonalizedFed(client, partyid, epoch, train_set, model, optimizer, lf, iter, device, num_batches, kp):

        model.train()
        aggregator = Aggregator_client.FedAvg
        for epoch in range(epoch):
            # 模型训练

            client_net, epoch_loss = aggregator(train_set, model, optimizer, lf, iter, device)

            print('partyid: ', partyid, ', epoch ', epoch, ': ', epoch_loss / num_batches)

            # tensor to list
            client_net = dict(client_net)
            for key in client_net:
                client_net[key] = client_net[key].cpu().numpy().tolist()

            # 拼接传输的数据内容
            data = {}
            data["net"] = client_net
            data["partyid"] = partyid
            data["num_dataset"] = len(train_set.dataset)

            client.send(data)
            new_net = client.rec()

            total = (len(new_net.keys()))
            # 加载新的网络参数
            n = 0
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
                n += 1
                if n > int(total * (1 - kp)):
                    new_net[key] = torch.tensor((client_net[key])).to(device)

            new_net = collections.OrderedDict(new_net)
            model.load_state_dict(new_net)

        return model

    @staticmethod
    def train_FedDP(client, partyid, epoch, train_set, model, optimizer, lf, iter, device, num_batches, lr,
                    dp_mechanism, dp_clip, dp_epsilon, dp_delta):
        model.train()
        aggregator = Aggregator_client.FedDP
        for epoch in range(epoch):
            # 模型训练

            client_net, epoch_loss = aggregator(train_set, model, optimizer, lf, iter, device, dp_mechanism, dp_clip)

            print('partyid: ', partyid, ', epoch ', epoch, ': ', epoch_loss / num_batches)

            # 添加差分隐私
            num_dataset = len(train_set.dataset)
            client_net = add_dp(client_net, lr, dp_clip, dp_mechanism, dp_epsilon, dp_delta, device)

            # tensor to list
            client_net = dict(client_net.state_dict())
            for key in client_net:
                client_net[key] = client_net[key].cpu().numpy().tolist()

            # 拼接传输的数据内容
            data = {}
            data["net"] = client_net
            data["partyid"] = partyid
            data["num_dataset"] = len(train_set.dataset)

            client.send(data)
            new_net = client.rec()

            # 加载新的网络参数
            for key in new_net:
                new_net[key] = torch.tensor((new_net[key])).to(device)
            new_net = collections.OrderedDict(new_net)
            model.load_state_dict(new_net)

        return model

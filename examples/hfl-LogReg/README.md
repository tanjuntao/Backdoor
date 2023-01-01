# 一、运行

1. 进入当前目录
2. 在一个终端窗口中，`python server_LogReg.py`，启动server
3. 新建另一个终端窗口，`python client1_LogRge.py`，启动client1
4. 新建另一个终端窗口，`python client1_LogRge.py`，启动client2
5. 程序自动完成参数设置，数据读取，横向联邦逻辑回归模型训练和测试任务。

# 二、从零开始实现纵向 SBT

## 0. 设置参数

**server**
```python
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
HOST = '127.0.0.1'
PORT = 23705
world_size = 2
partyid = 0

dataset_name = "census"
# dataset_name = "mnist"
epoch = 20
aggregator = 'FedAvg'
# aggregator = 'FedAvg_seq'


#逻辑回归模型
model_name = 'LogisticRegression'
in_features = 81
num_classes = 2
model = LogReg(in_features,num_classes)


model.to(device)

learningrate = 0.01
lossfunction = nn.CrossEntropyLoss()
role = 'server'

# # FedProx
# aggregator = 'FedProx'
# mu = 0.02
#
# # Scaffold
# aggregator = 'Scaffold'
# E = 30
#
# # PersonalizedFed
# aggregator = 'PersonalizedFed'
# kp = 0  # rate of personalized lyaer
#
# Differential Privacy Based Federated Learning
# aggregator = 'FedDP'
```

**client 1**
```python
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

# 设置相关参数
HOST = '127.0.0.1'
PORT = 23705
world_size = 2
partyid = 1

dataset_name = "census"
# dataset_name = "mnist"
learningrate = 0.01
epoch = 20
iter = 5
batch_size = 64


#逻辑回归模型
model_name = 'LogisticRegression'
in_features = 81
num_classes = 2
model = LogReg(in_features,num_classes)


model.to(device)
aggregator = 'FedAvg'
# aggregator = 'FedAvg_seq'
optimizer = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.5)
lossfunction = nn.CrossEntropyLoss()

# # FedProx
# aggregator = 'FedProx'
# mu = 0.02
#
# # Scaffold
# aggregator = 'Scaffold'
# E = 30
# optimizer = ScaffoldOptimizer(model.parameters(), lr=learningrate, weight_decay=1e-4)
#
# # PersonalizedFed
# aggregator = 'PersonalizedFed'
# kp = 0  # rate of personalized lyaer
#
# Differential Privacy Based Federated Learning
# aggregator = 'FedDP'
# dp_mechanism = 'Laplace'
# dp_clip = 10
# dp_epsilon = 100/math.sqrt(epoch)
# dp_delta = 1e-5
```
**client 2**
```python
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

# 设置相关参数
HOST = '127.0.0.1'
PORT = 23705
world_size = 2
partyid = 2

dataset_name = "census"
# dataset_name = "mnist"
learningrate = 0.01
epoch = 20
iter = 5
batch_size = 64


#逻辑回归模型
model_name = 'LogisticRegression'
in_features = 81
num_classes = 2
model = LogReg(in_features,num_classes)


model.to(device)
aggregator = 'FedAvg'
# aggregator = 'FedAvg_seq'
optimizer = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.5)
lossfunction = nn.CrossEntropyLoss()

# # FedProx
# aggregator = 'FedProx'
# mu = 0.02
#
# # Scaffold
# aggregator = 'Scaffold'
# E = 30
# optimizer = ScaffoldOptimizer(model.parameters(), lr=learningrate, weight_decay=1e-4)
#
# # PersonalizedFed
# aggregator = 'PersonalizedFed'
# kp = 0  # rate of personalized lyaer
#
# Differential Privacy Based Federated Learning
# aggregator = 'FedDP'
# dp_mechanism = 'Laplace'
# dp_clip = 10
# dp_epsilon = 100/math.sqrt(epoch)
# dp_delta = 1e-5
```

## 1. 加载数据集

**server加载测试数据集**
```python
Testset = myData(name=dataset_name,
                          root='../../data',
                          train=False,
                          download=True,)
```

**client1 加载训练和测试数据集**
```python
    Trainset = myData(name=dataset_name,
                          root='../../data',
                          train=True,
                          download=True,)
    Testset = myData(name=dataset_name,
                          root='../../data',
                          train=False,
                          download=True,)
```

**client2 加载训练和测试数据集**
```python
    Trainset = myData(name=dataset_name,
                          root='../../data',
                          train=True,
                          download=True,)
    Testset = myData(name=dataset_name,
                          root='../../data',
                          train=False,
                          download=True,)
```



## 2. 横向逻辑回归模型训练

**server**
```python
server = setServer()
print(" Server training...")
model = server.train(Testset)
print("Server training done.")
```

**client 1**
```python
client = setClient()
print("Client training...")
model_parameters = client.train(Trainset,Testset)
print("Client training done.")
```

**client 2**
```python
client = setClient()
print("Client training...")
model_parameters = client.train(Trainset,Testset)
print("Client training done.")
```
## 3. 横向逻辑回归测试
每个参与方可单独进行测试和预测，不需要和其他方合作，不需要通信
**server**
```python
test_accuracy, test_loss = server.test(Testset)
```

**client1**
```python
test_accuracy, test_loss = client.test(Testset)
```

**client2**
```python
test_accuracy, test_loss = client.test(Testset)
```


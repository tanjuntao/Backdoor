# 一、运行

1. 进入当前目录
2. 在一个终端窗口中 `python active.py`，启动主动方
3. 新建另一个终端窗口，`python passive.py`，启动被动方
4. 程序自动完成参数设置，数据读取，特征工程，隐私集合求交，vertical NN 模型训练和测试任务。

# 二、从零开始实现纵向 vertical NN

## 0. 设置参数

**主动方**
```python
trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_active_train.csv'
testset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_active_test.csv'
passive_feat_frac = 0.5
feat_perm_option = Const.SEQUENCE
active_ip = 'localhost'
active_port = 20000
passive_ip = 'localhost'
passive_port = 30000
_epochs = 80
_batch_size = 64
_learning_rate = 0.01
_crypto_type = Const.PLAIN
_key_size = 1024
_loss_fn = nn.CrossEntropyLoss()
bottom_nodes = [5, 3, 3]
intersect_nodes = [3, 3, 3]
top_nodes = [3, 2]
```

**被动方**
```python

trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_passive_train.csv'
testset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_passive_test.csv'
passive_feat_frac = 0.5
feat_perm_option = Const.SEQUENCE
active_ip = 'localhost'
active_port = 20000
passive_ip = 'localhost'
passive_port = 30000
_epochs = 80
_batch_size = 64
_learning_rate = 0.01
_crypto_type = Const.PLAIN
_key_size = 1024
bottom_nodes = [5, 3, 3]
```

## 1. 加载数据集

**主动方**
```python
active_trainset = TorchDataset(role=Const.ACTIVE_NAME, abs_path=trainset_path)
active_testset = TorchDataset(role=Const.ACTIVE_NAME, abs_path=testset_path)
print(colored('1. Finish loading dataset.', 'red'))
```

**被动方**
```python
passive_trainset = TorchDataset(role=Const.PASSIVE_NAME, abs_path=trainset_path)
passive_testset = TorchDataset(role=Const.PASSIVE_NAME, abs_path=testset_path)
print(colored('1. Finish loading dataset.', 'red'))
```

## 2. 特征工程

**主动方**
```python
# 此示例数据集(give_me_some_credit)无需特征工程
```

**被动方**
```python
# 此示例数据集(give_me_some_credit)无需特征工程
```

## 3. 隐私集合求交

**主动方**
```python
messenger = FastSocket(role=Const.ACTIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
psi_crypto = RSACrypto()
active_psi = RSAPSIActive(active_trainset.ids, messenger, psi_crypto)
common_ids = active_psi.run()
active_trainset.filter(common_ids)
print(colored('3. Finish psi protocol', 'red'))
```

**被动方**
```python
messenger = FastSocket(role=Const.PASSIVE_NAME,
                           active_ip=active_ip,
                           active_port=active_port,
                           passive_ip=passive_ip,
                           passive_port=passive_port)
passive_psi = RSAPSIPassive(passive_trainset.ids, messenger)
common_ids = passive_psi.run()
passive_trainset.filter(common_ids)
print(colored('3. Finish psi protocol', 'red'))
```

## 4. vertical NN 训练

**主动方**
```python
bottom_model = ActiveBottomModel(bottom_nodes)
intersect_model = IntersectionModel(intersect_nodes)
top_model = TopModel(top_nodes)
_models = [bottom_model, intersect_model, top_model]
_optimizers = [torch.optim.SGD(model.parameters(), lr=_learning_rate)
                for model in _models]
vfl_crypto = crypto_factory(crypto_type=_crypto_type,
                            key_size=_key_size,
                            num_enc_zeros=10000,
                            gen_from_set=False)
active_vfl = ActiveNeuralNetwork(epochs=_epochs,
                                    batch_size=_batch_size,
                                    models=_models,
                                    optimizers=_optimizers,
                                    loss_fn=_loss_fn,
                                    messenger=messenger,
                                    crypto_type=_crypto_type,
                                    saving_model=True)
active_vfl.train(active_trainset, active_testset)
```

**被动方**
```python
bottom_model = PassiveBottomModel(bottom_nodes)
optimizer = torch.optim.SGD(bottom_model.parameters(), lr=_learning_rate)
passive_vfl = PassiveNeuralNetwork(epochs=_epochs,
                                    batch_size=_batch_size,
                                    model=bottom_model,
                                    optimizer=optimizer,
                                    messenger=messenger,
                                    crypto_type=_crypto_type,
                                    saving_model=True)
passive_vfl.train(passive_trainset, passive_testset)
print(colored('4. Finish collaborative model training', 'red'))
```

## 5. vertical NN 预测

**主动方**
```python
scores = active_vfl.validate(active_testset)
print(scores)
```

**被动方**
```python
scores = passive_vfl.validate(passive_testset)
print(scores)
```

## 6. 结束训练

**主动方**
```python
messenger.close()
print(colored("All Done.", "red"))
```

**被动方**
```python
messenger.close()
print(colored('All Done.', 'red'))
```




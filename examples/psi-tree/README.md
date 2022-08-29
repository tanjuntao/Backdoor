# 一、运行

1. 进入当前目录
2. 在一个终端窗口中，`python active.py`，启动主动方
3. 新建另一个终端窗口，`python passive.py`，启动被动方
4. 程序自动完成参数设置，数据读取，特征工程，隐私集合求交，SBT 模型训练和测试任务。

# 二、从零开始实现纵向 SBT

## 0. 设置参数

**主动方**
```python
trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_active_train.csv'
testset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_active_test.csv'

passive_feat_frac = 0.5
feat_perm_option = Const.SEQUENCE

active_ip = "localhost"
active_port = 20000
passive_ip = "localhost"
passive_port = 30000

_n_trees = 5
_task = "binary"
_n_labels = 2
_crypto_type = Const.PAILLIER
_learning_rate = 0.3
_max_bin = 16
_max_depth = 4
_reg_lambda = 0.1
_min_split_samples = 3
_min_split_gain = 1e-7
_fix_point_precision = 53
_sampling_method = "uniform"
_n_processes = 6

_key_size = 1024
```

**被动方**
```python
trainset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_passive_train.csv'
testset_path = '/Users/tanjuntao/LinkeFL/linkefl/data/tabular/give_me_some_credit_passive_test.csv'

passive_feat_frac = 0.5
feat_perm_option = Const.SEQUENCE

active_ip = "localhost"
active_port = 20000
passive_ip = "localhost"
passive_port = 30000

_task = "binary"
_crypto_type = Const.PAILLIER
_max_bin = 16
_n_processes = 6

_key_size = 1024
```

## 1. 加载数据集

**主动方**
```python
active_trainset = NumpyDataset(role=Const.ACTIVE_NAME, abs_path=trainset_path)
active_testset = NumpyDataset(role=Const.ACTIVE_NAME, abs_path=testset_path)
print(colored("1. Finish loading dataset.", "red"))
```

**被动方**
```python
passive_trainset = NumpyDataset(role=Const.PASSIVE_NAME, abs_path=trainset_path)
passive_testset = NumpyDataset(role=Const.PASSIVE_NAME, abs_path=testset_path)
print(colored('1. Finish loading dataset.', 'red'))
```

## 2. 特征工程

**主动方**
```python
active_trainset = parse_label(active_trainset)
active_testset = parse_label(active_testset)
print(colored("2. Finish transforming features", "red"))
```

**被动方**
```python
print(colored('2. Finish transforming features', 'red'))
```

## 3. 隐私集合求交

**主动方**
```python
messenger = FastSocket(
    role=Const.ACTIVE_NAME,
    active_ip=active_ip,
    active_port=active_port,
    passive_ip=passive_ip,
    passive_port=passive_port,
)
psi_crypto = RSACrypto()
active_psi = RSAPSIActive(active_trainset.ids, messenger, psi_crypto, num_workers=_n_processes)
common_ids = active_psi.run()
active_trainset.filter(common_ids)
print(colored("3. Finish psi protocol", "red"))
```

**被动方**
```python
messenger = FastSocket(role=Const.PASSIVE_NAME,
                       active_ip=active_ip,
                       active_port=active_port,
                       passive_ip=passive_ip,
                       passive_port=passive_port)
passive_psi = RSAPSIPassive(passive_trainset.ids, messenger, num_workers=_n_processes)
common_ids = passive_psi.run()
passive_trainset.filter(common_ids)
print(colored('3. Finish psi protocol', 'red'))
```

## 4. SBT 训练

**主动方**
```python
vfl_crypto = crypto_factory(crypto_type=_crypto_type, key_size=_key_size, num_enc_zeros=10000, gen_from_set=False)
active_vfl = ActiveTreeParty(
    n_trees=_n_trees,
    task=_task,
    n_labels=_n_labels,
    crypto_type=_crypto_type,
    crypto_system=vfl_crypto,
    messenger=messenger,
    learning_rate=_learning_rate,
    max_bin=_max_bin,
    max_depth=_max_depth,
    reg_lambda=_reg_lambda,
    min_split_samples=_min_split_samples,
    min_split_gain=_min_split_gain,
    fix_point_precision=_fix_point_precision,
    sampling_method=_sampling_method,
    n_processes=_n_processes,
)
active_vfl.train(active_trainset, active_testset)
print(colored("4. Finish collaborative model training", "red"))
```

**被动方**
```python
passive_vfl = PassiveTreeParty(task=_task, crypto_type=_crypto_type, messenger=messenger, max_bin=_max_bin)
passive_vfl.train(passive_trainset, passive_testset)
print(colored('4. Finish collaborative model training', 'red'))
```

## 5. SBT 预测

**主动方**
```python
scores = active_vfl.predict(active_testset)
print(scores)
# print("Acc: {:.5f} \nAuc: {:.5f} \nf1: {:.5f}".format(scores["acc"], scores["auc"], scores["f1"]))
print(colored("5. Finish collaborative inference", "red"))
```

**被动方**
```python
passive_vfl.predict(passive_testset)
print(colored('5. Finish collaborative inference', 'red'))
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


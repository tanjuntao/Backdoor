# 一、运行

1. 进入当前目录
2. 在一个终端窗口中 `python active.py`，启动主动方
3. 新建另一个终端窗口，`python passive.py`，启动被动方
4. 程序自动完成参数设置，数据读取，特征工程，隐私集合求交，vertical LR 模型训练和测试任务。

# 二、从零开始实现纵向 vertical LR 

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
_epochs = 200
_batch_size = 100
_learning_rate = 0.01
_penalty = Const.L2
_reg_lambda = 0.001
_crypto_type = Const.PLAIN
_random_state = None
_key_size = 1024
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
_epochs = 200
_batch_size = 100
_learning_rate = 0.01
_penalty = Const.L2
_reg_lambda = 0.001
_crypto_type = Const.PLAIN
_random_state = None
```

## 1. 加载数据集

**主动方**
```python
active_trainset = NumpyDataset(role=Const.ACTIVE_NAME, abs_path=trainset_path)
active_testset = NumpyDataset(role=Const.ACTIVE_NAME, abs_path=testset_path)
print(colored('1. Finish loading dataset.', 'red'))
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
active_trainset = scale(add_intercept(active_trainset))
active_testset = scale(add_intercept(active_testset))
print(colored('2. Finish transforming features', 'red'))
```

**被动方**
```python
passive_trainset = scale(passive_trainset)
passive_testset = scale(passive_testset)
print(colored('2. Finish transforming features', 'red'))
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

## 4. vertical LR 训练

**主动方**
```python
vfl_crypto = crypto_factory(crypto_type=_crypto_type,
                                key_size=_key_size,
                                num_enc_zeros=10000,
                                gen_from_set=False)
active_vfl = ActiveLogReg(epochs=_epochs,
                            batch_size=_batch_size,
                            learning_rate=_learning_rate,
                            messenger=messenger,
                            cryptosystem=vfl_crypto,
                            penalty=_penalty,
                            reg_lambda=_reg_lambda,
                            random_state=_random_state,
                            using_pool=False)
active_vfl.train(active_trainset, active_testset)
print(colored('4. Finish collaborative model training', 'red'))
```

**被动方**
```python
passive_vfl = PassiveLogReg(epochs=_epochs,
                                batch_size=_batch_size,
                                learning_rate=_learning_rate,
                                messenger=messenger,
                                crypto_type=_crypto_type,
                                penalty=_penalty,
                                reg_lambda=_reg_lambda,
                                random_state=_random_state,
                                using_pool=False)
passive_vfl.train(passive_trainset, passive_testset)
print(colored('4. Finish collaborative model training', 'red'))
```

## 5. vertical LR 预测

**主动方**
```python
scores = active_vfl.predict(active_testset)
print('Acc: {:.5f} \nAuc: {:.5f} \nf1: {:.5f}'.format(scores['acc'],
                                                        scores['auc'],
                                                        scores['f1']))
print(colored('5. Finish collaborative inference', 'red'))
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


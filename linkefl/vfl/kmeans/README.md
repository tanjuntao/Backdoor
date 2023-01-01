## 一. 运行

1. 进入当前目录中
2. 在一个终端窗口中，`python passive.py`，启动被动方
3. 新建另一个终端窗口，`python active.py`，启动主动方
4. 程序自动完成参数设置，无监督聚类任务

## 二. 代码实现

### 0. 参数设置

##### 主动方

```python
    active_ip = 'localhost'
    active_port = 20001
    passive_ip = 'localhost'
    passive_port = 30001

    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.ACTIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
```

##### 被动方

```python
    active_ip = 'localhost'
    active_port = 20001
    passive_ip = 'localhost'
    passive_port = 30001

    _messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                   role=Const.PASSIVE_NAME,
                                   active_ip=active_ip,
                                   active_port=active_port,
                                   passive_ip=passive_ip,
                                   passive_port=passive_port)
```

### 1. 加载数据集

##### 主动方

```python
    dataset = np.genfromtxt('./watermelon_4.0.txt', delimiter=',')
    X = dataset[:, 1:]
    X_active = dataset[:, 1:2] # the first column are IDs
    y = [-1 for _ in range(X_active.shape[0])] # by default, all samples has no label

    y[3], y[24] = 0, 0
    y[11], y[19] = 1, 1
    y[13], y[16] = 2, 2
```

##### 被动方

```python
    dataset = np.genfromtxt('./watermelon_4.0.txt', delimiter=',')
    X_passive = dataset[:, 2:] # the first column are IDs
```

### 2. KMeans实现

##### 主动方

```python
    active = ActiveConstrainedSeedKMeans(messenger=_messenger,crypto_type=None,n_clusters=3, n_init=10, verbose=False)

    active.fit(X_active, y)
```

##### 被动方

```python
    passive = PassiveConstrainedSeedKMeans(messenger=_messenger,crypto_type=None,n_clusters=3, n_init=10, verbose=False)

    passive.fit(X_passive)
```

### 3. 对样本进行预测

##### 主动方

```python
    active.fit_predict(X_active, y)

    score = active.score(X_active)
    print('score', score)
```

##### 被动方

```python
    passive.fit_predict(X_passive)

    passive.score(X_passive)
```

### 


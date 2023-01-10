## 一. 运行

1. 进入当前目录中
2. 在一个终端窗口中，`python active.py`，启动主动方
3. 新建另一个终端窗口，`python passive.py`，启动被动方
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
    dataset_name = 'epsilon'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    _random_state = None

    active_trainset = NumpyDataset.buildin_dataset(dataset_name=dataset_name,
                                                   role=Const.ACTIVE_NAME,
                                                   root='../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option,
                                                   seed=_random_state)
```

##### 被动方

```python
    dataset_name = 'epsilon'
    passive_feat_frac = 0.5
    feat_perm_option = Const.SEQUENCE
    _random_state = None

    passive_trainset = NumpyDataset.buildin_dataset(dataset_name=dataset_name,
                                                   role=Const.PASSIVE_NAME,
                                                   root='../data',
                                                   train=True,
                                                   download=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option,
                                                   seed=_random_state)
```

### 2. KMeans实现

##### 主动方

```python
    active = ActiveConstrainedSeedKMeans(messenger=_messenger, crypto_type=None, n_clusters=3, n_init=10, verbose=False)

    active.fit(X_active, y)
```

##### 被动方

```python
    passive = PassiveConstrainedSeedKMeans(messenger=_messenger, crypto_type=None, n_clusters=3, n_init=10, verbose=False)

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

### 4. 对原始数据进行PCA降维后的可视化

##### 主动方

```python
    pca_active = PCA(n_components=2) 
    pca_active.fit(X_active)
    X_active_projection = pca_active.transform(X_active)

    plot(X_active_projection, active, color_num = n_cluster, name='active_kmeans')
```

##### 被动方

```python
    pca_passive = PCA(n_components=2) 
    pca_passive.fit(X_passive)
    X_passive_projection = pca_passive.transform(X_passive)

    plot(X_passive_projection, passive, color_num = n_cluster, name='passive_kmeans')
```


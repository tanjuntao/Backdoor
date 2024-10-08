# LinkeFL 纵向 SBT 算法说明文档

## 1. 集中式 XGBoost

XGBoost 是经典树模型算法 GBDT 的工程实现，在其基础上做了许多优化，对于输入的样本，将其在各棵树上计算出的结果相加，即得到最终的预测结果。

具体而言，给定训练集 $\{(x_i,y_i)\}_{i=1}^N$，损失函数 $L(y,F(x))$，$M$ 棵树和学习率 $\alpha$，XGBoost 算法实现如下：

1. 初始化模型 $\hat{f}_{(0)}(x)=\arg\min_\theta\sum_{i=1}^N L(y_i,\theta)$

2. 从 $m=1\to M$

   1. 计算一阶导，二阶导

      $\hat{g}_m(x_i)=\left[\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}\right]_{f(x)=\hat{f}_{(m-1)}(x)}$

      $\hat{h}_m(x_i)=\left[\frac{\partial^2 L(y_i,f(x_i))}{\partial f(x_i)^2}\right]_{f(x)=\hat{f}_{(m-1)}(x)}$

   2. 计算一棵树

      $\hat{\phi}_m=\arg\min_\phi\sum_{i=1}^N \frac{1}{2}\hat{h}_m(x_i)\left[-\frac{\hat{g}_m(x_i)}{\hat{h}_m(x_i)}-\phi(x_i)\right]^2$

      $\hat{f}_{(m)}(x)=\alpha\hat{\phi}_m(x)$

   3. 更新模型

      $\hat{f}_{(m)}(x)=\hat{f}_{(m-1)}(x)+\hat{f}_m(x)$

3. 输出 $\hat{f}(x)=\hat{f}_{(M)}(x)=\sum_{m=0}^M \hat{f}_m(x)$

## 2. 纵向 SecureBoost (SBT)

在一个纵向联邦学习系统中，不是一般性，假设有两个参与方 Alice 和 Bob，其中 Alice 方只拥有特征信息，称为被动方（Passive Party），Bob 方同时拥有特征信息和标签信息，称为主动方（Active Party）。双方通过事先商量好的训练协议，在保证本方数据不出本地的前提下，通过交互中间数据，实现模型的联合训练。

纵向 SBT 与 XGBoost 没有本质不同，区别在于被动方计算一、二阶导求和时需要在密态下进行，然后传回主动方解密后再计算增益从而选择最佳划分点。

|  步骤  | 主动方 Bob                                                   | 被动方 Alice                                                 |
| :----: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| step 0 | 创建 Paillier 秘钥对，并将公钥发送给被动方                   | 接收 Paillier 公钥                                           |
| step 1 | 每棵树训练开始前，根据当前预测结果与真实 label 计算 loss 的一、二阶导，加密发送给被动方 | 接受一、二阶导                                               |
| step 2 | 对树中每个节点，计算自己这一侧的划分增益；接受被动方的密态结果并计算其划分增益 | 在密态下计算当前节点拥有的数据的划分方案，返回给主动方       |
| step 3 | 选择最大增益点，以此 feature 和阈值作为该节点的划分方案：如果在本地，直接给出；如果在被动方，告知其划分并返回划分结果 | 如果划分结果在自己这一侧，给出以此 feature 和阈值划分后的 sample 分布返回给主动方 |

**安全性分析**

在整个训练协议中，主动方发送给被动方的一、二阶导信息通过 Paillier 同态加密，被动方无私钥，因此无法知道主动方的导数信息；而被动方为划分节点时，仅传输在被动方本地计算好的划分点标识而无具体内容，主动方必须依赖被动方才能获取该节点处的划分方向。

**模型性能分析**

通过上述协议可知，最终得到的模型测试准确率同不加密中心式训练方式相比是无损的。

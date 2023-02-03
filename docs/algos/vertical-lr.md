# LinkeFL 纵向逻辑回归算法说明文档

## 1. Centralized Logistic Regression
逻辑回归模型（logistic regression model）是一种经典的、广泛使用的、可解释的机器学习算法，属于监督学习范畴下的线性二分类模型。

记逻辑回归模型为 $f(\boldsymbol{W}) = \mathcal{X} \rightarrow \mathcal{Y}$，其表示从输入空间 $\mathcal{X}$ 到输出空间 $\mathcal{Y}$ 的一种映射，该映射关系由参数 $\boldsymbol{W}$ 所确定。

记训练数据集为 $T = \{(\boldsymbol{x}_1, y_1), (\boldsymbol{x}_2, y_1), \cdots, (\boldsymbol{x}_N, y_N)\}$，逻辑回归模型的学习目标是最小化如下的损失函数：

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} l(f(\boldsymbol{x}_i; \boldsymbol{W}), y_i) + \lambda \Omega(\boldsymbol{W})$$

其中 $l(\cdot)$ 表示损失函数 (Loss Function)，$\lambda \Omega(\boldsymbol{W})$ 表示正则化项，用于防止模型过拟合。

在逻辑回归二分类模型中，标签空间 $\mathcal{Y} = \{0, 1\}$，一般使用交叉熵函数（Cross-Entropy Loss），也称为 Log-Loss 函数来作为损失函数，并使用 `sigmoid` 函数作为激活函数。

更具体的，在逻辑回归模型中，映射函数表示和损失函数分别表示为：

$$f(\boldsymbol{x};\boldsymbol{W}) = \sigma(\boldsymbol{W}^T \boldsymbol{x}), \text{where} \; \sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\begin{equation}
\mathcal{L} = - \frac{1}{N}\sum_{i=1}^N y_i \log(f(\boldsymbol{x}_i)) + (1-y_i)\log(1 - f(\boldsymbol{x}_i)) + \lambda \Omega(\boldsymbol{W})
\end{equation}$$

梯度计算公式为：

$$\begin{equation}\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = -\frac{1}{N} \sum_{i=1}^N (y_i - f(\boldsymbol{x}_i)) \boldsymbol{x}_i + \lambda \frac{\partial \mathcal{L}}{\partial \Omega(\boldsymbol{W})}\end{equation}$$

其中 $y_i - f(\boldsymbol{x}_i)$ 称之为残差（residue）。

* 当使用 $L_1$ 正则项时，$\Omega(\boldsymbol{W}) = \left \| \boldsymbol{W}\right \|_1$，$\frac{\partial \mathcal{L}}{\partial \Omega(\boldsymbol{W})} = \text{sign}(\boldsymbol{W})$
* 当使用 $L_2$ 正则项时，$\Omega(\boldsymbol{W}) =\frac{1}{2} \left \| \boldsymbol{W}\right \|_2^2$，$\frac{\partial \mathcal{L}}{\partial \Omega(\boldsymbol{W})} = \boldsymbol{W}$

在模型训练阶段，由于使用的损失函数是交叉熵函数，该函数是凸函数（convex function），因此可以使用随机梯度下降（SGD）或者批量随机梯度下降（mini-batch SGD）来优化模型损失函数，并最终保证收敛到全局最优解。在模型预测阶段，对于给定的一条测试样本点 $\boldsymbol{x}_j$，映射函数的输出 $f(\boldsymbol{x}_j; \boldsymbol{W}) = \sigma(\boldsymbol{W}^T \boldsymbol{x}_j)$ 表示为该条样本被预测为正类（positive class）的概率。

## 2. Vertical Federated Logistic Regression

在一个纵向联邦学习系统中，不是一般性，假设有两个参与方 Alice 和 Bob，其中 Alice 方只拥有特征信息，称为被动方（Passive Party），Bob 方同时拥有特征信息和标签信息，称为主动方（Active Party）。双方通过事先商量好的训练协议，在保证本方数据不出本地的前提下，通过交互中间数据，实现模型的联合训练。

设主动方的训练数据集为 $T^B = \{(\boldsymbol{x}^B_i, y_1), (\boldsymbol{x}^B_2, y_2), \cdots, (\boldsymbol{x}^B_N, y_N)\}$，模型参数为 $\boldsymbol{W}^A$；被动方的训练数据集为 $T^A = \{\boldsymbol{x}^A_1,\boldsymbol{x}^A_2, \cdots, \boldsymbol{x}^A_N\}$，模型参数为 $\boldsymbol{W}^B$。双方通过如下的协议来完成逻辑回归模型的联合训练。

| 步骤 | 主动方 Bob | 被动方 Alice|
|:---:| -------|------|
|step 0 | 创建 Paillier 秘钥对，并将公钥发送给被动方 | 接收 Paillier 公钥|
|step 1 |  对于 $\forall i \in [N]$ 计算  $\boldsymbol{W}^B\cdot \boldsymbol{x}^B_i$，并从被动方接收$\boldsymbol{W}^A \cdot \boldsymbol{x}^A_i$，进而计算出完整的线性预测值 $\boldsymbol{W}\cdot \boldsymbol{x}_i = \boldsymbol{W}^A\cdot \boldsymbol{x}^A_i + \boldsymbol{W}^B\cdot \boldsymbol{x}^B_i$，并得到激活值 $f(\boldsymbol{x}_i; \boldsymbol{W}) = \sigma(\boldsymbol{W}\cdot \boldsymbol{x}_i)$，并根据公式（1）计算每个样本的损失值 | 对于 $\forall i \in [N]$ 计算 $\boldsymbol{W}^A \cdot \boldsymbol{x}^A_i$ 并将其发送给主动方|
|step 2| 对于 $\forall i \in [N]$，计算残差值 $r_i = y_i - f(\boldsymbol{x}_i; \boldsymbol{W})$，使用同态加密后将 $\langle r_i \rangle$ 发送给被动方 | 接收密态残差值 $\langle r_i \rangle$ 并根据公式（2） 计算密态梯度值 $\langle \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^B} \rangle $，加上随机噪声掩码之后将 $\langle \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^B} \rangle  + \langle \boldsymbol{R}^B \rangle$ 发送给主动方
| step 3| 首先本地根据公式（2）计算明文梯度 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^A}$，接着解密被动方发送过来的梯度得到 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^B} + \boldsymbol{R}^B$，将其返回给被动方 | 接收解密后的梯度，去除随机噪声掩码得到明文真实梯度 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^B}$
|step 4| 更新模型参数 $\boldsymbol{W}^A = \boldsymbol{W}^A - \eta \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^A}$ | 更新模型参数 $\boldsymbol{W}^B = \boldsymbol{W}^B - \eta \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^B}$


**安全性分析**
在整个训练协议中，主动方发送给被动方的残差信息通过 Paillier 同态加密，被动方无私钥，因此无法知道主动方的真实标签信息；被动方发送给主动方的密文梯度加入了随机噪声掩码，主动方也无法知晓被动方真实梯度，进而保护被动方的隐私数据。

**模型性能分析**
通过上述协议可知，上述两方的纵向逻辑回归训练协议，最终得到的模型测试准确率同中心式训练方式相比是无损的。

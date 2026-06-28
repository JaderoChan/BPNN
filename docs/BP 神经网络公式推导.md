# BP 神经网络公式推导

考虑一个 **三层全连接 BP 神经网络**。

以下内容使用 **Sigmoid** 和 **Softmax** 分别作为隐藏层和输出层的激活函数，使用 **多分类交叉熵 CCE** 作为损失函数进行推导（用于单标签多分类任务）。对于使用其他的函数的情况也可以根据这个流程进行推导，参见 [其他激活函数/损失函数的推导（追加）](#其他激活函数损失函数的推导追加)。

$$
\begin{aligned}
输入层：\underbrace{x_1,\;x_2,\;\cdots \;x_i}_n\\
\downarrow W_1\\
隐藏层：\underbrace{h_1,\;h_2,\;\cdots \;h_j}_m\\
\downarrow W_2\\
输出层：\underbrace{\hat{y}_1,\;\hat{y}_2,\;\cdots \;\hat{y}_k}_r\\
\end{aligned}
$$

## 符号定义

| 类别 | 符号 | 数量 | 意义 |
| ---- | ---- | ---- | --- |
| **输入层** | $x_i$ | $n,i\in[1,n]$ | 第 $i$ 个输入节点的输入值 |
| **隐藏层** | $h_j$ | $m,j\in[1,m]$ | 第 $j$ 个隐藏节点激活后的值 |
| **输出层** | $\hat{y}_k$ | $r,k\in[1,r]$ | 第 $k$ 个输出节点激活后的预测值 |
| **真实标签值** | $y_k$ | - | 第 $k$ 个输出节点的真实标签值 |
| **输入-隐藏层权重** | $w_{ij}$ | - | 第 $i$ 个输入层节点对第 $j$ 个隐藏层节点的权重 |
| **隐藏-输出层权重** | $w_{jk}$ | - | 第 $j$ 个隐藏层节点对第 $k$ 个输出层节点的权重 |
| **隐藏层偏置值** | $b_j$ | - | 第 $j$ 个隐藏层节点的偏置值 |
| **输出层偏置值** | $b_k$ | - | 第 $k$ 个输出层节点的偏置值 |

### 中间量符号

- $z_j$ 表示第 $j$ 个隐藏层节点的 **加权求和值**（隐藏层节点激活前的值）
- $z_k$ 表示第 $k$ 个输出层节点的 **加权求和值**（输出层节点激活前的值）

**加权求和值** 包括了偏置值在内。

### 预定义符号

- $L$ 表示 **损失函数**，用于衡量预测值与真实标签值间的差距（训练目的就是让损失函数的结果尽可能小）。本文使用 **多分类交叉熵 CCE** 损失函数：

$$L=-\sum_{k=1}^ry_k\log(\hat{y}_k)$$

- $f(\cdot)$ 表示隐藏层 **激活函数**。
- $g(\cdot)$ 表示输出层 **激活函数**。

本文中隐藏层使用 **Sigmoid** 函数，输出层使用 **Softmax** 函数：

$$h_j=f(z_j)=\frac{1}{1+e^{-z_j}}$$
$$\hat{y}_k=g(z)_k=\frac{e^{z_k}}{\sum_{k'=1}^re^{z_{k'}}}$$

- $\eta$ 表示 **学习率**，控制每次迭代的更新步长。

## 前向传播

### 计算隐藏层节点值

$$
z_j=\sum_{i=1}^nx_iw_{ij}+b_j\\
h_j=f(z_j)
$$

### 计算输出层节点值

$$
z_k=\sum_{j=1}^mh_jw_{jk}+b_k\\
\hat{y}_k=\frac{e^{z_k}}{\sum_{k'=1}^re^{z_{k'}}}
$$

## 反向传播

定义 $\delta$ 为损失函数对某节点的加权求和值（$z_j$ 或 $z_k$）的偏导，通过下标 $j$ 和 $k$ 区分其是隐藏层还是输出层的偏导。

### 计算损失函数对输出层节点的加权求和的偏导

$$
\delta_k=\frac{\partial L}{\partial z_k}
$$

Softmax 的各分量彼此耦合，因此需要对所有输出分量一起求导。对任意 $k'$，有

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=\hat{y}_{k'}(\mathbb{I}_{k'k}-\hat{y}_k)
$$

其中 $\mathbb{I}_{k'k}$ 为示性函数，当 $k'=k$ 时取 $1$，有

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=\hat{y}_k(1-\hat{y}_k)
$$

否则取 $0$，有

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=-\hat{y}_{k'}\cdot \hat{y}_k
$$

故

$$
\begin{aligned}
\delta_k
&=\frac{\partial L}{\partial z_k}
=\sum_{k'=1}^r\frac{\partial L}{\partial \hat{y}_{k'}}\frac{\partial \hat{y}_{k'}}{\partial z_k}\\
&=\sum_{k'=1}^r\left(-\frac{y_{k'}}{\hat{y}_{k'}}\right)\hat{y}_{k'}(\mathbb{I}_{k'k}-\hat{y}_k)\\
&=(-(y_k)(1-\hat{y}_k))+(y_{k'}\hat{y}_k)+...\\
&=-y_k+\hat{y}_k\sum_{k'=1}^ry_{k'}
\end{aligned}
$$

单标签多分类任务中，标签（常用 One-Hot）满足 $\sum_{k'=1}^ry_{k'}=1$，因此

$$
\delta_k=\hat{y}_k-y_k
$$

### 计算损失函数对隐藏层节点的加权求和的偏导

$$\delta_j=\frac{\partial L}{\partial z_j}$$

因为第 $j$ 个隐藏节点会影响所有 $r$ 个输出层节点，所以

$$
\delta_j=\sum_{k=1}^r\frac{\partial L}{\partial z_k}\cdot \frac{\partial z_k}{\partial h_j}\cdot \frac{\partial h_j}{\partial z_j}
$$

其中

$$\frac{\partial L}{\partial z_k}=\delta_k$$
$$\frac{\partial z_k}{\partial h_j}=w_{jk}$$
$$\frac{\partial h_j}{\partial z_j}=f'(z_j)$$

故

$$\delta_j=\sum_{k=1}^r\delta_k\cdot w_{jk}\cdot f'(z_j)$$

将 $f'(z_j)$ 提出得

$$\delta_j=f'(z_j)\sum_{k=1}^r\delta_kw_{jk}$$

### 计算损失函数对各参数的偏导

得到损失函数对隐藏层与输出层节点的加权求和的偏导 $\delta_j$ 和 $\delta_k$ 后，计算损失函数对各层节点的权重与偏置值的偏导就会容易许多。

下文将损失函数对各参数的偏导统称为 **梯度**，在每一次迭代过程中，沿着逆梯度方向移动一个步长（即学习率 $\eta$）即可逼近损失函数的局部最小值。

#### 隐藏-输出层权重 $w_{jk}$ 梯度

$$\frac{\partial L}{\partial w_{jk}}=\frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial w_{jk}}=\delta_kh_j$$

#### 输出层偏置值 $b_k$ 梯度

$$\frac{\partial L}{\partial b_k}=\frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial b_k}=\delta_k$$

#### 输入-隐藏层权重 $w_{ij}$ 梯度

$$\frac{\partial L}{\partial w_{ij}}=\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial w_{ij}}=\delta_jx_i$$

#### 隐藏层偏置值 $b_j$ 梯度

$$\frac{\partial L}{\partial b_j}=\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial b_j}=\delta_j$$

### 更新参数

上面已经求得了各参数的梯度，下面将原参数沿着逆梯度方向移动 $\eta$ 步长，即

$$w_{jk}'=w_{jk}-\eta\cdot \frac{\partial L}{\partial w_{jk}}=w_{jk}-\eta \delta_kh_j$$
$$b_k'=b_k-\eta\delta_k$$
$$w_{ij}'=w_{ij}-\eta \delta_jx_i$$
$$b_j'=b_j-\eta \delta_j$$

## 简易实现代码

```python
import math

# Const
n, m, r     # input node num, hide node num, output ndoe num
eta         # learn_rate

# Pre-data
xs      : list          # Input nodes value (x_i)
ws1     : list[list]    # Weights of input layer to hide layer (w_{ij})
ws2     : list[list]    # Weights of hide layer to output layer (w_{jk})
bs1     : list          # Bias of hide nodes (b_j)
bs2     : list          # Bias of output nodes (b_k)
labels  : list          # Real labels value (y_k)

# Intermediate data
zs1     : list          # Hide nodes value before activate (z_j)
zs2     : list          # Output nodes value before activate (z_k)
hs      : list          # Hide nodes value (h_j)
ys      : list          # Output nodes value (\hat{y}_k)
deltas1 : list          # (\delta_j)
deltas2 : list          # (\delta_k)


###########
# Utility #
###########

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_deriv(x):
    return math.exp(-x) / math.pow(1.0 + math.exp(-x), 2)

def loss():
    # assert ys.size() == labels.size()

    loss_sum = 0.0
    tiny = 1e-12
    for k in range(r):
        # log() maybe got extreme when the parameter passed close to 0, fix up by limit parameter's minimum.
        y_prob = max(ys[k], tiny)
        loss_sum += labels[k] * max.log(y_prob)

    return -loss_sum

#######################
# Forward propagation #
#######################

# Compute z_j and h_j
def comp_zs1():
    for j in range(m):
        zs1[j] = 0.0
        for i in range(n):
            zs1[j] += (xs[i] * ws1[i][j])
        zs1[j] += bs1[j]

def comp_hs():
    for j in range(m):
        hs[j] = sigmoid(zs1[j])

# Compute z_k and \hat{y}_k
def comp_zs2():
    for k in range(r):
        zs2[k] = 0.0
        for j in range(m):
            zs2[k] += (hs[j] * ws2[j][k])
        zs2[k] += bs2[k]

def comp_ys():
    max_z = zs2[0]
    for k in range(1, r):
        if zs2[k] > max_z:
            max_z = zs2[k]

    exp_sum = 0.0
    for k in range(r):
        exp_sum += math.exp(zs2[k] - max_z)
    for k in range(r):
        ys[k] = math.exp(zs2[k] - max_z) / exp_sum

####################
# Back propagation #
####################

# Compute \delta_k
def comp_deltas2():
    for k in range(r):
        deltas2[k] = ys[k] - labels[k]

# Compute \delta_j
def comp_deltas1():
    for j in range(m):
        deltas1[j] = 0.0
        for k in range(r):
            deltas1[j] += deltas2[k] * ws2[j][k]
        deltas1[j] *= sigmoid_deriv(zs1[j])

def update_ws1():
    for i in range(n):
        for j in range(m):
            ws1[i][j] -= (eta * deltas1[j] * xs[i])

def update_ws2():
    for j in range(m):
        for k in range(r):
            ws2[j][k] -= (eta * deltas2[k] * hs[j])

def update_bs1():
    for j in range(m):
        bs1[j] -= eta * deltas1[j]

def update_bs2():
    for k in range(r):
        bs2[k] -= eta * deltas2[k]

#############
# Main loop #
#############

def train(epoch, eps):
    last_loss = float('inf')
    while (epoch):
        # Forward propagation
        comp_zs1()
        comp_hs()
        comp_zs2()
        comp_ys()

        # Back propagation
        comp_deltas2()
        comp_deltas1()
        update_ws1()
        update_ws2()
        update_bs1()
        update_bs2()

        curr_loss = loss()
        if (math.fabs(curr_loss - last_loss) < eps):
            break
        last_loss = curr_loss
        epoch -= 1

def predict():
    # Forward propagation
    comp_zs1()
    comp_hs()
    comp_zs2()
    comp_ys()

    # parse ys
    # ...
```

## 其他激活函数/损失函数的推导（追加）

### 使用其他函数的前向传播

---

使用其他激活函数或损失函数的前向传播过程与前文推导基本一致。对于隐藏层激活函数为 `f(x)`，输出层激活函数为 `g(x)` 的情况，可得：

**隐藏层节点值**：

$$
z_j=\sum_{i=1}^nx_iw_{ij}+b_j\\
h_j=f(z_j)
$$

**输出层节点值**：

$$
z_k=\sum_{j=1}^mh_jw_{jk}+b_k\\
\hat{y}_k=g(z_k)
$$

### 使用其他函数的反向传播

---

使用其他激活函数或损失函数的反向传播过程，只有求 $\delta$（损失函数对某节点加权求和值的偏导）部分有所不同。

#### 推导思路

观察 $\delta_k$ 和 $\delta_j$ 的链式展开结构：

对于 **非 Softmax** 的输出层激活函数，各输出节点的激活值彼此独立，根据链式法则可将 $\delta_k$ 分解为两个相互独立的因子之积：

$$\delta_k = \frac{\partial L}{\partial z_k} = \underbrace{\frac{\partial L}{\partial \hat{y}_k}}_{\text{仅由损失函数决定}} \cdot \underbrace{g'(z_k)}_{\text{仅由输出层激活函数决定}}$$

$\delta_j$ 的结构同理：

$$\delta_j = \underbrace{f'(z_j)}_{\text{仅由隐藏层激活函数决定}} \cdot \sum_{k=1}^r\delta_k w_{jk}$$

三个因子完全相互独立，对于不同激活函数与损失函数的搭配只需要模块化组合不同因子即可。

#### 各函数对应的因子

> 代码中出现的另外两个损失函数的定义：
>
> | 损失函数 | 定义 |
> | ------- | --- |
> | 均方误差 MSE | $L=\frac{1}{r}\displaystyle\sum_{k=1}^{r}(y_k-\hat{y}_k)^2$ |
> | 二元交叉熵 BCE | $L=-\frac{1}{r}\displaystyle\sum_{k=1}^r\left[y_k\log(\hat{y}_k)+(1-y_k)\log(1-\hat{y}_k)\right]$ |

**损失函数的 $\frac{\partial L}{\partial \hat{y}_k}$**：

| 损失函数 | $\frac{\partial L}{\partial \hat{y}_k}$ |
| ------- | --------------------------------------- |
| CCE | $-\frac{y_k}{\hat{y}_k}$ |
| MSE | $\frac{2(\hat{y}_k-y_k)}{r}$ |
| BCE | $\frac{1}{r}\cdot\frac{\hat{y}_k-y_k}{\hat{y}_k(1-\hat{y}_k)}$ |

**输出层激活函数的 $g'(z_k)$**：

| 输出层激活函数 | $g'(z_k)$ |
| ------------ | --------- |
| Sigmoid | $\hat{y}_k(1-\hat{y}_k)$ |
| Tanh | $1-\hat{y}_k^2$ |
| ReLU | $\mathbb{I}[z_k>0]$ |
| Leaky ReLU | $\begin{cases}1 & z_k>0\\ \alpha & z_k\leq 0\end{cases}$ |
| Linear | $1$ |

**各隐藏层激活函数的 $f'(z_j)$**：

| 隐藏层激活函数 | $f'(z_j)$ |
| ------------ | --------- |
| Sigmoid | $h_j(1-h_j)$ |
| Tanh | $1-h_j^2$ |
| ReLU | $\mathbb{I}[z_j>0]$ |
| Leaky ReLU | $\begin{cases}1 & z_j>0\\ \alpha & z_j\leq 0\end{cases}$ |
| Linear | $1$ |

#### 组合使用

查表得到所需因子后，代入下式即可：

$$\delta_k = \frac{\partial L}{\partial \hat{y}_k} \cdot g'(z_k)$$

$$\delta_j = f'(z_j)\sum_{k=1}^r\delta_k w_{jk}$$

之后的权重与偏置梯度计算，以及参数更新方式与正文完全相同，不再复述。

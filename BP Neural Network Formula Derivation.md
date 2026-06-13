# BP Neural Network Formula Derivation

> This document was translated by AI.

Consider a **three-layer fully-connected BP neural network**.

The following content is derived using **Sigmoid** and **Softmax** as activation functions of the hidden layer and output layer respectively, and **Multi-classification cross-entropy** as the loss function (for single-label multi-classification tasks). The use of other functions can also be deduced based on this process, which will not be described here. (May be added in the future)

$$
\begin{aligned}
\text{Input layer:}\;&\underbrace{x_1,\;x_2,\;\cdots\;x_i}_n\\
&\downarrow W_1\\
\text{Hidden layer:}\;&\underbrace{h_1,\;h_2,\;\cdots\;h_j}_m\\
&\downarrow W_2\\
\text{Output layer:}\;&\underbrace{\hat{y}_1,\;\hat{y}_2,\;\cdots\;\hat{y}_k}_r\\
\end{aligned}
$$

## Symbol Definitions

| Category | Symbol | Count | Meaning |
| --- | --- | --- | --- |
| **Input layer** | $x_i$ | $n,\;i\in[1,n]$ | Value of the $i$-th input node |
| **Hidden layer** | $h_j$ | $m,\;j\in[1,m]$ | Post-activation value of the $j$-th hidden node |
| **Output layer** | $\hat{y}_k$ | $r,\;k\in[1,r]$ | Post-activation predicted value of the $k$-th output node |
| **True label** | $y_k$ | - | True label value for the $k$-th output node |
| **Input–hidden weights** | $w_{ij}$ | - | Weight from the $i$-th input node to the $j$-th hidden node |
| **Hidden–output weights** | $w_{jk}$ | - | Weight from the $j$-th hidden node to the $k$-th output node |
| **Hidden layer biases** | $b_j$ | - | Bias of the $j$-th hidden node |
| **Output layer biases** | $b_k$ | - | Bias of the $k$-th output node |

### Intermediate Symbols

- $z_j$: the **weighted sum** of the $j$-th hidden node (pre-activation value of the hidden layer)
- $z_k$: the **weighted sum** of the $k$-th output node (pre-activation value of the output layer)

The **weighted sum** includes the bias term.

### Predefined Symbols

- $L$ denotes the **loss function**, which measures the discrepancy between predicted values and true labels (the goal of training is to minimize $L$). This document uses the **multi-class cross-entropy** loss corresponding to Softmax:

$$L=-\sum_{k=1}^ry_k\log(\hat{y}_k)$$

- $f(\cdot)$ denotes the hidden layer **activation function**.
- $g(\cdot)$ denotes the output layer **activation function**.

This document uses **Sigmoid** for the hidden layer and **Softmax** for the output layer:

$$h_j=f(z_j)=\frac{1}{1+e^{-z_j}}$$
$$\hat{y}_k=g(z)_k=\frac{e^{z_k}}{\sum_{k'=1}^re^{z_{k'}}}$$

- $\eta$ denotes the **learning rate**, controlling the step size of each update.

## Forward Propagation

### Computing Hidden Layer Values

$$
z_j=\sum_{i=1}^nx_iw_{ij}+b_j\\
h_j=f(z_j)
$$

### Computing Output Layer Values

$$
z_k=\sum_{j=1}^mh_jw_{jk}+b_k\\
\hat{y}_k=\frac{e^{z_k}}{\sum_{k'=1}^re^{z_{k'}}}
$$

## Backward Propagation

Define $\delta$ as the partial derivative of the loss function with respect to the weighted sum ($z_j$ or $z_k$) of a given node, distinguished by subscripts $j$ (hidden layer) and $k$ (output layer).

### Partial Derivative of Loss w.r.t. Output Layer Weighted Sums

$$
\delta_k=\frac{\partial L}{\partial z_k}
$$

Since Softmax components are mutually coupled, we differentiate over all output components simultaneously. For any $k'$:

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=\hat{y}_{k'}(\mathbb{I}_{k'k}-\hat{y}_k)
$$

where $\mathbb{I}_{k'k}$ is an indicator function equal to $1$ when $k'=k$:

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=\hat{y}_k(1-\hat{y}_k)
$$

and $0$ otherwise:

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=-\hat{y}_{k'}\cdot \hat{y}_k
$$

Therefore:

$$
\begin{aligned}
\delta_k
&=\frac{\partial L}{\partial z_k}
=\sum_{k'=1}^r\frac{\partial L}{\partial \hat{y}_{k'}}\frac{\partial \hat{y}_{k'}}{\partial z_k}\\
&=\sum_{k'=1}^r\left(-\frac{y_{k'}}{\hat{y}_{k'}}\right)\hat{y}_{k'}(\mathbb{I}_{k'k}-\hat{y}_k)\\
&=(-(y_k)(1-\hat{y}_k))+(y_{k'}\hat{y}_k)+\cdots\\
&=-y_k+\hat{y}_k\sum_{k'=1}^ry_{k'}
\end{aligned}
$$

For a single-label multi-class task, the One-Hot label satisfies $\sum_{k'=1}^ry_{k'}=1$, so:

$$
\delta_k=\hat{y}_k-y_k
$$

### Partial Derivative of Loss w.r.t. Hidden Layer Weighted Sums

$$\delta_j=\frac{\partial L}{\partial z_j}$$

Since the $j$-th hidden node influences all $r$ output nodes:

$$
\delta_j=\sum_{k=1}^r\frac{\partial L}{\partial z_k}\cdot \frac{\partial z_k}{\partial h_j}\cdot \frac{\partial h_j}{\partial z_j}
$$

where:

$$\frac{\partial L}{\partial z_k}=\delta_k$$
$$\frac{\partial z_k}{\partial h_j}=w_{jk}$$
$$\frac{\partial h_j}{\partial z_j}=f'(z_j)$$

Therefore:

$$\delta_j=\sum_{k=1}^r\delta_k\cdot w_{jk}\cdot f'(z_j)$$

Factoring out $f'(z_j)$:

$$\delta_j=f'(z_j)\sum_{k=1}^r\delta_kw_{jk}$$

### Gradients of the Loss w.r.t. Each Parameter

Having obtained $\delta_j$ and $\delta_k$, computing the gradients of the loss with respect to all weights and biases becomes straightforward.

The partial derivatives of the loss with respect to each parameter are collectively called **gradients**. At each iteration, moving one step of size $\eta$ in the negative gradient direction approximates a local minimum of the loss function.

#### Gradient of Hidden–Output Weights $w_{jk}$

$$\frac{\partial L}{\partial w_{jk}}=\frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial w_{jk}}=\delta_kh_j$$

#### Gradient of Output Layer Biases $b_k$

$$\frac{\partial L}{\partial b_k}=\frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial b_k}=\delta_k$$

#### Gradient of Input–Hidden Weights $w_{ij}$

$$\frac{\partial L}{\partial w_{ij}}=\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial w_{ij}}=\delta_jx_i$$

#### Gradient of Hidden Layer Biases $b_j$

$$\frac{\partial L}{\partial b_j}=\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial b_j}=\delta_j$$

### Parameter Update

With all gradients computed, update each parameter by moving $\eta$ steps in the negative gradient direction:

$$w_{jk}'=w_{jk}-\eta\cdot \frac{\partial L}{\partial w_{jk}}=w_{jk}-\eta \delta_kh_j$$
$$b_k'=b_k-\eta\delta_k$$
$$w_{ij}'=w_{ij}-\eta \delta_jx_i$$
$$b_j'=b_j-\eta \delta_j$$

## Simple Implementation

```python
import math

# Const
n, m, r     # input node num, hide node num, output node num
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
        # log() may get extreme when the parameter is close to 0; clamp to a minimum value.
        y_prob = max(ys[k], tiny)
        loss_sum += labels[k] * math.log(y_prob)

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

def identify():
    # Forward propagation
    comp_zs1()
    comp_hs()
    comp_zs2()
    comp_ys()

    # parse ys
    # ...
```

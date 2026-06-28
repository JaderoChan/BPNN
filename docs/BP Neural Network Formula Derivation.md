# BP Neural Network Formula Derivation

> This document was translated by AI.

Consider a **three-layer fully connected BP neural network**.

The following derivation uses **Sigmoid** and **Softmax** as the activation functions for the hidden layer and output layer respectively, and **Categorical Cross-Entropy (CCE)** as the loss function (for single-label multi-class classification tasks). The same process can be applied when using other functions, see also [Derivation for Other Activation/Loss Functions (Appendix)](#derivation-for-other-activationloss-functions-appendix).

$$
\begin{aligned}
\text{Input layer:}\quad\underbrace{x_1,\;x_2,\;\cdots \;x_i}_n\\
\downarrow W_1\\
\text{Hidden layer:}\quad\underbrace{h_1,\;h_2,\;\cdots \;h_j}_m\\
\downarrow W_2\\
\text{Output layer:}\quad\underbrace{\hat{y}_1,\;\hat{y}_2,\;\cdots \;\hat{y}_k}_r\\
\end{aligned}
$$

## Symbol Definitions

| Category | Symbol | Count | Meaning |
| -------- | ------ | ----- | ------- |
| **Input layer** | $x_i$ | $n,i\in[1,n]$ | Input value of the $i$-th input node |
| **Hidden layer** | $h_j$ | $m,j\in[1,m]$ | Activated value of the $j$-th hidden node |
| **Output layer** | $\hat{y}_k$ | $r,k\in[1,r]$ | Predicted value of the $k$-th output node after activation |
| **True label** | $y_k$ | - | True label value of the $k$-th output node |
| **Input-to-hidden weights** | $w_{ij}$ | - | Weight from the $i$-th input node to the $j$-th hidden node |
| **Hidden-to-output weights** | $w_{jk}$ | - | Weight from the $j$-th hidden node to the $k$-th output node |
| **Hidden layer bias** | $b_j$ | - | Bias of the $j$-th hidden node |
| **Output layer bias** | $b_k$ | - | Bias of the $k$-th output node |

### Intermediate Variable Symbols

- $z_j$ denotes the **weighted sum** of the $j$-th hidden node (value before activation)
- $z_k$ denotes the **weighted sum** of the $k$-th output node (value before activation)

The **weighted sum** includes the bias term.

### Predefined Symbols

- $L$ denotes the **loss function**, used to measure the discrepancy between predicted values and true labels (the training objective is to minimize the loss). This document uses the **Categorical Cross-Entropy (CCE)** loss function:

$$L=-\sum_{k=1}^ry_k\log(\hat{y}_k)$$

- $f(\cdot)$ denotes the hidden layer **activation function**.
- $g(\cdot)$ denotes the output layer **activation function**.

In this document, the hidden layer uses the **Sigmoid** function, and the output layer uses the **Softmax** function:

$$h_j=f(z_j)=\frac{1}{1+e^{-z_j}}$$
$$\hat{y}_k=g(z)_k=\frac{e^{z_k}}{\sum_{k'=1}^re^{z_{k'}}}$$

- $\eta$ denotes the **learning rate**, controlling the step size of each update.

## Forward Propagation

### Computing Hidden Layer Node Values

$$
z_j=\sum_{i=1}^nx_iw_{ij}+b_j\\
h_j=f(z_j)
$$

### Computing Output Layer Node Values

$$
z_k=\sum_{j=1}^mh_jw_{jk}+b_k\\
\hat{y}_k=\frac{e^{z_k}}{\sum_{k'=1}^re^{z_{k'}}}
$$

## Backpropagation

Define $\delta$ as the partial derivative of the loss function with respect to the weighted sum of a given node ($z_j$ or $z_k$). Subscripts $j$ and $k$ distinguish between the hidden layer and output layer respectively.

### Partial Derivative of the Loss w.r.t. the Output Layer Weighted Sum

$$
\delta_k=\frac{\partial L}{\partial z_k}
$$

Since the components of Softmax are coupled with each other, they must be differentiated jointly. For any $k'$:

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=\hat{y}_{k'}(\mathbb{I}_{k'k}-\hat{y}_k)
$$

where $\mathbb{I}_{k'k}$ is the indicator function. When $k'=k$, it equals $1$:

$$
\frac{\partial \hat{y}_{k'}}{\partial z_k}=\hat{y}_k(1-\hat{y}_k)
$$

Otherwise it equals $0$:

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
&=(-(y_k)(1-\hat{y}_k))+(y_{k'}\hat{y}_k)+...\\
&=-y_k+\hat{y}_k\sum_{k'=1}^ry_{k'}
\end{aligned}
$$

In a single-label multi-class task, the labels (commonly One-Hot encoded) satisfy $\sum_{k'=1}^ry_{k'}=1$, so:

$$
\delta_k=\hat{y}_k-y_k
$$

### Partial Derivative of the Loss w.r.t. the Hidden Layer Weighted Sum

$$\delta_j=\frac{\partial L}{\partial z_j}$$

Because the $j$-th hidden node influences all $r$ output nodes:

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

### Partial Derivatives of the Loss w.r.t. Each Parameter

Having derived $\delta_j$ and $\delta_k$, computing the partial derivatives of the loss with respect to the weights and biases of each layer becomes straightforward.

The partial derivatives of the loss with respect to all parameters are collectively referred to as **gradients**. In each iteration, moving one step of size $\eta$ (the learning rate) in the negative gradient direction brings us closer to a local minimum of the loss function.

#### Gradient of Hidden-to-Output Weights $w_{jk}$

$$\frac{\partial L}{\partial w_{jk}}=\frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial w_{jk}}=\delta_kh_j$$

#### Gradient of Output Layer Bias $b_k$

$$\frac{\partial L}{\partial b_k}=\frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial b_k}=\delta_k$$

#### Gradient of Input-to-Hidden Weights $w_{ij}$

$$\frac{\partial L}{\partial w_{ij}}=\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial w_{ij}}=\delta_jx_i$$

#### Gradient of Hidden Layer Bias $b_j$

$$\frac{\partial L}{\partial b_j}=\frac{\partial L}{\partial z_j}\frac{\partial z_j}{\partial b_j}=\delta_j$$

### Updating Parameters

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
        # log() maybe got extreme when the parameter passed close to 0, fix up by limit parameter's minimum.
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

def predict():
    # Forward propagation
    comp_zs1()
    comp_hs()
    comp_zs2()
    comp_ys()

    # parse ys
    # ...
```

## Derivation for Other Activation/Loss Functions (Appendix)

### Forward Propagation with Other Functions

---

The forward propagation process when using other activation or loss functions is essentially the same as derived above. For a hidden layer activation function $f(x)$ and an output layer activation function $g(x)$:

**Hidden layer node values**:

$$
z_j=\sum_{i=1}^nx_iw_{ij}+b_j\\
h_j=f(z_j)
$$

**Output layer node values**:

$$
z_k=\sum_{j=1}^mh_jw_{jk}+b_k\\
\hat{y}_k=g(z_k)
$$

### Backpropagation with Other Functions

---

When using other activation or loss functions, only the computation of $\delta$ (the partial derivative of the loss w.r.t. a node's weighted sum) differs.

#### Derivation Strategy

Observe the chain rule expansion of $\delta_k$ and $\delta_j$:

For **non-Softmax** output layer activation functions, the activated values of each output node are mutually independent. By the chain rule, $\delta_k$ can be decomposed into the product of two independent factors:

$$\delta_k = \frac{\partial L}{\partial z_k} = \underbrace{\frac{\partial L}{\partial \hat{y}_k}}_{\text{determined solely by the loss function}} \cdot \underbrace{g'(z_k)}_{\text{determined solely by the output activation function}}$$

The structure of $\delta_j$ is analogous:

$$\delta_j = \underbrace{f'(z_j)}_{\text{determined solely by the hidden activation function}} \cdot \sum_{k=1}^r\delta_k w_{jk}$$

The three factors are completely independent of each other. For different combinations of activation and loss functions, simply look up and substitute the corresponding factors.

#### Factors for Each Function

> Definitions of the other two loss functions appearing in the code:
>
> | Loss Function | Definition |
> | ------------- | ---------- |
> | Mean Squared Error (MSE) | $L=\frac{1}{r}\displaystyle\sum_{k=1}^{r}(y_k-\hat{y}_k)^2$ |
> | Binary Cross-Entropy (BCE) | $L=-\frac{1}{r}\displaystyle\sum_{k=1}^r\left[y_k\log(\hat{y}_k)+(1-y_k)\log(1-\hat{y}_k)\right]$ |

**$\frac{\partial L}{\partial \hat{y}_k}$ for each loss function**:

| Loss Function | $\frac{\partial L}{\partial \hat{y}_k}$ |
| ------------- | --------------------------------------- |
| CCE | $-\frac{y_k}{\hat{y}_k}$ |
| MSE | $\frac{2(\hat{y}_k-y_k)}{r}$ |
| BCE | $\frac{1}{r}\cdot\frac{\hat{y}_k-y_k}{\hat{y}_k(1-\hat{y}_k)}$ |

**$g'(z_k)$ for each output layer activation function**:

| Output Activation Function | $g'(z_k)$ |
| -------------------------- | --------- |
| Sigmoid | $\hat{y}_k(1-\hat{y}_k)$ |
| Tanh | $1-\hat{y}_k^2$ |
| ReLU | $\mathbb{I}[z_k>0]$ |
| Leaky ReLU | $\begin{cases}1 & z_k>0\\ \alpha & z_k\leq 0\end{cases}$ |
| Linear | $1$ |

**$f'(z_j)$ for each hidden layer activation function**:

| Hidden Activation Function | $f'(z_j)$ |
| -------------------------- | --------- |
| Sigmoid | $h_j(1-h_j)$ |
| Tanh | $1-h_j^2$ |
| ReLU | $\mathbb{I}[z_j>0]$ |
| Leaky ReLU | $\begin{cases}1 & z_j>0\\ \alpha & z_j\leq 0\end{cases}$ |
| Linear | $1$ |

#### Combining Factors

After looking up the required factors, substitute them into the following formulas:

$$\delta_k = \frac{\partial L}{\partial \hat{y}_k} \cdot g'(z_k)$$

$$\delta_j = f'(z_j)\sum_{k=1}^r\delta_k w_{jk}$$

The subsequent gradient computations for weights and biases, as well as the parameter update procedure, are identical to the main derivation and are not repeated here.

# Function Comparison and Usage Recommendations

> This document was translated by AI.

This document describes the activation functions included in the code, along with their common pairings with loss functions and applicable scenarios.

## Activation Function Comparison

| Activation Function | Formula | Advantages | Disadvantages | Applicable Layer | Notes |
| ------------------- | ------- | ---------- | ------------- | ---------------- | ----- |
| Sigmoid | $f(x)=\frac{1}{1+e^{-x}}$ | Smooth; probabilistic interpretation | Vanishing gradient problem; non-zero-centered output | Hidden / Output | Output range: $(0,1)$. Suitable for binary classification and multi-label classification output layers. |
| Tanh | $f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | Smooth; zero-centered | Vanishing gradient problem; no probabilistic interpretation | Hidden | Output range: $(-1,1)$. Generally not used for output layers unless the regression target lies within $(-1,1)$. |
| ReLU | $f(x)=\max(0,x)$ | Fast computation; alleviates vanishing gradient | Neurons may "die" (output permanently 0) | Hidden | Output range: $[0,+\infty)$. Almost exclusively used in hidden layers. |
| Leaky ReLU | $f(x)=\max(\alpha x,x)$ | Solves the dying neuron problem of ReLU | Introduces an extra hyperparameter $\alpha$; marginal improvement over ReLU in practice | Hidden | Output range: $[0,+\infty)$. Almost exclusively used in hidden layers. |
| Softmax | $f(x)=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}}$ | — | — | Output | Output range: $(0,1)$, summing to $1$. Suitable for single-label multi-class classification output layers. Compresses an arbitrary real-valued vector into a probability distribution. ($K$ is the number of output nodes; $z_k$ is the pre-activation value of the $k$-th output node.) |
| Linear | $f(x)=x$ | — | — | Output | Identity function (no activation); output range: $(-\infty,+\infty)$. Suitable for general regression output layers. |

## Task Pairings

| Task Type | Applicable Scenario | Output Layer Activation | Loss Function |
| --------- | ------------------- | ----------------------- | ------------- |
| Binary Classification | Yes/No decisions | Sigmoid (single output node) | Binary Cross-Entropy (BCE) |
| Single-Label Multi-Class Classification | Mutually exclusive categories (e.g., digit recognition) | Softmax | Categorical Cross-Entropy (CCE) |
| Multi-Label Multi-Class Classification | Multi-label scenarios (e.g., image tagging) | Sigmoid (one output node per class) | Binary Cross-Entropy (BCE) |
| Regression | Curve fitting | Linear | Mean Squared Error (MSE) |

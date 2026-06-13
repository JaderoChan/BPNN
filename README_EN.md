# BP Neural Network

[中文简体 | [**English**](README.md)]

> This document was translated by AI.

A three-layer fully-connected BP neural network library implemented in pure C99, with no third-party dependencies, created for learning purposes.

> **Available enumeration**
>
> - **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax, Linear
> - **Loss functions**：MCE (Multi-class cross entropy), BCE (Binary cross entropy), MSE (Mean square error)

For formula derivations, see [BP Neural Network Formula Derivation](BP%20Neural%20Network%20Formula%20Derivation.md).

## Features

- Zero dependencies, standard C library only
- Provides common types of activation functions and loss functions to meet the needs of a variety of tasks
- Forward and backward propagation fully decoupled — each step can be called independently
- Model parameter saving and loading support
- Training supports custom callbacks for real-time loss monitoring

## Quick Start

### Build the Library

```sh
git clone https://github.com/JaderoChan/bpnn.git
cd bpnn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBPNN_BUILD_EXAMPLE=OFF
cmake --build build -j --config=Release
```

Output: static library `libbpnn.a`, header at `src/bpnn.h`.

### Build the Example

To build the example program, first extract the dataset archives under `example/data_set/`, then enable the option:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBPNN_BUILD_EXAMPLE=ON
cmake --build build -j --config=Release
```

## API Overview

### Parameter Management

| Function | Description |
| --- | --- |
| `bpnn_params_construct_v1()` | Create parameters (weights and biases initialized to 0) |
| `bpnn_params_construct_v2()` | Create parameters (with specified initial weights and biases) |
| `bpnn_params_randomize()` | Xavier random initialization of weights |
| `bpnn_params_save_to_file()` | Save parameters to file |
| `bpnn_params_load_from_file()` | Load parameters from file |
| `bpnn_params_destroy()` | Free parameter memory |

### Training and Inference

```c
// Train
void bpnn_train(
    bpnn_params_t* params, const double* ins_group, const double* labels_group, uint32_t group_num,
    double learn_rate, LossFn loss_fn, uint32_t epoch, double esp,
    bpnn_train_callback_t callback, void* userdata)

// Inference
void bpnn_use(const bpnn_params_t* params, const double* ins, double* outs);
```

`bpnn_train`'s `callback` parameter may be `NULL`; if provided, it is called at the end of each epoch with the current loss value and the delta from the previous epoch.

## Example

An MNIST handwritten digit recognition program built on this library (Use the Sigmoid function as the hidden layer activation function, Softmax as the output layer activation function, and multi-class cross entropy as the loss function.), achieving **98%** accuracy after 50 training epochs. See [BPNN Digit Recognizer](./example/README_EN.md).

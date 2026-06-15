# BP 神经网络

[**中文简体** | [English](README_EN.md)]

一个纯 C99 实现的三层全连接 BP 神经网络库，无任何第三方依赖，出于学习目的创建。

> **可用枚举**
>
> - **激活函数**：Sigmoid、Tanh、ReLU、Leaky ReLU、Softmax、Linear
> - **损失函数**：CCE（多分类交叉熵）、BCE（二元交叉熵）、MSE（均方误差）

公式推导详见 [BP 神经网络公式推导](BP%20神经网络公式推导.md)。

## 特性

- 零依赖，仅使用 C 标准库
- 提供常见的各类激活函数与损失函数，可满足多种任务需求
- 前向传播与反向传播完全解耦，可单独调用各步骤
- 支持模型参数的保存与加载
- 训练过程支持自定义回调函数，可实时监控损失值

## 快速开始

### 构建库

```sh
git clone https://github.com/JaderoChan/bpnn.git
cd bpnn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBPNN_BUILD_EXAMPLE=OFF
cmake --build build -j --config=Release
```

构建产物为静态库 `libbpnn.a`，头文件位于 `src/bpnn.h`。

### 构建示例程序

若需同时构建示例程序，请先解压 `example/digit_recognizer/data_set/` 下的数据集压缩包，再开启对应选项：

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBPNN_BUILD_EXAMPLE=ON
cmake --build build -j --config=Release
```

## API 概览

### 参数管理

| 函数 | 说明 |
| --- | --- |
| `bpnn_params_construct_v1()` | 创建 BP 神经网络参数（权重和偏置初始化为 0） |
| `bpnn_params_construct_v2()` | 创建 BP 神经网络参数（指定初始权重和偏置） |
| `bpnn_params_randomize()` | Xavier 随机初始化权重 |
| `bpnn_params_save_to_file()` | 保存 BP 神经网络参数到文件 |
| `bpnn_params_load_from_file()` | 从文件加载 BP 神经网络参数 |
| `bpnn_params_destroy()` | 释放 BP 神经网络参数内存 |

### 训练与推理

```c
// 训练
void bpnn_train(
    bpnn_params_t* params,
    const double* ins_samples, const double* labels_samples, uint32_t sample_num,
    double learn_rate, loss_fn_t loss_fn, uint32_t epoch, double esp,
    bpnn_train_sample_callback_t sample_callback, void* sc_userdata,
    bpnn_train_epoch_callback_t  epoch_callback,  void* ec_userdata);

// 推理
void bpnn_use(const bpnn_params_t* params, const double* ins, double* outs);
```

## 示例

基于本库实现的 MNIST 手写数字识别程序（使用 Sigmoid 函数作为隐藏层激活函数，Softmax 作为输出层激活函数，多分类交叉熵作为损失函数），50 轮训练后准确率可达 **98%**，详见 [BPNN Digit Recognizer](./example/digit_recognizer/README.md)。

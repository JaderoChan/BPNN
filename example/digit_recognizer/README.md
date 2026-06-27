# BPNN Digit Recognizer

[**中文简体** | [English](README_EN.md)]

基于 BPNN 库实现的 MNIST 手写数字识别示例，涵盖模型训练与推理完整流程。

## 网络结构

使用 MNIST 手写数字数据集（0–9 共 10 类）：

| 层 | 节点数 |
| --- | --- |
| 输入层 | 784（28×28 像素） |
| 隐藏层 | 128 |
| 输出层 | 10 |

## 数据集

数据集图像均为 28×28 灰度图，像素值以 `uint8_t` 存储，文件采用 IDX 二进制格式。

| 文件 | 说明 | 数量 |
| --- | --- | --- |
| `train-images-idx3-ubyte` | 训练图像（跳过开头 16 字节，之后每 784 字节为一张） | 60,000 |
| `train-labels-idx1-ubyte` | 训练标签（跳过开头 8 字节，之后每字节为对应图像的真实类别，取值 0–9） | 60,000 |
| `t10k-images-idx3-ubyte` | 测试图像（格式同训练集） | 10,000 |
| `t10k-labels-idx1-ubyte` | 测试标签（格式同训练集） | 10,000 |

使用前请先解压 `train/data_set/` 目录下的压缩包。

## 使用

`./use` 目录下具有一个基于 **Qt** 实现的数字识别器项目，`./use/bpnn_net_params/net_params.np` 是一个 50 轮训练的结果，可供数字识别器使用。

## 构建

默认情况下将仅构建 `train` 项目

```sh
cmake -B build
cmake --build build
```

如果需要构建 `use` 项目（基于 **Qt** 的数字识别器），需保证 **Qt** （Qt5/Qt6 均可使用）在构建环境中可见

```sh
cmake -B build -DBUILD_DIGIT_RECOGNIZER_USE=ON
cmke --build build
```

## 结果

迭代 50 轮后，测试集准确率可达 **98%**，训练大约需要五分钟时间。

![train_result](./images/train_result.png)
![use_result](./images/use_result.png)

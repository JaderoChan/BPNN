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

## 构建

> 项目构建前请先解压 `data_set/` 目录下的压缩包。

默认情况下仅构建模型训练与测试模块。

```sh
cmake -B build
cmake --build build -j --config=Release
```

如果需要构建 **数字识别器 GUI** 项目，需保证 **Qt** （Qt5/Qt6）在构建环境中可用。

```sh
cmake -B build -DBUILD_DIGIT_RECOGNIZER_GUI=ON
cmke --build build -j --config=Release
```

## 使用

1. 运行模型训练程序以进行模型的训练、测试与保存
2. 运行 GUI 程序，加载训练好的模型数据，手写数字进行识别

> `models` 目录下具有训练好的模型文件，可供 GUI 程序加载使用。

## 结果

迭代 50 轮后，测试集准确率可达 **98%**，训练大约需要五分钟时间。

![train_result](./images/train_result.png)
![predict_result](./images/predict_result.png)

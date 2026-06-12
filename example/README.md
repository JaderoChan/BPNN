# BPNN 示例

这是 BPNN 库的示例程序，实现了一个数字识别程序，涵盖了模型的训练与使用。

使用 MNIST 手写数字集作为训练数据与测试数据，输入层大小为 `784`，隐藏层大小为 `128`，输出层向量大小为 `10` （0-9 十个数字）。

## 数据集

数据集中的图像大小均为 `28*28` 的灰度图，使用 `uint8_t` 存储像素值。

- train-images-idx3-ubyte：跳过开头 16 个字节，之后每 784 个字节为一张图像。总共 60000 张。用于模型训练。
- train-labels-idx1-ubyte：跳过开头 8 个字节，之后每一个字节为 `train-images-idx3-ubyte` 相应位置图像的期望值（真实标签值），取值范围为 `0-9`。总共 60000 张。与 `train-images-idx3-ubyte` 共同用于模型训练。
- t10k-images-idx3-ubyte：数据结构与 `train-images-idx3-ubyte` 相同，但图像数量为 10000 张。用于模型测试。
- t10k-labels-idx1-ubyte：数据结构与 `train-labels-idx1-ubyte` 相同，但图像数量为 10000 张。与 `t10k-images-idx3-ubyte` 共同用于模型测试。

## 结果

迭代 50 轮后，最终的精确度能达到 **97%**

![image.png](./image/image.png)

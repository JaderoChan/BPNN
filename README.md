# BP 神经网络

这是一个三层全连接 BP 神经网络的实现，出于学习目的，没有依赖任何第三方库。各个部分都进行了解耦，可以进行多种切实的任务训练。

其中隐藏层激活函数使用 Sigmoid，输出层激活函数使用 Softmax，损失函数使用多分类交叉熵。

关于各参数的公式推导部分，可以参见此文档 [BP 神经网络公式推导](BP%20神经网络公式推导.md)。

## 示例

使用此库实现了一个数字识别的程序，参见 [BPNN Digit Recognizer](./example/README.md)

## 如何构建

```sh
git clone https://github.com/JaderoChan/bpnn.git
cd bpnn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBPNN_BUILD_EXAMPLE=ON
cmake --build build -j --config=Release
```

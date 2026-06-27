# BPNN Digit Recognizer

[[中文简体](README.md) | **English**]

> This document was translated by AI.

An MNIST handwritten digit recognition example based on the BPNN library, covering the complete training and inference pipeline.

## Network Architecture

Using the MNIST handwritten digit dataset (10 classes: 0–9):

| Layer | Nodes |
| --- | --- |
| Input | 784 (28×28 pixels) |
| Hidden | 128 |
| Output | 10 |

## Dataset

All images are 28×28 grayscale, pixel values stored as `uint8_t`, files in IDX binary format.

| File | Description | Count |
| --- | --- | --- |
| `train-images-idx3-ubyte` | Training images (skip first 16 bytes, then 784 bytes per image) | 60,000 |
| `train-labels-idx1-ubyte` | Training labels (skip first 8 bytes, then 1 byte per label, value 0–9) | 60,000 |
| `t10k-images-idx3-ubyte` | Test images (same format as training set) | 10,000 |
| `t10k-labels-idx1-ubyte` | Test labels (same format as training set) | 10,000 |

Extract the archives under `train/data_set/` before use.

## Use

There is a digit recognizer project implemented based on **Qt** in the `./use` directory. `./use/bpnn_net_params/net_params.np` is the result of 50 rounds of training and can be used by the digit recognizer.

## Build

By default only the `train` project will be built

```sh
cmake -B build
cmake --build build
```

If you need to build a `use` project (number recognizer based on **Qt**), you need to ensure that **Qt** (can be used by Qt5/Qt6) is visible in the build environment

```sh
cmake -B build -DBUILD_DIGIT_RECOGNIZER_USE=ON
cmke --build build
```

## Results

After 50 training epochs, the model achieves **98%** accuracy on the test set, the train takes about five minutes.

![train_result](./images/train_result.png)
![use_result](./images/use_result.png)

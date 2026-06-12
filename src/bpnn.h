// MIT License
//
// Copyright (c) 2026 頔珞 JaderoChan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Contact email: c_dl_cn@outlook.com
// Repository URL: https://github.com/JaderoChan/bpnn

/**
 * @file bpnn.h
 * @brief 手撕 BP 神经网络。
 * @details
 * 使用 Sigmoid 函数作为隐藏层激活函数；
 * 使用 Softmax 函数作为输出层激活函数；
 * 使用 Softmax 对应的多分类交叉熵函数作为损失函数。
 * @author 頔珞 JaderoChan
 * @version 0.1.0
 */

#ifndef BPNN_H
#define BPNN_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ======================
// > BP 神经网络相关数据结构
// ======================

/** @brief BP 神经网络参数 */
typedef struct bpnn_params_t
{
    uint32_t in_num;    /**< 输入层节点数量 */
    uint32_t hide_num;  /**< 隐藏层节点数量 */
    uint32_t out_num;   /**< 输出层节点数量 */

    double* in_hide_weights;    /**< 输入层-隐藏层权重矩阵 */
    double* hide_out_weights;   /**< 隐藏层-输出层权重矩阵 */
    double* hide_biases;        /**< 隐藏层偏置值向量 */
    double* out_biases;         /**< 输出层偏置值向量 */
} bpnn_params_t;

#define BPNN_PARAMS_INIT { \
    .in_num = 0, .hide_num = 0, .out_num = 0, \
    .in_hide_weights = NULL, .hide_out_weights = NULL, \
    .hide_biases = NULL, .out_biases = NULL \
}

/** @brief 根据给定参数构造 \ref bpnn_params_t，权重矩阵与偏置值向量初始化为 0 */
bool bpnn_params_construct_v1(
    bpnn_params_t* params, uint32_t in_num, uint32_t hide_num, uint32_t out_num);

/** @brief 根据给定参数构造 \ref bpnn_params_t */
bool bpnn_params_construct_v2(
    bpnn_params_t* params, uint32_t in_num, uint32_t hide_num, uint32_t out_num,
    const double* in_hide_weights, const double* hide_out_weights,
    const double* hide_biases, const double* out_biases);

/** @brief 释放指定 \ref bpnn_params_t 对象的所有成员 */
void bpnn_params_destroy(bpnn_params_t* params);

/** @brief 指定的 \ref bpnn_params_t 对象是否包含有效数据 */
bool bpnn_params_valid(const bpnn_params_t* params);

/** @brief 随机化指定 \ref bpnn_params_t 对象的权重参数至 Xavier 区间 */
void bpnn_params_randomize(bpnn_params_t* params);

bool bpnn_params_load(bpnn_params_t* params, FILE* file);

bool bpnn_params_load_from_file(bpnn_params_t* params, const char* filepath);

bool bpnn_params_save(const bpnn_params_t* params, FILE* file);

bool bpnn_params_save_to_file(const bpnn_params_t* params, const char* filepath);

/** @brief BP 神经网络，保存网络各节点数据，用于记录前向传播与逆向传播的中间状态 */
typedef struct bpnnet_t
{
    /** 是否仅用于模型使用（仅前向传播）。若为真，则 labels、hide_ds 和 out_ds 为空指针且学习率无意义。 */
    bool only_for_use;

    const bpnn_params_t* params;    /**< 神经网络参数，无所有权 */
    const double* ins;              /**< 输入向量，无所有权 */
    const double* labels;           /**< 真实标签向量，无所有权 */

    double* unact_hides;    /**< 激活前的隐藏层向量 */
    double* unact_outs;     /**< 激活前的输出层向量 */
    double* hides;          /**< 激活后的隐藏层向量 */
    double* outs;           /**< 激活后的输出层向量 */
    double* hide_ds;        /**< 损失函数对激活前隐藏层的偏导向量 */
    double* out_ds;         /**< 损失函数对激活前输出层的偏导向量 */

    double  learn_rate;     /**< 学习率 */
} bpnnet_t;

#define BPNNET_INIT { \
    .params = NULL, .ins = NULL, .labels = NULL, \
    .unact_hides = NULL, .unact_outs = NULL, \
    .hides = NULL,       .outs = NULL, \
    .hide_ds = NULL,     .out_ds = NULL, \
    .learn_rate = 0.0, .only_for_use = false \
}

bool bpnnet_construct_for_train(
    bpnnet_t* net, bpnn_params_t* params,
    const double* ins /**< Can be NULL */, const double* labels /**< Can be NULL */,
    double learn_rate);

bool bpnnet_construct_for_use(
    bpnnet_t* net, const bpnn_params_t* params, const double* ins /**< Can be NULL */);

/** @brief 释放指定 \ref bpnnet_t 对象具备所有权的成员 */
void bpnnet_destroy(bpnnet_t* net);

/** @brief 指定的 \ref bpnnet_t 对象是否包含有效数据 */
bool bpnnet_valid(const bpnnet_t* net);

// =========
// > 工具函数
// =========

/**
 * @brief Sigmoid 函数，值域 (0, 1)
 *
 * $$
 * f(x)=\frac{1}{1+e^{-x}}
 * $$
 */
static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/**
 *
 * @brief Sigmoid 导函数
 *
 * $$
 * f(x)=\frac{e^{-x}}{(1+e^{-x})^2}
 * $$
 */
static inline double sigmoid_deriv(double x)
{
    const double t = exp(-x);
    return t / ((1.0 + t) * (1.0 + t));
}

/**
 * @brief ReLU 函数
 *
 * $$
 * f(x)=\max(0,x)
 * $$
 */
static inline double relu(double x)
{
    return (x < 0.0 ? 0.0 : x);
}

/**
 * @brief Leaky ReLU
 *
 * $$
 * f(x)=\begin{cases}x&x\ge0\\\alpha x&x<0\end{cases},\alpha\ll1\qquad(a=1^{-12})
 * $$
 */
static inline double leaky_relu(double x)
{
    double ret = x >= 0 ? x : (1e-9 * x);
    return ret;
}

/** @brief Softmax 对应的多分类交叉熵损失函数
 *
 * $$
 * L=-\sum_{k=1}^{r}y_k\cdot\log(\hat{y}_k)
 * $$
 */
double loss(const bpnnet_t* net);

// =========
// > 前向传播
// =========

/** @brief 计算隐藏层节点激活前的值 */
void bpnnet_comp_unact_hides(bpnnet_t* net);

/** @brief 计算隐藏层节点激活后的值 */
void bpnnet_comp_hides(bpnnet_t* net);

/** @brief 计算输出层节点激活前的值 */
void bpnnet_comp_unact_outs(bpnnet_t* net);

/** @brief 计算输出层节点激活后的值 */
void bpnnet_comp_outs(bpnnet_t* net);

/** @brief 上面四个函数的组合 */
void bpnnet_forward_propagation(bpnnet_t* net);

// =========
// > 逆向传播
// =========

/** @brief 计算损失函数对输出层节点激活前的值的偏导 */
void bpnnet_comp_out_ds(bpnnet_t* net);

/** @brief 计算损失函数对隐藏层节点激活前的值的偏导 */
void bpnnet_comp_hide_ds(bpnnet_t* net);

/** @brief 更新 BP 神经网络参数 */
void bpnnet_update_params(bpnnet_t* net);

/** @brief 上面三个函数的组合 */
void bpnnet_back_propagation(bpnnet_t* net);

// ===========
// > 训练与使用
// ===========

/**
 * @brief 训练回调函数类型
 * @param epoch       当前 epoch 编号（从 1 开始）
 * @param total_epoch 总 epoch 数
 * @param curr_loss   本 epoch 的总损失值
 * @param delta_loss  本 epoch 与上一 epoch 的损失差值（第一个 epoch 为 NAN）
 * @param net         本 epoch 的神经网络状态
 * @param userdata    调用方传入的自定义指针
 */
typedef void (*bpnn_train_callback_t)(
    uint32_t epoch, uint32_t total_epoch,
    double curr_loss, double delta_loss,
    bpnnet_t* net, void* userdata);

void bpnn_train(
    bpnn_params_t* params      /**< [in, out] */,
    const double* ins_group    /**< [in] 多组输入向量 */,
    const double* labels_group /**< [in] 多组真实标签向量 */,
    uint32_t group_num, double learn_rate, uint32_t epoch, double esp,
    bpnn_train_callback_t callback /**< [in] 每个 epoch 结束后调用，可为 NULL */,
    void* userdata                 /**< [in] 传递给回调的自定义指针，可为 NULL */);

void bpnn_use(const bpnn_params_t* params, const double* ins, double* outs);

#endif // !BPNN_H

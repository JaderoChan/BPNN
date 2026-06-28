#ifndef BPNN_EXAMPLE_MNIST_LOADER_H
#define BPNN_EXAMPLE_MNIST_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/** @brief 从图像数据集中读取输入层样本数据 */
bool load_ins_samples_from_image_set(
    double* ins_samples /**< [out] */, FILE* file, /**< [in] */
    size_t skip_bytes, size_t bytes_per_img, size_t sample_num);

/** @brief 从文件中读取输入层样本数据 */
bool load_ins_samples_from_file(
    double* ins_samples, const char* filepath,
    size_t skip_bytes, size_t bytes_per_img, size_t sample_num);

/** @brief 从真实标签数据集中读取真实标签数据 */
bool load_labels_samples_from_label_set(
    double* labels_samples /**< [out] */, FILE* file /**< [in] */,
    size_t skip_bytes, size_t sample_num);

/** @brief 从文件中读取真实标签数据 */
bool load_labels_samples_from_file(
    double* labels_samples, const char* filepath,
    size_t skip_bytes, size_t sample_num);

#endif // !BPNN_EXAMPLE_MNIST_LOADER_H

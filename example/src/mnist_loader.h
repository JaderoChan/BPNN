#ifndef BPNN_EXAMPLE_MNIST_LOADER_H
#define BPNN_EXAMPLE_MNIST_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

bool load_ins_group_from_image_set(
    double* ins_group /**< [out] */, FILE* file, /**< [in] */
    size_t skip_bytes, size_t bytes_per_img, size_t group_num);

bool load_ins_group_from_file(
    double* ins_group, const char* filepath,
    size_t skip_bytes, size_t bytes_per_img, size_t group_num);

bool load_labels_group_from_label_set(
    double* labels_group /**< [out] */, FILE* file /**< [in] */,
    size_t skip_bytes, size_t group_num);

bool load_labels_group_from_file(
    double* labels_group, const char* filepath,
    size_t skip_bytes, size_t group_num);

#endif // !BPNN_EXAMPLE_MNIST_LOADER_H

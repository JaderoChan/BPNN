#ifndef BPNN_EXAMPLE_MNIST_LOADER_H
#define BPNN_EXAMPLE_MNIST_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

bool load_ins_samples_from_image_set(
    double* ins_samples /**< [out] */, FILE* file, /**< [in] */
    size_t skip_bytes, size_t bytes_per_img, size_t sample_num);

bool load_ins_samples_from_file(
    double* ins_samples, const char* filepath,
    size_t skip_bytes, size_t bytes_per_img, size_t sample_num);

bool load_labels_samples_from_label_set(
    double* labels_samples /**< [out] */, FILE* file /**< [in] */,
    size_t skip_bytes, size_t sample_num);

bool load_labels_samples_from_file(
    double* labels_samples, const char* filepath,
    size_t skip_bytes, size_t sample_num);

#endif // !BPNN_EXAMPLE_MNIST_LOADER_H

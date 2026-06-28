#include "mnist_loader.h"

#include <stdlib.h>
#include <string.h>

#include <config.h>

bool load_ins_samples_from_image_set(
    double* ins_samples, FILE* file,
    size_t skip_bytes, size_t bytes_per_img, size_t sample_num)
{
    if (!ins_samples || !file || ferror(file) )
        return false;

    // 跳过开头忽略的字节
    for (size_t i = 0; i < skip_bytes; ++i)
    {
        if (fgetc(file) == EOF)
            return false;
    }

    // 预分配内存
    const size_t ins_samples_bytes = sample_num * bytes_per_img * sizeof(double);
    double* data = (double*) malloc(ins_samples_bytes);
    if (!data) return false;

    // 读取数据
    for (size_t n = 0; n < sample_num; ++n)
    {
        for (size_t i = 0; i < bytes_per_img; ++i)
        {
            const int c = fgetc(file);
            if (c == EOF)
            {
                free(data);
                return false;
            }

            const size_t idx = n * bytes_per_img + i;
            data[idx]        = (double) c / 255.0;
        }
    }

    memcpy(ins_samples, data, ins_samples_bytes);
    free(data);

    return true;
}

bool load_ins_samples_from_file(
    double* ins_samples, const char* filepath,
    size_t skip_bytes, size_t bytes_per_img, size_t sample_num)
{
    if (!ins_samples || !filepath)
        return false;

    FILE* file = fopen(filepath, "rb");
    if (!file)
        return false;

    const bool ok = load_ins_samples_from_image_set(
        ins_samples, file, skip_bytes, bytes_per_img, sample_num);
    fclose(file);
    return ok;
}

bool load_labels_samples_from_label_set(
    double* labels_samples, FILE* file,
    size_t skip_bytes, size_t sample_num)
{
    if (!labels_samples || !file || ferror(file) )
        return false;

    // 跳过开头忽略的字节
    for (size_t i = 0; i < skip_bytes; ++i)
    {
        if (fgetc(file) == EOF)
            return false;
    }

    // 预分配内存
    const size_t labels_num = OUTPUT_LAYER_SIZE; // 0-9 digits
    const size_t labels_samples_bytes = sample_num * labels_num * sizeof(double);
    double* data = (double*) malloc(labels_samples_bytes);
    if (!data) return false;
    memset(data, 0, labels_samples_bytes);

    // 读取数据
    for (size_t n = 0; n < sample_num; ++n)
    {
        const int i = fgetc(file);
        if (i == EOF || i < 0 || i > 9)
        {
            free(data);
            return false;
        }

        const size_t idx = (n * OUTPUT_LAYER_SIZE) + i;
        data[idx]        = 1.0;
    }

    memcpy(labels_samples, data, labels_samples_bytes);
    free(data);

    return true;
}

bool load_labels_samples_from_file(
    double* labels_samples, const char* filepath,
    size_t skip_bytes, size_t sample_num)
{
    if (!labels_samples || !filepath)
        return false;

    FILE* file = fopen(filepath, "rb");
    if (!file)
        return false;

    const bool ok = load_labels_samples_from_label_set(
        labels_samples, file, skip_bytes, sample_num);
    fclose(file);
    return ok;
}

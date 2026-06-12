#include "mnist_loader.h"

#include <stdlib.h>
#include <string.h>

#include "config.h"

bool load_ins_group_from_image_set(
    double* ins_group, FILE* file,
    size_t skip_bytes, size_t bytes_per_img, size_t group_num)
{
    if (!ins_group || !file || ferror(file) )
        return false;

    for (size_t i = 0; i < skip_bytes; ++i)
    {
        if (fgetc(file) == EOF)
            return false;
    }

    const size_t ins_group_bytes = group_num * bytes_per_img * sizeof(double);
    double* data = (double*) malloc(ins_group_bytes);
    if (!data) return false;

    for (size_t n = 0; n < group_num; ++n)
    {
        for (size_t i = 0; i < bytes_per_img; ++i)
        {
            const int c = fgetc(file);
            if (c == EOF)
            {
                free((void*) data);
                return false;
            }

            const size_t idx = n * bytes_per_img + i;
            data[idx]        = (double) c;
        }
    }

    memcpy(ins_group, data, ins_group_bytes);
    free((void*) data);

    return true;
}

bool load_ins_group_from_file(
    double* ins_group, const char* filepath,
    size_t skip_bytes, size_t bytes_per_img, size_t group_num)
{
    if (!ins_group || !filepath)
        return false;

    FILE* file = fopen(filepath, "rb");
    if (!file)
        return false;

    const bool ok = load_ins_group_from_image_set(
        ins_group, file, skip_bytes, bytes_per_img, group_num);
    fclose(file);
    return ok;
}

bool load_labels_group_from_label_set(
    double* labels_group, FILE* file,
    size_t skip_bytes, size_t group_num)
{
    if (!labels_group || !file || ferror(file) )
        return false;

    for (size_t i = 0; i < skip_bytes; ++i)
    {
        if (fgetc(file) == EOF)
            return false;
    }

    const size_t labels_num = OUTPUT_LAYER_SIZE; // 0-9 digits
    const size_t labels_group_bytes = group_num * labels_num * sizeof(double);
    double* data = (double*) malloc(labels_group_bytes);
    if (!data) return false;
    memset(data, 0, labels_group_bytes);

    for (size_t n = 0; n < group_num; ++n)
    {
        const int i = fgetc(file);
        if (i == EOF || i < 0 || i > 9)
        {
            free((void*) data);
            return false;
        }

        const size_t idx = (n * OUTPUT_LAYER_SIZE) + i;
        data[idx]        = 1.0;
    }

    memcpy(labels_group, data, labels_group_bytes);
    free((void*) data);

    return true;
}

bool load_labels_group_from_file(
    double* labels_group, const char* filepath,
     size_t skip_bytes, size_t group_num)
{
    if (!labels_group || !filepath)
        return false;

    FILE* file = fopen(filepath, "rb");
    if (!file)
        return false;

    const bool ok = load_labels_group_from_label_set(
        labels_group, filepath, skip_bytes, group_num);
    fclose(file);
    return ok;
}

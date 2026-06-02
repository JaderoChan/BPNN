#include "bpnn.h"

#include <assert.h> // assert
#include <stdlib.h> // malloc, free
#include <time.h>   // time

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

bool bpnn_params_construct(
    bpnn_params_t* params, uint32_t in_num, uint32_t hide_num, uint32_t out_num,
    const double* in_hide_weights, const double* hide_out_weights,
    const double* hide_biases, const double* out_biases)
{
    if (!params || in_num == 0 || hide_num == 0 || out_num == 0 ||
        !in_hide_weights || !hide_out_weights || !hide_biases || !out_biases)
        return false;

    // ws1 : in_hide_weights
    // ws2 : hide_out_weights
    // bs1 : hide_biases
    // bs2 : out_biases

    const size_t ws1_bytes = (size_t) in_num   * (size_t) hide_num * sizeof(double);
    const size_t ws2_bytes = (size_t) hide_num * (size_t) out_num  * sizeof(double);
    const size_t bs1_bytes = (size_t) hide_num * sizeof(double);
    const size_t bs2_bytes = (size_t) out_num  * sizeof(double);

    double* new_ws1 = (double*) malloc(ws1_bytes);
    double* new_ws2 = (double*) malloc(ws2_bytes);
    double* new_bs1 = (double*) malloc(bs1_bytes);
    double* new_bs2 = (double*) malloc(bs2_bytes);

    if (!new_ws1 || !new_ws2 || !new_bs1 || !new_bs2)
    {
        if (new_ws1) free(new_ws1);
        if (new_ws2) free(new_ws2);
        if (new_bs1) free(new_bs1);
        if (new_bs2) free(new_bs2);
        return false;
    }

    memcpy(new_ws1, in_hide_weights,  ws1_bytes);
    memcpy(new_ws2, hide_out_weights, ws2_bytes);
    memcpy(new_bs1, hide_biases,      bs1_bytes);
    memcpy(new_bs2, out_biases,       bs2_bytes);

    params->in_num           = in_num;
    params->hide_num         = hide_num;
    params->out_num          = out_num;
    params->in_hide_weights  = new_ws1;
    params->hide_out_weights = new_ws2;
    params->hide_biases      = new_bs1;
    params->out_biases       = new_bs2;

    return true;
}

void bpnn_params_destroy(bpnn_params_t* params)
{
    if (params)
    {
        if (params->in_hide_weights)    free(params->in_hide_weights);
        if (params->hide_out_weights)   free(params->hide_out_weights);
        if (params->hide_biases)        free(params->hide_biases);
        if (params->out_biases)         free(params->out_biases);

        params->in_num = params->hide_num = params->out_num = 0;
        params->in_hide_weights  = NULL;
        params->hide_out_weights = NULL;
        params->hide_biases      = NULL;
        params->out_biases       = NULL;
    }
}

bool bpnn_params_valid(const bpnn_params_t* params)
{
    return (
        params && params->in_num != 0 && params->hide_num != 0 && params->out_num != 0 &&
        params->in_hide_weights && params->hide_out_weights &&
        params->hide_biases && params->out_biases
    );
}

void bpnn_params_randomize(bpnn_params_t* params)
{
    if (!bpnn_params_valid(params))
        return;

    static bool seeded = false;
    if (!seeded)
    {
        srand((unsigned int) time(NULL));
        seeded = true;
    }

    const uint32_t in_hide_weights_num  = params->in_num   * params->hide_num;
    const uint32_t hide_out_weights_num = params->hide_num * params->out_num;

    // 使用 Xavier 自适应缩放区间进行随机化，以加快收敛速度。
    // 对于任意权重值，假设其输入向量大小为 ins，输出向量大小为 outs，随机化区间为 [-a, a]，
    // 根据 Xavier 可得 $a = \sqrt{\frac{6}{ins, outs}}$。

    const double a1 = sqrt(6.0 / ((double) params->in_num   + (double) params->hide_num));
    const double a2 = sqrt(6.0 / ((double) params->hide_num + (double) params->out_num));

    for (uint32_t i = 0; i < in_hide_weights_num; ++i)
    {
        const double p = (double) rand() / (double) RAND_MAX;   // p \in [0.0, 1.0]
        params->in_hide_weights[i] = (2.0 * p - 1.0) * a1;      // map to [-a, a]
    }

    for (uint32_t i = 0; i < hide_out_weights_num; ++i)
    {
        const double p = (double) rand() / (double) RAND_MAX;
        params->hide_out_weights[i] = (2.0 * p - 1.0) * a2;
    }
}

bool bpnnet_construct_for_train(
    bpnnet_t* net, bpnn_params_t* params, const double* ins, const double* labels,
    double learn_rate)
{
    if (!net || !params || learn_rate == 0.0 || !bpnn_params_valid(params))
        return false;

    const size_t hide_bytes = (size_t) params->hide_num * sizeof(double);
    const size_t out_bytes  = (size_t) params->out_num  * sizeof(double);

    double* unact_hides = (double*) malloc(hide_bytes);
    double* unact_outs  = (double*) malloc(out_bytes);
    double* hides       = (double*) malloc(hide_bytes);
    double* outs        = (double*) malloc(out_bytes);
    double* hide_ds     = (double*) malloc(hide_bytes);
    double* out_ds      = (double*) malloc(out_bytes);

    if (!unact_hides || !unact_outs || !hides || !outs || !hide_ds || !out_ds)
    {
        if (unact_hides)    free(unact_hides);
        if (unact_outs)     free(unact_outs);
        if (hides)          free(hides);
        if (outs)           free(outs);
        if (hide_ds)        free(hide_ds);
        if (out_ds)         free(out_ds);
        return false;
    }

    net->only_for_use = false;
    net->params       = params;
    net->ins          = ins;
    net->labels       = labels;
    net->learn_rate   = learn_rate;

    net->unact_hides  = unact_hides;
    net->unact_outs   = unact_outs;
    net->hides        = hides;
    net->outs         = outs;
    net->hide_ds      = hide_ds;
    net->out_ds       = out_ds;

    return true;
}

bool bpnnet_construct_for_use(bpnnet_t* net, const bpnn_params_t* params, const double* ins)
{
    if (!net || !params || !bpnn_params_valid(params))
        return false;

    const size_t hide_bytes = (size_t) params->hide_num * sizeof(double);
    const size_t out_bytes  = (size_t) params->out_num  * sizeof(double);

    double* unact_hides = (double*) malloc(hide_bytes);
    double* unact_outs  = (double*) malloc(out_bytes);
    double* hides       = (double*) malloc(hide_bytes);
    double* outs        = (double*) malloc(out_bytes);

    if (!unact_hides || !unact_outs || !hides || !outs)
    {
        if (unact_hides)    free(unact_hides);
        if (unact_outs)     free(unact_outs);
        if (hides)          free(hides);
        if (outs)           free(outs);
        return false;
    }

    net->only_for_use = true;
    net->params       = params;
    net->ins          = ins;
    net->labels       = NULL;
    net->learn_rate   = 0.0;

    net->unact_hides  = unact_hides;
    net->unact_outs   = unact_outs;
    net->hides        = hides;
    net->outs         = outs;
    net->hide_ds      = NULL;
    net->out_ds       = NULL;

    return true;
}

void bpnnet_destroy(bpnnet_t* net)
{
    if (net)
    {
        if (net->unact_hides)   free(net->unact_hides);
        if (net->unact_outs)    free(net->unact_outs);
        if (net->hides)         free(net->hides);
        if (net->outs)          free(net->outs);
        if (net->hide_ds)       free(net->hide_ds);
        if (net->out_ds)        free(net->out_ds);

        net->params = NULL;
        net->ins = net->labels = NULL;
        net->unact_hides = net->unact_outs   = NULL;
        net->hides       = net->outs         = NULL;
        net->hide_ds     = net->out_ds       = NULL;
        net->learn_rate  = 0.0;
    }
}

bool bpnnet_valid(const bpnnet_t* net)
{
    if (net)
    {
        bool ok = (
            net->only_for_use ?
            true :
            (net->labels && net->hide_ds && net->out_ds && net->learn_rate != 0.0));
        return (
            ok && net->params && net->ins &&
            net->unact_hides && net->unact_outs &&
            net->hides       && net->outs &&
            bpnn_params_valid(net->params)
        );
    }
    return false;
}

double loss(const bpnnet_t* net)
{
    assert(bpnnet_valid(net));

    const double tiny = 1e-12;
    double sum = 0.0;
    for (uint32_t i = 0; i < net->params->out_num; ++i)
    {
        // log() 函数在实参值极小的情况下可能产生极端值/无效值，通过 MAX 进行钳位。
        const double out = MAX(net->outs[i], tiny);
        sum += net->labels[i] * log(out);
    }
    return -sum;
}

void bpnnet_comp_unact_hides(bpnnet_t* net)
{
    assert(bpnnet_valid(net));

    const uint32_t n = net->params->in_num;
    const uint32_t m = net->params->hide_num;
    for (uint32_t j = 0; j < m; ++j)
    {
        net->unact_hides[j] = 0.0;
        for (uint32_t i = 0; i < n; ++i)
        {
            const size_t w_idx = i * m + j;
            net->unact_hides[j] += net->ins[i] * net->params->in_hide_weights[w_idx];
        }
        net->unact_hides[j] += net->params->hide_biases[j];
    }
}

void bpnnet_comp_hides(bpnnet_t* net)
{
    assert(bpnnet_valid(net));

    for (uint32_t i = 0; i < net->params->hide_num; ++i)
        net->hides[i] = sigmoid(net->unact_hides[i]);
}

void bpnnet_comp_unact_outs(bpnnet_t* net)
{
    assert(bpnnet_valid(net));

    const uint32_t n = net->params->hide_num;
    const uint32_t m = net->params->out_num;
    for (uint32_t j = 0; j < m; ++j)
    {
        net->unact_outs[j] = 0.0;
        for (uint32_t i = 0; i < n; ++i)
        {
            const size_t w_idx = i * m + j;
            net->unact_outs[j] += net->hides[i] * net->params->hide_out_weights[w_idx];
        }
        net->unact_outs[j] += net->params->out_biases[j];
    }
}

void bpnnet_comp_outs(bpnnet_t* net)
{
    assert(bpnnet_valid(net));

    const uint32_t n = net->params->out_num;
    double max_unact_out = net->unact_outs[0];
    for (uint32_t i = 0; i < n; ++i)
        if (net->unact_outs[i] > max_unact_out) max_unact_out = net->unact_outs[i];

    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        sum += exp(net->unact_outs[i] - max_unact_out);
    for (uint32_t i = 0; i < n; ++i)
        net->outs[i] = exp(net->unact_outs[i] - max_unact_out) / sum;
}

void bpnnet_forward_propagation(bpnnet_t* net)
{
    bpnnet_comp_unact_hides(net);
    bpnnet_comp_hides(net);
    bpnnet_comp_unact_outs(net);
    bpnnet_comp_outs(net);
}

void bpnnet_comp_out_ds(bpnnet_t* net)
{
    assert(bpnnet_valid(net) && !net->only_for_use);

    for (uint32_t i = 0; i < net->params->out_num; ++i)
        net->out_ds[i] = net->outs[i] - net->labels[i];
}

void bpnnet_comp_hide_ds(bpnnet_t* net)
{
    assert(bpnnet_valid(net) && !net->only_for_use);

    const uint32_t n = net->params->hide_num;
    const uint32_t m = net->params->out_num;
    for (uint32_t i = 0; i < n; ++i)
    {
        net->hide_ds[i] = 0.0;
        for (uint32_t j = 0; j < m; ++j)
        {
            const size_t w_idx = i * m + j;
            net->hide_ds[i] += net->out_ds[j] * net->params->hide_out_weights[w_idx];
        }
        net->hide_ds[i] *= sigmoid_deriv(net->unact_hides[i]);
    }
}

void bpnnet_update_params(bpnnet_t* net)
{
    assert(bpnnet_valid(net) && !net->only_for_use);

    bpnn_params_t* params = (bpnn_params_t*) net->params;
    const uint32_t n = params->in_num;
    const uint32_t m = params->hide_num;
    const uint32_t r = params->out_num;

    // 更新输入-隐藏层权重
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t j = 0; j < m; ++j)
        {
            const size_t w_idx = i * m + j;
            params->in_hide_weights[w_idx] -= net->learn_rate * net->hide_ds[j] * net->ins[i];
        }
    }

    // 更新隐藏层-输出层权重
    for (uint32_t j = 0; j < m; ++j)
    {
        for (uint32_t k = 0; k < r; ++k)
        {
            const size_t w_idx = j * r + k;
            params->hide_out_weights[w_idx] -=net->learn_rate * net->out_ds[k] * net->hides[j];
        }
    }

    // 更新隐藏层偏置值
    for (uint32_t j = 0; j < m; ++j)
        params->hide_biases[j] -= net->learn_rate * net->hide_ds[j];

    // 更新输出层偏置值
    for (uint32_t k = 0; k < r; ++k)
        params->out_biases[k] -= net->learn_rate * net->out_ds[k];
}

void bpnnet_back_propagation(bpnnet_t* net)
{
    bpnnet_comp_out_ds(net);
    bpnnet_comp_hide_ds(net);
    bpnnet_update_params(net);
}

void bpnn_train(
    bpnn_params_t* params, const double* ins_group, const double* labels_group,
    uint32_t group_num, double learn_rate, uint32_t epoch, double esp)
{
    if (!params || !ins_group || !labels_group ||
        group_num == 0 || learn_rate == 0.0 ||
        !bpnn_params_valid(params))
        return;

    bpnnet_t net = BPNNET_INIT;
    if (!bpnnet_construct_for_train(&net, params, NULL, NULL, learn_rate))
        return;

    double last_loss = NAN;
    for (uint32_t i = 0; i < epoch; ++i)
    {
        for (uint32_t j = 0; j < group_num; ++j)
        {
            // 更新此次迭代的输入向量与真实标签向量。
            net.ins    = &ins_group[j * params->in_num];
            net.labels = &labels_group[j * params->out_num];
            // 进行正向传播与逆向传播。
            bpnnet_forward_propagation(&net);
            bpnnet_back_propagation(&net);
            // 判断是否达到预期收敛值。
            double curr_loss = loss(&net);
            if (!isnan(last_loss) && fabs(curr_loss - last_loss) < esp)
            {
                bpnnet_destroy(&net);
                return;
            }
            last_loss = curr_loss;
        }
    }

    bpnnet_destroy(&net);
}

void bpnn_use(const bpnn_params_t* params, const double* ins, double* outs)
{
    if (!params || !ins || !outs || !bpnn_params_valid(params))
        return;

    bpnnet_t net = BPNNET_INIT;
    if (!bpnnet_construct_for_use(&net, params, ins))
        return;

    bpnnet_forward_propagation(&net);
    memcpy(outs, net.outs, (size_t) params->out_num * sizeof(double));

    bpnnet_destroy(&net);
}

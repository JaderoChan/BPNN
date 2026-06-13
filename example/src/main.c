#include <math.h>
#include <stdlib.h>

#include <bpnn.h>
#include "config.h"
#include "mnist_loader.h"

#define ERROR_EXIT(msg, ec) do { fprintf(stderr, (msg)); return (ec); } while(0)

static void train_callback(
    uint32_t epoch, uint32_t total_epoch,
    double curr_loss, double delta_loss,
    const bpnnet_t* net, bool* stop, void* userdata)
{
    (void) net;
    (void) stop;
    (void) userdata;

    // 打印训练轮数与损失值相关信息。
    if (isnan(delta_loss))
        printf("[Epoch %3u/%u] loss: %.4f\n", epoch, total_epoch, curr_loss);
    else
        printf("[Epoch %3u/%u] loss: %.4f delta: %+.4f\n",
            epoch, total_epoch, curr_loss, delta_loss);
}

int main(int argc, char* argv[])
{
    // ======
    // > 训练
    // ======

    printf(">>> Train\n");

    // 初始化 BP 神经网络参数。使用 Sigmoid 和 Softmax 作为隐藏层和输出层的激活函数。
    bpnn_params_t params = BPNN_PARAMS_INIT;
    {
        const bool ok = bpnn_params_construct_v1(
            &params, ACT_FN_SIGMOID, ACT_FN_SOFTMAX,
            INPUT_LAYER_SIZE, HIDE_LAYER_SIZE, OUTPUT_LAYER_SIZE);
        if (!ok) ERROR_EXIT("Failed to construct bpnn_params_t.\n", 1);
    }

    printf("- Randomize params\n");
    bpnn_params_randomize(&params);

    {
        // 读取训练数据
        printf("- Load train data\n");

        // 预分配内存
        const size_t ins_group_bytes    = TRAIN_GROUP_NUM * INPUT_LAYER_SIZE * sizeof(double);
        double*      ins_group          = (double*) malloc(ins_group_bytes);
        const size_t labels_group_bytes = TRAIN_GROUP_NUM * OUTPUT_LAYER_SIZE * sizeof(double);
        double*      labels_group       = (double*) malloc(labels_group_bytes);

        if (!ins_group || !labels_group)
        {
            bpnn_params_destroy(&params);
            if (ins_group)    free(ins_group);
            if (labels_group) free(labels_group);
            ERROR_EXIT("Failed to malloc ins_group and labels_group for train.\n", 1);
        }

        {
            // 读取训练用输入数据集。
            const bool ok1 = load_ins_group_from_file(
                ins_group,
                TRAIN_INPUTS_FILE,
                INPUTS_SKIP_BYTES,
                INPUT_LAYER_SIZE,
                TRAIN_GROUP_NUM);
            // 读取训练用真实标签数据集。
            const bool ok2 = load_labels_group_from_file(
                labels_group,
                TRAIN_LABELS_FILE,
                LABELS_SKIP_BYTES,
                TRAIN_GROUP_NUM);
            if (!ok1 || !ok2)
            {
                bpnn_params_destroy(&params);
                free(ins_group);
                free(labels_group);
                ERROR_EXIT("Failed to load mnist data for train.\n", 1);
            }
        }

        // 训练。使用多分类交叉熵函数作为损失函数。
        printf("- Train\n");
        bpnn_train(
            &params, ins_group, labels_group, TRAIN_GROUP_NUM,
            TRAIN_LEARN_RATE, LOSS_FN_MCE, TRAIN_EPOCH, TRAIN_ESP,
            train_callback, NULL);
    }

    // ======
    // > 验证
    // ======

    printf(">>> Verify\n");

    {
        // 读取验证数据
        printf("- Load verify data\n");

        // 预分配内存
        const size_t ins_group_bytes    = VERIFY_GROUP_NUM * INPUT_LAYER_SIZE * sizeof(double);
        double*      ins_group          = (double*) malloc(ins_group_bytes);
        const size_t labels_group_bytes = VERIFY_GROUP_NUM * OUTPUT_LAYER_SIZE * sizeof(double);
        double*      labels_group       = (double*) malloc(labels_group_bytes);

        if (!ins_group || !labels_group)
        {
            bpnn_params_destroy(&params);
            if (ins_group)    free(ins_group);
            if (labels_group) free(labels_group);
            ERROR_EXIT("Failed to malloc ins_group and labels_group for verify.\n", 1);
        }

        {
            // 读取验证/测试用输入数据集
            const bool ok1 = load_ins_group_from_file(
                ins_group,
                VERIFY_INPUTS_FILE,
                INPUTS_SKIP_BYTES,
                INPUT_LAYER_SIZE,
                VERIFY_GROUP_NUM);
            // 读取验证/测试用真实标签数据集
            const bool ok2 = load_labels_group_from_file(
                labels_group,
                VERIFY_LABELS_FILE,
                LABELS_SKIP_BYTES,
                VERIFY_GROUP_NUM);
            if (!ok1 || !ok2)
            {
                bpnn_params_destroy(&params);
                free(ins_group);
                free(labels_group);
                ERROR_EXIT("Failed to load mnist data for verify.\n", 1);
            }
        }

        // 验证
        printf("- Verify\n");

        size_t correct = 0; // 正确数量
        double outs[OUTPUT_LAYER_SIZE] = {0.0}; // 输出向量
        for (size_t i = 0; i < VERIFY_GROUP_NUM; ++i)
        {
            const double* ins    = &ins_group[i * params.in_num];
            const double* labels = &labels_group[i * params.out_num];
            bpnn_use(&params, ins, outs);

            // 读取当前输出的数字
            int num = -1;
            for (int n = 0; n < OUTPUT_LAYER_SIZE; ++n)
            {
                if (num == -1 || outs[n] > outs[num])
                    num = n;
            }

            // 读取当前真实标签数字
            int label = -1;
            for (int n = 0; n < OUTPUT_LAYER_SIZE; ++n)
            {
                if (label == -1 || labels[n] > labels[label])
                    label = n;
            }

            correct += (num == label ? 1 : 0);
        }

        // 计算并输出准确率
        printf("Precision: %lf.\n", (double) correct / (double) VERIFY_GROUP_NUM);
    }

    return 0;
}

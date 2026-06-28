#include <math.h>
#include <stdlib.h>

#include <bpnn.h>
#include <config.h>
#include "mnist_loader.h"
#include "elapsed_timer.h"

#define ERROR_EXIT(msg, ec) do { fprintf(stderr, (msg)); return (ec); } while(0)

static void train_epoch_callback(
    uint32_t epoch, uint32_t total_epoch,
    double curr_loss, double delta_loss,
    const bpnnet_t* net, bool* stop, void* userdata)
{
    (void) net;
    (void) stop;

    elapsed_timer_t* et = (elapsed_timer_t*) userdata;

    // 打印训练轮数与损失值相关信息。
    if (isnan(delta_loss))
        printf("[Epoch %3u/%u] loss: %.4f", epoch, total_epoch, curr_loss);
    else
        printf("[Epoch %3u/%u] loss: %.4f delta: %+.4f",
            epoch, total_epoch, curr_loss, delta_loss);

    uint64_t ms = elapsed_timer_elapsed_ms_reset(et);
    printf(" [%llu ms]\n", ms);
}

int main(int argc, char* argv[])
{
    elapsed_timer_t et = ELAPSED_TIMER_INIT;
    elapsed_timer_create(&et);

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

        printf("- Randomize params\n");
        bpnn_params_randomize(&params);
    }


    {
        // 读取训练数据
        printf("- Load train data\n");

        // 预分配内存
        const size_t ins_samples_bytes    = TRAIN_SAMPLE_NUM * INPUT_LAYER_SIZE  * sizeof(double);
        double*      ins_samples          = (double*) malloc(ins_samples_bytes);
        const size_t labels_samples_bytes = TRAIN_SAMPLE_NUM * OUTPUT_LAYER_SIZE * sizeof(double);
        double*      labels_samples       = (double*) malloc(labels_samples_bytes);

        if (!ins_samples || !labels_samples)
        {
            bpnn_params_destroy(&params);
            if (ins_samples)    free(ins_samples);
            if (labels_samples) free(labels_samples);
            ERROR_EXIT("Failed to malloc ins_samples and labels_samples for train.\n", 1);
        }

        {
            // 读取训练用输入数据集。
            const bool ok1 = load_ins_samples_from_file(
                ins_samples,
                TRAIN_INPUTS_FILE,
                INPUTS_FILE_SKIP_BYTES,
                INPUT_LAYER_SIZE,
                TRAIN_SAMPLE_NUM);
            // 读取训练用真实标签数据集。
            const bool ok2 = load_labels_samples_from_file(
                labels_samples,
                TRAIN_LABELS_FILE,
                LABELS_FILE_SKIP_BYTES,
                TRAIN_SAMPLE_NUM);
            if (!ok1 || !ok2)
            {
                bpnn_params_destroy(&params);
                free(ins_samples);
                free(labels_samples);
                ERROR_EXIT("Failed to load mnist data for train.\n", 1);
            }
        }

        // 模型训练（使用多分类交叉熵函数作为损失函数）
        printf("- Train\n");

        elapsed_timer_t et2 = ELAPSED_TIMER_INIT;
        elapsed_timer_create(&et2);
        elapsed_timer_reset(&et);

        const bool ok = bpnn_train(
            &params, ins_samples, labels_samples, TRAIN_SAMPLE_NUM,
            TRAIN_LEARN_RATE, LOSS_FN_CCE, TRAIN_EPOCH, TRAIN_ESP,
            NULL, NULL, train_epoch_callback, &et2);
        if (!ok)
        {
            bpnn_params_destroy(&params);
            ERROR_EXIT("Failed to train.\n", 1);
        }

        double sec = elapsed_timer_elapsed_sec(&et);
        printf("[Train elapsed: %lf sec]\n", sec);
    }

    // ======
    // > 测试
    // ======

    printf(">>> Test\n");

    {
        // 读取测试数据
        printf("- Load test data\n");

        // 预分配内存
        const size_t ins_samples_bytes    = TEST_SAMPLE_NUM * INPUT_LAYER_SIZE  * sizeof(double);
        double*      ins_samples          = (double*) malloc(ins_samples_bytes);
        const size_t labels_samples_bytes = TEST_SAMPLE_NUM * OUTPUT_LAYER_SIZE * sizeof(double);
        double*      labels_samples       = (double*) malloc(labels_samples_bytes);

        if (!ins_samples || !labels_samples)
        {
            bpnn_params_destroy(&params);
            if (ins_samples)    free(ins_samples);
            if (labels_samples) free(labels_samples);
            ERROR_EXIT("Failed to malloc ins_samples and labels_samples for test.\n", 1);
        }

        {
            // 读取测试用输入数据集
            const bool ok1 = load_ins_samples_from_file(
                ins_samples,
                TEST_INPUTS_FILE,
                INPUTS_FILE_SKIP_BYTES,
                INPUT_LAYER_SIZE,
                TEST_SAMPLE_NUM);
            // 读取测试用真实标签数据集
            const bool ok2 = load_labels_samples_from_file(
                labels_samples,
                TEST_LABELS_FILE,
                LABELS_FILE_SKIP_BYTES,
                TEST_SAMPLE_NUM);
            if (!ok1 || !ok2)
            {
                bpnn_params_destroy(&params);
                free(ins_samples);
                free(labels_samples);
                ERROR_EXIT("Failed to load mnist data for test.\n", 1);
            }
        }

        // 模型测试
        printf("- Test\n");
        elapsed_timer_reset(&et);

        size_t correct = 0; // 正确数量
        double outs[OUTPUT_LAYER_SIZE] = {0.0}; // 输出向量
        for (size_t i = 0; i < TEST_SAMPLE_NUM; ++i)
        {
            const double* ins    = &ins_samples[i * params.in_num];
            const double* labels = &labels_samples[i * params.out_num];
            const bool ok = bpnn_predict(&params, ins, outs);
            if (!ok)
            {
                bpnn_params_destroy(&params);
                ERROR_EXIT("Failed to predict.\n", 1);
            }

            // 解析当前输出的数字
            int num = -1;
            for (int n = 0; n < OUTPUT_LAYER_SIZE; ++n)
            {
                if (num == -1 || outs[n] > outs[num])
                    num = n;
            }

            // 解析当前真实标签数字
            int label = -1;
            for (int n = 0; n < OUTPUT_LAYER_SIZE; ++n)
            {
                if (label == -1 || labels[n] > labels[label])
                    label = n;
            }

            correct += (num == label ? 1 : 0);
        }

        double sec = elapsed_timer_elapsed_sec(&et);
        printf("[Test elapsed: %lf sec]\n", sec);

        // 输出模型准确率
        printf("Precision: %lf.\n", (double) correct / (double) TEST_SAMPLE_NUM);
    }

    // =======
    // 保存模型
    // =======

    {
        printf("==============================\n");
        printf("DO you want to save the model? [Yes(Y)/No(N)]\n");

        bool needSave = false;
        do
        {
            char key = 0;
            if (scanf("%c", &key) != 1)
                continue;
            if (key == 'Y' || key == 'y')
            {
                needSave = true;
                break;
            }
            else if (key == 'N' || key =='n')
            {
                needSave = false;
                break;
            }
            else
            {
                printf("Please input the Y/N.\n");
            }
        } while (true);

        if (needSave)
        {
            printf("Please input the save file path:\n");
            char filepath[256] = {0};
            while (scanf("%s", filepath) != 1);
            if (bpnn_params_save_to_file(&params, filepath))
                printf("Successfull save the model to %s.\n", filepath);
            else
                printf("Failed to save the model to %s.\n", filepath);
        }
    }

    bpnn_params_destroy(&params);
    return 0;
}

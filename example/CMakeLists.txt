project(BPNN_Digit_Recognizer LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

####################
# 配置 config 头文件 #
####################

set(DATA_SET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_set)

# 训练用数据集及数量
set(TRAIN_INPUTS_FILE  ${DATA_SET_DIR}/train-images-idx3-ubyte)
set(TRAIN_LABELS_FILE  ${DATA_SET_DIR}/train-labels-idx1-ubyte)
set(TRAIN_GROUP_NUM    60000)

# 验证/测试用数据集及数量
set(VERIFY_INPUTS_FILE ${DATA_SET_DIR}/t10k-images-idx3-ubyte)
set(VERIFY_LABELS_FILE ${DATA_SET_DIR}/t10k-labels-idx1-ubyte)
set(VERIFY_GROUP_NUM   10000)

# 输入与真实标签数据集的跳过字节数（参见 README 文档）
set(INPUTS_SKIP_BYTES  16)
set(LABELS_SKIP_BYTES  8)

set(INPUT_IMAGE_SIZE   28)   # 数据集图片大小 28*28
set(HIDE_LAYER_SIZE    128)  # 隐藏层节点数量
set(TRAIN_LEARN_RATE   0.01) # 训练的学习率
set(TRAIN_EPOCH        50)   # 训练的轮数
set(TRAIN_ESP          0.5)  # 训练终止精度（当相邻轮损失值误差小于此值时退出训练）

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h
    @ONLY
)

#######
# 生成 #
#######

set(
    SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mnist_loader.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/elapsed_timer.c
)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE bpnn)

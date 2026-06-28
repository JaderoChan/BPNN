set(DATA_SET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_set)
set(MODELS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/models)

# 数据集图片大小 28*28
set(DATA_SET_IMAGE_SIZE    28)

# 训练用数据集文件路径及样本数量
set(TRAIN_INPUTS_FILE      ${DATA_SET_DIR}/train-images-idx3-ubyte)
set(TRAIN_LABELS_FILE      ${DATA_SET_DIR}/train-labels-idx1-ubyte)
set(TRAIN_SAMPLE_NUM       60000)

# 测试用数据集文件路径及样本数量
set(TEST_INPUTS_FILE       ${DATA_SET_DIR}/t10k-images-idx3-ubyte)
set(TEST_LABELS_FILE       ${DATA_SET_DIR}/t10k-labels-idx1-ubyte)
set(TEST_SAMPLE_NUM        10000)

# 输入数据集与真实标签数据集文件的跳过字节数（参见 README 文档）
set(INPUTS_FILE_SKIP_BYTES 16)
set(LABELS_FILE_SKIP_BYTES 8)

# 预定义值
set(HIDE_LAYER_SIZE        128)  # 隐藏层节点数量
set(TRAIN_LEARN_RATE       0.01) # 训练的学习率
set(TRAIN_EPOCH            50)   # 训练的轮数
set(TRAIN_ESP              0.5)  # 训练终止精度（当相邻轮损失值误差小于此值时退出训练）

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
    @ONLY
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(train)
option(BUILD_DIGIT_RECOGNIZER_GUI "Build the digit recognizer gui." ON)
if(BUILD_DIGIT_RECOGNIZER_GUI)
    add_subdirectory(digit_recognizer_gui)
endif()

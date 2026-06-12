project(BPNN_Digit_Recognizer LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

####################
# 配置 config 头文件 #
####################

set(DATA_SET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_set)

set(TRAIN_INPUTS_FILE  ${DATA_SET_DIR}/train-images-idx3-ubyte)
set(TRAIN_LABELS_FILE  ${DATA_SET_DIR}/train-labels-idx1-ubyte)
set(TRAIN_GROUP_NUM    60000)

set(VERIFY_INPUTS_FILE ${DATA_SET_DIR}/t10k-images-idx3-ubyte)
set(VERIFY_LABELS_FILE ${DATA_SET_DIR}/t10k-labels-idx1-ubyte)
set(VERIFY_GROUP_NUM   10000)

set(INPUTS_SKIP_BYTES  16)
set(LABELS_SKIP_BYTES  8)

set(INPUT_IMAGE_SIZE   28)
set(HIDE_LAYER_SIZE    128)
set(TRAIN_LEARN_RATE   0.01)
set(TRAIN_EPOCH        1000)
set(TRAIN_ESP          0.01)

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
)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE bpnn)

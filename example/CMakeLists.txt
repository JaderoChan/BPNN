project(BPNN_Digit_Recognizer LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(DATA_SET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_set)
set(TRAIN_INPUTS_FILE  ${DATA_SET_DIR}/train-images-idx3-ubyte)
set(TRAIN_LABELS_FILE  ${DATA_SET_DIR}/train-labels-idx1-ubyte)
set(VERIFY_INPUTS_FILE ${DATA_SET_DIR}/t10k-images-idx3-ubyte)
set(VERIFY_LABELS_FILE ${DATA_SET_DIR}/t10k-labels-idx1-ubyte)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h
    @ONLY
)

set(
    SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mnist_loader.c
)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE bpnn)

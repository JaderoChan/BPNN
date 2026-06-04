project(BPNN_Digit_Recognizer LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(DATA_SET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_set)
set(
    SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mnist_parse.c
)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE bpnn)

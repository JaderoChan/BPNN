cmake_minimum_required(VERSION 3.17)

project(bpnn LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(
    SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bpnn.c
)
add_library(${PROJECT_NAME} STATIC ${SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

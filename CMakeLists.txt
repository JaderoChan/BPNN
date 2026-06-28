cmake_minimum_required(VERSION 3.17)

project(bpnn VERSION 1.1.1 LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(
    SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bpnn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bpnn.c
)
add_library(${PROJECT_NAME} STATIC ${SRCS})

if(MSVC)
    add_compile_options(/utf-8)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

option(BPNN_BUILD_EXAMPLE "Build the digit recognizer example use BPNN." ON)
if(BPNN_BUILD_EXAMPLE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example/digit_recognizer)
endif()

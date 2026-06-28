project(digit_recognizer_model_train LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(
    SRCS
    mnist_loader.h
    mnist_loader.c
    elapsed_timer.h
    elapsed_timer.c
    main.c
)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE bpnn)

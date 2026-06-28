project(digit_recognizer_gui LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt5 Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(
    SRCS
    pixel_drawing_board.h
    pixel_drawing_board.cpp
    main_widget.h
    main_widget.cpp
    main.cpp
)

qt_standard_project_setup()
qt_add_executable(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt${QT_VERSION_MAJOR}::Widgets bpnn)

option(BUILD_DIGIT_RECOGNIZER_EXAMPLE "Build digit recognizer example" ON)
if(BUILD_DIGIT_RECOGNIZER_EXAMPLE)
    add_subdirectory(digit_recognizer)
endif()

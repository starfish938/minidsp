cmake_minimum_required(VERSION 3.10)
project(DSPLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(googletest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/include)


# 主库（header-only）
add_library(dsp_lib INTERFACE)
target_include_directories(dsp_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# 单元测试
add_executable(test_dsp_lib gtest.cpp)
target_link_libraries(test_dsp_lib   gtest  gmock)

enable_testing()
add_test(NAME test_dsp_lib COMMAND test_dsp_lib)


# 使用示例
add_executable(example example.cpp)
target_link_libraries(example dsp_lib)
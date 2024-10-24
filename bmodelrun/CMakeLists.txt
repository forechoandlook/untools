cmake_minimum_required(VERSION 3.10)
project(test)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
# debug
set(CMAKE_BUILD_TYPE "Debug")

# # 查找 oneDNN 库
# find_package(DNNL REQUIRED)

include_directories(/home/zwy/172.28.141.89/sg2260/image/daily_build/Master_20241008_034000/bm1690/tpuv7-runtime/PCIe/tpuv7-runtime_0.1.0_x86_64/opt/tpuv7/tpuv7-runtime_0.1.0/include)
link_directories(/home/zwy/172.28.141.89/sg2260/image/daily_build/Master_20241008_034000/bm1690/tpuv7-runtime/PCIe/tpuv7-runtime_0.1.0_x86_64/opt/tpuv7/tpuv7-runtime_0.1.0/lib)

# 添加源文件
# add_executable(convbwd main.cpp)
# 遍历所有的cc文件
# file(GLOB_RECURSE SOURCES "*.cpp")
# 生成共享库
# add_library(test SHARED ${SOURCES})

add_executable(test test.cpp)
# 链接 oneDNN 库
target_link_libraries(test tpuv7_rt tpuv7_modelrt )
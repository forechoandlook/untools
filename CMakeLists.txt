
cmake_minimum_required(VERSION 3.10)
project(test_project)

set(CMAKE_CXX_STANDARD 11)

# debug
set(CMAKE_BUILD_TYPE Debug)
# one-dnn
set(DNNL_ROOT /opt/homebrew/Cellar/onednn/3.5.3/)
include_directories(SYSTEM ${DNNL_ROOT}/include)
link_directories(${DNNL_ROOT}/lib)

# all cc files
file(GLOB_RECURSE SOURCES "*.cc")
# add_executable(test_project ${SOURCES})
# share so
add_library(test_project SHARED ${SOURCES})
target_link_libraries(test_project dnnl pthread)
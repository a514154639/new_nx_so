cmake_minimum_required(VERSION 2.6)

# 指定编译器GCC 7.5.0 aarch64-linux-gnu,如使用系统默认cc,编译时会报错floating constant is out of range
set (CMAKE_C_COMPILER "/usr/bin/gcc-7")
set (CMAKE_CXX_COMPILER "/usr/bin/g++-7")

project(yolov5_infer_lib)
set(PROJECT_NAME yolov5_infer_rtsp)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# 编译类型：debug
set(CMAKE_BUILD_TYPE Debug) 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 


find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread")
cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成库
cuda_add_library(${PROJECT_NAME} SHARED yolov5.cpp preprocess.cu)
# 生成demo
cuda_add_executable(demo demo.cpp yolov5.h)
# set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,./lib")

target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} myplugins)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_link_libraries(demo ${OpenCV_LIBS} ${PROJECT_NAME} dl)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)



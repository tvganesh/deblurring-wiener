cmake_minimum_required(VERSION 3.10)
project(deblurring)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(deblurring deblurring.cpp)
target_link_libraries(deblurring ${OpenCV_LIBS})

# minimum required version of cmake
cmake_minimum_required(VERSION 3.31)

# project name
project(main)

# set the C++ standard to C++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std==c++17 -Wall -g")

# find the OpenCV package
find_package(OpenCV REQUIRED)

# include the OpenCV headers
include_directories( ${OpenCV_INCLUDE_DIRS})

# create an executable file named Main
add_executable(Main main.cpp)

# link the OpenCV library to the executable
target_link_libraries(Main ${OpenCV_LIBS})
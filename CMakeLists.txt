# minimum required version of cmake
cmake_minimum_required(VERSION 3.31)

# project name
project(main LANGUAGES CXX)

# set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# find the OpenCV package
find_package(OpenCV REQUIRED)

# set the name of the executable file
set(SRC_FILES pixelwise/pixelwise.cpp)

# include the OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS} pixelwise)

# create an executable file named Main
add_executable(Main main.cpp ${SRC_FILES})

# link the OpenCV library to the executable
target_link_libraries(Main ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.8)
project(source)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR C:/Dev/dependencies/opencv/msvcbuild)
set(Glog_DIR C:/Program Files/glog)
set(GFlags_DIR C:/Program Files/gflags)

FIND_PACKAGE(OpenCV REQUIRED)
if( OpenCV_FOUND )
    message( "Found OpenCV" )
    message( "Includes: " ${OpenCV_INCLUDE_DIRS} )
else()
    message( "OpenCV not found, woot" )
endif( OpenCV_FOUND )

find_package (Glog REQUIRED)
find_package (GFlags REQUIRED)
find_package (GTest REQUIRED)
find_package (Benchmark REQUIRED)
#find_package (Halide REQUIRED)
message( status ${GTEST_INCLUDE_DIR} )
message( status ${GTEST_LIBRARY} )
message( status ${GTest_INCLUDE_DIR} )
include_directories(source ${GLOG_INCLUDE_DIRS} ${GTEST_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIR} ${Benchmark_INCLUDE_DIR} ${Halide_INCLUDE_DIR})
#FIND_PACKAGE(benchmark REQUIRED)
#FIND_PACKAGE(gtest REQUIRED)
#   message(status ${Halide_LIBRARY_RELEASE})

add_executable(source main.cpp foundation/logging.h foundation/testing.h)

TARGET_LINK_LIBRARIES(source ${OpenCV_LIBS} ${GLOG_LIBRARY_RELEASE} ${GTEST_MAIN_LIBRARY} ${GTEST_LIBRARY} ${GFLAGS_LIBRARY})


cmake_minimum_required (VERSION 2.6)

project(test2b)

find_package( Boost 1.53 COMPONENTS timer system REQUIRED )
find_package( CUDA REQUIRED )
include_directories( include/ ${Boost_INCLUDE_DIR} )

# Just choose one of the following, either twobodyForce to run the polynomials or twobodyForceNN to run Neural Nets
cuda_add_library(twobodyForce twobodyForce.cu)
#cuda_add_library(twobodyForce twobodyForceNN.cu)

add_executable(run_test run_test.cpp)

target_link_libraries(run_test twobodyForce ${Boost_LIBRARIES})

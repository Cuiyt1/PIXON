cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pixon LANGUAGES CXX)
add_executable(pixon src/main.cpp src/utilities.cpp)

find_path(NLOPT_INCLUDE_DIR nlopt.h)
find_library(NLOPT_LIB nlopt)
include_directories(${NLOPT_INCLUDE_DIR})

find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIB fftw3)
include_directories(${FFTW3_INCLUDE_DIR})

target_link_libraries(pixon ${NLOPT_LIB} ${FFTW3_LIB})
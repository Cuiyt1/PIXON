cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pixon LANGUAGES CXX C)

set(SRC "src")

add_executable(pixon ${SRC}/main.cpp)

add_subdirectory(${SRC})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_path(NLOPT_INCLUDE_DIR nlopt.h)
find_library(NLOPT_LIB nlopt)
include_directories(${NLOPT_INCLUDE_DIR})

find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIB fftw3)
include_directories(${FFTW3_INCLUDE_DIR})

find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h PATHS /usr/include/lapacke)
find_library(LAPACKE_LIB lapacke)
include_directories(${LAPACKE_INCLUDE_DIR})

find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS /usr/include/cblas)
find_library(CBLAS_LIB cblas)
include_directories(${CBLAS_INCLUDE_DIR})

find_path(CDNEST_INCLUDE_DIR NAMES dnestvars.h PATHS /home/liyropt/Projects/GIT/CDNestSingle)
find_library(CDNEST_LIB NAMES dnest PATHS /home/liyropt/Projects/GIT/CDNestSingle)
include_directories(${CDNEST_INCLUDE_DIR})

target_link_libraries(pixon utilities cont_model run test ${NLOPT_LIB} ${FFTW3_LIB} ${LAPACKE_LIB} ${CBLAS_LIB} ${CDNEST_LIB})
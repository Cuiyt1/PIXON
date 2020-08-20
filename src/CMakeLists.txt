
find_path(CDNEST_INCLUDE_DIR NAMES dnestvars.h PATHS /home/liyropt/Projects/GIT/CDNestSingle)
find_library(CDNEST_LIB NAMES dnest PATHS /home/liyropt/Projects/GIT/CDNestSingle)
include_directories(${CDNEST_INCLUDE_DIR})

find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h PATHS /usr/include/lapacke)
find_library(LAPACKE_LIB lapacke)
include_directories(${LAPACKE_INCLUDE_DIR})

find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS /usr/include/cblas)
find_library(CBLAS_LIB cblas)
include_directories(${CBLAS_INCLUDE_DIR})

add_library(utilities 
   SHARED 
   utilities.hpp
   utilities.cpp 
   tnc.h
   tnc.c
   mathfun.h 
   mathfun.c
   )

add_library(cont_model
   SHARED 
   cont_model.hpp
   cont_model.cpp
   )
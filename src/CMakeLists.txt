find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h PATHS /usr/include/lapacke)
find_library(LAPACKE_LIB lapacke)
include_directories(${LAPACKE_INCLUDE_DIR})

find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS /usr/include/cblas)
find_library(CBLAS_LIB cblas)
include_directories(${CBLAS_INCLUDE_DIR})

include_directories("../cdnest")

add_library(utilities 
   SHARED 
   proto.hpp
   utilities.hpp
   utilities.cpp 
   pixon_cont.hpp 
   pixon_cont.cpp
   drw_cont.hpp
   drw_cont.cpp
   tnc.h
   tnc.c
   mathfun.h 
   mathfun.c
   cfgparser.hpp
   )

add_library(cont_model
   SHARED 
   cont_model.hpp
   cont_model.cpp
   )

add_library(run
   SHARED
   run.cpp
)

add_library(test
   SHARED 
   test.cpp
   )
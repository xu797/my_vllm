set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test DIR_TEST_ARMA)
aux_source_directory(./src DIR_SOURCE_ARMA)

add_executable(tensor  tensor_main.cpp ${DIR_TEST_ARMA} ${DIR_SOURCE_ARMA})
target_link_libraries(tensor  ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(tensor PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(tensor PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(tensor PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(tensor PUBLIC ./include)

# add_library(tensor ${DIR_SOURCE_ARMA})

enable_testing()
# # clear
# find_package(OpenMP REQUIRED)
# find_package(Armadillo REQUIRED)
# find_package(glog REQUIRED)
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)
# find_package(GTest REQUIRED)

# set(link_lib glog::glog GTest::gtest)
# if (!WIN32)
#     set(link_lib "${link_lib} pthread")
# endif ()
# set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
# aux_source_directory(./test/topo_test DIR_TEST_ARMA)
# aux_source_directory(./src DIR_SOURCE_ARMA)
# aux_source_directory(${PROJECT_SOURCE_DIR}/tensor/src TENSOR_SRC)

# add_executable(graph_main graph_main.cpp ${DIR_TEST_ARMA} ${DIR_SOURCE_ARMA} ${TENSOR_SRC})
# target_link_libraries(graph_main ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

# target_include_directories(graph_main PUBLIC ${glog_INCLUDE_DIR})
# target_include_directories(graph_main PUBLIC ${GTest_INCLUDE_DIR})
# target_include_directories(graph_main PUBLIC ${Armadillo_INCLUDE_DIR})
# target_include_directories(graph_main PUBLIC ./include)
# target_include_directories(graph_main PUBLIC ${PROJECT_SOURCE_DIR}/tensor/include)

# enable_testing()

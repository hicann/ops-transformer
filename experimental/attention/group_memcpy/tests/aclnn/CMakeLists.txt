cmake_minimum_required(VERSION 3.18.4)
project(group_memcpy_aclnn_tests LANGUAGES CXX)
enable_testing()
set(ASCEND_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
set(GROUP_MEMCPY_OPAPI_ROOT "" CACHE PATH "Installed vendor op_api root (include/ and lib/ or lib64/)")
find_path(ACL_INCLUDE_DIR acl/acl.h HINTS "${ASCEND_ROOT}/include" REQUIRED)
find_path(OPAPI_INCLUDE_DIR aclnn_group_memcpy.h
    HINTS "${GROUP_MEMCPY_OPAPI_ROOT}/include" "${ASCEND_ROOT}/include/aclnnop"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../op_host/op_api" REQUIRED)
find_library(ACL_LIBRARY ascendcl HINTS "${ASCEND_ROOT}/lib64" REQUIRED)
find_library(NNOPBASE_LIBRARY nnopbase HINTS "${ASCEND_ROOT}/lib64" REQUIRED)
find_library(OPAPI_LIBRARY NAMES cust_opapi opapi_transformer
    HINTS "${GROUP_MEMCPY_OPAPI_ROOT}/lib" "${GROUP_MEMCPY_OPAPI_ROOT}/lib64"
          "${ASCEND_ROOT}/lib64" REQUIRED)
add_executable(opapi_test main.cpp)
target_compile_features(opapi_test PRIVATE cxx_std_17)
target_include_directories(opapi_test PRIVATE "${ACL_INCLUDE_DIR}" "${OPAPI_INCLUDE_DIR}")
target_link_libraries(opapi_test PRIVATE "${OPAPI_LIBRARY}" "${ACL_LIBRARY}" "${NNOPBASE_LIBRARY}")
add_test(NAME group_memcpy_aclnn COMMAND opapi_test)

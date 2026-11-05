cmake_minimum_required(VERSION 3.18.4)
project(group_memcpy_package LANGUAGES C CXX)

# Use the packaging framework shipped with CANN; no downloaded cann-cmake needed.
find_package(ASC REQUIRED)
get_filename_component(OP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(package_name group_memcpy)
npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH "${CMAKE_BINARY_DIR}")

npu_op_code_gen(SRC "${OP_ROOT}/op_host/group_memcpy_def.cpp"
    PACKAGE ${package_name} OUT_DIR "${ASCEND_AUTOGEN_PATH}" JOIN_OP_DEF True
    OPTIONS OPS_ACLNN_GEN False)

# Keep the public TensorList API implemented in the operator source.
npu_op_library(cust_opapi ACLNN "${OP_ROOT}/op_host/op_api/aclnn_group_memcpy.cpp")
target_include_directories(cust_opapi PRIVATE "$ENV{ASCEND_HOME_PATH}/include/aclnn")
target_compile_definitions(cust_opapi PRIVATE FUNC_VISIBILITY)

file(GLOB proto_sources "${ASCEND_AUTOGEN_PATH}/op_proto.cc"
    "${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc")
npu_op_library(cust_op_proto GRAPH ${proto_sources}
    "${OP_ROOT}/op_host/group_memcpy_def.cpp"
    "${OP_ROOT}/op_host/group_memcpy_infershape.cpp")
npu_op_library(cust_optiling TILING "${OP_ROOT}/op_host/group_memcpy_tiling.cpp")
target_include_directories(cust_optiling PRIVATE "${OP_ROOT}/op_host")
npu_op_package_add(${package_name} LIBRARY cust_opapi cust_op_proto cust_optiling)

file(REMOVE "${ASCEND_AUTOGEN_PATH}/ascendc_kernels_custom_source_files.ini")
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_options(ascendc_kernels ALL OPTIONS --cce-auto-sync=on)
npu_op_kernel_library(ascendc_kernels SRC_BASE "${OP_ROOT}/op_kernel"
    TILING_LIBRARY cust_optiling)
npu_op_package_add(${package_name} LIBRARY ascendc_kernels)

install(FILES "${OP_ROOT}/op_host/op_api/aclnn_group_memcpy.h"
    DESTINATION "packages/vendors/group_memcpy/op_api/include")

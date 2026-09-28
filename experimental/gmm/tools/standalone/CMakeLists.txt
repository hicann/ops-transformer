# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(experimental_group_matmul LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(ASCEND_COMPUTE_UNIT ascend910b CACHE STRING "ascend910b or ascend910_93")
find_package(ASC REQUIRED HINTS "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
get_filename_component(GMM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(pkg experimental_group_matmul)
npu_op_package(${pkg} TYPE RUN CONFIG ENABLE_SOURCE_PACKAGE TRUE ENABLE_BINARY_PACKAGE TRUE
  INSTALL_PATH "${CMAKE_BINARY_DIR}/package")
file(GLOB defs "${GMM_ROOT}/*/op_host/*_def.cpp")
file(GLOB tilings "${GMM_ROOT}/*/op_host/*_tiling.cpp")
file(GLOB infers "${GMM_ROOT}/*/op_host/*_infershape.cpp")
npu_op_code_gen(SRC ${defs} PACKAGE ${pkg} OUT_DIR "${ASCEND_AUTOGEN_PATH}" OPTIONS OPS_ACLNN_GEN 1)
file(GLOB apis "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
file(GLOB protos "${ASCEND_AUTOGEN_PATH}/op_proto.cc" "${ASCEND_AUTOGEN_PATH}/group_proto/*.cc")
npu_op_library(gmm_api ACLNN ${apis})
npu_op_library(gmm_tiling TILING ${tilings} ${infers})
npu_op_library(gmm_graph GRAPH ${protos} ${infers})
foreach(target gmm_api gmm_tiling gmm_graph)
  target_include_directories(${target} PRIVATE
    "$ENV{ASCEND_HOME_PATH}/include/tiling"
    "$ENV{ASCEND_HOME_PATH}/pkg_inc"
    "$ENV{ASCEND_HOME_PATH}/pkg_inc/base"
    "$ENV{ASCEND_HOME_PATH}/pkg_inc/op_common"
    "$ENV{ASCEND_HOME_PATH}/pkg_inc/op_common/op_host"
    "$ENV{ASCEND_HOME_PATH}/include/experiment"
    "$ENV{ASCEND_HOME_PATH}/include/experiment/metadef"
    "$ENV{ASCEND_HOME_PATH}/include/experiment/platform")
  target_link_libraries(${target} PRIVATE ops_base)
endforeach()
if(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
  set(kernel_soc ascend910b1)
elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93")
  set(kernel_soc ascend910_9391)
else()
  message(FATAL_ERROR "Supported SoCs: ascend910b, ascend910_93")
endif()
# CANN creates one source-copy target per directory. Build the four kernels
# together, collecting their unchanged source files into the build tree.
set(kernel_src "${CMAKE_BINARY_DIR}/kernel_src")
file(MAKE_DIRECTORY "${kernel_src}")
file(GLOB kernel_files "${GMM_ROOT}/*/op_kernel/*.cpp" "${GMM_ROOT}/*/op_kernel/*.h")
foreach(src ${kernel_files})
  get_filename_component(name "${src}" NAME)
  configure_file("${src}" "${kernel_src}/${name}" COPYONLY)
endforeach()
npu_op_kernel_library(gmm_kernel SRC_BASE "${kernel_src}" TILING_LIBRARY gmm_tiling)
npu_op_kernel_sources(gmm_kernel OP_TYPE GmmAdd KERNEL_FILE gmm_add.cpp COMPUTE_UNIT ${kernel_soc})
npu_op_kernel_sources(gmm_kernel OP_TYPE GmmKDim KERNEL_FILE gmm_k_dim.cpp COMPUTE_UNIT ${kernel_soc})
npu_op_kernel_sources(gmm_kernel OP_TYPE GmmLocalExp KERNEL_FILE gmm_local_exp.cpp COMPUTE_UNIT ${kernel_soc})
npu_op_kernel_sources(gmm_kernel OP_TYPE GmmLocalExpWithZero KERNEL_FILE gmm_local_exp_with_zero.cpp COMPUTE_UNIT ${kernel_soc})
npu_op_package_add(${pkg} LIBRARY gmm_api gmm_tiling gmm_graph gmm_kernel)

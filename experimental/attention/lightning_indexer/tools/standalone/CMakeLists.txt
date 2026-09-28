# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(experimental_lightning_indexer LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(ASCEND_COMPUTE_UNIT ascend910b CACHE STRING "ascend910b or ascend910_93")
find_package(ASC REQUIRED HINTS "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
get_filename_component(OP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(pkg experimental_lightning_indexer)
npu_op_package(${pkg} TYPE RUN CONFIG ENABLE_SOURCE_PACKAGE TRUE ENABLE_BINARY_PACKAGE TRUE
  INSTALL_PATH "${CMAKE_BINARY_DIR}/package")
npu_op_code_gen(SRC "${OP_ROOT}/op_host/lightning_indexer_def.cpp"
                PACKAGE ${pkg} OUT_DIR "${ASCEND_AUTOGEN_PATH}"
                OPTIONS OPS_ACLNN_GEN 1)
file(GLOB api_sources "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
file(GLOB proto_sources "${ASCEND_AUTOGEN_PATH}/op_proto.cc" "${ASCEND_AUTOGEN_PATH}/group_proto/*.cc")
npu_op_library(li_api ACLNN ${api_sources})
npu_op_library(li_tiling TILING
  "${OP_ROOT}/op_host/lightning_indexer_tiling.cpp"
  "${OP_ROOT}/op_host/lightning_indexer_proto.cpp")
npu_op_library(li_graph GRAPH ${proto_sources} "${OP_ROOT}/op_host/lightning_indexer_proto.cpp")
foreach(target li_api li_tiling li_graph)
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
  message(FATAL_ERROR "This operator supports ascend910b and ascend910_93")
endif()
npu_op_kernel_library(li_kernel SRC_BASE "${OP_ROOT}/op_kernel" TILING_LIBRARY li_tiling)
npu_op_kernel_sources(li_kernel OP_TYPE LightningIndexer KERNEL_FILE lightning_indexer.cpp COMPUTE_UNIT ${kernel_soc})
npu_op_package_add(${pkg} LIBRARY li_api li_tiling li_graph li_kernel)

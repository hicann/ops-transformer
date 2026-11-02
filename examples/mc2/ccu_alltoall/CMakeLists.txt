# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ============================================================
# CMakeLists.txt — CCU AllToAll + MXFP8 MatMul 示例（自包含构建入口）
#
# examples 树无统一构建入口（仓库根 build.sh 不扫描），独立 cmake：
#   cmake -S <本目录> -B <本目录>/build && cmake --build <本目录>/build --target ccu_alltoall
# ============================================================
cmake_minimum_required(VERSION 3.18.4)

# ---- 查找 CANN 工具链 ----
if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        set(_ascend_processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    if(NOT _ascend_processor)
        execute_process(COMMAND uname -m OUTPUT_VARIABLE _ascend_processor OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux" CACHE STRING "Ascend system prefix")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
    message(WARNING "Environment variable ASCEND_HOME_PATH is not set. Using default path.")
    if("$ENV{USER}" STREQUAL "root")
        if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
        elseif(EXISTS /usr/local/Ascend/latest)
            set(ASCEND_DIR /usr/local/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Please set ASCEND_HOME_PATH.")
        endif()
    else()
        if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
        elseif(EXISTS $ENV{HOME}/Ascend/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Please set ASCEND_HOME_PATH.")
        endif()
    endif()
endif()

message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${BISHENG}" CACHE FILEPATH "C compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${BISHENG}" CACHE FILEPATH "CXX compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_LINKER)
    set(CMAKE_LINKER "${BISHENG}" CACHE FILEPATH "Linker" FORCE)
endif()

# ---- ASC 编译器模块 ----
list(APPEND CMAKE_MODULE_PATH
    "${ASCEND_DIR}/${SYSTEM_PREFIX}/tikcpp/ascendc_kernel_cmake/asc_modules")
find_package(ASC REQUIRED)

project(ccu_alltoall LANGUAGES CXX ASC)

# ---- 仓库路径（CACHE PATH：支持非常规位置构建时 -D 覆盖）----
set(OPS_TRANSFORMER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "ops-transformer 仓库根")
set(OP_KERNEL_ROOT "${OPS_TRANSFORMER_ROOT}/mc2/common/op_kernel")
set(APACE_ROOT "${OP_KERNEL_ROOT}/apace")

# ---- blaze / tensor_api（ops-tensor，pin 43d554d）----
#   "blaze/gemm/..."        -> ops-tensor/include/
#   "tensor_api/tensor.h"   -> ops-tensor/include/tensor_api/include/
#   "include/tensor_api/..."-> ops-tensor/include/tensor_api/
# 优先复用 apace/.wtdeps 下已就绪的 ops-tensor（存在才用），否则 FetchContent 拉到构建目录。
if(NOT DEFINED CANN_3RD_LIB_PATH)
    if(EXISTS "${APACE_ROOT}/.wtdeps/ops-tensor")
        set(CANN_3RD_LIB_PATH "${APACE_ROOT}/.wtdeps" CACHE PATH "third-party deps root (ops-tensor)")
    else()
        set(CANN_3RD_LIB_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE PATH "third-party deps root (ops-tensor)")
    endif()
endif()
include("${OPS_TRANSFORMER_ROOT}/cmake/third_party/ops-tensor.cmake")
set(OPS_TENSOR_INCLUDE_DIR "${OPTENSOR_SOURCE_PATH}/include")
if(NOT EXISTS "${OPS_TENSOR_INCLUDE_DIR}/blaze")
    message(FATAL_ERROR "blaze not found at: ${OPS_TENSOR_INCLUDE_DIR}/blaze")
endif()
if(NOT EXISTS "${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include/tensor_api/tensor.h")
    message(FATAL_ERROR "tensor_api/tensor.h not found at: ${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include")
endif()
message(STATUS "ops-tensor root: ${OPTENSOR_SOURCE_PATH}")

set(BLAZE_COMMON "${OPS_TENSOR_INCLUDE_DIR}")

# kernel 模式（-xasc）include 路径：apace kernel + Blaze + hcomm/ccu（CCU 直调增量）。
# ops-tensor 三个根必须置于 CANN 目录之前：本机 CANN 9.2.0 的 asc/include 下自带旧版
# tensor_api（无 asc::te 命名空间），会抢先劫持 blaze 的 "tensor_api/tensor.h" 解析。
set(KERNEL_INCLUDES
    ${BLAZE_COMMON}                                                         # blaze/ + ops-tensor 根
    ${BLAZE_COMMON}/tensor_api/include                                      # tensor_api/...
    ${BLAZE_COMMON}/tensor_api
    ${OP_KERNEL_ROOT}                                                       # apace/ 头文件（原位引用）
    ${APACE_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include                              # basic_api/ adv_api/ comm_api/
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/basic_api                     # kernel_basic_intf.h（裸名 include）
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/interface
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling                        # platform/platform_ascendc.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/comm_api/ccu                  # ccu_host_launch.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/comm_api                      # ccu/hcomm/*（新资源 API 头）
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api             # kernel_tiling/kernel_tiling.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include                                   # hccl/ acl/
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/hccl
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/hcomm
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/hcomm/ccu                         # ccu/ccu_*.hpp
    ${ASCEND_DIR}/pkg_inc
    ${ASCEND_DIR}/pkg_inc/hcomm
    ${ASCEND_DIR}/pkg_inc/hcomm/ccu                                          # hccl_ccu_res.h / ccu_res.h
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/include
)

set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3")

# 链接库：CCU 直调组合（hcomm/asccomm_ccu/c_sec/acl_rt/ascendc_runtime/register）
set(COMMON_LIBS dl platform tiling_api ascendcl acl_rt runtime hccl stdc++ hcomm asccomm_ccu c_sec
    ascendc_runtime register pthread)

add_executable(ccu_alltoall
    src/main.cpp
)

# main.cpp: kernel 模式编译（单翻译单元，经 include 引入 .asc 链）
set_source_files_properties(src/main.cpp PROPERTIES
    COMPILE_OPTIONS "${KERNEL_OPTS}"
    INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}"
)

target_link_directories(ccu_alltoall PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_DIR}/lib64
)

target_link_libraries(ccu_alltoall PRIVATE ${COMMON_LIBS})

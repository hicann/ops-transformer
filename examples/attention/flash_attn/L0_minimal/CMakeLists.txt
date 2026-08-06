# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)
project(AscendOps)
option(BUILD_TORCH_OPS "Build torch ops project (PyTorch extension style)" OFF)
message(STATUS "Build torch ops: ${BUILD_TORCH_OPS}")

if(NOT BUILD_TORCH_OPS)
    message(STATUS "BUILD_TORCH_OPS is OFF, exiting CMake configuration")
    return()
endif()

#================================
# Torch Ops 项目特定配置
#================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release/Debug)" FORCE)
endif()

#================================
# Ascend 环境配置
#================================
message(STATUS "$ENV{ASCEND_HOME_PATH}")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend installation path" FORCE)
else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend installation path")
endif()

set(BISHENG "${ASCEND_HOME}/tools/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "ASCEND_HOME = ${ASCEND_HOME}")
message(STATUS "BISHENG     = ${BISHENG}")
# 设置编译器为 bisheng
set(CMAKE_C_COMPILER   ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER       ${BISHENG})

# Python 配置
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable" FORCE)
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
message(STATUS "Python3_EXECUTABLE   = ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES    = ${Python3_LIBRARIES}")

# Torch 配置
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to query torch CMake path: ${TORCH_CMAKE_ERROR}")
endif()
string(REPLACE "\r\n" "\n" TORCH_CMAKE_OUTPUT "${TORCH_CMAKE_OUTPUT}")
string(REGEX MATCH "[^\n]+$" TORCH_CMAKE_PATH "${TORCH_CMAKE_OUTPUT}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
set(Torch_DIR "${TORCH_CMAKE_PATH}/Torch" CACHE PATH "Torch CMake package path" FORCE)
find_package(Torch REQUIRED)
message(STATUS "Torch_DIR       = ${Torch_DIR}")
message(STATUS "Torch_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "Torch_INCLUDES  = ${TORCH_INCLUDE_DIRS}")

# Torch-NPU 配置
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    RESULT_VARIABLE TORCH_NPU_RESULT
    OUTPUT_VARIABLE TORCH_NPU_OUTPUT
    ERROR_VARIABLE TORCH_NPU_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_NPU_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to query torch_npu path: ${TORCH_NPU_ERROR}")
endif()
string(REPLACE "\r\n" "\n" TORCH_NPU_OUTPUT "${TORCH_NPU_OUTPUT}")
string(REGEX MATCH "[^\n]+$" TORCH_NPU_PATH "${TORCH_NPU_OUTPUT}")
set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIB_PATH     "${TORCH_NPU_PATH}/lib")

message(STATUS "TORCH_NPU_PATH         = ${TORCH_NPU_PATH}")
message(STATUS "TORCH_NPU_INCLUDE_DIR  = ${TORCH_NPU_INCLUDE_PATH}")
message(STATUS "TORCH_NPU_LIB_DIR      = ${TORCH_NPU_LIB_PATH}")

# Torch Ops 公共配置
set(COMMON_INCLUDE_DIRS
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ops/csrc
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api
    ${ASCEND_HOME}/include/ascendc
    ${ASCEND_HOME}/include/ascendc/basic_api/interface
    ${ASCEND_HOME}/include/ascendc/basic_api
    ${ASCEND_HOME}/pkg_inc/op_common
    ${ASCEND_HOME}/pkg_inc/base
    ${ASCEND_HOME}/pkg_inc
)

set(COMMON_LINK_DIRS
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_HOME}/lib64
)

set(COMMON_LINK_LIBS
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    ${Python3_LIBRARIES}
)

set(COMMON_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O2
    -fdiagnostics-color=always
    -DPy_LIMITED_API=0x03090000
    -w
    -DTORCH_MODE
)

message(STATUS "Building Torch Ops project...")

file(GLOB OPERATOR_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/ascend_ops/csrc/*")
set(OPERATOR_TARGETS "")

foreach(OP_DIR ${OPERATOR_DIRS})
    if(IS_DIRECTORY ${OP_DIR})
        # 获取目录名
        get_filename_component(OP_NAME ${OP_DIR} NAME)

        # 检查目录中是否有 CMakeLists.txt
        if(EXISTS "${OP_DIR}/CMakeLists.txt")
            message(STATUS "Found operator: ${OP_NAME}")
            add_subdirectory(ascend_ops/csrc/${OP_NAME})
            list(APPEND OPERATOR_TARGETS $<TARGET_OBJECTS:${OP_NAME}_objects>)
        endif()
    endif()
endforeach()

message(STATUS "Discovered torch operator targets: ${OPERATOR_TARGETS}")

# 创建主库
add_library(_C SHARED
    ${OPERATOR_TARGETS}
)

set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LINK_FLAGS "--cce-fatobj-link"
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
)

target_compile_options(_C PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(_C PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(_C PRIVATE ${COMMON_LINK_LIBS})
return()

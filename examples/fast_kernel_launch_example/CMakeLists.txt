# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)
find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
set(PKG_NAME AscendOps)
project(${PKG_NAME} LANGUAGES ASC AICPU CXX VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)
include(../../cmake/third_party/opbase.cmake)

# set NPU SOC version, default to ascend910b
set(NPU_SOC_VERSION "ascend910b" CACHE STRING "NPU SOC version")
# set EXTENSION_MODULE_NAME to `ascend_ops`
set(EXTENSION_MODULE_NAME "ascend_ops" CACHE STRING "Extension module name")

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend installation path" FORCE)
else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend installation path")
endif()

# include directories
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/inc
    ${OPBASE_SOURCE_PATH}/include/op_common
    ${OPBASE_SOURCE_PATH}/pkg_inc/op_common
    ${ASCEND_HOME}/pkg_inc/base
)

# link directories
set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
)

# link libraries
set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    hccl
    hcomm
)

# compile options
set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -w
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
    -DTORCH_MODE
)

# define a LIST variable to hold object files
set(OBJECTS_LIST "" CACHE INTERNAL "List of objects")
message(STATUS "OBJECTS_LIST before adding sources: ${OBJECTS_LIST}")
add_subdirectory(csrc)
message(STATUS "OBJECTS_LIST after adding sources: ${OBJECTS_LIST}")

# create shared library
set("EXTENSION_CPP" ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp)
add_library(_C SHARED
    ${EXTENSION_CPP}
    ${OBJECTS_LIST}
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
)
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying compiled extension $<TARGET_FILE_NAME:_C> to ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/"
)

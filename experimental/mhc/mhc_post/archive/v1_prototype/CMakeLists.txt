# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.

cmake_minimum_required(VERSION 3.18.4)
project(mhc_post_v1)

set(CMAKE_CXX_STANDARD 17)
set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_PATH})
set(SOC_VERSION "ascend910b2" CACHE STRING "SoC version")

include(${ASCEND_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(mhc_post_kernel STATIC kernel/mhc_post_kernel.cpp)

macro(add_test_exe NAME)
    add_executable(${NAME} test/${NAME}.cpp)
    target_link_directories(${NAME} PRIVATE ${ASCEND_PATH}/lib64)
    target_link_libraries(${NAME} mhc_post_kernel ascendcl runtime)
    target_include_directories(${NAME} PRIVATE ${ASCEND_PATH}/include)
endmacro()

add_test_exe(test_mhc_post)

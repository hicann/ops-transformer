# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Run with: cmake -DHI_PYTHON=/path/to/python -DTEST_TMP=/tmp/compile-units -P cmake/tests/compile_units.cmake
cmake_minimum_required(VERSION 3.16)
if(NOT DEFINED HI_PYTHON OR NOT DEFINED TEST_TMP)
    message(FATAL_ERROR "Set HI_PYTHON and TEST_TMP")
endif()
file(MAKE_DIRECTORY "${TEST_TMP}")
set(ASCEND_CUSTOM_OPTIONS "${TEST_TMP}/custom_compile_options.ini")
set(ASCENDC_CMAKE_UTIL_DIR "${CMAKE_CURRENT_LIST_DIR}/../scripts/util")
include("${CMAKE_CURRENT_LIST_DIR}/../func.cmake")

foreach(_backend IN ITEMS legacy v2)
    file(WRITE "${ASCEND_CUSTOM_OPTIONS}" "")
    file(REMOVE "${TEST_TMP}/custom_opc_options.ini")
    if(_backend STREQUAL "v2")
        set(ADD_OPS_COMPILE_OPTION_V2 TRUE)
    else()
        set(ADD_OPS_COMPILE_OPTION_V2 FALSE)
    endif()
    add_ops_compile_options(OP_NAME MixedComputeUnit
        COMPUTE_UNIT dav_3510 Ascend960DT_968B7 OPTIONS -g --oom)
    file(READ "${ASCEND_CUSTOM_OPTIONS}" _options)
    foreach(_expected IN ITEMS ascend950pr_9599 ascend350_355e Ascend960DT_968B7)
        if(NOT _options MATCHES "${_expected}")
            message(FATAL_ERROR "${_backend}: missing ${_expected}: ${_options}")
        endif()
    endforeach()
    if(_options MATCHES "ascend960pr|dav_3510")
        message(FATAL_ERROR "${_backend}: incorrect expansion: ${_options}")
    endif()
    if(_backend STREQUAL "v2")
        file(READ "${TEST_TMP}/custom_opc_options.ini" _opc)
        if(NOT _opc MATCHES "ascend950pr_9599;ascend350_355e;Ascend960DT_968B7@--op_debug_config=oom")
            message(FATAL_ERROR "v2: incorrect OPC configuration: ${_opc}")
        endif()
    endif()
endforeach()

expand_compile_compute_unit("ascend910b" _legacy_soc)
if(NOT _legacy_soc STREQUAL "ascend910b")
    message(FATAL_ERROR "Existing SoC entries must pass through unchanged")
endif()

get_npu_arch_by_soc_version("ASCEND960PR_969A1" _arch)
if(NOT _arch STREQUAL "dav_9202")
    message(FATAL_ERROR "incorrect SoC mapping: ${_arch}")
endif()
get_npu_arch_by_soc_version("unknown" _arch)
if(NOT _arch STREQUAL "")
    message(FATAL_ERROR "unknown SoC must not have an architecture")
endif()
message(STATUS "Mixed compile unit tests passed")

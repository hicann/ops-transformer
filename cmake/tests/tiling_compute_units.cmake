# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
include("${CMAKE_CURRENT_LIST_DIR}/../obj_func.cmake")

if(DEFINED TILING_WARNING_CASE)
  set(OPS_NAME "tiling_test_operator")
  if(TILING_WARNING_CASE STREQUAL "partial")
    set(_socs "ascend950;ascend910b")
    set(_expected "arch35")
  elseif(TILING_WARNING_CASE STREQUAL "unmatched")
    set(_socs "ascend910b")
    set(_expected "")
  elseif(TILING_WARNING_CASE STREQUAL "matched")
    set(_socs "ascend950;ascend350")
    set(_expected "arch35")
  elseif(TILING_WARNING_CASE STREQUAL "empty")
    set(_socs "")
    set(_expected "")
  else()
    message(FATAL_ERROR "Unknown test case: ${TILING_WARNING_CASE}")
  endif()
  select_tiling_directories(_actual "dav_3510" "arch35" "${_socs}")
  if(NOT "${_actual}" STREQUAL "${_expected}")
    message(FATAL_ERROR "${TILING_WARNING_CASE}: expected '${_expected}', got '${_actual}'")
  endif()
  return()
endif()

function(assert_warning_case case expect_warning)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "-DTILING_WARNING_CASE=${case}" -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _result ERROR_VARIABLE _stderr OUTPUT_VARIABLE _stdout)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${case}: subprocess failed: ${_stderr}")
  endif()
  if(expect_warning)
    if(NOT _stderr MATCHES "No TILING_DIR match for SoC" OR
       NOT _stderr MATCHES "ascend910b" OR NOT _stderr MATCHES "tiling_test_operator")
      message(FATAL_ERROR "${case}: missing expected warning: ${_stderr}")
    endif()
    string(REGEX MATCHALL "No TILING_DIR match for SoC" _warnings "${_stderr}")
    list(LENGTH _warnings _warning_count)
    if(NOT _warning_count EQUAL 1 OR _stderr MATCHES "SoC 'ascend950'")
      message(FATAL_ERROR "${case}: unexpected warning count or matched SoC: ${_stderr}")
    endif()
  elseif(_stderr MATCHES "No TILING_DIR match for SoC")
    message(FATAL_ERROR "${case}: unexpected warning: ${_stderr}")
  endif()
endfunction()

assert_warning_case(partial TRUE)
assert_warning_case(unmatched TRUE)
assert_warning_case(matched FALSE)
assert_warning_case(empty FALSE)

function(assert_tiling expected units dirs socs)
  select_tiling_directories(actual "${units}" "${dirs}" "${socs}")
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR "Expected '${expected}', got '${actual}' for ${units} / ${socs}")
  endif()
endfunction()

assert_tiling("arch35;arch92" "ascend950;dav_9201;dav_9202" "arch35;arch92;arch92" "ascend350;ascend960dt;ascend960pr")
assert_tiling("first" "ascend950;dav_3510" "first;second" "ascend350;ascend950")
assert_tiling("" "dav_3510" "arch35" "ascend960dt")
assert_tiling("arch22" "DAV_2201;ascend950" "arch22;arch35" "ascend910_93")
assert_tiling("arch20" "ascend310p;dav_3510" "arch20;arch35" "ASCEND310P")
assert_tiling("arch38" "mc62;dav_3510" "arch38;arch35" "mc62")
assert_tiling("" "ascend310p" "arch20" "mc62")

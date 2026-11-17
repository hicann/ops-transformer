# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(RegistrationArchHelperTest LANGUAGES CXX)

set(OPBASE_SOURCE_PATH "${CMAKE_CURRENT_BINARY_DIR}/fake-opbase")
file(MAKE_DIRECTORY "${OPBASE_SOURCE_PATH}/src/op_common/op_host/util")
file(MAKE_DIRECTORY "${OPBASE_SOURCE_PATH}/include")
file(WRITE "${OPBASE_SOURCE_PATH}/src/op_common/op_host/util/op_arch_helper.cpp" "")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/operator_def.cpp" "")
include("${CMAKE_CURRENT_LIST_DIR}/../../op_arch_helper.cmake")

add_library(test_registration SHARED "${CMAKE_CURRENT_BINARY_DIR}/operator_def.cpp")
target_include_directories(test_registration PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/old-headers")
add_registration_arch_helper(test_registration)
add_registration_arch_helper(test_registration)
get_target_property(_sources test_registration SOURCES)
set(_helper_count 0)
foreach(_source IN LISTS _sources)
    if(_source STREQUAL "${OPBASE_SOURCE_PATH}/src/op_common/op_host/util/op_arch_helper.cpp")
        math(EXPR _helper_count "${_helper_count} + 1")
    endif()
endforeach()
if(NOT _helper_count EQUAL 1)
    message(FATAL_ERROR "Expected one helper source, got ${_helper_count}")
endif()
get_target_property(_includes test_registration INCLUDE_DIRECTORIES)
list(GET _includes 0 _first_include)
if(NOT _first_include STREQUAL "${OPBASE_SOURCE_PATH}/include")
    message(FATAL_ERROR "Pinned opbase headers must precede installed headers")
endif()

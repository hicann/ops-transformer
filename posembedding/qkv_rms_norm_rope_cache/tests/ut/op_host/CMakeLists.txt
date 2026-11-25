# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # The shared host UT target is created after operator test sources are registered.
    # errmsg negative cases read raw messages from liberror_manager.so.
    set_property(
        DIRECTORY "${CMAKE_SOURCE_DIR}"
        APPEND PROPERTY LINK_LIBRARIES
        "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,${PKG_NAME}_op_host_ut>:-Wl,--no-as-needed>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,${PKG_NAME}_op_host_ut>:error_manager>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,${PKG_NAME}_op_host_ut>:nnopbase>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,${PKG_NAME}_op_host_ut>:-Wl,--as-needed>")
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

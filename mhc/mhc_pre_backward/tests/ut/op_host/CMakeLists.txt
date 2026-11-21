# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
    target_include_directories(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../common/include)
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

# 覆盖率插桩时保留函数调用，避免优化内联导致独立函数计数为零。
if(ENABLE_GCOV AND (UT_TEST_ALL OR OP_HOST_UT))
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/mhc_pre_backward_infershape.cpp
        TARGET_DIRECTORY ophost_transformer_infer_obj
        PROPERTIES COMPILE_FLAGS "-O0 -fno-inline"
    )
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_tiling/arch35/mhc_pre_backward_tiling.cpp
        TARGET_DIRECTORY ophost_transformer_tiling_obj
        PROPERTIES COMPILE_FLAGS "-O0 -fno-inline"
    )
endif()

if(ENABLE_GCOV AND (UT_TEST_ALL OR OP_HOST_UT))
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/test_mhc_pre_backward_a5_coverage_tiling.cpp
        TARGET_DIRECTORY ${OP_TILING_MODULE_NAME}_cases_obj
        PROPERTIES COMPILE_FLAGS "-O0 -fno-inline"
    )
endif()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Test the real registered templates in the same binary as their existing production objects.
# This avoids a second hidden registry singleton across the normal host-test shared library.
if((UT_TEST_ALL OR OP_HOST_UT) AND "arch35" IN_LIST ARCH_DIRECTORY)
    list(FIND ASCEND_OP_NAME "mhc_post" _mhc_selected)
    if(ASCEND_OP_NAME STREQUAL "ALL" OR NOT _mhc_selected EQUAL -1)
        add_executable(mhc_post_tiling_base_isolated_ut mhc_post_base_isolated.cpp)
        target_include_directories(mhc_post_tiling_base_isolated_ut PRIVATE
            ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common
            ${OP_TILING_INCLUDE} ${ASCEND_DIR}/pkg_inc
            ${ASCEND_DIR}/include/base/context_builder)
        target_link_directories(mhc_post_tiling_base_isolated_ut PRIVATE ${ASCEND_DIR}/lib64)
        target_link_libraries(mhc_post_tiling_base_isolated_ut PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            ${OPHOST_NAME}_tiling_obj ${COMMON_NAME}_obj
            opbase_util_objs opbase_infer_objs opbase_tiling_objs
            ${OP_TILING_MODULE_NAME}_common_obj
            -Wl,--push-state,--no-as-needed metadef nnopbase mmpa runtime acl_rt
            error_manager register opp_registry graph graph_base platform tiling_api
            ascendalog unified_dlog -Wl,--pop-state dl gtest)
    endif()
endif()

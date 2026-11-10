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
endif()

if((UT_TEST_ALL OR OP_HOST_UT) AND TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    get_target_property(_mc2_case_sources ${OP_TILING_MODULE_NAME}_cases_obj SOURCES)
    if("${_mc2_case_sources}" MATCHES "test_moe_update_expert_tiling\\.cpp")
        target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE error_manager)
    endif()
endif()

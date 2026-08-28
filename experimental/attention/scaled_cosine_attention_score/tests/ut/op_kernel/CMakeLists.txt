# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------

if((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    set(scas_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/scaled_cosine_attention_score_tiling.cpp)
    AddOpTestCase(scaled_cosine_attention_score "ascend910B1" "" "${scas_tiling_files}")
    if(TARGET scaled_cosine_attention_score_Ascend910B1_cases_obj)
        target_include_directories(scaled_cosine_attention_score_Ascend910B1_cases_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel)
    endif()
endif()

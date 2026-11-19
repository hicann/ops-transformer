# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(BUILD_OPEN_PROJECT)
    enable_cannbotdsl_kernel(
        fused_qkv_proj_norm_rope_cache
        "${CMAKE_CURRENT_SOURCE_DIR}/op_dsl"
    )
    get_property(_fused_qkv_dsl_available GLOBAL PROPERTY CANNBOTDSL_AVAILABLE)
    if(NOT _fused_qkv_dsl_available)
        message(STATUS "fused_qkv_proj_norm_rope_cache: DSL is unavailable for the current SOC/toolchain; skip operator registration")
        return()
    endif()
endif()

add_op_to_compiled_list(OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_modules_sources(
    OPTYPE fused_qkv_proj_norm_rope_cache
    ACLNNTYPE aclnn_exclude
)

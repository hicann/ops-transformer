# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        moe_ffn_def.cpp
        moe_ffn_tiling.cpp
    )
    add_ops_compile_options(
        OP_NAME MoeFFN
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

target_sources(optiling PRIVATE
    moe_ffn_tiling.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_ffn ACLNNTYPE aclnn)
endif()

# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        prompt_flash_attention_def.cpp
    )
    add_ops_compile_options(
        OP_NAME PromptFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
    )
    set(prompt_flash_attention_depends attention/common PARENT_SCOPE)
endif()

if(NOT BUILD_OPS_RTY_KERNEL) #
    add_modules_sources(OPTYPE prompt_flash_attention ACLNNTYPE aclnn_inner)
    add_ops_compile_options(
        OP_NAME PromptFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
    )
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        prompt_flash_attention_tiling.cpp
        fallback_prompt_flash_attention.cpp
    )
endif()
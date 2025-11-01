# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)
set(fused_infer_attention_score_depends attention/incre_flash_attention attention/prompt_flash_attention attention/flash_attention_score attention/common PARENT_SCOPE)
    target_sources(op_host_aclnnInner PRIVATE
        fused_infer_attention_score_def.cpp
    )
    add_ops_compile_options(
        OP_NAME FusedInferAttentionScore
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL) #
    add_modules_sources(OPTYPE fused_infer_attention_score ACLNNTYPE aclnn_inner)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ../../common/op_host/fia_tiling_shape.cpp
        ../../common/op_host/fia_tiling_info.cpp
        ../../common/op_host/arch32/fia_tiling_nonquant_mla.cpp
        fallback_fused_infer_attention_score.cpp
    )
endif()

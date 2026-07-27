# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        sparse_flash_attention_grad_def.cpp
    )
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    set(sparse_flash_attention_grad_depends attention/common CACHE INTERNAL "Dependencies for sparse_flash_attention_grad")
    add_ops_compile_options(
        OP_NAME sparse_flash_attention_grad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME SparseFlashAttentionGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
else()
   add_ops_compile_options(
        OP_NAME SparseFlashAttentionGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
    add_modules_sources(
        OPTYPE sparse_flash_attention_grad
        ACLNNTYPE aclnn_inner
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ./arch22/sparse_flash_attention_grad_tiling_bs1_basic.cpp
        ./arch22/sparse_flash_attention_grad_tiling_bs1_basic.h
        ./arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp
        ./arch35/sparse_flash_attention_grad_tiling_bs1_regbase.h
    )
endif()

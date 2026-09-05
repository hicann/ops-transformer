# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

set(generic_block_sparse_attention_grad_depends attention/generic_block_sparse_attention_grad_metadata CACHE INTERNAL
    "Dependencies for generic_block_sparse_attention_grad")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        generic_block_sparse_attention_grad_def.cpp
    )
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME GenericBlockSparseAttentionGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME GenericBlockSparseAttentionGrad
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -fpermissive
    )
else()
    add_modules_sources(OPTYPE generic_block_sparse_attention_grad ACLNNTYPE aclnn_exclude)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ./arch35/generic_block_sparse_attention_grad_tiling_arch35.cpp
    )
endif()

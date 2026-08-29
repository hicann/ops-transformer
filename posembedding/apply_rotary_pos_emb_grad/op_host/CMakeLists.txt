# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        apply_rotary_pos_emb_grad_def.cpp
    )
    add_modules_sources(OPTYPE apply_rotary_pos_emb_grad ACLNNTYPE aclnn)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    set(apply_rotary_pos_emb_grad_apt_depends posembedding/apply_rotary_pos_emb CACHE INTERNAL "Dependencies for apply_rotary_pos_emb_grad")
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME ApplyRotaryPosEmbGrad
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_modules_sources(OPTYPE apply_rotary_pos_emb_grad ACLNNTYPE aclnn)
endif()

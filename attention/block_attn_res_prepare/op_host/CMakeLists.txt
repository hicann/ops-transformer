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

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        block_attn_res_prepare_def.cpp
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME BlockAttnResPrepare
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS --cce-auto-sync=off
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
                    -Wno-deprecated-declarations
                    -Werror
        )
    else()
        add_ops_compile_options(
            OP_NAME BlockAttnResPrepare
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE block_attn_res_prepare ACLNNTYPE aclnn_exclude)
endif()

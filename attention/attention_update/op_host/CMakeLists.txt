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
    target_sources(op_host_aclnnExc PRIVATE
    attention_update_def.cpp
    )
    add_ops_compile_options(
        OP_NAME AttentionUpdate
        COMPUTE_UNIT dav_3510 dav_9201 dav_9202
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
    add_ops_compile_options(
        OP_NAME AttentionUpdate
        COMPUTE_UNIT dav_2201
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )

endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE attention_update ACLNNTYPE aclnn_exclude)
endif()

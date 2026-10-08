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
    target_sources(op_host_aclnnInner PRIVATE
        inplace_partial_rotary_mul_def.cpp
    )
    add_modules_sources(OPTYPE inplace_partial_rotary_mul ACLNNTYPE aclnn_inner)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME InplacePartialRotaryMul
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
elseif("${CONDITION_UNIT}" STREQUAL "ascend350")
    add_ops_compile_options(
        OP_NAME InplacePartialRotaryMul
        COMPUTE_UNIT Ascend350_355e
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
elseif("${CONDITION_UNIT}" STREQUAL "ascend960dt")
    add_ops_compile_options(
        OP_NAME InplacePartialRotaryMul
        COMPUTE_UNIT Ascend960DT_968B7
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME InplacePartialRotaryMul
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm -cce-aicore-hoist-movemask=false
                    --op_relocatable_kernel_binary=true
    )
else()
    add_modules_sources(OPTYPE inplace_partial_rotary_mul ACLNNTYPE aclnn_inner)
endif()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
require_pypto_pro(flash_attn_grad)
add_op_to_compiled_list()

set(FLASH_ATTN_GRAD_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for flash_attn_grad")
set(FLASH_ATTN_GRAD_INNER_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging inner header for flash_attn_grad")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
    flash_attn_grad_def.cpp
    )
    add_modules_sources(OPTYPE flash_attn_grad ACLNNTYPE aclnn_inner)
    enable_pypto_kernel(flash_attn_grad)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME FlashAttnGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME FlashAttnGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME FlashAttnGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE flash_attn_grad ACLNNTYPE aclnn_inner)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()

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
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        set(grouped_matmul_add_depends gmm/common CACHE INTERNAL "Dependencies for grouped_matmul_add")
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend350")
        set(grouped_matmul_add_depends gmm/common CACHE INTERNAL "Dependencies for grouped_matmul_add")
    endif()
    target_sources(op_host_aclnnExc PRIVATE
        grouped_matmul_add_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
            OP_NAME GroupedMatmulAdd
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        set(grouped_matmul_add_depends gmm/common CACHE INTERNAL "Dependencies for grouped_matmul_add")
        add_ops_compile_options(
            OP_NAME GroupedMatmulAdd
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend350")
        set(grouped_matmul_add_depends gmm/common CACHE INTERNAL "Dependencies for grouped_matmul_add")
        add_ops_compile_options(
            OP_NAME GroupedMatmulAdd
            COMPUTE_UNIT Ascend350_355e
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
else() # custom
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE grouped_matmul_add ACLNNTYPE aclnn_exclude)
endif()

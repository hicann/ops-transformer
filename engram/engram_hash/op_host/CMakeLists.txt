# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 纯标量计算类算子, 不依赖通信基础设施 (原属 mc2 模块, 2026-11 评审后移至 engram 模块)。
# ACLNNTYPE aclnn: aclnn 两段式接口与 GE proto 由 opbuild 工具从 engram_hash_def.cpp 自动生成,
# 无需手写 op_api 目录。
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        engram_hash_def.cpp
    )

    # ASCEND_COMPUTE_UNIT 在多 soc 合编 (--soc=a,b,c) 时是列表,
    # STREQUAL 整串比较会全部落空, 必须逐项匹配后再下发编译选项。
    foreach(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
        if("${CONDITION_UNIT}" STREQUAL "ascend950")
            add_ops_compile_options(
                OP_NAME EngramHash
                COMPUTE_UNIT Ascend950PR_9599
                OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
            )
        elseif("${CONDITION_UNIT}" STREQUAL "ascend960dt")
            add_ops_compile_options(
                OP_NAME EngramHash
                COMPUTE_UNIT Ascend960DT_968B7
                OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
            )
        elseif("${CONDITION_UNIT}" STREQUAL "ascend960pr")
            add_ops_compile_options(
                OP_NAME EngramHash
                COMPUTE_UNIT Ascend960PR_969A1
                OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
            )
        endif()
    endforeach()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_op_to_compiled_list()
    add_modules_sources(OPTYPE engram_hash ACLNNTYPE aclnn)
endif()

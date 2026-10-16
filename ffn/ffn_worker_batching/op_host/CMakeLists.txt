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
    # Generate only the internal ABI; public V1/V2 signatures are maintained in op_api.
    target_sources(op_host_aclnnInner PRIVATE
        ffn_worker_batching_def.cpp
    )
    if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
        add_ops_compile_options(
                OP_NAME FfnWorkerBatching
                OPTIONS --cce-auto-sync=on
                        -Wno-deprecated-declarations
                        -Werror
        )
    else() # custom
        # ascend950/ascend350(arch35) 专属编译选项（dcci）；A2/A3 不进此分支，零回归。
        if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950" OR ASCEND_COMPUTE_UNIT STREQUAL "ascend350")
            add_ops_compile_options(
                    OP_NAME FfnWorkerBatching
                    OPTIONS --cce-auto-sync=on
                            -Wno-deprecated-declarations
                            -Werror
                            -mllvm
                            -cce-aicore-dcci-before-kernel-end=false
            )
        endif()
        add_modules_sources(OPTYPE ffn_worker_batching ACLNNTYPE aclnn_inner)
    endif()
else() # 回黄host
    add_modules_sources(OPTYPE ffn_worker_batching ACLNNTYPE aclnn_inner)
endif()

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
    target_sources(op_host_aclnn PRIVATE
        attention_worker_combine_def.cpp
    )
    if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
        add_ops_compile_options(
                OP_NAME AttentionWorkerCombine
                OPTIONS --cce-auto-sync=on
                        -Wno-deprecated-declarations
        )
    else() # custom
        add_modules_sources(OPTYPE attention_worker_combine ACLNNTYPE aclnn)
    endif()
else() # 回黄host
    add_modules_sources(OPTYPE attention_worker_combine ACLNNTYPE aclnn)
endif()

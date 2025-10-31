# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    set(mla_prolog_v3_depends attention/mla_prolog PARENT_SCOPE)
    target_sources(op_host_aclnnInner PRIVATE
        mla_prolog_v3_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME MlaPrologV3
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )

    add_ops_compile_options(
        OP_NAME MlaPrologV3
        COMPUTE_UNIT ascend310p1
        OPTIONS -mllvm
                -cce-aicore-jump-expand=true
    )
else()
    add_modules_sources(OPTYPE mla_prolog_v3 ACLNNTYPE aclnn_inner)
endif()
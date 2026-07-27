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
    target_sources(op_host_aclnnInner PRIVATE
        apply_rotary_pos_emb_def.cpp
    )
endif()

set(apply_rotary_pos_emb_mc62_arch_directory arch35 arch38 CACHE INTERNAL "apply_rotary_pos_emb mc62 arch")

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME ApplyRotaryPosEmb
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
    )
else()
    add_modules_sources(OPTYPE apply_rotary_pos_emb ACLNNTYPE aclnn_inner)
endif()

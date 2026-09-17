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
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_compute_wy_def.cpp
    )
endif()

# --cce-auto-sync=off: the kernel is hand-synchronised (explicit Set/WaitFlag around the
# Mmad path in op_kernel/arch20/compute_wy_micro_mm.h). Leave it off.
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleComputeWy
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_opbase_modules()
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE chunk_gated_delta_rule_compute_wy
        ACLNNTYPE aclnn_exclude
    )
endif()

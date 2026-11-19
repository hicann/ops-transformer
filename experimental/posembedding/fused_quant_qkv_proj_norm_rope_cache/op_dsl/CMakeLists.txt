# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
include_guard(GLOBAL)

# Keep the explicit operator root when this file is included from op_host.
add_op_to_compiled_list(OP_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

if(BUILD_OPEN_PROJECT)
    # Native-only packages still need the framework Host stub targets.
    add_modules_sources()
    enable_cannbotdsl_kernel(
        fused_quant_qkv_proj_norm_rope_cache
        "${CMAKE_CURRENT_LIST_DIR}"
    )
endif()

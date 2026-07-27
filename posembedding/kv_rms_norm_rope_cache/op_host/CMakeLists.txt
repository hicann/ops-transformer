# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


message(STATUS "=== Debug: start ops.posembedding.kv_rms_norm_rope_cache.op_host.CMakeLists.txt")

if (BUILD_OPEN_PROJECT)
    message(STATUS "=== Debug: BUILD is BUILD_OPEN_PROJECT")
    target_sources(op_host_aclnn PRIVATE
        kv_rms_norm_rope_cache_def.cpp
    )
    add_ops_compile_options(
            OP_NAME KvRmsNormRopeCache
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    message(STATUS "=== Debug: BUILD is NOT BUILD_OPS_RTY_KERNEL")
    add_modules_sources(OPTYPE kv_rms_norm_rope_cache ACLNNTYPE aclnn ACLNN_EXTRA_VERSION v2)
endif()

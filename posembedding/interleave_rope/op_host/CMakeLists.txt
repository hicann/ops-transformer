# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.posembedding.interleave_rope.op_host.CMakeLists.txt ")

if (BUILD_OPEN_PROJECT)
    message(STATUS "=== Debug: BUILD is BUILD_OPEN_PROJECT")
    set(interleave_rope_depends posembedding/rotary_position_embedding CACHE INTERNAL "Dependencies for rotary_position_embedding")
    target_sources(op_host_aclnnInner PRIVATE
        interleave_rope_def.cpp
    )
    add_ops_compile_options(
            OP_NAME InterleaveRope
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    message(STATUS "=== Debug: BUILD is NOT BUILD_OPS_RTY_KERNEL")
    add_modules_sources(OPTYPE interleave_rope ACLNNTYPE aclnn_inner)
endif()

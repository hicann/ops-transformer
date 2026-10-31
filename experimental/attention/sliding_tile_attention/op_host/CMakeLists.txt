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

# SlidingTileAttention uses the standard BlockSparseAttention implementation
# for the task-book D=64 fast path.  Experimental builds only discover
# dependencies under experimental/ by default, so register the standard
# dependency explicitly and package its host, tiling and kernel assets in the
# same custom OPP.
set(sliding_tile_attention_depends attention/block_sparse_attention
    CACHE INTERNAL "Dependencies for sliding_tile_attention")
if(NOT block_sparse_attention IN_LIST COMPILED_OPS)
    list(APPEND ASCEND_OP_NAME block_sparse_attention)
    set(ASCEND_OP_NAME "${ASCEND_OP_NAME}" CACHE STRING "Operators that need to be compiled" FORCE)
    add_subdirectory(
        ${CMAKE_SOURCE_DIR}/attention/block_sparse_attention/op_host
        ${CMAKE_BINARY_DIR}/attention/block_sparse_attention/op_host
    )
endif()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        sliding_tile_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SlidingTileAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
)

add_modules_sources(OPTYPE sliding_tile_attention ACLNNTYPE aclnn_exclude)

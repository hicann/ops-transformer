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
    # 本算子不依赖其它算子: 源码未引用任何跨算子头文件, 因此不声明 depends
    # (实验分支下 depends 会被加上 experimental/ 前缀, 主树同名依赖路径不存在 -> 构建报 copy 错误)
    target_sources(op_host_aclnn PRIVATE
        quant_sparse_lightning_indexer_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME QuantSparseLightningIndexer
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE quant_sparse_lightning_indexer ACLNNTYPE aclnn)
endif()

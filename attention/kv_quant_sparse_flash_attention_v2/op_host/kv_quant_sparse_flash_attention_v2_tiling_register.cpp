/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_flash_attention_v2_tiling_register.cpp
 * \brief KvQuantSparseFlashAttentionV2 tiling 入口注册, 复用 kv_quant_sparse_flash_attention 的 tiling 实现
 */

#include "kv_quant_sparse_flash_attention_v2_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
namespace optiling {
IMPL_OP_OPTILING(KvQuantSparseFlashAttentionV2)
    .Tiling(TilingKvQuantSparseFlashAttention)
    .TilingParse<KvQuantSparseFlashAttentionCompileInfo>(TilingPrepareForKvQuantSparseFlashAttention);
} // namespace optiling

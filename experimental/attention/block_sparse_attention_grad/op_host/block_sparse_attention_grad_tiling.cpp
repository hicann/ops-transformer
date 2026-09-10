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
 * \file block_sparse_attention_grad_tiling.cpp
 * \brief Tiling registration for BlockSparseAttentionGrad.
 */

#include <register/op_impl_registry.h>
#include "op_host/data_copy_transpose_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "block_sparse_attention_grad_tiling.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace bsag {

ASCENDC_EXTERN_C ge::graphStatus TilingBlockSparseAttentionGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("BlockSparseAttentionGrad", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareBlockSparseAttentionGrad(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockSparseAttentionGrad)
    .Tiling(TilingBlockSparseAttentionGrad)
    .TilingParse<BlockSparseAttentionGradCompileInfo>(TilingPrepareBlockSparseAttentionGrad); // 向框架注册入口函数
} // namespace bsag
} // namespace optiling

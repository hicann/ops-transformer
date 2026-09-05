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
 * \file generic_block_sparse_attention_grad_tiling.cpp
 * \brief Tiling registration for GenericBlockSparseAttentionGrad.
 */

#include <register/op_impl_registry.h>
#include "op_host/tiling_templates_registry.h"
#include "generic_block_sparse_attention_grad_tiling.h"
#include "err/ops_err.h"

using namespace ge;

namespace optiling {
namespace gsag {

ASCENDC_EXTERN_C ge::graphStatus TilingGenericBlockSparseAttentionGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttentionGrad", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareGenericBlockSparseAttentionGrad(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericBlockSparseAttentionGrad)
    .Tiling(TilingGenericBlockSparseAttentionGrad)
    .TilingParse<GenericBlockSparseAttentionGradCompileInfo>(TilingPrepareGenericBlockSparseAttentionGrad);

} // namespace gsag
} // namespace optiling

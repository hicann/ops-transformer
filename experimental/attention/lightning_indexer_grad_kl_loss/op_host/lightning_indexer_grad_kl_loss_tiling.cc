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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling.cc
 * \brief
 */

#include <graph/utils/type_utils.h>
// #include "../../utils/inc/error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_lightning_indexer_grad_kl_loss_template_tiling_key.h"
#include "../op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "lightning_indexer_grad_kl_loss_tiling_general.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus TilingSparseLightningIndexerGradKLLoss(gert::TilingContext *context)
{
    SparseLightningIndexerGradKLLossTilingBase sligKLLossTiling(context);
    auto ret = sligKLLossTiling.DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSparseLightningIndexerGradKLLoss(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "Start registering tiling.");
    auto compileInfoPtr = context->GetCompiledInfo<SparseLightningIndexerGradKLLossCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LightningIndexerGradKLLoss)
    .Tiling(TilingSparseLightningIndexerGradKLLoss)
    .TilingParse<SparseLightningIndexerGradKLLossCompileInfo>(TilingPrepareForSparseLightningIndexerGradKLLoss);
} // namespace optiling

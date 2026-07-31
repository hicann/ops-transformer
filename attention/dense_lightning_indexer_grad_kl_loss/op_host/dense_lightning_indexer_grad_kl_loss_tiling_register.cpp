/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_grad_kl_loss_tiling_register.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling_data.h"
#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling_key.h"
#include "dense_lightning_indexer_grad_kl_loss_tiling.h"

namespace optiling {

// tiling 分发入口
static ge::graphStatus DenseLightningIndexerGradKLLossTilingFunc(gert::TilingContext* context)
{
    DenseLightningIndexerGradKLLossTilingBase dligKLLossTiling(context);
    return dligKLLossTiling.DoTiling();
}

static ge::graphStatus TilingParseForDenseLightningIndexerGradKLLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start parsing tiling for DenseLightningIndexerGradKLLoss.");
    auto compileInfoPtr = context->GetCompiledInfo<DenseLightningIndexerGradKLLossCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(DenseLightningIndexerGradKLLoss)
    .Tiling(DenseLightningIndexerGradKLLossTilingFunc)
    .TilingParse<DenseLightningIndexerGradKLLossCompileInfo>(TilingParseForDenseLightningIndexerGradKLLoss);
} // namespace optiling

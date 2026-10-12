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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling_common.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;

namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

ge::graphStatus TilingSparseLightningIndexerGradKLLoss(gert::TilingContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto sligPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (sligPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
    } else {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
    }
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSparseLightningIndexerGradKLLoss(gert::TilingParseContext* context)
{
    OP_LOGW(context, "Start registering tiling.");
    auto compileInfoPtr = context->GetCompiledInfo<SparseLightningIndexerGradKLLossCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseLightningIndexerGradKLLoss)
    .Tiling(TilingSparseLightningIndexerGradKLLoss)
    .TilingParse<SparseLightningIndexerGradKLLossCompileInfo>(TilingPrepareForSparseLightningIndexerGradKLLoss);
} // namespace optiling

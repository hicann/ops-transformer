/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file dense_lightning_indexer_kl_loss_grad_tiling.cpp
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
#include "dense_lightning_indexer_kl_loss_grad_tiling_common.h"
#include "./arch35/dense_lightning_indexer_kl_loss_grad_tiling_general_regbase.h"
using std::map;
using std::pair;
using std::string;

using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";

ge::graphStatus TilingDenseLightningIndexerKLLossGrad(gert::TilingContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto dlikgPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForDenseLightningIndexerKLLossGrad(gert::TilingParseContext *context)
{
    OP_LOGW(context, "Start registering tiling.");
    auto compileInfoPtr = context->GetCompiledInfo<DenseLightningIndexerKLLossGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DenseLightningIndexerKLLossGrad)
    .Tiling(TilingDenseLightningIndexerKLLossGrad)
    .TilingParse<DenseLightningIndexerKLLossGradCompileInfo>(TilingPrepareForDenseLightningIndexerKLLossGrad);
} // namespace optiling

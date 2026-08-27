/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update_tiling.h"

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace {
constexpr const char *OP_NAME = "BlockAttnResUpdate";
// Platform data is read directly from TilingContext, so the parse stage carries no compile-time payload.
struct TilingParsePlaceholder {};
} // namespace

ge::graphStatus Tiling4BlockAttnResUpdate(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "Context is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetNodeType() == nullptr, OP_LOGE(OP_NAME, "Node type is nullptr."), return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4BlockAttnResUpdate(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "Context is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttnResUpdate)
    .Tiling(Tiling4BlockAttnResUpdate)
    .TilingParse<TilingParsePlaceholder>(TilingPrepare4BlockAttnResUpdate);

} // namespace optiling

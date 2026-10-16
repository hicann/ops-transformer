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
 * \file rotary_position_embedding.cc
 * \brief
 */
#include <vector>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "partial_rotary_mul_quant_tiling.h"
namespace optiling {

ge::graphStatus Tiling4PartialRotaryMulQuant(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4PartialRotaryMulQuant running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForPartialRotaryMulQuant(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepareForPartialRotaryMulQuant context success");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PartialRotaryMulQuant)
    .Tiling(Tiling4PartialRotaryMulQuant)
    .TilingParse<QuantRotaryPositionEmbeddingCompileInfo>(TilingPrepareForPartialRotaryMulQuant);
} // namespace optiling

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
 * \file masked_causal_conv1d_backward.cpp
 * \brief Main tiling entry for MaskedCausalConv1dBackward
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "masked_causal_conv1d_backward_tiling_arch35.h"

namespace optiling {

static ge::graphStatus TilingMaskedCausalConv1dBackward(gert::TilingContext* context)
{
    MaskedCausalConv1dBackwardTilingArch35 tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMaskedCausalConv1dBackward(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MaskedCausalConv1dBackward", "context is null"), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Register main tiling entry
IMPL_OP_OPTILING(MaskedCausalConv1dBackward)
    .Tiling(TilingMaskedCausalConv1dBackward)
    .TilingParse<MaskedCausalConv1dBackwardArch35CompileInfo>(TilingPrepareForMaskedCausalConv1dBackward);

} // namespace optiling

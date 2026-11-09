/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "text_gated_residual_grad_tiling.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "kernel_tiling/kernel_tiling.h"

using Ops::Transformer::OpTiling::TilingRegistryNew;
using Ops::Transformer::OpTiling::TilingRegistryArch;

namespace optiling {

static ge::graphStatus Tiling4TextGatedResidualGrad(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_2201) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    OP_LOGE(context,
            "TextGatedResidualGrad: only DAV_3510 (ascend950) and DAV_2201 (ascend910b/910_93) are supported.");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepareForTextGatedResidualGrad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TextGatedResidualGrad)
    .Tiling(Tiling4TextGatedResidualGrad)
    .TilingParse<TextGatedResidualGradCompileInfo>(TilingPrepareForTextGatedResidualGrad);

} // namespace optiling

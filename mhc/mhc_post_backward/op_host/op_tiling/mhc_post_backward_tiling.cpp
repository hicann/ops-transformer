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
 * \file mhc_post_backward_tiling.cpp
 * \brief
 */

#include "mhc_post_backward_tiling.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "kernel_tiling/kernel_tiling.h"

using Ops::Transformer::OpTiling::TilingRegistryNew;
using Ops::Transformer::OpTiling::TilingRegistryArch;

namespace optiling {
static ge::graphStatus Tiling4MhcPostBackward(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMhcPostBackward(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPostBackward)
    .Tiling(Tiling4MhcPostBackward)
    .TilingParse<MhcPostBackwardCompileInfo>(TilingPrepareForMhcPostBackward);

} // namespace optiling

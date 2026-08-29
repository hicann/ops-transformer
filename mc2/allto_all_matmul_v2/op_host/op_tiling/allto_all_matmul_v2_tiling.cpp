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
 * \file allto_all_matmul_v2_tiling.cpp
 * \brief Tiling implementation for AlltoAllMatmulV2
 */

#include "arch35/allto_all_matmul_v2_tiling_base.h"

#include <cstdio>
#include <string>
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

namespace MC2Tiling {

using Ops::Transformer::OpTiling::TilingRegistryArch;

static ge::graphStatus AlltoAllMatmulV2TilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "[platform] platformInfo is nullptr");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "[platform] AlltoAllMatmulV2 only supports DAV_3510, current arch is not supported");
        return ge::GRAPH_FAILED;
    }
    return TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmulV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulV2CompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmulV2)
    .Tiling(AlltoAllMatmulV2TilingFunc)
    .TilingParse<AlltoAllMatmulV2CompileInfo>(TilingParseForAlltoAllMatmulV2);

} // namespace MC2Tiling

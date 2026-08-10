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
 * \file moe_re_routing_v2_tiling_base.cpp
 * \brief
 */
#include "moe_re_routing_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"

namespace optiling {

ge::graphStatus Tiling4MoeReRoutingV2(gert::TilingContext *context)
{
    OP_LOGD(context, "TilingForMoeReRoutingV2 running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4MoeReRoutingV2(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4MoeReRoutingV2 running.");
    auto compileInfo = context->GetCompiledInfo<MoeReRoutingCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "compileInfo->coreNum", std::to_string(compileInfo->coreNum).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "compileInfo->ubSize", std::to_string(compileInfo->ubSize).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4MoeReRoutingV2 success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeReRoutingV2)
    .Tiling(Tiling4MoeReRoutingV2)
    .TilingParse<MoeReRoutingCompileInfo>(TilingPrepare4MoeReRoutingV2);

}  // namespace optiling

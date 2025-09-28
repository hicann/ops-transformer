/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_re_routing_tiling.cpp
 * \brief
 */
#include "moe_re_routing_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"

namespace optiling {

ge::graphStatus Tiling4MoeReRouting(gert::TilingContext *context)
{
    OP_LOGD(context, "TilingForMoeReRouting running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4MoeReRouting(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4MoeReRouting running.");
    auto compileInfo = context->GetCompiledInfo<MoeReRoutingCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(context->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0,
        OP_LOGE(context->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4MoeReRouting success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeReRouting)
    .Tiling(Tiling4MoeReRouting)
    .TilingParse<MoeReRoutingCompileInfo>(TilingPrepare4MoeReRouting);

}  // namespace optiling
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
 * \file moe_init_routing_v3_tiling_base.cpp
 * \brief
 */
#include "moe_init_routing_v3_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

using Ops::Transformer::OpTiling::TilingRegistry;

namespace optiling {
static ge::graphStatus TilingForMoeInitRoutingV3(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeInitRountingV3(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepareForMoeInitRountingV3: %s", context->GetNodeName());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingV3)
    .Tiling(TilingForMoeInitRoutingV3)
    .TilingParse<MoeInitRoutingV3CompileInfo>(TilingPrepareForMoeInitRountingV3);
} // namespace optiling
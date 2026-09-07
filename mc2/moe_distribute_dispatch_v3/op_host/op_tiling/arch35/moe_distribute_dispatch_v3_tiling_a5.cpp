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
 * \file moe_distribute_dispatch_v3_tiling_a5.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v3_tiling_a5.h"
#include "../../../../moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.h"

namespace optiling {
ge::graphStatus MoeDistributeDispatchV3TilingFuncA5::MoeDistributeDispatchA3TilingFuncImplPublic(
    gert::TilingContext *context, DispatchV2Config &config)
{
    MoeDistributeDispatchV2TilingFuncA5 impl;
    return impl.MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV3TilingFuncImplA5(gert::TilingContext *context)
{
    MoeDistributeDispatchV3TilingFuncA5 impl;
    return impl.MoeDistributeDispatchV3TilingFunc(context);
}
IMPL_OP_OPTILING(MoeDistributeDispatchV3)
    .Tiling(MoeDistributeDispatchV3TilingFuncImplA5)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV3);
} // namespace optiling

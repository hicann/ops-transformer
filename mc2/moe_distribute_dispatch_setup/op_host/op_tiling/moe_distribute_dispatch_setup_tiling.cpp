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
 * \file moe_distribute_dispatch_setup_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "mc2_log.h"

using namespace Ops::Transformer::OpTiling;
using namespace ge;

namespace optiling {

ge::graphStatus MoeDistributeDispatchSetupTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingParseForMoeDistributeDispatchSetup(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeDispatchSetupCompileInfo {};
IMPL_OP_OPTILING(MoeDistributeDispatchSetup)
    .Tiling(MoeDistributeDispatchSetupTilingFunc)
    .TilingParse<MoeDistributeDispatchSetupCompileInfo>(TilingParseForMoeDistributeDispatchSetup);
} // namespace optiling

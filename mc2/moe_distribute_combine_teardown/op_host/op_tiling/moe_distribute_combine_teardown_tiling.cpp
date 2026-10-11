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
 * \file moe_distribute_combine_teardown_tiling.cpp
 * \brief host侧tiling实现
 */

#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "mc2_log.h"

using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
using namespace ge;

namespace MC2Tiling {

ge::graphStatus MoeDistributeCombineTeardownTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(
        context == nullptr,
        OP_LOGE("MoeDistributeCombineTeardown", "failed to get tiling context in moe_distribute_combine_teardown."),
        return ge::GRAPH_FAILED);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct MoeDistributeCombineTeardownCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineTeardown(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineTeardown)
    .Tiling(MoeDistributeCombineTeardownTilingFunc)
    .TilingParse<MoeDistributeCombineTeardownCompileInfo>(TilingParseForMoeDistributeCombineTeardown);
} // namespace MC2Tiling

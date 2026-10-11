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
 * \file allto_allv_grouped_mat_mul_tiling_a5.cpp
 * \brief
 */
#include "allto_allv_grouped_mat_mul_tiling_a5.h"

namespace {
static const char* A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";
}

namespace optiling {
std::vector<int64_t> AlltoAllvGmmTilingA5::GetEpWorldSizeOptional() const
{
    return {2, 4, 8, 16, 32, 64}; // A5限制epWorldSize为{2，4，8，16，32，64}
}

bool AlltoAllvGmmTilingA5::NeedToCheckCounts() const
{
    return false;
}

ge::graphStatus AlltoAllvGmmTilingFuncA5::AlltoAllvGmmOpTilingFunc(gert::TilingContext* context)
{
    AlltoAllvGmmTilingA5 tiling(context);
    OP_TILING_CHECK(tiling.Init(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM tiling init failed."),
                    return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

ge::graphStatus AlltoAllvGmmTilingStructA5::DoOpTiling()
{
    AlltoAllvGmmTilingFuncA5 funcA5;
    return funcA5.AlltoAllvGmmOpTilingFunc(context_);
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmTilingStructA5, 0);
} // namespace optiling

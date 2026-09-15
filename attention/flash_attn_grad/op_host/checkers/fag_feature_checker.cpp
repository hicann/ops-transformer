/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fag_feature_checker.h"

#include "log/log.h"

namespace optiling {

// dq/dk/dv 目前用 fp32 atomicAdd 累加，跨核累加顺序不固定，因此开启确定性
// 计算时结果并不确定。这不是参数非法，而是特性未支持 —— 明确拒绝，不静默放行，
// 避免用户以为拿到了确定性结果。待确定性模板落地后（独立 tilingkey 分支、
// 前缀表放 AICPU），把这里改成正例即可。
ge::graphStatus FagDeterministicChecker::Check(FagCheckCtx &ctx)
{
    OP_CHECK_IF(ctx.context->GetDeterministic() == 1,
                OP_LOGE(ctx.opName, "FlashAttnGrad does not support deterministic computation yet "
                                    "(dq/dk/dv are accumulated with fp32 atomicAdd). Please disable the deterministic "
                                    "switch, or use FlashAttentionScoreGrad instead."),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

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

#include <string>

#include "log/log.h"

namespace optiling {

// GS1S2 走核上闭式分核；BN2 单块一头一核已确定，放行现网。TND 第一期仍拒。
ge::graphStatus FagDeterministicChecker::Check(FagCheckCtx& ctx)
{
    if (ctx.context->GetDeterministic() != 1) {
        return ge::GRAPH_SUCCESS;
    }
    const auto attrs = ctx.context->GetAttrs();
    if (attrs != nullptr) {
        const auto layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
        if (layoutQPtr != nullptr && std::string(layoutQPtr) == "TND") {
            OP_LOGE(ctx.opName, "FlashAttnGrad deterministic computation does not support TND yet.");
            return ge::GRAPH_PARAM_INVALID;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

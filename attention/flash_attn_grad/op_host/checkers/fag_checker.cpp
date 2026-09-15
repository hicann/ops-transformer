/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fag_checker.h"

#include "fag_attr_checker.h"
#include "fag_feature_checker.h"
#include "fag_input_checker.h"
#include "fag_shape_checker.h"
#include "log/log.h"

namespace optiling {

// 顺序是有约束的，不能随意调：
//   Deterministic 最前，未支持的特性不必再查参数细节；
//   InputExistence 必须在所有解引用 shape 的 checker 之前；
//   AttrRange 是唯一读属性的地方，Mask / Layout / Shape 都依赖它填好的 ctx。
FagChecker::FagChecker()
{
    checkers_.emplace_back(std::make_unique<FagDeterministicChecker>());
    checkers_.emplace_back(std::make_unique<FagInputExistenceChecker>());
    checkers_.emplace_back(std::make_unique<FagDtypeChecker>());
    checkers_.emplace_back(std::make_unique<FagAttrRangeChecker>());
    checkers_.emplace_back(std::make_unique<FagMaskChecker>());
    checkers_.emplace_back(std::make_unique<FagLayoutChecker>());
    checkers_.emplace_back(std::make_unique<FagShapeChecker>());
}

ge::graphStatus FagChecker::Process(gert::TilingContext *context)
{
    FagCheckCtx ctx;
    ctx.context = context;
    ctx.opName = context->GetNodeName();

    for (const auto &checker : checkers_) {
        const ge::graphStatus ret = checker->Check(ctx);
        // 具体原因由 checker 自己打过，这里只补一条定位信息。
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx.opName, "%s check failed.", checker->Name()), return ret);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

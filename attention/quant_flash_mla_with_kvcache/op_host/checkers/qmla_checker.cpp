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
 * \file qmla_checker.cpp
 * \brief QmlaChecker implementation using Composite Pattern
 */

#include "log/log.h"
#include "../qmla_tiling_info.h"
#include "qmla_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

// RegisterCheckers - 注册所有 checker
// 新增 checker 时在此方法中添加: checkers_.push_back(std::make_unique<XxxChecker>());
void QmlaChecker::RegisterCheckers()
{
    checkers_.push_back(std::make_unique<CommonChecker>());
    checkers_.push_back(std::make_unique<QuantChecker>());
    checkers_.push_back(std::make_unique<MaskChecker>());
    checkers_.push_back(std::make_unique<SeqLenChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<SoftmaxLSEChecker>());
}

ge::graphStatus QmlaChecker::Init(const QmlaTilingInfo& qmlaInfo)
{
    (void)qmlaInfo;
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::RunCheck(const CheckMethod& method, const QmlaTilingInfo& qmlaInfo)
{
    for (const auto& checker : checkers_) {
        if (ge::GRAPH_SUCCESS != method(checker.get(), qmlaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::Process(const QmlaTilingInfo& qmlaInfo)
{
    // 按文档参数组校验顺序执行:
    //   1. CheckSinglePara — 每个参数的 dtype/format/shape/dim 单参数校验
    //   2. CheckParaExistence — 必选/可选参数存在性校验
    //   3. CheckFeature — 特性交叉校验 (轴校验, layout关系, 非连续等)
    //   4. CheckMultiPara — 一致性校验 (dtype一致, shape匹配等)
    if (ge::GRAPH_SUCCESS !=
        RunCheck([](QmlaBaseChecker* c, const QmlaTilingInfo& info) { return c->CheckSinglePara(info); }, qmlaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS !=
        RunCheck([](QmlaBaseChecker* c, const QmlaTilingInfo& info) { return c->CheckParaExistence(info); },
                 qmlaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS !=
        RunCheck([](QmlaBaseChecker* c, const QmlaTilingInfo& info) { return c->CheckFeature(info); }, qmlaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS !=
        RunCheck([](QmlaBaseChecker* c, const QmlaTilingInfo& info) { return c->CheckMultiPara(info); }, qmlaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qfa_checker.cpp
 * \brief QfaChecker implementation using Composite Pattern
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../quant_flash_attn_tiling_info.h"
#include "qfa_checker.h"
#include "common_checker.h"
#include "quant_checker.h"
#include "mask_checker.h"
#include "seq_len_checker.h"
#include "paged_attention_checker.h"
#include "sinks_checker.h"
#include "softmax_lse_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;

// RegisterCheckers - 注册所有 checker
// 新增 checker 时在此方法中添加: checkers_.push_back(std::make_unique<XxxChecker>());
void QfaChecker::RegisterCheckers()
{
    checkers_.push_back(std::make_unique<CommonChecker>());
    checkers_.push_back(std::make_unique<QuantChecker>());
    checkers_.push_back(std::make_unique<MaskChecker>());
    checkers_.push_back(std::make_unique<SeqLenChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<SinksChecker>());
    checkers_.push_back(std::make_unique<SoftmaxLSEChecker>());
}

ge::graphStatus QfaChecker::Init(const QuantFlashAttnTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::RunCheck(const CheckMethod &method, const QuantFlashAttnTilingInfo &qfaInfo)
{
    for (const auto &checker : checkers_) {
        if (ge::GRAPH_SUCCESS != method(checker.get(), qfaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    return RunCheck([](QfaBaseChecker *c, const QuantFlashAttnTilingInfo &info) { return c->CheckSinglePara(info); },
                    qfaInfo);
}

ge::graphStatus QfaChecker::CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    return RunCheck([](QfaBaseChecker *c, const QuantFlashAttnTilingInfo &info) { return c->CheckParaExistence(info); },
                    qfaInfo);
}

ge::graphStatus QfaChecker::CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
{
    return RunCheck([](QfaBaseChecker *c, const QuantFlashAttnTilingInfo &info) { return c->CheckFeature(info); },
                    qfaInfo);
}

ge::graphStatus QfaChecker::CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    return RunCheck([](QfaBaseChecker *c, const QuantFlashAttnTilingInfo &info) { return c->CheckMultiPara(info); },
                    qfaInfo);
}

ge::graphStatus QfaChecker::Process(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSinglePara(qfaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckParaExistence(qfaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckFeature(qfaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckMultiPara(qfaInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling

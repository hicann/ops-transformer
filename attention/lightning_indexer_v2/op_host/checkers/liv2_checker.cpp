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
 * \file liv2_checker.cpp
 * \brief Implements the Lightning Indexer V2 checker entry point and type checks.
 */

#include "liv2_checker.h"

#include "checker_runner_lightning_indexer_v2.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {
namespace lightning_indexer_v2_checker {
namespace {
class LightningIndexerV2TypeChecker : public LightningIndexerV2BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const override
    {
        if (CheckDtype(info, info.query, "q", {ge::DT_FLOAT16, ge::DT_BF16}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.key, "k", {ge::DT_FLOAT16, ge::DT_BF16}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.weights, "w", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.sparseValues, "sparse_values", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (info.query.desc == nullptr || info.key.desc == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const ge::DataType qType = info.query.desc->GetDataType();
        const ge::DataType kType = info.key.desc->GetDataType();
        if (qType != kType) {
            const std::string actual = Ops::Base::ToString(qType) + " and " + Ops::Base::ToString(kType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(info.opName, "q and k", actual.c_str(),
                                                   "Q and k must have the same dtype");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
};
} // namespace

ge::graphStatus LIV2Checker::Process() const
{
    CheckerRunner runner;
    RegisterCommonCheckers(runner);
    runner.Add(std::make_unique<LightningIndexerV2TypeChecker>());
    return runner.Process(info_);
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

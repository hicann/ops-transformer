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
 * \file checker_runner.cpp
 * \brief Implements phased execution and registration of Lightning Indexer V2 checkers.
 */

#include "checker_runner_lightning_indexer_v2.h"

#include <utility>

#include "common_checker_lightning_indexer_v2.h"
#include "compression_checker_lightning_indexer_v2.h"
#include "log/log.h"
#include "paged_attention_checker_lightning_indexer_v2.h"
#include "seq_len_checker_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {
namespace {
constexpr char CHECKER_RUNNER_NAME[] = "LightningIndexerV2Checker";
}

void CheckerRunner::Add(std::unique_ptr<LightningIndexerV2BaseChecker> checker)
{
    checkers_.push_back(std::move(checker));
}

ge::graphStatus CheckerRunner::Run(CheckMethod method, const LightningIndexerV2CheckerInfo &info) const
{
    for (const auto &checker : checkers_) {
        const auto *checkerPtr = checker.get();
        if (checkerPtr == nullptr) {
            const char *opName = info.opName == nullptr ? CHECKER_RUNNER_NAME : info.opName;
            OP_LOGE(opName, "Checker must not be null.");
            return ge::GRAPH_FAILED;
        }
        if ((checkerPtr->*method)(info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckerRunner::Process(const LightningIndexerV2CheckerInfo &info) const
{
    if (Run(&LightningIndexerV2BaseChecker::CheckSinglePara, info) != ge::GRAPH_SUCCESS ||
        Run(&LightningIndexerV2BaseChecker::CheckParaExistence, info) != ge::GRAPH_SUCCESS ||
        Run(&LightningIndexerV2BaseChecker::CheckFeature, info) != ge::GRAPH_SUCCESS ||
        Run(&LightningIndexerV2BaseChecker::CheckMultiPara, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void RegisterCommonCheckers(CheckerRunner &runner)
{
    runner.Add(std::make_unique<CommonChecker>());
    runner.Add(std::make_unique<SeqLenChecker>());
    runner.Add(std::make_unique<CompressionChecker>());
    runner.Add(std::make_unique<PagedAttentionChecker>());
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

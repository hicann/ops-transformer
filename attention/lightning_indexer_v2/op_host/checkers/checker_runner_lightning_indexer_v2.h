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
 * \file checker_runner_lightning_indexer_v2.h
 * \brief Declares phased execution and registration of Lightning Indexer V2 checkers.
 */

#ifndef CHECKER_RUNNER_LIGHTNING_INDEXER_V2_H
#define CHECKER_RUNNER_LIGHTNING_INDEXER_V2_H

#include <memory>
#include <vector>

#include "base_checker_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

class CheckerRunner {
public:
    CheckerRunner() = default;
    ~CheckerRunner() = default;

    void Add(std::unique_ptr<LightningIndexerV2BaseChecker> checker);
    ge::graphStatus Process(const LightningIndexerV2CheckerInfo &info) const;

private:
    using CheckMethod = ge::graphStatus (LightningIndexerV2BaseChecker::*)(const LightningIndexerV2CheckerInfo &) const;

    ge::graphStatus Run(CheckMethod method, const LightningIndexerV2CheckerInfo &info) const;
    std::vector<std::unique_ptr<LightningIndexerV2BaseChecker>> checkers_;
};

void RegisterCommonCheckers(CheckerRunner &runner);

} // namespace lightning_indexer_v2_checker
} // namespace optiling

#endif // CHECKER_RUNNER_LIGHTNING_INDEXER_V2_H

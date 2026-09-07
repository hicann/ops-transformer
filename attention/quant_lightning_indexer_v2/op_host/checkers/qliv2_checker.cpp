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
 * \file qliv2_checker.cpp
 * \brief Implements the Quant Lightning Indexer V2 checker entry point.
 */

#include "qliv2_checker.h"

#include "qliv2_variant_checker.h"
#include "../../../lightning_indexer_v2/op_host/checkers/checker_runner_lightning_indexer_v2.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

ge::graphStatus QLIV2Checker::Process() const
{
    CheckerRunner runner;
    RegisterCommonCheckers(runner);
    runner.Add(std::make_unique<QLIV2VariantChecker>());
    return runner.Process(info_);
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

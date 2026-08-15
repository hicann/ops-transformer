/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mixed_quant_sparse_flash_mla_checker.h"
#include <memory>
#include "mixed_quant_variant_checker.h"
#include "../../../sparse_flash_mla/op_host/checkers/checker_adapter.h"
#include "../../../sparse_flash_mla/op_host/checkers/checker_runner.h"

namespace optiling {
namespace {
using sparse_mla_checker::CheckContext;
CheckContext BuildContext(const MQSMLATilingInfo &info)
{
    CheckContext context;
    sparse_mla_checker::PopulateCommonContext(context, info);
    context.variant = sparse_mla_checker::OperatorVariant::MIXED_QUANT;
    context.qHeadDim = info.qkHeadDim;
    context.quantMode = info.quantMode;
    context.ropeHeadDim = info.ropeHeadDim;
    context.oriKvStrides = info.oriKvStrides;
    context.cmpKvStrides = info.cmpKvStrides;
    return context;
}
} // namespace

ge::graphStatus MixedQuantSparseFlashMlaChecker::Process() const
{
    sparse_mla_checker::CheckerRunner runner;
    sparse_mla_checker::RegisterCommonCheckers(runner);
    runner.Add(std::make_unique<sparse_mla_checker::MixedQuantVariantChecker>());
    return runner.Process(BuildContext(info_));
}

} // namespace optiling

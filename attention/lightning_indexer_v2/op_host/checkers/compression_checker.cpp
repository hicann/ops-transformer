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
 * \file compression_checker.cpp
 * \brief Implements compression parameter checks for Lightning Indexer V2 operators.
 */

#include "compression_checker_lightning_indexer_v2.h"

#include "log/log.h"

namespace optiling {
namespace lightning_indexer_v2_checker {
namespace {
constexpr uint32_t CMP_RESIDUAL_RANK = 1U;
} // namespace

ge::graphStatus CompressionChecker::CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const
{
    const bool needResidual = info.cmpRatio != 1 && info.maskMode != 0;
    if (needResidual != info.cmpResidualK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "cmp_residual_k",
            "Cmp_residual_k must be provided if and only if cmp_ratio is not 1 and mask_mode is not 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressionChecker::CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckRank(info, info.cmpResidualK, "cmp_residual_k", CMP_RESIDUAL_RANK) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.cmpResidualK, "cmp_residual_k", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressionChecker::CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckElementCount(info, info.cmpResidualK, "cmp_residual_k", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

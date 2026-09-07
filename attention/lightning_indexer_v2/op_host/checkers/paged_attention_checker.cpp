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
 * \file paged_attention_checker.cpp
 * \brief Implements Paged Attention parameter checks for Lightning Indexer V2 operators.
 */

#include "paged_attention_checker_lightning_indexer_v2.h"

#include <limits>

#include "log/log.h"

namespace optiling {
namespace lightning_indexer_v2_checker {
namespace {
constexpr int64_t BLOCK_SIZE_FACTOR = 16;
constexpr int64_t BLOCK_SIZE_LIMIT = 1024;
constexpr uint32_t BLOCK_TABLE_RANK = 2U;
} // namespace

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const
{
    const bool paged = info.layoutK == CheckerLayout::PA_BBND;
    if (paged != info.blockTable.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "block_table",
                                                 "Block_table must be provided if and only if layout_k is PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if (paged && !info.sequsedK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "seqused_k",
                                                 "Seqused_k must be provided when layout_k is PA_BBND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.layoutK != CheckerLayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckRank(info, info.blockTable, "block_table", BLOCK_TABLE_RANK) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.blockTable, "block_table", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.blockSize < BLOCK_SIZE_FACTOR || info.blockSize > BLOCK_SIZE_LIMIT ||
        info.blockSize % BLOCK_SIZE_FACTOR != 0) {
        const std::string actual = std::to_string(info.blockSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "block_size", actual.c_str(),
                                              "Block_size must be a multiple of 16 in [16, 1024]");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *keyShape = info.key.GetShape();
    if (keyShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (keyShape->GetDim(0) <= 0) {
        const std::string actual = ShapeToString(info.key);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "k", actual.c_str(), "Block_num must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckContiguous(const LightningIndexerV2CheckerInfo &info,
                                                       const CheckerTensor &tensor, const char *name,
                                                       uint32_t firstCheckedAxis) const
{
    const gert::Shape *shape = tensor.GetShape();
    if (!tensor.IsPresent() || tensor.stride == nullptr || tensor.stride->GetDimNum() == 0 || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t expected = 1;
    for (int64_t i = static_cast<int64_t>(shape->GetDimNum()) - 1; i >= static_cast<int64_t>(firstCheckedAxis); --i) {
        if (static_cast<uint32_t>(i) < tensor.stride->GetDimNum() && tensor.stride->GetStride(i) != expected) {
            const std::string actual = std::to_string(tensor.stride->GetStride(i));
            const std::string reasonPrefix = firstCheckedAxis == 1U ?
                                                 "Only axis 0 may be non-contiguous in PA_BBND; expected stride " :
                                                 "The tensor must be contiguous; expected stride ";
            const std::string reason = reasonPrefix + std::to_string(expected) + " at axis " + std::to_string(i);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        const int64_t dim = shape->GetDim(i);
        if (dim < 0 || (dim != 0 && expected > std::numeric_limits<int64_t>::max() / dim)) {
            const std::string actual = ShapeToString(tensor);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                info.opName, name, actual.c_str(),
                "Shape dimensions must be non-negative and their product must fit in int64_t");
            return ge::GRAPH_FAILED;
        }
        expected *= dim;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.layoutK == CheckerLayout::PA_BBND) {
        const gert::Shape *blockShape = info.blockTable.GetShape();
        if (blockShape->GetDim(0) != info.batch) {
            const std::string actual = ShapeToString(info.blockTable);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "block_table", actual.c_str(),
                                                  "Block_table dim 0 must equal the query batch size");
            return ge::GRAPH_FAILED;
        }
    }
    const uint32_t firstAxis = info.layoutK == CheckerLayout::PA_BBND ? 1U : 0U;
    if (CheckContiguous(info, info.key, "k", firstAxis) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.quantized && info.quantMode != QUANT_MODE_HIFLOAT8 &&
        CheckContiguous(info, info.keyDescale, "k_descale", firstAxis) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

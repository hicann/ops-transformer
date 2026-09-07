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
 * \file common_checker.cpp
 * \brief Implements common shape, type, attribute, and feature checks for Lightning Indexer V2 operators.
 */

#include "common_checker_lightning_indexer_v2.h"

#include <limits>
#include <utility>
#include "log/log.h"

namespace optiling {
namespace lightning_indexer_v2_checker {
namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t HEAD_GROUP_LIMIT = 64;
constexpr int64_t TOPK_8K = 8192;
constexpr int64_t TOPK_MIN = 1;
constexpr int64_t METADATA_SIZE = 1024;
constexpr int64_t MAX_SEQLEN_Q = std::numeric_limits<int32_t>::max();
constexpr int64_t MAX_SEQLEN_Q_DEFAULT = -1;
constexpr int64_t MASK_MODE_NONE = 0;
constexpr int64_t MASK_MODE_COMPRESS = 3;
constexpr int64_t CMP_RATIO_MIN = 1;
constexpr int64_t CMP_RATIO_MAX = 128;
constexpr int64_t RETURN_VALUE_INDICES = 0;
constexpr int64_t RETURN_VALUE_WITH_VALUES = 1;
constexpr int64_t K_HEAD_COUNT = 1;
constexpr int64_t Q_HEAD_COUNT_MIN = 1;
constexpr uint32_t BLOCK_TABLE_RANK = 2U;
constexpr uint32_t BSND_RANK = 4U;
constexpr uint32_t TND_RANK = 3U;
constexpr uint32_t BSND_BATCH_AXIS = 0U;
constexpr uint32_t BSND_SEQ_AXIS = 1U;
constexpr uint32_t BSND_HEAD_AXIS = 2U;
constexpr uint32_t BSND_HEAD_DIM_AXIS = 3U;
constexpr uint32_t TND_TOKEN_AXIS = 0U;
constexpr uint32_t TND_HEAD_AXIS = 1U;
constexpr uint32_t TND_HEAD_DIM_AXIS = 2U;
constexpr uint32_t BLOCK_TABLE_BLOCK_COUNT_AXIS = 1U;

bool SameDim(const gert::Shape &lhs, uint32_t lhsIndex, const gert::Shape &rhs, uint32_t rhsIndex)
{
    return lhs.GetDim(lhsIndex) == rhs.GetDim(rhsIndex);
}

ge::graphStatus LogShapeMismatch(const LightningIndexerV2CheckerInfo &info, const char *names,
                                 const std::string &actual, const char *reason)
{
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(info.opName, names, actual.c_str(), reason);
    return ge::GRAPH_FAILED;
}

ge::graphStatus CheckUint32Range(const LightningIndexerV2CheckerInfo &info, const char *name, int64_t value)
{
    if (value >= 0 && value <= std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string actual = std::to_string(value);
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, name, actual.c_str(), "Value must fit in uint32_t");
    return ge::GRAPH_FAILED;
}
} // namespace

ge::graphStatus CommonChecker::CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const
{
    const std::pair<const CheckerTensor *, const char *> required[] = {
        {&info.query, "q"}, {&info.key, "k"}, {&info.weights, "w"}, {&info.sparseIndices, "sparse_indices"}};
    for (const auto &item : required) {
        if (!item.first->IsPresent() || item.first->desc == nullptr || item.first->GetShape() == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, item.second,
                                                     "Required tensor, descriptor and shape must be provided");
            return ge::GRAPH_FAILED;
        }
    }
    if (info.returnValue == RETURN_VALUE_WITH_VALUES &&
        (!info.sparseValues.IsPresent() || info.sparseValues.desc == nullptr ||
         info.sparseValues.GetShape() == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "sparse_values",
                                                 "Sparse_values must be provided when return_value is 1");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQText.empty() || info.layoutKText.empty()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "layout_q and layout_k",
                                                 "Layout_q and layout_k must be provided");
        return ge::GRAPH_FAILED;
    }
    if (!info.metadata.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "metadata", "Metadata must be provided");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckTensorBasics(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckDtype(info, info.sparseIndices, "sparse_indices", {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckDtype(info, info.outputIdxOffset, "output_idx_offset", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAttrs(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckLayouts(info) != ge::GRAPH_SUCCESS || CheckTopk(info) != ge::GRAPH_SUCCESS ||
        CheckCompressionAttrs(info) != ge::GRAPH_SUCCESS || CheckReturnValue(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckLayouts(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.layoutQ != CheckerLayout::BSND && info.layoutQ != CheckerLayout::TND) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "layout_q", info.layoutQText.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK != CheckerLayout::BSND && info.layoutK != CheckerLayout::TND &&
        info.layoutK != CheckerLayout::PA_BBND) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "layout_k", info.layoutKText.c_str(), "BSND, TND or PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK != CheckerLayout::PA_BBND && info.layoutQText != info.layoutKText) {
        const std::string actual = info.layoutQText + " and " + info.layoutKText;
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "layout_q and layout_k", actual.c_str(),
                                               "Layout_q and layout_k must be equal outside PagedAttention");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckTopk(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.topk < TOPK_MIN || info.topk > TOPK_8K) {
        const std::string actual = std::to_string(info.topk);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "topk", actual.c_str(), "Topk must be in [1, 8192]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckCompressionAttrs(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.maskMode != MASK_MODE_NONE && info.maskMode != MASK_MODE_COMPRESS) {
        const std::string actual = std::to_string(info.maskMode);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "mask_mode", actual.c_str(), "0 or 3");
        return ge::GRAPH_FAILED;
    }
    if (info.cmpRatio < CMP_RATIO_MIN || info.cmpRatio > CMP_RATIO_MAX) {
        const std::string actual = std::to_string(info.cmpRatio);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "cmp_ratio", actual.c_str(),
                                              "Cmp_ratio must be in [1, 128]");
        return ge::GRAPH_FAILED;
    }
    if (info.maxSeqlenQ < MAX_SEQLEN_Q_DEFAULT) {
        const std::string actual = std::to_string(info.maxSeqlenQ);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "max_seqlen_q", actual.c_str(), ">= -1");
        return ge::GRAPH_FAILED;
    }
    if (info.maxSeqlenQ > MAX_SEQLEN_Q) {
        const std::string actual = std::to_string(info.maxSeqlenQ);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "max_seqlen_q", actual.c_str(), "<= 2147483647");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckReturnValue(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.returnValue != RETURN_VALUE_INDICES && info.returnValue != RETURN_VALUE_WITH_VALUES) {
        const std::string actual = std::to_string(info.returnValue);
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "return_value", actual.c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckTensorBasics(info) != ge::GRAPH_SUCCESS || CheckAttrs(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t qRank = info.layoutQ == CheckerLayout::BSND ? BSND_RANK : TND_RANK;
    const uint32_t kRank = info.layoutK == CheckerLayout::TND ? TND_RANK : BSND_RANK;
    const std::string qCondition = "layout_q is " + info.layoutQText;
    const std::string kCondition = "layout_k is " + info.layoutKText;
    if (CheckRank(info, info.query, "q", qRank, qCondition) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.key, "k", kRank, kCondition) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.weights, "w", qRank - 1U, qCondition) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.sparseIndices, "sparse_indices", qRank, qCondition) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.returnValue == RETURN_VALUE_WITH_VALUES &&
        CheckRank(info, info.sparseValues, "sparse_values", qRank, qCondition) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.outputIdxOffset.IsPresent() &&
        CheckRank(info, info.outputIdxOffset, "output_idx_offset", qRank - 1U, qCondition) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckHeadFeature(info) != ge::GRAPH_SUCCESS || CheckFeatureRanges(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckHeadFeature(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.kHeads != K_HEAD_COUNT || info.headDim != HEAD_DIM) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "K head count must be 1 and head_dim must be 128");
    }
    if (info.qHeads < Q_HEAD_COUNT_MIN || info.qHeads / info.kHeads > HEAD_GROUP_LIMIT) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "Q head count must be in [1, 64]");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeatureRanges(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckUint32Range(info, "batch", info.batch) != ge::GRAPH_SUCCESS ||
        CheckUint32Range(info, "k sequence length", info.kSeq) != ge::GRAPH_SUCCESS ||
        CheckUint32Range(info, "q head group", info.qHeads / info.kHeads) != ge::GRAPH_SUCCESS ||
        (info.layoutQ == CheckerLayout::BSND &&
         CheckUint32Range(info, "q sequence length", info.qSeq) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::PA_BBND) {
        const gert::Shape *blockTableShape = info.blockTable.GetShape();
        if (blockTableShape != nullptr && blockTableShape->GetDimNum() >= BLOCK_TABLE_RANK &&
            CheckUint32Range(info, "block_table dim 1", blockTableShape->GetDim(BLOCK_TABLE_BLOCK_COUNT_AXIS)) !=
                ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (info.key.stride != nullptr && info.key.stride->GetDimNum() > 0 &&
        CheckUint32Range(info, "k stride 0", info.key.stride->GetStride(0)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.keyDescale.stride != nullptr && info.keyDescale.stride->GetDimNum() > 0 &&
        CheckUint32Range(info, "k_descale stride 0", info.keyDescale.stride->GetStride(0)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMainShapes(const LightningIndexerV2CheckerInfo &info) const
{
    const gert::Shape &q = *info.query.GetShape();
    const gert::Shape &k = *info.key.GetShape();
    const gert::Shape &w = *info.weights.GetShape();
    const uint32_t qS = info.layoutQ == CheckerLayout::BSND ? BSND_SEQ_AXIS : TND_TOKEN_AXIS;
    const uint32_t qN = info.layoutQ == CheckerLayout::BSND ? BSND_HEAD_AXIS : TND_HEAD_AXIS;
    const uint32_t qD = info.layoutQ == CheckerLayout::BSND ? BSND_HEAD_DIM_AXIS : TND_HEAD_DIM_AXIS;
    const uint32_t kN = info.layoutK == CheckerLayout::TND ? TND_HEAD_AXIS : BSND_HEAD_AXIS;
    const uint32_t kD = info.layoutK == CheckerLayout::TND ? TND_HEAD_DIM_AXIS : BSND_HEAD_DIM_AXIS;
    const uint32_t wS = info.layoutQ == CheckerLayout::BSND ? BSND_SEQ_AXIS : TND_TOKEN_AXIS;
    const uint32_t wN = info.layoutQ == CheckerLayout::BSND ? BSND_HEAD_AXIS : TND_HEAD_AXIS;
    if (!SameDim(q, qS, w, wS) || !SameDim(q, qN, w, wN) || !SameDim(q, qD, k, kD) ||
        (info.layoutQ == CheckerLayout::BSND && !SameDim(q, BSND_BATCH_AXIS, w, BSND_BATCH_AXIS))) {
        const std::string actual =
            ShapeToString(info.query) + ", " + ShapeToString(info.key) + " and " + ShapeToString(info.weights);
        return LogShapeMismatch(info, "q, k and w", actual,
                                "Q and w must share sequence/head axes and, for BSND, the batch axis; "
                                "q and k must share head_dim");
    }
    if (info.layoutQ == CheckerLayout::BSND && info.layoutK == CheckerLayout::BSND &&
        !SameDim(q, BSND_BATCH_AXIS, k, BSND_BATCH_AXIS)) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.key);
        return LogShapeMismatch(info, "q and k", actual, "Batch dimensions of BSND q and k must be equal");
    }
    if (k.GetDim(kN) != K_HEAD_COUNT) {
        const std::string actual = ShapeToString(info.key);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "k", actual.c_str(), "The head count of k must be 1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckOutputShapes(const LightningIndexerV2CheckerInfo &info) const
{
    const gert::Shape &q = *info.query.GetShape();
    const gert::Shape &out = *info.sparseIndices.GetShape();
    const uint32_t outN = info.layoutQ == CheckerLayout::BSND ? BSND_HEAD_AXIS : TND_HEAD_AXIS;
    if (out.GetDim(TND_TOKEN_AXIS) != q.GetDim(TND_TOKEN_AXIS) ||
        (info.layoutQ == CheckerLayout::BSND && out.GetDim(BSND_SEQ_AXIS) != q.GetDim(BSND_SEQ_AXIS)) ||
        out.GetDim(outN) != info.kHeads || out.GetDim(outN + 1U) != info.topk) {
        const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.sparseIndices);
        return LogShapeMismatch(info, "q and sparse_indices", actual,
                                "Sparse_indices must be [B,S1,N2,topk] or [T1,N2,topk]");
    }
    if (info.returnValue == RETURN_VALUE_WITH_VALUES) {
        const gert::Shape &values = *info.sparseValues.GetShape();
        if ((info.layoutQ == CheckerLayout::TND && values.GetDim(TND_TOKEN_AXIS) != q.GetDim(TND_TOKEN_AXIS)) ||
            values.GetDim(outN) != info.kHeads || values.GetDim(outN + 1U) != info.topk) {
            const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.sparseValues);
            return LogShapeMismatch(info, "q and sparse_values", actual,
                                    "Sparse_values must match T1 (for TND), N2 and topk");
        }
    }
    if (info.outputIdxOffset.IsPresent()) {
        const gert::Shape &offset = *info.outputIdxOffset.GetShape();
        const uint32_t offsetN = info.layoutQ == CheckerLayout::BSND ? BSND_HEAD_AXIS : TND_HEAD_AXIS;
        if (!SameDim(q, TND_TOKEN_AXIS, offset, TND_TOKEN_AXIS) ||
            (info.layoutQ == CheckerLayout::BSND && !SameDim(q, BSND_SEQ_AXIS, offset, BSND_SEQ_AXIS)) ||
            offset.GetDim(offsetN) != info.kHeads) {
            const std::string actual = ShapeToString(info.query) + " and " + ShapeToString(info.outputIdxOffset);
            return LogShapeMismatch(info, "q and output_idx_offset", actual,
                                    "Output_idx_offset must be [B,S1,N2] or [T1,N2]");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckMainShapes(info) != ge::GRAPH_SUCCESS || CheckOutputShapes(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.metadata.IsPresent() && info.metadata.GetShape()->GetShapeSize() != METADATA_SIZE) {
        const std::string actual = ShapeToString(info.metadata);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, "metadata", actual.c_str(), "Metadata size must be 1024");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

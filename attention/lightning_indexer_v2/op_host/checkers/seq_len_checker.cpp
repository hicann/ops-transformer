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
 * \file seq_len_checker.cpp
 * \brief Implements sequence-length parameter checks for Lightning Indexer V2 operators.
 */

#include "seq_len_checker_lightning_indexer_v2.h"

#include "log/log.h"

namespace optiling {
namespace lightning_indexer_v2_checker {
namespace {
constexpr int64_t MIN_CU_SEQLENS_SIZE = 1;
constexpr uint32_t SEQ_LEN_RANK = 1U;
} // namespace

ge::graphStatus SeqLenChecker::CheckSinglePara(const LightningIndexerV2CheckerInfo &info) const
{
    if (CheckRank(info, info.cuSeqlensQ, "cu_seqlens_q", SEQ_LEN_RANK) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.cuSeqlensK, "cu_seqlens_k", SEQ_LEN_RANK) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.sequsedQ, "seqused_q", SEQ_LEN_RANK) != ge::GRAPH_SUCCESS ||
        CheckRank(info, info.sequsedK, "seqused_k", SEQ_LEN_RANK) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::TND &&
        CheckDtype(info, info.cuSeqlensK, "cu_seqlens_k", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.sequsedK.IsPresent() &&
        CheckDtype(info, info.sequsedK, "seqused_k", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQ == CheckerLayout::TND &&
        CheckDtype(info, info.cuSeqlensQ, "cu_seqlens_q", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.sequsedQ.IsPresent() &&
        CheckDtype(info, info.sequsedQ, "seqused_q", {ge::DT_INT32}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckParaExistence(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.layoutK == CheckerLayout::TND && !info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                 "Cu_seqlens_k must be provided when layout_k is TND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::BSND && info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                 "Cu_seqlens_k must not be provided when layout_k is BSND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutK == CheckerLayout::PA_BBND && info.cuSeqlensK.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                 "Cu_seqlens_k must not be provided when layout_k is PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQ == CheckerLayout::TND && !info.cuSeqlensQ.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                 "Cu_seqlens_q must be provided when layout_q is TND");
        return ge::GRAPH_FAILED;
    }
    if (info.layoutQ == CheckerLayout::BSND && info.cuSeqlensQ.IsPresent()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                 "Cu_seqlens_q must not be provided when layout_q is BSND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckMultiPara(const LightningIndexerV2CheckerInfo &info) const
{
    if (info.layoutQ == CheckerLayout::TND) {
        const int64_t cuQSize = info.cuSeqlensQ.GetShape()->GetShapeSize();
        if (cuQSize <= MIN_CU_SEQLENS_SIZE) {
            const std::string actual = std::to_string(cuQSize);
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(info.opName, "cu_seqlens_q", actual.c_str(),
                                                      "The shape size of cu_seqlens_q must be greater than 1");
            return ge::GRAPH_FAILED;
        }
        if (info.layoutK == CheckerLayout::PA_BBND) {
            const int64_t sequsedKSize = info.sequsedK.GetShape()->GetShapeSize();
            if (sequsedKSize < 0 || cuQSize - MIN_CU_SEQLENS_SIZE != sequsedKSize) {
                const std::string actual = ShapeToString(info.cuSeqlensQ) + " and " + ShapeToString(info.sequsedK);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    info.opName, "cu_seqlens_q and seqused_k", actual.c_str(),
                    "The size of cu_seqlens_q must equal the size of seqused_k plus 1");
                return ge::GRAPH_FAILED;
            }
        }
        if (info.layoutK == CheckerLayout::TND && cuQSize != info.cuSeqlensK.GetShape()->GetShapeSize()) {
            const std::string actual = ShapeToString(info.cuSeqlensQ) + " and " + ShapeToString(info.cuSeqlensK);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(info.opName, "cu_seqlens_q and cu_seqlens_k", actual.c_str(),
                                                   "Cu_seqlens_q and cu_seqlens_k must have the same size");
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckElementCount(info, info.sequsedQ, "seqused_q", info.batch) != ge::GRAPH_SUCCESS ||
        CheckElementCount(info, info.sequsedK, "seqused_k", info.batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling

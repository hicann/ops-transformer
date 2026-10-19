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
 * \file qfmla_paged_attention_checker.cpp
 * \brief Checker for block_table, cache_seqlens, k_cache non-contiguous ( Paged Attention参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_paged_attention_checker.h"

namespace optiling {
namespace quant_flash_mla {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus PagedAttentionChecker::CheckSingleParaBlockTable(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // dtype: int32
    OP_CHECK_IF(qfmlaInfo.opParamInfo.blockTable.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, BLOCK_TABLE_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.blockTable.desc->GetDataType()).c_str(),
                    "The dtype of block_table must be int32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.blockTable.desc, BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (B, Bn)
    const auto& shape = qfmlaInfo.opParamInfo.blockTable.shape->GetStorageShape();
    OP_CHECK_IF(
        shape.GetDimNum() != 2 || static_cast<int64_t>(shape.GetDim(0)) != qfmlaInfo.bSize,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, BLOCK_TABLE_NAME, ToString(shape).c_str(),
                                              "The shape of block_table must be (B, Bn), B must match q batch size"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(1) <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, BLOCK_TABLE_NAME, ToString(shape).c_str(),
                                                      "The dim1 of block_table must be greater than 0"),
                return ge::GRAPH_FAILED);
    // dim1 (Bn) must match maxBlockNumPerBatch
    OP_CHECK_IF(static_cast<int64_t>(shape.GetDim(1)) != qfmlaInfo.maxBlockNumPerBatch,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, BLOCK_TABLE_NAME, ToString(shape).c_str(),
                    ("The dim1 of block_table must be " + std::to_string(qfmlaInfo.maxBlockNumPerBatch) +
                     " (Bn, maxBlockNumPerBatch)")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSingleParaCacheSeqlens(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // dtype: int32
    OP_CHECK_IF(
        qfmlaInfo.opParamInfo.cacheSeqlens.desc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName, CACHE_SEQLENS_NAME,
            ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.cacheSeqlens.desc->GetDataType()).c_str(),
            "The dtype of cache_seqlens must be int32"),
        return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.cacheSeqlens.desc, CACHE_SEQLENS_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (B,)
    const auto& shape = qfmlaInfo.opParamInfo.cacheSeqlens.shape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<int64_t>(shape.GetDim(0)) != qfmlaInfo.bSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, CACHE_SEQLENS_NAME, ToString(shape).c_str(),
                                                      "The shape of cache_seqlens must be (B,)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockSize(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    constexpr int64_t MLA_BLOCK_SIZE = 128;
    OP_CHECK_IF(
        qfmlaInfo.blockSize != MLA_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, K_CACHE_NAME, std::to_string(qfmlaInfo.blockSize).c_str(),
            "The block_size (derived from k_cache shape) must be 128, but got " + std::to_string(qfmlaInfo.blockSize)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMaxSeqlenKvCapacity(const QfmlaTilingInfo& qfmlaInfo)
{
    if (qfmlaInfo.maxSeqKv <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t capacity = qfmlaInfo.maxBlockNumPerBatch * qfmlaInfo.blockSize;
    OP_CHECK_IF(qfmlaInfo.maxSeqKv > capacity,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qfmlaInfo.opName, "max_seqlen_kv", std::to_string(qfmlaInfo.maxSeqKv).c_str(),
                    ("max_seqlen_kv must not exceed block_table capacity " + std::to_string(capacity)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckSingleParaBlockTable(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCacheSeqlens(qfmlaInfo) != ge::GRAPH_SUCCESS || CheckBlockSize(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckMaxSeqlenKvCapacity(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — k_cache 非连续 Tensor 支持校验
// 规则: PA_BBND 仅 0 轴可非连续; PA_BNBD/PA_NZ 仅 0、1 轴可非连续
// ============================================================================

ge::graphStatus PagedAttentionChecker::CheckNonContiguousSupport(const QfmlaTilingInfo& qfmlaInfo)
{
    if (!qfmlaInfo.hasStride) {
        return ge::GRAPH_SUCCESS;
    }
    const char* opName = qfmlaInfo.opName;
    const gert::StorageShape* kCacheShape = qfmlaInfo.opParamInfo.kCache.shape;
    OP_CHECK_IF(kCacheShape == nullptr, OP_LOGE(opName, "k_cache shape is nullptr"), return ge::GRAPH_FAILED);
    int32_t dimIndex = -1;
    if (CheckTensorContiguous(kCacheShape->GetStorageShape(), qfmlaInfo.keyStrides, dimIndex) == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    // PA_BBND 仅 0 轴可非连续, PA_BNBD/PA_NZ 仅 0、1 轴可非连续
    const bool isBbnd = (qfmlaInfo.layoutKv == QfmlaKvLayout::PA_BBND);
    const int32_t maxNonContiguousAxis = isBbnd ? 0 : 1;
    if (dimIndex > maxNonContiguousAxis) {
        const char* layoutName = isBbnd                                       ? "PA_BBND (BnBsH)" :
                                 (qfmlaInfo.layoutKv == QfmlaKvLayout::PA_NZ) ? "PA_NZ" :
                                                                                "PA_BNBD (BnNBsD)";
        const std::string reason = "k_cache layout is " + std::string(layoutName) + ", only axis 0" +
                                   (isBbnd ? "" : " and axis 1") + " can be non-contiguous, but axis " +
                                   std::to_string(dimIndex) + " is non-contiguous";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, K_CACHE_NAME, std::to_string(dimIndex).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeature(const QfmlaTilingInfo& qfmlaInfo)
{
    // PA 场景: block_table 必须不为空 (parser 已检查必选参数存在性)
    // k_cache 非连续校验
    if (CheckNonContiguousSupport(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

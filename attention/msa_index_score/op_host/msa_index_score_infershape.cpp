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
 * \file msa_index_score_infershape.cpp
 * \brief MsaIndexScore InferShape / InferDataType。
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

#include <cstring>
#include "../op_kernel/msa_index_score_common.h"

using MsaIndexScoreNs::MSA_KEY_LAYOUT_BBND;
using MsaIndexScoreNs::MSA_KEY_LAYOUT_BNBD;
using MsaIndexScoreNs::MSA_KEY_LAYOUT_BNSD;
using MsaIndexScoreNs::MSA_KEY_LAYOUT_BSND;
using MsaIndexScoreNs::MSA_KEY_LAYOUT_TND;
using MsaIndexScoreNs::MSA_QUERY_LAYOUT_BNSD;
using MsaIndexScoreNs::MSA_QUERY_LAYOUT_BSND;
using MsaIndexScoreNs::MSA_QUERY_LAYOUT_TND;
using MsaIndexScoreNs::MsaIsSeqKeyLayout;

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t MSA_QUERY_INDEX = 0;
constexpr uint32_t MSA_KEY_INDEX = 1;
constexpr uint32_t MSA_BLOCK_TABLE_INDEX = 2;
constexpr uint32_t MSA_ACTUAL_SEQ_QLEN_INDEX = 5;
constexpr uint32_t MSA_ACTUAL_SEQ_KLEN_INDEX = 6;
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_TND_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_PA_DIM_NUM = 4;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr uint32_t MSA_SCORE_DIM_BLOCK = 2;
constexpr int64_t MSA_KLEN_PREFIX_MIN = 2; // actual_seq_klen 为 [B+1]
constexpr uint32_t MSA_ATTR_LAYOUT_KEY = 0;
constexpr uint32_t MSA_ATTR_LAYOUT_QUERY = 4;
constexpr uint32_t MSA_ATTR_POOL_BLOCK_SIZE = 5;
constexpr int64_t MSA_SCORE_STRIDE_ALIGN = 16;
constexpr int64_t MSA_BLOCK_SIZE = 128;
constexpr int64_t MSA_BLOCK16_SIZE = 16;

inline int64_t RoundUpTo(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}

} // namespace

static ge::graphStatus InferShapeMsaIndexScore(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MsaIndexScore", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter MsaIndexScore InferShape impl.");

    const gert::Shape* queryShape = context->GetInputShape(MSA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape* keyShape = context->GetInputShape(MSA_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape* blockTableShape = context->GetOptionalInputShape(MSA_BLOCK_TABLE_INDEX);
    gert::Shape* scoreShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);

    auto attrs = context->GetAttrs();
    const char* layoutKeyPtr = (attrs != nullptr) ? attrs->GetStr(MSA_ATTR_LAYOUT_KEY) : nullptr;
    const char* layoutKey = (layoutKeyPtr == nullptr || layoutKeyPtr[0] == '\0') ? "BBND" : layoutKeyPtr;
    const char* layoutQueryPtr = (attrs != nullptr) ? attrs->GetStr(MSA_ATTR_LAYOUT_QUERY) : nullptr;
    const char* layoutQuery = (layoutQueryPtr == nullptr || layoutQueryPtr[0] == '\0') ? "TND" : layoutQueryPtr;
    // pool_block_size：缺省 128，只允许 16 / 128。BSND/BNSD 固定 16。
    const int64_t* poolBlockSizeAttrPtr = (attrs != nullptr) ? attrs->GetInt(MSA_ATTR_POOL_BLOCK_SIZE) : nullptr;
    const int64_t poolBlockSizeAttr =
        (poolBlockSizeAttrPtr == nullptr || *poolBlockSizeAttrPtr <= 0) ? MSA_BLOCK_SIZE : *poolBlockSizeAttrPtr;
    OP_CHECK_IF(poolBlockSizeAttr != MSA_BLOCK_SIZE && poolBlockSizeAttr != MSA_BLOCK16_SIZE,
                OP_LOGE(context, "pool_block_size must be %ld or %ld, got %ld.", MSA_BLOCK_SIZE, MSA_BLOCK16_SIZE,
                        poolBlockSizeAttr),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (std::strcmp(layoutKey, "BSND") == 0 || std::strcmp(layoutKey, "BNSD") == 0) &&
            poolBlockSizeAttr != MSA_BLOCK16_SIZE,
        OP_LOGE(context, "BSND/BNSD requires pool_block_size=%ld, got %ld.", MSA_BLOCK16_SIZE, poolBlockSizeAttr),
        return ge::GRAPH_FAILED);
    uint32_t keyLayout = MSA_KEY_LAYOUT_BBND;
    uint32_t queryLayout = MSA_QUERY_LAYOUT_TND;
    if (std::strcmp(layoutKey, "TND") == 0) {
        keyLayout = MSA_KEY_LAYOUT_TND;
    } else if (std::strcmp(layoutKey, "BSND") == 0) {
        keyLayout = MSA_KEY_LAYOUT_BSND;
    } else if (std::strcmp(layoutKey, "BNSD") == 0) {
        keyLayout = MSA_KEY_LAYOUT_BNSD;
    } else if (std::strcmp(layoutKey, "BNBD") == 0) {
        keyLayout = MSA_KEY_LAYOUT_BNBD;
    } else if (std::strcmp(layoutKey, "BBND") == 0) {
        keyLayout = MSA_KEY_LAYOUT_BBND;
    } else {
        OP_LOGE(context, "layout_key must be TND, BSND, BNSD, BBND or BNBD, got %s.", layoutKey);
        return ge::GRAPH_FAILED;
    }
    if (std::strcmp(layoutQuery, "BSND") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_BSND;
    } else if (std::strcmp(layoutQuery, "BNSD") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_BNSD;
    } else if (std::strcmp(layoutQuery, "TND") != 0) {
        OP_LOGE(context, "layout_query must be TND, BSND or BNSD, got %s.", layoutQuery);
        return ge::GRAPH_FAILED;
    }
    if (MsaIsSeqKeyLayout(keyLayout)) {
        const bool same = (keyLayout == MSA_KEY_LAYOUT_TND && queryLayout == MSA_QUERY_LAYOUT_TND) ||
                          (keyLayout == MSA_KEY_LAYOUT_BSND && queryLayout == MSA_QUERY_LAYOUT_BSND) ||
                          (keyLayout == MSA_KEY_LAYOUT_BNSD && queryLayout == MSA_QUERY_LAYOUT_BNSD);
        OP_CHECK_IF(!same, OP_LOGE(context, "non-paged key layout must match layout_query."), return ge::GRAPH_FAILED);
    }

    int64_t totalQ = 0;
    int64_t numQHeads = 0;
    if (queryLayout == MSA_QUERY_LAYOUT_TND) {
        OP_CHECK_IF(queryShape->GetDimNum() != MSA_QUERY_DIM_NUM,
                    OP_LOGE(context, "layout_query=TND requires query rank 3, got %zu.", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        totalQ = queryShape->GetDim(0);
        numQHeads = queryShape->GetDim(1);
    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != MSA_KEY_PA_DIM_NUM,
                    OP_LOGE(context, "layout_query=BSND/BNSD requires query rank 4, got %zu.", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        const int64_t batch = queryShape->GetDim(0);
        const int64_t seq = (queryLayout == MSA_QUERY_LAYOUT_BSND) ? queryShape->GetDim(1) : queryShape->GetDim(2);
        numQHeads = (queryLayout == MSA_QUERY_LAYOUT_BSND) ? queryShape->GetDim(2) : queryShape->GetDim(1);
        totalQ = batch * seq;
        const gert::Tensor* qlenTensor = context->GetOptionalInputTensor(MSA_ACTUAL_SEQ_QLEN_INDEX);
        const gert::Shape* qlenShape = context->GetOptionalInputShape(MSA_ACTUAL_SEQ_QLEN_INDEX);
        if (qlenTensor != nullptr && qlenShape != nullptr && qlenShape->GetDimNum() == 1 && qlenShape->GetDim(0) >= 2) {
            const auto qPlacement = qlenTensor->GetPlacement();
            if (qPlacement == gert::kOnHost || qPlacement == gert::kFollowing) {
                const int32_t* qlenData = qlenTensor->GetData<int32_t>();
                if (qlenData != nullptr) {
                    const int64_t prefixN = qlenShape->GetDim(0);
                    for (int64_t i = 0; i + 1 < prefixN; ++i) {
                        const int64_t qlenB = static_cast<int64_t>(qlenData[i + 1]) - static_cast<int64_t>(qlenData[i]);
                        OP_CHECK_IF(qlenB < 0 || qlenB > seq,
                                    OP_LOGE(context, "query length %ld exceeds layout S=%ld.", qlenB, seq),
                                    return ge::GRAPH_FAILED);
                    }
                    totalQ = static_cast<int64_t>(qlenData[prefixN - 1]);
                }
            }
        }
    }
    const bool isSeqKey = MsaIsSeqKeyLayout(keyLayout);
    int64_t maxBlocks = 0;
    if (!isSeqKey) {
        // 手写 op_api 对未传的可选输入给 AllocTensor 空占位（0 维）。占位视同未传。
        OP_CHECK_IF(blockTableShape == nullptr || blockTableShape->GetDimNum() == 0,
                    OP_LOGE(context, "layout_key=BBND/BNBD requires block_table."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(blockTableShape->GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                    OP_LOGE(context, "block_table must have 2 dims, but got %zu.", blockTableShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        // 页长仍须整除 pool_block_size：一个打分块从同一张物理页连续取出。
        const int64_t pageLen = (std::strcmp(layoutKey, "BBND") == 0) ? keyShape->GetDim(1) : keyShape->GetDim(2);
        OP_CHECK_IF(pageLen <= 0 || poolBlockSizeAttr <= 0 || pageLen % poolBlockSizeAttr != 0,
                    OP_LOGE(context, "BBND/BNBD page size(%ld) must be a positive multiple of pool_block_size(%ld).",
                            pageLen, poolBlockSizeAttr),
                    return ge::GRAPH_FAILED);
        maxBlocks = blockTableShape->GetDim(1);
    } else {
        const size_t expectRank = (keyLayout == MSA_KEY_LAYOUT_TND) ? MSA_KEY_TND_DIM_NUM : MSA_KEY_PA_DIM_NUM;
        OP_CHECK_IF(keyShape->GetDimNum() != expectRank,
                    OP_LOGE(context, "non-paged key rank mismatch, got %zu.", keyShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        const gert::Shape* klenShape = context->GetOptionalInputShape(MSA_ACTUAL_SEQ_KLEN_INDEX);
        const gert::Tensor* klenTensor = context->GetOptionalInputTensor(MSA_ACTUAL_SEQ_KLEN_INDEX);
        const int32_t* klenData = nullptr;
        if (klenTensor != nullptr) {
            const auto kPlacement = klenTensor->GetPlacement();
            if (kPlacement == gert::kOnHost || kPlacement == gert::kFollowing) {
                klenData = klenTensor->GetData<int32_t>();
            }
        }
        // 逻辑块长 = pool_block_size attr：TND 128 / 16，BSND/BNSD 固定 16。
        const int64_t seqBlock = poolBlockSizeAttr;
        if (klenData != nullptr && klenShape != nullptr && klenShape->GetDimNum() == 1 &&
            klenShape->GetDim(0) >= MSA_KLEN_PREFIX_MIN) {
            const int64_t prefixN = klenShape->GetDim(0);
            const bool capSeq = (keyLayout == MSA_KEY_LAYOUT_BSND || keyLayout == MSA_KEY_LAYOUT_BNSD);
            const int64_t keySeqCap = !capSeq                            ? 0 :
                                      (keyLayout == MSA_KEY_LAYOUT_BSND) ? keyShape->GetDim(1) :
                                                                           keyShape->GetDim(2);
            for (int64_t i = 0; i + 1 < prefixN; ++i) {
                const int64_t kv = static_cast<int64_t>(klenData[i + 1]) - static_cast<int64_t>(klenData[i]);
                OP_CHECK_IF(kv < 0 || (capSeq && kv > keySeqCap),
                            OP_LOGE(context, "key length %ld exceeds layout S=%ld.", kv, keySeqCap),
                            return ge::GRAPH_FAILED);
                const int64_t blocks = (kv <= 0) ? 0 : ((kv + seqBlock - 1) / seqBlock);
                if (blocks > maxBlocks) {
                    maxBlocks = blocks;
                }
            }
        } else if (keyLayout == MSA_KEY_LAYOUT_TND) {
            const int64_t totalK = keyShape->GetDim(0);
            maxBlocks = (totalK + seqBlock - 1) / seqBlock;
        } else {
            const int64_t seq = (keyLayout == MSA_KEY_LAYOUT_BSND) ? keyShape->GetDim(1) : keyShape->GetDim(2);
            maxBlocks = (seq + seqBlock - 1) / seqBlock;
        }
    }
    // 不拦截全 kv_len=0：按 1 个 dummy block 再 16 对齐，score 末维为 16；
    // 核内跳过 QK，把该请求的 score 填 -inf（T1=0 时输出为空张量）。
    if (maxBlocks <= 0) {
        maxBlocks = 1;
    }

    // maxpool 已经把一个 block 内的 poolBlockSize 个 token 归约成 1 个分数，
    // 因此输出末维是 block 数（16 对齐）而不是 kv token 数。
    // pool_block_size=16 再做 head 间 max，每个块只剩一个分数，head 维为 1。
    const int64_t scoreHeads = (poolBlockSizeAttr == MSA_BLOCK16_SIZE) ? 1 : numQHeads;
    scoreShape->SetDimNum(MSA_QUERY_DIM_NUM);
    scoreShape->SetDim(0, scoreHeads);
    scoreShape->SetDim(1, totalQ); // 1: totalQ
    scoreShape->SetDim(MSA_SCORE_DIM_BLOCK, RoundUpTo(maxBlocks, MSA_SCORE_STRIDE_ALIGN));

    OP_LOGI(context->GetNodeName(), "MsaIndexScore InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMsaIndexScore(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MsaIndexScore", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    // score 恒为 fp32：Cube 以 fp32 累加，下游 topk 直接消费 fp32。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MsaIndexScore).InferShape(InferShapeMsaIndexScore).InferDataType(InferDataTypeMsaIndexScore);
} // namespace ops

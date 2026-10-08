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
 * \file qmla_checker.cpp
 * \brief QuantFlashMlaWithKvcache 参数校验实现
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qmla_tiling_info.h"
#include "../qmla_tiling_info_parser.h"
#include "qmla_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

using std::map;
using std::string;
using namespace ge;
using namespace Ops::Base;

// MLA 支持的Q_N取值
static const std::vector<int64_t> MLA_SUPPORT_N1 = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};

ge::graphStatus QmlaChecker::CheckTensorContiguous(const gert::Shape& inputShape, const gert::Stride* strides,
                                                   int32_t& index) const
{
    // 根据 shape 从最后一维向前累乘推算连续场景的期望 stride, 若某一维实际 stride 不等于期望值则不连续
    if (strides == nullptr || strides->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t tensorDimNum = static_cast<int32_t>(inputShape.GetDimNum());
    // 维度为 0 或 1 的 tensor 始终连续
    if (tensorDimNum == 0 || tensorDimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t preStride = 1; // 连续场景最后一维的 stride 默认为 1
    for (index = tensorDimNum - 1; index >= 0; index--) {
        if (inputShape.GetDim(index) == 1) { // dim=1 时步长不影响连续性
            continue;
        }
        if (preStride != strides->GetStride(index)) {
            return ge::GRAPH_FAILED;
        }
        preStride *= inputShape.GetDim(index);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckNonContiguousSupport(const QmlaTilingInfo& qmlaInfo)
{
    if (!qmlaInfo.hasStride) {
        return ge::GRAPH_SUCCESS;
    }
    const char* opName = qmlaInfo.opName;
    const gert::StorageShape* kCacheShape = qmlaInfo.opParamInfo.kCache.shape;
    OP_CHECK_IF(kCacheShape == nullptr, OP_LOGE(opName, "k_cache shape is nullptr"), return ge::GRAPH_FAILED);
    int32_t dimIndex = -1;
    if (CheckTensorContiguous(kCacheShape->GetStorageShape(), qmlaInfo.keyStrides, dimIndex) == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    // PA_BBND(BnBsH) 仅 0 轴可非连续, PA_BNBD(BnNBsD)/PA_NZ 仅 0、1 轴可非连续
    const bool isBbnd = (qmlaInfo.layoutKv == QmlaKvLayout::PA_BBND);
    const int32_t maxNonContiguousAxis = isBbnd ? 0 : 1;
    if (dimIndex > maxNonContiguousAxis) {
        const char* layoutName = isBbnd                                     ? "PA_BBND (BnBsH)" :
                                 (qmlaInfo.layoutKv == QmlaKvLayout::PA_NZ) ? "PA_NZ" :
                                                                              "PA_BNBD (BnNBsD)";
        const std::string reason = "k_cache layout is " + std::string(layoutName) + ", only axis 0" +
                                   (isBbnd ? "" : " and axis 1") + " can be non-contiguous, but axis " +
                                   std::to_string(dimIndex) + " is non-contiguous";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, K_CACHE_NAME, std::to_string(dimIndex).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::Init(const QmlaTilingInfo& qmlaInfo)
{
    (void)qmlaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckAxisPara(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // MLA: head_dim_qk = 576
    OP_CHECK_IF(qmlaInfo.headDimQk != QMLA_HEAD_DIM_QK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "q", std::to_string(qmlaInfo.headDimQk).c_str(),
                    "In MLA scenario, the head dim of q (head_dim_qk) must be 576 (nope 512 + rope 64)"),
                return ge::GRAPH_FAILED);
    // MLA: head_dim_v = 512
    OP_CHECK_IF(qmlaInfo.headDimV != QMLA_HEAD_DIM_V,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "head_dim_v", std::to_string(qmlaInfo.headDimV).c_str(),
                                                      "In MLA scenario, head_dim_v must be 512"),
                return ge::GRAPH_FAILED);
    // MLA: KV_N = 1
    OP_CHECK_IF(qmlaInfo.n2Size != QMLA_KV_N,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "k_cache", std::to_string(qmlaInfo.n2Size).c_str(),
                                                      "In MLA scenario, KV_N of k_cache must be 1"),
                return ge::GRAPH_FAILED);
    // Q_N 取值约束
    if (std::find(MLA_SUPPORT_N1.begin(), MLA_SUPPORT_N1.end(), qmlaInfo.n1Size) == MLA_SUPPORT_N1.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "q", std::to_string(qmlaInfo.n1Size).c_str(),
            "In MLA scenario, Q_N must be one of {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}");
        return ge::GRAPH_FAILED;
    }
    // S1约束: 1 <= S <= 16
    OP_CHECK_IF(qmlaInfo.s1Size < 1 || qmlaInfo.s1Size > 16,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qmlaInfo.s1Size).c_str(),
                                                      "In MLA scenario, Q_S must be in range [1, 16]"),
                return ge::GRAPH_FAILED);
    // S2约束: >= 1
    OP_CHECK_IF(qmlaInfo.s2Size < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "cache_seqlens", std::to_string(qmlaInfo.s2Size).c_str(),
                                                      "In MLA scenario, KV seq len must be >= 1"),
                return ge::GRAPH_FAILED);
    // B约束: >= 1
    OP_CHECK_IF(qmlaInfo.bSize < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qmlaInfo.bSize).c_str(),
                                                      "Batch size must be >= 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckSeqLenPara(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // cu_seqlens_q: layout_q为TND时必传，非TND时不支持
    if (qmlaInfo.layoutQ == QmlaLayout::TND) {
        OP_CHECK_IF(!qmlaInfo.cuSeqLenQFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, "provided",
                                                          "When layout_q is TND, cu_seqlens_q must be provided"),
                    return ge::GRAPH_FAILED);
        // TND时seqused_q与max_seqlen_q至少传入其中一个
        OP_CHECK_IF(!qmlaInfo.seqUsedQFlag && qmlaInfo.maxSeqQ <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "seqused_q", "provided",
                        "When layout_q is TND, seqused_q and max_seqlen_q must be provided at least one of them"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(qmlaInfo.cuSeqLenQFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, "not provided",
                                                          "When layout_q is not TND, cu_seqlens_q is not supported"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckMaskPara(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // mask_mode: 仅支持0/3
    OP_CHECK_IF(qmlaInfo.maskMode != QMLA_MASK_MODE_NO_MASK && qmlaInfo.maskMode != QMLA_MASK_MODE_CAUSAL,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "mask_mode", std::to_string(qmlaInfo.maskMode).c_str(),
                                                      "mask_mode must be 0 (NO_MASK) or 3 (CAUSAL)"),
                return ge::GRAPH_FAILED);
    // mask_mode=0时，不支持传入attn_mask；mask_mode=3时，必须传入attn_mask
    if (qmlaInfo.maskMode == QMLA_MASK_MODE_NO_MASK) {
        OP_CHECK_IF(qmlaInfo.attnMaskFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, ATTN_MASK_NAME, "not provided",
                                                          "When mask_mode is 0 (NO_MASK), attn_mask is not supported"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(!qmlaInfo.attnMaskFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, ATTN_MASK_NAME, "provided",
                                                          "When mask_mode is 3 (CAUSAL), attn_mask must be provided"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckQuantPara(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // quant_mode已在parser中校验，这里校验dtype一致性
    // FP8场景: q/k_cache为fp8_e4m3，q_descale/k_descale为float，attn_out为bf16
    OP_CHECK_IF(qmlaInfo.qType != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, "q", ge::TypeUtils::DataTypeToSerialString(qmlaInfo.qType).c_str(),
                    "In MLA_FP8_E4M3_FULLQUANT scenario, dtype of q must be float8_e4m3fn"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qmlaInfo.kvType != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, "k_cache", ge::TypeUtils::DataTypeToSerialString(qmlaInfo.kvType).c_str(),
                    "In MLA_FP8_E4M3_FULLQUANT scenario, dtype of k_cache must be float8_e4m3fn"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qmlaInfo.outType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "attn_out",
                                                      ge::TypeUtils::DataTypeToSerialString(qmlaInfo.outType).c_str(),
                                                      "The dtype of attn_out must be bfloat16"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckLayoutPara(const QmlaTilingInfo& qmlaInfo)
{
    // layout_kv仅支持PA系列，已在parser校验；此处校验q_descale与layout_q一致性
    const char* opName = qmlaInfo.opName;
    const auto& qDescaleShape = qmlaInfo.opParamInfo.qDescale.shape->GetStorageShape();
    if (qmlaInfo.layoutQ == QmlaLayout::TND) {
        // (Q_T, Q_N)
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qmlaInfo.qTSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qmlaInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (Q_T, Q_N) when layout_q is TND"),
            return ge::GRAPH_FAILED);
    } else if (qmlaInfo.layoutQ == QmlaLayout::BSND) {
        // (B, Q_S, Q_N)
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qmlaInfo.bSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qmlaInfo.s1Size ||
                static_cast<int64_t>(qDescaleShape.GetDim(2)) != qmlaInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (B, Q_S, Q_N) when layout_q is BSND"),
            return ge::GRAPH_FAILED);
    } else {
        // (B, Q_N, Q_S)
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qmlaInfo.bSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qmlaInfo.n1Size ||
                static_cast<int64_t>(qDescaleShape.GetDim(2)) != qmlaInfo.s1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (B, Q_N, Q_S) when layout_q is BNSD"),
            return ge::GRAPH_FAILED);
    }

    // attn_mask shape (2048, 2048)
    if (qmlaInfo.attnMaskFlag) {
        OP_CHECK_IF(qmlaInfo.attenMaskS1Size != 2048 || qmlaInfo.attenMaskS2Size != 2048,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, ATTN_MASK_NAME,
                                                          std::to_string(qmlaInfo.attenMaskS1Size)
                                                              .append("x")
                                                              .append(std::to_string(qmlaInfo.attenMaskS2Size))
                                                              .c_str(),
                                                          "The shape of attn_mask must be (2048, 2048)"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::CheckSoftmaxLsePara(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    if (qmlaInfo.returnSoftmaxLse) {
        OP_CHECK_IF(
            qmlaInfo.opParamInfo.softmaxLse.shape == nullptr ||
                qmlaInfo.opParamInfo.softmaxLse.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, SOFTMAX_LSE_NAME, "not empty",
                                                  "When return_softmax_lse is True, softmax_lse must not be empty"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaChecker::Process(const QmlaTilingInfo& qmlaInfo)
{
    if (CheckAxisPara(qmlaInfo) != ge::GRAPH_SUCCESS || CheckSeqLenPara(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckMaskPara(qmlaInfo) != ge::GRAPH_SUCCESS || CheckQuantPara(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckLayoutPara(qmlaInfo) != ge::GRAPH_SUCCESS || CheckSoftmaxLsePara(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckNonContiguousSupport(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling

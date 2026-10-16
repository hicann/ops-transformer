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
 * \file qfmla_common_checker.cpp
 * \brief Common checker for layout, dtype, shape, axis parameters ( 公共参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_common_checker.h"

namespace optiling {
namespace quant_flash_mla {

using std::string;
using namespace ge;
using namespace Ops::Base;

// MLA 支持的 Q_N 取值
static const std::vector<int64_t> MLA_SUPPORT_N1 = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};

// ============================================================================
// SinglePara — Layout
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaLayout(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // layout_q: 当前仅支持 TND（不支持 BSND/BNSD/TND_NTD）
    OP_CHECK_IF(qfmlaInfo.layoutQ != QfmlaLayout::TND,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "layout_q", "BSND/BNSD/TND_NTD",
                    "Currently only layout_q=TND is supported, BSND/BNSD/TND_NTD are not supported"),
                return ge::GRAPH_FAILED);

    // layout_kv: 支持 PA_BBND/PA_BNBD/PA_NZ (parser 已校验枚举合法性，此处补检查)
    OP_CHECK_IF(qfmlaInfo.layoutKv != QfmlaKvLayout::PA_BBND && qfmlaInfo.layoutKv != QfmlaKvLayout::PA_BNBD &&
                    qfmlaInfo.layoutKv != QfmlaKvLayout::PA_NZ,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_kv", "invalid",
                                                      "layout_kv must be PA_BBND/PA_BNBD/PA_NZ"),
                return ge::GRAPH_FAILED);

    // layout_out: 支持 BSND/BNSD/TND/NTD
    OP_CHECK_IF(
        qfmlaInfo.layoutOut != QfmlaOutLayout::BSND && qfmlaInfo.layoutOut != QfmlaOutLayout::BNSD &&
            qfmlaInfo.layoutOut != QfmlaOutLayout::TND && qfmlaInfo.layoutOut != QfmlaOutLayout::NTD,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_out", "invalid", "layout_out must be BSND/BNSD/TND/NTD"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// SinglePara — Dtype
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaDtype(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // q: 仅支持 float8_e4m3fn (不支持 hifloat8)
    OP_CHECK_IF(qfmlaInfo.qType != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, "q", ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.qType).c_str(),
                    "In MLA scenario, dtype of q must be float8_e4m3fn (hifloat8 is not supported)"),
                return ge::GRAPH_FAILED);
    // k_cache: 仅支持 float8_e4m3fn
    OP_CHECK_IF(qfmlaInfo.kvType != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, K_CACHE_NAME, ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.kvType).c_str(),
                    "In MLA scenario, dtype of k_cache must be float8_e4m3fn (hifloat8 is not supported)"),
                return ge::GRAPH_FAILED);
    // attn_out: 仅支持 bfloat16
    OP_CHECK_IF(qfmlaInfo.outType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, ATTN_OUT_NAME,
                                                      ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.outType).c_str(),
                                                      "The dtype of attn_out must be bfloat16"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.query.desc, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(qfmlaInfo.opParamInfo.kCache.desc, K_CACHE_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(qfmlaInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// SinglePara — ShapeDim
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaShapeDim(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // q: layout_q=TND 时 shape dim 必须为 3D
    const auto& qShape = qfmlaInfo.opParamInfo.query.shape->GetStorageShape();
    OP_CHECK_IF(
        qShape.GetDimNum() != 3,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, QUERY_NAME, (std::to_string(qShape.GetDimNum()) + "D").c_str(),
                                                 "The shape dim of q must be 3D when layout_q is TND"),
        return ge::GRAPH_FAILED);

    // k_cache: PA_BBND/PA_BNBD 为 4D, PA_NZ 为 5D
    const auto& kCacheShape = qfmlaInfo.opParamInfo.kCache.shape->GetStorageShape();
    uint32_t kCacheDimNum = kCacheShape.GetDimNum();
    if (qfmlaInfo.layoutKv == QfmlaKvLayout::PA_NZ) {
        OP_CHECK_IF(
            kCacheDimNum != 5,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, K_CACHE_NAME, (std::to_string(kCacheDimNum) + "D").c_str(),
                                                     "The shape dim of k_cache must be 5D when layout_kv is PA_NZ"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(kCacheDimNum != 4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName, K_CACHE_NAME, (std::to_string(kCacheDimNum) + "D").c_str(),
                        "The shape dim of k_cache must be 4D when layout_kv is PA_BBND/PA_BNBD"),
                    return ge::GRAPH_FAILED);
    }

    // attn_out: layout_q=TND 时为 3D
    const auto& attnOutShape = qfmlaInfo.opParamInfo.attnOut.shape->GetStorageShape();
    OP_CHECK_IF(attnOutShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, ATTN_OUT_NAME,
                                                         (std::to_string(attnOutShape.GetDimNum()) + "D").c_str(),
                                                         "The shape dim of attn_out must be 3D when layout_q is TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// SinglePara — softmax_scale
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaSoftmaxScale(const QfmlaTilingInfo& qfmlaInfo)
{
    // softmax_scale 可选传入，未传入时默认 -1.0f (算子内部取 1/sqrt(headdim))
    // 传入时必须 > 0
    OP_CHECK_IF(qfmlaInfo.softmaxScale != -1.0f && qfmlaInfo.softmaxScale <= 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfmlaInfo.opName, "softmax_scale",
                                                      std::to_string(qfmlaInfo.softmaxScale).c_str(),
                                                      "softmax_scale must be greater than 0 when provided"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckSingleParaLayout(qfmlaInfo) != ge::GRAPH_SUCCESS || CheckSingleParaDtype(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaShapeDim(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaSoftmaxScale(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus CommonChecker::CheckParaExistence(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // q/k_cache/attn_out 必选参数, desc 和 shape 均不能为空 (parser 已检查，此处补检查)
    OP_CHECK_IF(qfmlaInfo.opParamInfo.query.desc == nullptr || qfmlaInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName, QUERY_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfmlaInfo.opParamInfo.kCache.desc == nullptr || qfmlaInfo.opParamInfo.kCache.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName, K_CACHE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfmlaInfo.opParamInfo.attnOut.desc == nullptr || qfmlaInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName, ATTN_OUT_NAME), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — 轴校验
// ============================================================================

ge::graphStatus CommonChecker::CheckAxis(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // head_dim_qk = 576 (nope 512 + rope 64)
    OP_CHECK_IF(qfmlaInfo.headDimQk != QFMLA_HEAD_DIM_QK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.headDimQk).c_str(),
                                                      "In MLA scenario, head_dim_qk must be 576 (nope 512 + rope 64)"),
                return ge::GRAPH_FAILED);
    // head_dim_v = 512
    OP_CHECK_IF(qfmlaInfo.headDimV != QFMLA_HEAD_DIM_V,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "head_dim_v", std::to_string(qfmlaInfo.headDimV).c_str(),
                                                      "In MLA scenario, head_dim_v must be 512"),
                return ge::GRAPH_FAILED);
    // KV_N = 1
    OP_CHECK_IF(qfmlaInfo.n2Size != QFMLA_KV_N,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, K_CACHE_NAME, std::to_string(qfmlaInfo.n2Size).c_str(),
                                                      "In MLA scenario, KV_N of k_cache must be 1"),
                return ge::GRAPH_FAILED);
    // Q_N 取值约束
    if (std::find(MLA_SUPPORT_N1.begin(), MLA_SUPPORT_N1.end(), qfmlaInfo.n1Size) == MLA_SUPPORT_N1.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "q", std::to_string(qfmlaInfo.n1Size).c_str(),
            "In MLA scenario, Q_N must be one of {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}");
        return ge::GRAPH_FAILED;
    }
    // S1 约束: TND 时 s1Size=-1 表示未传 max_seqlen_q (哨兵值), 跳过范围校验;
    //          传入 max_seqlen_q (s1Size > 0) 时校验 [1, 16]
    OP_CHECK_IF(qfmlaInfo.s1Size != -1 && (qfmlaInfo.s1Size < 1 || qfmlaInfo.s1Size > 16),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "q", std::to_string(qfmlaInfo.s1Size).c_str(),
                    "In MLA scenario, Q_S must be in range [1, 16] when max_seqlen_q is provided"),
                return ge::GRAPH_FAILED);
    // S2 约束: >= 1
    OP_CHECK_IF(
        qfmlaInfo.s2Size < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CACHE_SEQLENS_NAME, std::to_string(qfmlaInfo.s2Size).c_str(),
                                              "In MLA scenario, KV seq len must be >= 1"),
        return ge::GRAPH_FAILED);
    // B 约束: >= 1
    OP_CHECK_IF(qfmlaInfo.bSize < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.bSize).c_str(),
                                                      "Batch size must be >= 1"),
                return ge::GRAPH_FAILED);
    // Q_T 约束: >= 0
    OP_CHECK_IF(qfmlaInfo.qTSize < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.qTSize).c_str(),
                                                      "Q_T must be >= 0"),
                return ge::GRAPH_FAILED);
    // k_cache D 维度校验: head_dim_qk=576
    const auto& kCacheShape = qfmlaInfo.opParamInfo.kCache.shape->GetStorageShape();
    int64_t kCacheD = 0;
    if (qfmlaInfo.layoutKv == QfmlaKvLayout::PA_BBND || qfmlaInfo.layoutKv == QfmlaKvLayout::PA_BNBD) {
        kCacheD = static_cast<int64_t>(kCacheShape.GetDim(3));
    } else {
        kCacheD = static_cast<int64_t>(kCacheShape.GetDim(2)) * 32;
    }
    OP_CHECK_IF(kCacheD != QFMLA_HEAD_DIM_QK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, K_CACHE_NAME, std::to_string(kCacheD).c_str(),
                                                      "In MLA scenario, head_dim_qk of k_cache must be 576"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q shape 维度值校验
// ============================================================================

ge::graphStatus CommonChecker::CheckQueryShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // TND: q shape 应为 (Q_T, Q_N, head_dim_qk=576)
    const auto& qShape = qfmlaInfo.opParamInfo.query.shape->GetStorageShape();
    OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(0)) != qfmlaInfo.qTSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                      "The dim0 of q shape must be Q_T"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(1)) != qfmlaInfo.n1Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                      "The dim1 of q shape must be Q_N"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(2)) != QFMLA_HEAD_DIM_QK,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                      "The dim2 of q shape must be head_dim_qk(576)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — attn_out shape 维度值校验
// ============================================================================

ge::graphStatus CommonChecker::CheckAttnOutShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    const auto& attnOutShape = qfmlaInfo.opParamInfo.attnOut.shape->GetStorageShape();
    // layout_out=TND: (Q_T, Q_N, head_dim_v=512)
    // layout_out=NTD: (Q_N, Q_T, head_dim_v=512)
    if (qfmlaInfo.layoutOut == QfmlaOutLayout::TND) {
        OP_CHECK_IF(
            static_cast<int64_t>(attnOutShape.GetDim(0)) != qfmlaInfo.qTSize ||
                static_cast<int64_t>(attnOutShape.GetDim(1)) != qfmlaInfo.n1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(2)) != QFMLA_HEAD_DIM_V,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, ATTN_OUT_NAME, ToString(attnOutShape).c_str(),
                                                  "When layout_out is TND, attn_out shape must be (Q_T, Q_N, 512)"),
            return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutOut == QfmlaOutLayout::NTD) {
        OP_CHECK_IF(
            static_cast<int64_t>(attnOutShape.GetDim(0)) != qfmlaInfo.n1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(1)) != qfmlaInfo.qTSize ||
                static_cast<int64_t>(attnOutShape.GetDim(2)) != QFMLA_HEAD_DIM_V,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, ATTN_OUT_NAME, ToString(attnOutShape).c_str(),
                                                  "When layout_out is NTD, attn_out shape must be (Q_N, Q_T, 512)"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckAxis(qfmlaInfo) != ge::GRAPH_SUCCESS || CheckQueryShape(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckAttnOutShape(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — q/k_cache dtype 一致性
// ============================================================================

ge::graphStatus CommonChecker::CheckDtypeConsistency(const QfmlaTilingInfo& qfmlaInfo)
{
    OP_CHECK_IF(qfmlaInfo.qType != qfmlaInfo.kvType,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    qfmlaInfo.opName, K_CACHE_NAME, ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.kvType).c_str(),
                    "The dtype of k_cache must be the same as q, q dtype is " +
                        std::string(ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.qType))),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckDtypeConsistency(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

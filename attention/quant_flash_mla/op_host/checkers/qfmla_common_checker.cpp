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
// arch22 int8全量化支持的 Q_N 取值（对齐IFA现状: G∈{1,2,4,8,16,32,64,128}）
static const std::vector<int64_t> MLA_INT8_SUPPORT_N1 = {1, 2, 4, 8, 16, 32, 64, 128};

// ============================================================================
// SinglePara — Layout
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaLayout(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // 芯片差异用 npuArch 区分: arch22(910B) 承接 int8 全量化 MLA 收编场景
    const bool isArch22 = qfmlaInfo.npuArch == NpuArch::DAV_2201;
    if (isArch22) {
        // arch22 int8全量化: layout_q 仅 BSH/TND; layout_kv 仅 PA_NZ; layout_out 仅 BSH/TND
        OP_CHECK_IF(qfmlaInfo.layoutQ != QfmlaLayout::TND && qfmlaInfo.layoutQ != QfmlaLayout::BSH,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_q", "invalid",
                                                          "In MLA_INT8_FULLQUANT scenario, only layout_q=BSH/TND "
                                                          "are supported"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(qfmlaInfo.layoutKv != QfmlaKvLayout::PA_NZ,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_kv", "invalid",
                                                          "In MLA_INT8_FULLQUANT scenario, layout_kv must be "
                                                          "PA_NZ"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(qfmlaInfo.layoutOut != QfmlaOutLayout::TND && qfmlaInfo.layoutOut != QfmlaOutLayout::BSH,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_out", "invalid",
                                                          "In MLA_INT8_FULLQUANT scenario, layout_out must be "
                                                          "BSH/TND"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // fp8/hif8: layout_q: 支持TND/BSND/BNSD
    OP_CHECK_IF(qfmlaInfo.layoutQ != QfmlaLayout::TND && qfmlaInfo.layoutQ != QfmlaLayout::BSND &&
                    qfmlaInfo.layoutQ != QfmlaLayout::BNSD,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_q", "invalid",
                                                      "Currently only layout_q=TND/BSND/BNSD are supported"),
                return ge::GRAPH_FAILED);

    // layout_kv: 支持 PA_BBND/PA_BNBD/PA_NZ (parser 已校验枚举合法性，此处补检查)
    OP_CHECK_IF(qfmlaInfo.layoutKv != QfmlaKvLayout::PA_BBND && qfmlaInfo.layoutKv != QfmlaKvLayout::PA_BNBD &&
                    qfmlaInfo.layoutKv != QfmlaKvLayout::PA_NZ,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "layout_kv", "invalid",
                                                      "layout_kv must be PA_BBND/PA_BNBD/PA_NZ"),
                return ge::GRAPH_FAILED);

    // layout_q: 支持TND/BSND/BNSD; layout_out: 支持BSND/BNSD/TND/NTD。枚举解析与
    // 能力校验分层，避免非法字符串与“枚举合法但当前不支持”混成同一类报错。
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
    // INT8场景(arch22): q/k_cache为int8; FP8场景: q/k_cache为fp8_e4m3; HIF8场景: q/k_cache为hifloat8
    ge::DataType expectDtype;
    const char* expectDtypeName;
    const char* sceneName;
    if (qfmlaInfo.quantMode == QfmlaQuantMode::MLA_INT8_FULLQUANT) {
        expectDtype = ge::DT_INT8;
        expectDtypeName = "int8";
        sceneName = "MLA_INT8_FULLQUANT";
    } else {
        const bool isHif8 = qfmlaInfo.quantMode == QfmlaQuantMode::MLA_HIF8_FULLQUANT;
        expectDtype = isHif8 ? ge::DT_HIFLOAT8 : ge::DT_FLOAT8_E4M3FN;
        expectDtypeName = isHif8 ? "hifloat8" : "float8_e4m3fn";
        sceneName = isHif8 ? "MLA_HIF8_FULLQUANT" : "MLA_FP8_E4M3_FULLQUANT";
    }
    // q: dtype 与 quant_mode 匹配
    OP_CHECK_IF(qfmlaInfo.qType != expectDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, "q", ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.qType).c_str(),
                    (std::string("In ") + sceneName + " scenario, dtype of q must be " + expectDtypeName).c_str()),
                return ge::GRAPH_FAILED);
    // k_cache: dtype 与 quant_mode 匹配
    OP_CHECK_IF(
        qfmlaInfo.kvType != expectDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName, K_CACHE_NAME, ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.kvType).c_str(),
            (std::string("In ") + sceneName + " scenario, dtype of k_cache must be " + expectDtypeName).c_str()),
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
    // q: layout_q=TND/BSH 时 shape dim 必须为 3D, BSND/BNSD 时为 4D
    const bool isQ4D = qfmlaInfo.layoutQ == QfmlaLayout::BSND || qfmlaInfo.layoutQ == QfmlaLayout::BNSD;
    const auto& qShape = qfmlaInfo.opParamInfo.query.shape->GetStorageShape();
    OP_CHECK_IF(
        qShape.GetDimNum() != (isQ4D ? 4 : 3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, QUERY_NAME, (std::to_string(qShape.GetDimNum()) + "D").c_str(),
                                                 isQ4D ? "The shape dim of q must be 4D when layout_q is BSND/BNSD" :
                                                         "The shape dim of q must be 3D when layout_q is TND/BSH"),
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

    // attn_out: layout_out=TND/NTD/BSH 时为 3D, BSND/BNSD 时为 4D
    const bool isOut4D = qfmlaInfo.layoutOut == QfmlaOutLayout::BSND || qfmlaInfo.layoutOut == QfmlaOutLayout::BNSD;
    const auto& attnOutShape = qfmlaInfo.opParamInfo.attnOut.shape->GetStorageShape();
    OP_CHECK_IF(attnOutShape.GetDimNum() != (isOut4D ? 4 : 3),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName, ATTN_OUT_NAME, (std::to_string(attnOutShape.GetDimNum()) + "D").c_str(),
                    isOut4D ? "The shape dim of attn_out must be 4D when layout_out is BSND/BNSD" :
                              "The shape dim of attn_out must be 3D when layout_out is TND/NTD/BSH"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// SinglePara — softmax_scale
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaSoftmaxScale(const QfmlaTilingInfo& qfmlaInfo)
{
    // softmax_scale 必须大于 0。
    OP_CHECK_IF(qfmlaInfo.softmaxScale <= 0.0f,
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
    // 芯片差异用 npuArch 区分: arch22(910B) 承接 int8 全量化 MLA 收编场景
    const bool isArch22 = qfmlaInfo.npuArch == NpuArch::DAV_2201;
    if (isArch22) {
        // arch22 int8全量化: q为nope分离输入, D=512; rope单独传入(D=64)
        OP_CHECK_IF(qfmlaInfo.headDimQk != 512,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.headDimQk).c_str(),
                                                          "In MLA_INT8_FULLQUANT scenario, head_dim of q must be 512 "
                                                          "(nope), rope is provided separately"),
                    return ge::GRAPH_FAILED);
        // S1 约束: [1, 32] (对齐IFA现状: 32 MTP最大QS)
        OP_CHECK_IF(qfmlaInfo.s1Size != -1 && (qfmlaInfo.s1Size < 1 || qfmlaInfo.s1Size > 32),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "q", std::to_string(qfmlaInfo.s1Size).c_str(),
                        "In MLA_INT8_FULLQUANT scenario, Q_S must be in range [1, 32] when max_seqlen_q is provided"),
                    return ge::GRAPH_FAILED);
        // k_cache D 维度校验: D=512 (nope)。PA_NZ: dim2*32==512 -> dim2==16
        const auto& kCacheShapeInt8 = qfmlaInfo.opParamInfo.kCache.shape->GetStorageShape();
        OP_CHECK_IF(static_cast<int64_t>(kCacheShapeInt8.GetDim(4)) != 32,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, K_CACHE_NAME, ToString(kCacheShapeInt8).c_str(),
                                                          "When layout_kv is PA_NZ, the innermost axis must be 32"),
                    return ge::GRAPH_FAILED);
        const int64_t kCacheDInt8 = static_cast<int64_t>(kCacheShapeInt8.GetDim(2)) * 32;
        OP_CHECK_IF(kCacheDInt8 != 512,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, K_CACHE_NAME, std::to_string(kCacheDInt8).c_str(),
                                                          "In MLA_INT8_FULLQUANT scenario, head_dim of k_cache must "
                                                          "be 512 (nope)"),
                    return ge::GRAPH_FAILED);
    } else {
        // head_dim_qk = 576 (nope 512 + rope 64)
        OP_CHECK_IF(
            qfmlaInfo.headDimQk != QFMLA_HEAD_DIM_QK,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.headDimQk).c_str(),
                                                  "In MLA scenario, head_dim_qk must be 576 (nope 512 + rope 64)"),
            return ge::GRAPH_FAILED);
        // S1 约束: [1, 16]
        OP_CHECK_IF(qfmlaInfo.s1Size != -1 && (qfmlaInfo.s1Size < 1 || qfmlaInfo.s1Size > 16),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "q", std::to_string(qfmlaInfo.s1Size).c_str(),
                        "In MLA scenario, Q_S must be in range [1, 16] when max_seqlen_q is provided"),
                    return ge::GRAPH_FAILED);
        // k_cache D 维度校验: head_dim_qk=576
        const auto& kCacheShape = qfmlaInfo.opParamInfo.kCache.shape->GetStorageShape();
        int64_t kCacheD = 0;
        if (qfmlaInfo.layoutKv == QfmlaKvLayout::PA_BBND || qfmlaInfo.layoutKv == QfmlaKvLayout::PA_BNBD) {
            kCacheD = static_cast<int64_t>(kCacheShape.GetDim(3));
        } else {
            OP_CHECK_IF(static_cast<int64_t>(kCacheShape.GetDim(4)) != 32,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, K_CACHE_NAME, ToString(kCacheShape).c_str(),
                                                              "When layout_kv is PA_NZ, the innermost axis must be 32"),
                        return ge::GRAPH_FAILED);
            kCacheD = static_cast<int64_t>(kCacheShape.GetDim(2)) * 32;
        }
        OP_CHECK_IF(kCacheD != QFMLA_HEAD_DIM_QK,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, K_CACHE_NAME, std::to_string(kCacheD).c_str(),
                                                          "In MLA scenario, head_dim_qk of k_cache must be 576"),
                    return ge::GRAPH_FAILED);
    }
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
    const auto& supportN1 = isArch22 ? MLA_INT8_SUPPORT_N1 : MLA_SUPPORT_N1;
    if (std::find(supportN1.begin(), supportN1.end(), qfmlaInfo.n1Size) == supportN1.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.n1Size).c_str(),
                                              isArch22 ?
                                                  "In MLA_INT8_FULLQUANT scenario, Q_N must be one of "
                                                  "{1, 2, 4, 8, 16, 32, 64, 128}" :
                                                  "In MLA scenario, Q_N must be one of {1, 2, 4, 6, 8, 12, 16, 24, 32, "
                                                  "48, 64, 96, 128}");
        return ge::GRAPH_FAILED;
    }
    // BSND/BNSD s1Size comes from the physical Q shape, independently of max_seqlen_q.
    // TND max_seqlen_q is validated by SeqLenChecker; actual lengths are in cu/seq tensors.
    // arch22 int8 全量化 (layout_q 为 BSH/TND) 的 S1 校验由上方 arch22 专用分支负责 ([1,32], 允许未传入)
    OP_CHECK_IF(!isArch22 && qfmlaInfo.layoutQ != QfmlaLayout::TND && (qfmlaInfo.s1Size < 1 || qfmlaInfo.s1Size > 16),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "q", std::to_string(qfmlaInfo.s1Size).c_str(),
                    "In MLA scenario, Q_S of q must be in range [1, 16] when layout_q is BSND or BNSD"),
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
    // Q_T 约束: > 0
    OP_CHECK_IF(
        qfmlaInfo.qTSize <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "q", std::to_string(qfmlaInfo.qTSize).c_str(), "Q_T must be > 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q shape 维度值校验
// ============================================================================

ge::graphStatus CommonChecker::CheckQueryShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    const auto& qShape = qfmlaInfo.opParamInfo.query.shape->GetStorageShape();
    if (qfmlaInfo.layoutQ == QfmlaLayout::BSND) {
        // BSND: q shape 应为 (B, Q_S, Q_N, head_dim_qk=576)
        OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(0)) != qfmlaInfo.bSize ||
                        static_cast<int64_t>(qShape.GetDim(1)) != qfmlaInfo.s1Size ||
                        static_cast<int64_t>(qShape.GetDim(2)) != qfmlaInfo.n1Size ||
                        static_cast<int64_t>(qShape.GetDim(3)) != QFMLA_HEAD_DIM_QK,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                          "When layout_q is BSND, q shape must be (B, Q_S, Q_N, 576)"),
                    return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutQ == QfmlaLayout::BNSD) {
        // BNSD: q shape 应为 (B, Q_N, Q_S, head_dim_qk=576)
        OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(0)) != qfmlaInfo.bSize ||
                        static_cast<int64_t>(qShape.GetDim(1)) != qfmlaInfo.n1Size ||
                        static_cast<int64_t>(qShape.GetDim(2)) != qfmlaInfo.s1Size ||
                        static_cast<int64_t>(qShape.GetDim(3)) != QFMLA_HEAD_DIM_QK,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                          "When layout_q is BNSD, q shape must be (B, Q_N, Q_S, 576)"),
                    return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutQ == QfmlaLayout::BSH) {
        // BSH(arch22): q shape 应为 (B, Q_S, Q_N*512)
        OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(0)) != qfmlaInfo.bSize ||
                        static_cast<int64_t>(qShape.GetDim(1)) != qfmlaInfo.s1Size ||
                        static_cast<int64_t>(qShape.GetDim(2)) != qfmlaInfo.n1Size * 512,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                          "When layout_q is BSH, q shape must be (B, Q_S, Q_N * 512)"),
                    return ge::GRAPH_FAILED);
    } else {
        // TND: q shape 应为 (Q_T, Q_N, head_dim_qk)；arch22 int8 全量化时 head_dim_qk 为 nope 维度 512
        const bool isArch22 = qfmlaInfo.npuArch == NpuArch::DAV_2201;
        const int64_t expectHeadDimQk = isArch22 ? 512 : QFMLA_HEAD_DIM_QK;
        OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(0)) != qfmlaInfo.qTSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                          "The dim0 of q shape must be Q_T"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(1)) != qfmlaInfo.n1Size,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, QUERY_NAME, ToString(qShape).c_str(),
                                                          "The dim1 of q shape must be Q_N"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<int64_t>(qShape.GetDim(2)) != expectHeadDimQk,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, QUERY_NAME, ToString(qShape).c_str(),
                        isArch22 ? "The dim2 of q shape must be head_dim_qk(512) in MLA_INT8_FULLQUANT scenario" :
                                   "The dim2 of q shape must be head_dim_qk(576)"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — attn_out shape 维度值校验
// ============================================================================

ge::graphStatus CommonChecker::CheckAttnOutShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    const auto& attnOutShape = qfmlaInfo.opParamInfo.attnOut.shape->GetStorageShape();
    // layout_out=BSND: (B, Q_S, Q_N, head_dim_v=512)
    if (qfmlaInfo.layoutOut == QfmlaOutLayout::BSND) {
        OP_CHECK_IF(
            static_cast<int64_t>(attnOutShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(attnOutShape.GetDim(1)) != qfmlaInfo.s1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(2)) != qfmlaInfo.n1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(3)) != QFMLA_HEAD_DIM_V,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, ATTN_OUT_NAME, ToString(attnOutShape).c_str(),
                                                  "When layout_out is BSND, attn_out shape must be (B, Q_S, Q_N, 512)"),
            return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutOut == QfmlaOutLayout::BNSD) {
        // layout_out=BNSD: (B, Q_N, Q_S, head_dim_v=512)
        OP_CHECK_IF(
            static_cast<int64_t>(attnOutShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(attnOutShape.GetDim(1)) != qfmlaInfo.n1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(2)) != qfmlaInfo.s1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(3)) != QFMLA_HEAD_DIM_V,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, ATTN_OUT_NAME, ToString(attnOutShape).c_str(),
                                                  "When layout_out is BNSD, attn_out shape must be (B, Q_N, Q_S, 512)"),
            return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutOut == QfmlaOutLayout::TND) {
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
    } else if (qfmlaInfo.layoutOut == QfmlaOutLayout::BSH) {
        // BSH(arch22): attn_out shape 应为 (B, Q_S, Q_N*512)
        OP_CHECK_IF(
            static_cast<int64_t>(attnOutShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(attnOutShape.GetDim(1)) != qfmlaInfo.s1Size ||
                static_cast<int64_t>(attnOutShape.GetDim(2)) != qfmlaInfo.n1Size * QFMLA_HEAD_DIM_V,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, ATTN_OUT_NAME, ToString(attnOutShape).c_str(),
                                                  "When layout_out is BSH, attn_out shape must be (B, Q_S, Q_N * 512)"),
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
// MultiPara — q/k_cache dtype 一致性、layout_q/layout_out 配对一致性
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

ge::graphStatus CommonChecker::CheckLayoutPairConsistency(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // layout_q 与 layout_out 配对一致性: 仅支持 4 种组合
    // BSND->BSND / BNSD->BNSD / TND->TND / TND->NTD(即 TND_NTD, 下划线格式表示输入输出 layout)
    // arch22 int8全量化: BSH->BSH / TND->TND
    const bool layoutPairValid =
        (qfmlaInfo.layoutQ == QfmlaLayout::TND && qfmlaInfo.layoutOut == QfmlaOutLayout::TND) ||
        (qfmlaInfo.layoutQ == QfmlaLayout::TND && qfmlaInfo.layoutOut == QfmlaOutLayout::NTD) ||
        (qfmlaInfo.layoutQ == QfmlaLayout::BSND && qfmlaInfo.layoutOut == QfmlaOutLayout::BSND) ||
        (qfmlaInfo.layoutQ == QfmlaLayout::BNSD && qfmlaInfo.layoutOut == QfmlaOutLayout::BNSD) ||
        (qfmlaInfo.layoutQ == QfmlaLayout::BSH && qfmlaInfo.layoutOut == QfmlaOutLayout::BSH);
    OP_CHECK_IF(!layoutPairValid,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "layout_out", "invalid",
                    "layout_q and layout_out must be one of BSND->BSND, BNSD->BNSD, TND->TND, TND->NTD (TND_NTD), "
                    "BSH->BSH (int8)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckDtypeConsistency(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckLayoutPairConsistency(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

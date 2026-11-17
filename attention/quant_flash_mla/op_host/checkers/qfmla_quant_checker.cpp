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
 * \file qfmla_quant_checker.cpp
 * \brief Checker for quant_mode, q_descale, k_descale ( 全量化参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_quant_checker.h"

namespace optiling {
namespace quant_flash_mla {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus QuantChecker::CheckSingleParaQuantMode(const QfmlaTilingInfo& qfmlaInfo)
{
    // quant_mode: 支持 1 (MLA_FP8_E4M3_FULLQUANT)、0 (MLA_HIF8_FULLQUANT)、2 (MLA_INT8_FULLQUANT, arch22)
    // (取值合法性已在 parser 中校验, q/k dtype 匹配在 CommonChecker::CheckSingleParaDtype 中校验)
    (void)qfmlaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaQDescale(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // q_descale dtype: float32
    OP_CHECK_IF(qfmlaInfo.opParamInfo.qDescale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, Q_DESCALE_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.qDescale.desc->GetDataType()).c_str(),
                    "The dtype of q_descale must be float32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.qDescale.desc, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaKDescale(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // k_descale dtype: float32
    OP_CHECK_IF(qfmlaInfo.opParamInfo.kDescale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, K_DESCALE_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.kDescale.desc->GetDataType()).c_str(),
                    "The dtype of k_descale must be float32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.kDescale.desc, K_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaRope(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // 芯片差异用 npuArch 区分: rope 分离输入为 arch22 int8 全量化场景能力
    const bool isArch22 = qfmlaInfo.npuArch == NpuArch::DAV_2201;
    // fp8/hif8: rope 合一进 q/k_cache, 不支持传入
    OP_CHECK_IF(
        !isArch22 && (qfmlaInfo.opParamInfo.qRope.desc != nullptr || qfmlaInfo.opParamInfo.kRope.desc != nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, Q_ROPE_NAME, "provided",
                                              "q_rope/k_rope is only supported in MLA_INT8_FULLQUANT "
                                              "scenario"),
        return ge::GRAPH_FAILED);
    if (!isArch22) {
        return ge::GRAPH_SUCCESS;
    }
    // int8: rope dtype 必须为 bf16（对齐IFA现状: quant场景 rope BF16）
    for (const auto& ropeInfo : {std::make_pair(&qfmlaInfo.opParamInfo.qRope, Q_ROPE_NAME),
                                 std::make_pair(&qfmlaInfo.opParamInfo.kRope, K_ROPE_NAME)}) {
        if (ropeInfo.first->desc == nullptr) {
            continue;
        }
        OP_CHECK_IF(ropeInfo.first->desc->GetDataType() != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, ropeInfo.second,
                        ge::TypeUtils::DataTypeToSerialString(ropeInfo.first->desc->GetDataType()).c_str(),
                        "The dtype of rope must be bfloat16 in MLA_INT8_FULLQUANT scenario"),
                    return ge::GRAPH_FAILED);
        if (CheckFormatSupport(ropeInfo.first->desc, ropeInfo.second) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckSingleParaQuantMode(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaQDescale(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaKDescale(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaRope(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus QuantChecker::CheckParaExistence(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // q_descale, k_descale 为必选参数 (parser 已检查, 此处补检查)
    OP_CHECK_IF(qfmlaInfo.opParamInfo.qDescale.desc == nullptr || qfmlaInfo.opParamInfo.qDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName, Q_DESCALE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfmlaInfo.opParamInfo.kDescale.desc == nullptr || qfmlaInfo.opParamInfo.kDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName, K_DESCALE_NAME), return ge::GRAPH_FAILED);
    // 芯片差异用 npuArch 区分: rope 分离输入为 arch22 int8 全量化场景能力
    const bool isArch22 = qfmlaInfo.npuArch == NpuArch::DAV_2201;
    if (isArch22) {
        // arch22 int8全量化: rope分离必传（对齐IFA现状: quant场景 rope 必传）; KV descale统一由k_descale承载
        OP_CHECK_IF(qfmlaInfo.opParamInfo.qRope.desc == nullptr || qfmlaInfo.opParamInfo.kRope.desc == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName, Q_ROPE_NAME), return ge::GRAPH_FAILED);
        // rope要么成对传入要么都不传
        OP_CHECK_IF((qfmlaInfo.opParamInfo.qRope.desc == nullptr) != (qfmlaInfo.opParamInfo.kRope.desc == nullptr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, Q_ROPE_NAME, "partial",
                                                          "q_rope and k_rope must be provided together"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q/k dtype 与 quant_mode 精确匹配
// ============================================================================

ge::graphStatus QuantChecker::CheckFeature(const QfmlaTilingInfo& qfmlaInfo)
{
    // quant_mode=1 (FP8_E4M3) 时, q/k_cache 必须为 float8_e4m3fn
    // (dtype 校验已在 CommonChecker::CheckSingleParaDtype 中完成, 此处不重复)
    (void)qfmlaInfo;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — q_descale shape 与 layout_q 匹配
// ============================================================================

ge::graphStatus QuantChecker::CheckQDescaleShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // q_descale shape: TND->(Q_T, Q_N), BSND->(B, Q_S, Q_N), BNSD->(B, Q_N, Q_S), BSH(arch22)->(B, Q_S, Q_N)
    const auto& qDescaleShape = qfmlaInfo.opParamInfo.qDescale.shape->GetStorageShape();
    if (qfmlaInfo.layoutQ == QfmlaLayout::BSND) {
        OP_CHECK_IF(qDescaleShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName, Q_DESCALE_NAME, (std::to_string(qDescaleShape.GetDimNum()) + "D").c_str(),
                        "The shape dim of q_descale must be 3D when layout_q is BSND"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qfmlaInfo.s1Size ||
                static_cast<int64_t>(qDescaleShape.GetDim(2)) != qfmlaInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (B, Q_S, Q_N) when layout_q is BSND"),
            return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutQ == QfmlaLayout::BNSD) {
        OP_CHECK_IF(qDescaleShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName, Q_DESCALE_NAME, (std::to_string(qDescaleShape.GetDimNum()) + "D").c_str(),
                        "The shape dim of q_descale must be 3D when layout_q is BNSD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qfmlaInfo.n1Size ||
                static_cast<int64_t>(qDescaleShape.GetDim(2)) != qfmlaInfo.s1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (B, Q_N, Q_S) when layout_q is BNSD"),
            return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.layoutQ == QfmlaLayout::BSH) {
        // BSH(arch22): q_descale shape 为 (B, Q_S, Q_N)（per-token-head, BSH时H维拆为N）
        OP_CHECK_IF(qDescaleShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName, Q_DESCALE_NAME, (std::to_string(qDescaleShape.GetDimNum()) + "D").c_str(),
                        "The shape dim of q_descale must be 3D when layout_q is BSH"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qfmlaInfo.s1Size ||
                static_cast<int64_t>(qDescaleShape.GetDim(2)) != qfmlaInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (B, Q_S, Q_N) when layout_q is BSH"),
            return ge::GRAPH_FAILED);
    } else {
        // TND 时 q_descale shape 为 (Q_T, Q_N)
        OP_CHECK_IF(qDescaleShape.GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName, Q_DESCALE_NAME, (std::to_string(qDescaleShape.GetDimNum()) + "D").c_str(),
                        "The shape dim of q_descale must be 2D when layout_q is TND"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<int64_t>(qDescaleShape.GetDim(0)) != qfmlaInfo.qTSize ||
                static_cast<int64_t>(qDescaleShape.GetDim(1)) != qfmlaInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                  "The shape of q_descale must be (Q_T, Q_N) when layout_q is TND"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckKDescaleShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // k_descale 为 per-tensor 静态量化，shape 固定为 (1,)
    const auto& kDescaleShape = qfmlaInfo.opParamInfo.kDescale.shape->GetStorageShape();
    OP_CHECK_IF(kDescaleShape.GetDimNum() != 1 || static_cast<int64_t>(kDescaleShape.GetDim(0)) != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, K_DESCALE_NAME, ToString(kDescaleShape).c_str(),
                    "The shape of k_descale must be (1,) as it is per-tensor static quantization"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckRopeShape(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // 芯片差异用 npuArch 区分: rope 分离输入为 arch22 int8 全量化场景能力
    const bool isArch22 = qfmlaInfo.npuArch == NpuArch::DAV_2201;
    if (!isArch22 || qfmlaInfo.opParamInfo.qRope.shape == nullptr || qfmlaInfo.opParamInfo.kRope.shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    constexpr int64_t ROPE_DIM = 64;
    // q_rope: BSH->(B, Q_S, Q_N, 64) 4D / TND->(Q_T, Q_N, 64) 3D
    const auto& qRopeShape = qfmlaInfo.opParamInfo.qRope.shape->GetStorageShape();
    if (qfmlaInfo.layoutQ == QfmlaLayout::BSH) {
        OP_CHECK_IF(
            qRopeShape.GetDimNum() != 4 || static_cast<int64_t>(qRopeShape.GetDim(0)) != qfmlaInfo.bSize ||
                static_cast<int64_t>(qRopeShape.GetDim(1)) != qfmlaInfo.s1Size ||
                static_cast<int64_t>(qRopeShape.GetDim(2)) != qfmlaInfo.n1Size ||
                static_cast<int64_t>(qRopeShape.GetDim(3)) != ROPE_DIM,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_ROPE_NAME, ToString(qRopeShape).c_str(),
                                                  "When layout_q is BSH, q_rope shape must be (B, Q_S, Q_N, 64)"),
            return ge::GRAPH_FAILED);
    } else { // TND
        OP_CHECK_IF(qRopeShape.GetDimNum() != 3 || static_cast<int64_t>(qRopeShape.GetDim(0)) != qfmlaInfo.qTSize ||
                        static_cast<int64_t>(qRopeShape.GetDim(1)) != qfmlaInfo.n1Size ||
                        static_cast<int64_t>(qRopeShape.GetDim(2)) != ROPE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_ROPE_NAME, ToString(qRopeShape).c_str(),
                                                          "When layout_q is TND, q_rope shape must be (Q_T, Q_N, 64)"),
                    return ge::GRAPH_FAILED);
    }
    // k_rope: PA_NZ->(Bn, 1, 4, Bs, 16) 5D, 与 k_cache PA_NZ 布局匹配
    const auto& kRopeShape = qfmlaInfo.opParamInfo.kRope.shape->GetStorageShape();
    OP_CHECK_IF(kRopeShape.GetDimNum() != 5,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, K_ROPE_NAME,
                                                         (std::to_string(kRopeShape.GetDimNum()) + "D").c_str(),
                                                         "The shape dim of k_rope must be 5D when layout_kv is PA_NZ"),
                return ge::GRAPH_FAILED);
    const auto& kCacheShape = qfmlaInfo.opParamInfo.kCache.shape->GetStorageShape();
    // NZ cache: (Bn, N, D/16, Bs, 16) -> rope cache: (Bn, N, 64/16=4, Bs, 16)
    OP_CHECK_IF(static_cast<int64_t>(kRopeShape.GetDim(0)) != qfmlaInfo.totalBlockNum ||
                    static_cast<int64_t>(kRopeShape.GetDim(1)) != qfmlaInfo.n2Size ||
                    static_cast<int64_t>(kRopeShape.GetDim(2)) != ROPE_DIM / 16 ||
                    static_cast<int64_t>(kRopeShape.GetDim(3)) != qfmlaInfo.blockSize ||
                    static_cast<int64_t>(kRopeShape.GetDim(4)) != 16,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, K_ROPE_NAME, ToString(kRopeShape).c_str(),
                    "When layout_kv is PA_NZ, k_rope shape must be (Bn, 1, 4, Bs, 16) matching k_cache"),
                return ge::GRAPH_FAILED);
    (void)kCacheShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckQDescaleShape(qfmlaInfo) != ge::GRAPH_SUCCESS || CheckKDescaleShape(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckRopeShape(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

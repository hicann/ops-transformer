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
    // quant_mode: 当前仅支持 1 (MLA_FP8_E4M3_FULLQUANT), 不支持 0 (HIF8)
    OP_CHECK_IF(qfmlaInfo.quantMode != QfmlaQuantMode::MLA_FP8_E4M3_FULLQUANT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qfmlaInfo.opName, "quant_mode", std::to_string(static_cast<int64_t>(qfmlaInfo.quantMode)).c_str(),
                    "quant_mode must be 1 (MLA_FP8_E4M3_FULLQUANT), HIF8 (quant_mode=0) is not supported"),
                return ge::GRAPH_FAILED);
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

ge::graphStatus QuantChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckSingleParaQuantMode(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaQDescale(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaKDescale(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus QuantChecker::CheckParaExistence(const QfmlaTilingInfo& qfmlaInfo)
{
    // q_descale, k_descale 为必选参数 (parser 已检查, 此处补检查)
    OP_CHECK_IF(qfmlaInfo.opParamInfo.qDescale.desc == nullptr || qfmlaInfo.opParamInfo.qDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfmlaInfo.opName, Q_DESCALE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfmlaInfo.opParamInfo.kDescale.desc == nullptr || qfmlaInfo.opParamInfo.kDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfmlaInfo.opName, K_DESCALE_NAME), return ge::GRAPH_FAILED);
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
    // TND 时 q_descale shape 为 (Q_T, Q_N)
    const auto& qDescaleShape = qfmlaInfo.opParamInfo.qDescale.shape->GetStorageShape();
    OP_CHECK_IF(qDescaleShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, Q_DESCALE_NAME,
                                                         (std::to_string(qDescaleShape.GetDimNum()) + "D").c_str(),
                                                         "The shape dim of q_descale must be 2D when layout_q is TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(qDescaleShape.GetDim(0)) != qfmlaInfo.qTSize ||
                    static_cast<int64_t>(qDescaleShape.GetDim(1)) != qfmlaInfo.n1Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                      "The shape of q_descale must be (Q_T, Q_N) when layout_q is TND"),
                return ge::GRAPH_FAILED);
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

ge::graphStatus QuantChecker::CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckQDescaleShape(qfmlaInfo) != ge::GRAPH_SUCCESS || CheckKDescaleShape(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

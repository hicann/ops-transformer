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
 * \file quant_checker.cpp
 * \brief Checker for quant_mode, q_descale, k_descale ( 全量化参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qmla_tiling_info.h"
#include "../qmla_tiling_info_parser.h"
#include "quant_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus QuantChecker::CheckSingleParaQuantMode(const QmlaTilingInfo& qmlaInfo)
{
    // quant_mode: 当前仅支持 1 (MLA_FP8_E4M3_FULLQUANT), 不支持 0 (HIF8)
    OP_CHECK_IF(qmlaInfo.quantMode != QmlaQuantMode::MLA_FP8_E4M3_FULLQUANT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qmlaInfo.opName, "quant_mode", std::to_string(static_cast<int64_t>(qmlaInfo.quantMode)).c_str(),
                    "quant_mode must be 1 (MLA_FP8_E4M3_FULLQUANT), HIF8 (quant_mode=0) is not supported"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaQDescale(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // q_descale dtype: float32
    OP_CHECK_IF(qmlaInfo.opParamInfo.qDescale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, Q_DESCALE_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qmlaInfo.opParamInfo.qDescale.desc->GetDataType()).c_str(),
                    "The dtype of q_descale must be float32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qmlaInfo.opParamInfo.qDescale.desc, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaKDescale(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // k_descale dtype: float32
    OP_CHECK_IF(qmlaInfo.opParamInfo.kDescale.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, K_DESCALE_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qmlaInfo.opParamInfo.kDescale.desc->GetDataType()).c_str(),
                    "The dtype of k_descale must be float32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qmlaInfo.opParamInfo.kDescale.desc, K_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSinglePara(const QmlaTilingInfo& qmlaInfo)
{
    if (CheckSingleParaQuantMode(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaQDescale(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaKDescale(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus QuantChecker::CheckParaExistence(const QmlaTilingInfo& qmlaInfo)
{
    // q_descale, k_descale 为必选参数 (parser 已检查, 此处补检查)
    OP_CHECK_IF(qmlaInfo.opParamInfo.qDescale.desc == nullptr || qmlaInfo.opParamInfo.qDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qmlaInfo.opName, Q_DESCALE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qmlaInfo.opParamInfo.kDescale.desc == nullptr || qmlaInfo.opParamInfo.kDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qmlaInfo.opName, K_DESCALE_NAME), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q/k dtype 与 quant_mode 精确匹配
// ============================================================================

ge::graphStatus QuantChecker::CheckFeature(const QmlaTilingInfo& qmlaInfo)
{
    // quant_mode=1 (FP8_E4M3) 时, q/k_cache 必须为 float8_e4m3fn
    // (dtype 校验已在 CommonChecker::CheckSingleParaDtype 中完成, 此处不重复)
    (void)qmlaInfo;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — q_descale shape 与 layout_q 匹配
// ============================================================================

ge::graphStatus QuantChecker::CheckQDescaleShape(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // TND 时 q_descale shape 为 (Q_T, Q_N)
    const auto& qDescaleShape = qmlaInfo.opParamInfo.qDescale.shape->GetStorageShape();
    OP_CHECK_IF(qDescaleShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, Q_DESCALE_NAME,
                                                         (std::to_string(qDescaleShape.GetDimNum()) + "D").c_str(),
                                                         "The shape dim of q_descale must be 2D when layout_q is TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(qDescaleShape.GetDim(0)) != qmlaInfo.qTSize ||
                    static_cast<int64_t>(qDescaleShape.GetDim(1)) != qmlaInfo.n1Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, Q_DESCALE_NAME, ToString(qDescaleShape).c_str(),
                                                      "The shape of q_descale must be (Q_T, Q_N) when layout_q is TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckKDescaleShape(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // k_descale 为 per-tensor 静态量化，shape 固定为 (1,)
    const auto& kDescaleShape = qmlaInfo.opParamInfo.kDescale.shape->GetStorageShape();
    OP_CHECK_IF(kDescaleShape.GetDimNum() != 1 || static_cast<int64_t>(kDescaleShape.GetDim(0)) != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, K_DESCALE_NAME, ToString(kDescaleShape).c_str(),
                    "The shape of k_descale must be (1,) as it is per-tensor static quantization"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const QmlaTilingInfo& qmlaInfo)
{
    if (CheckQDescaleShape(qmlaInfo) != ge::GRAPH_SUCCESS || CheckKDescaleShape(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling

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
 * \file qfmla_softmax_lse_checker.cpp
 * \brief Checker for return_softmax_lse, softmax_lse ( SoftmaxLSE参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_softmax_lse_checker.h"

namespace optiling {
namespace quant_flash_mla {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    // softmax_lse dtype: float32 (传入时校验)
    if (qfmlaInfo.opParamInfo.softmaxLse.desc != nullptr) {
        OP_CHECK_IF(
            qfmlaInfo.opParamInfo.softmaxLse.desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                qfmlaInfo.opName, SOFTMAX_LSE_NAME,
                ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.softmaxLse.desc->GetDataType()).c_str(),
                "The dtype of softmax_lse must be float32"),
            return ge::GRAPH_FAILED);
        // format: ND
        if (CheckFormatSupport(qfmlaInfo.opParamInfo.softmaxLse.desc, SOFTMAX_LSE_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — return_softmax_lse 与 softmax_lse 一致性
// ============================================================================

ge::graphStatus SoftmaxLSEChecker::CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    if (qfmlaInfo.returnSoftmaxLse) {
        // return_softmax_lse=True 时, softmax_lse 必须非空
        OP_CHECK_IF(
            qfmlaInfo.opParamInfo.softmaxLse.shape == nullptr ||
                qfmlaInfo.opParamInfo.softmaxLse.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, SOFTMAX_LSE_NAME, "not empty",
                                                  "When return_softmax_lse is True, softmax_lse must not be empty"),
            return ge::GRAPH_FAILED);
        const auto& lseShape = qfmlaInfo.opParamInfo.softmaxLse.shape->GetStorageShape();
        OP_CHECK_IF(lseShape.GetDimNum() != 2 || static_cast<int64_t>(lseShape.GetDim(0)) != qfmlaInfo.n1Size ||
                        static_cast<int64_t>(lseShape.GetDim(1)) != qfmlaInfo.qTSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, SOFTMAX_LSE_NAME, ToString(lseShape).c_str(),
                        "When return_softmax_lse is True and layout_q is TND, softmax_lse shape must be (Q_N, Q_T)"),
                    return ge::GRAPH_FAILED);
    } else if (qfmlaInfo.opParamInfo.softmaxLse.shape != nullptr) {
        const auto& lseShape = qfmlaInfo.opParamInfo.softmaxLse.shape->GetStorageShape();
        OP_CHECK_IF(
            lseShape.GetShapeSize() != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, SOFTMAX_LSE_NAME, ToString(lseShape).c_str(),
                                                  "When return_softmax_lse is False, softmax_lse must be empty"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

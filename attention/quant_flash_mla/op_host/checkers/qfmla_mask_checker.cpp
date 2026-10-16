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
 * \file qfmla_mask_checker.cpp
 * \brief Checker for mask_mode, attn_mask ( Mask参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_mask_checker.h"

namespace optiling {
namespace quant_flash_mla {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus MaskChecker::CheckSingleParaMaskMode(const QfmlaTilingInfo& qfmlaInfo)
{
    // mask_mode: 仅支持 0 (NO_MASK) / 3 (CAUSAL)
    OP_CHECK_IF(
        qfmlaInfo.maskMode != QFMLA_MASK_MODE_NO_MASK && qfmlaInfo.maskMode != QFMLA_MASK_MODE_CAUSAL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfmlaInfo.opName, "mask_mode", std::to_string(qfmlaInfo.maskMode).c_str(),
                                              "mask_mode must be 0 (NO_MASK) or 3 (CAUSAL)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaAttnMask(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // attn_mask 为可选输入，未传入时跳过
    if (!qfmlaInfo.attnMaskFlag) {
        return ge::GRAPH_SUCCESS;
    }
    // dtype: int8
    OP_CHECK_IF(qfmlaInfo.opParamInfo.attnMask.desc->GetDataType() != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, ATTN_MASK_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.attnMask.desc->GetDataType()).c_str(),
                    "The dtype of attn_mask must be int8"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.attnMask.desc, ATTN_MASK_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (2048, 2048)
    OP_CHECK_IF(
        qfmlaInfo.attenMaskS1Size != 2048 || qfmlaInfo.attenMaskS2Size != 2048,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, ATTN_MASK_NAME,
            (std::to_string(qfmlaInfo.attenMaskS1Size) + "x" + std::to_string(qfmlaInfo.attenMaskS2Size)).c_str(),
            "The shape of attn_mask must be (2048, 2048)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckSingleParaMaskMode(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaAttnMask(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — mask_mode 与 attn_mask 存在性关系
// ============================================================================

ge::graphStatus MaskChecker::CheckMaskModeAttnMaskConsistency(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // mask_mode=0 (NO_MASK): 不支持传入 attn_mask
    // mask_mode=3 (CAUSAL): 必须传入 attn_mask
    if (qfmlaInfo.maskMode == QFMLA_MASK_MODE_NO_MASK) {
        OP_CHECK_IF(qfmlaInfo.attnMaskFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, ATTN_MASK_NAME, "provided",
                                                          "When mask_mode is 0 (NO_MASK), attn_mask is not supported"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(!qfmlaInfo.attnMaskFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, ATTN_MASK_NAME, "provided",
                                                          "When mask_mode is 3 (CAUSAL), attn_mask must be provided"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckMaskModeAttnMaskConsistency(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling

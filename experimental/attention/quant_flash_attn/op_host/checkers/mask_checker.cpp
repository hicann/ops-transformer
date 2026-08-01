/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mask_checker.cpp
 * \brief Checker for mask_mode, attn_mask, win_left, win_right (文档约束: Mask参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../quant_flash_attn_tiling_info.h"
#include "mask_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;

ge::graphStatus MaskChecker::CheckSingleParaMaskMode(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // data_type 支持 INT32；支持输入为 0/3/4
    const std::vector<uint32_t> supportedMaskModes = {static_cast<uint32_t>(MaskMode::NO_MASK),
                                                      static_cast<uint32_t>(MaskMode::CAUSAL),
                                                      static_cast<uint32_t>(MaskMode::BAND)};
    OP_CHECK_IF(std::find(supportedMaskModes.begin(), supportedMaskModes.end(), qfaInfo.maskMode) ==
                    supportedMaskModes.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, MASK_MODE_NAME.c_str(),
                                                      std::to_string(qfaInfo.maskMode).c_str(),
                                                      "The value of mask_mode must be in {0, 3, 4}"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaAttnMask(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // tensor_type 支持 INT8；tensor_shape 为 (2048, 2048)
    const gert::Tensor *attnMaskTensor = qfaInfo.opParamInfo.attnMask.tensor;
    const gert::CompileTimeTensorDesc *attnMaskDesc = qfaInfo.opParamInfo.attnMask.desc;
    if (attnMaskTensor == nullptr || attnMaskDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(attnMaskDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, ATTEN_MASK_NAME.c_str(),
                                          DataTypeToSerialStr(attnMaskDesc->GetDataType()).c_str(), "INT8"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(attnMaskDesc, ATTEN_MASK_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // shape dim 必须为 2，且 shape 为 (2048, 2048)
    const gert::Shape &shape = attnMaskTensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, ATTEN_MASK_NAME.c_str(),
                                             (std::to_string(shape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    const int64_t ATTN_MASK_DIM0 = 2048;
    const int64_t ATTN_MASK_DIM1 = 2048;
    OP_CHECK_IF(shape.GetDim(0) != ATTN_MASK_DIM0 || shape.GetDim(1) != ATTN_MASK_DIM1,
                OP_LOGE(qfaInfo.opName, "%s shape must be (%ld, %ld), but got (%ld, %ld).", ATTEN_MASK_NAME.c_str(),
                        ATTN_MASK_DIM0, ATTN_MASK_DIM1, shape.GetDim(0), shape.GetDim(1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaWindowParams(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // data_type 支持 INT32；值需要 >= -1
    // win_left / win_right 为可选属性，默认值为 -1（表示无穷）
    OP_CHECK_IF(qfaInfo.winLeft < -1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, WIN_LEFT_NAME.c_str(),
                                                      std::to_string(qfaInfo.winLeft).c_str(),
                                                      "The value of win_left must be >= -1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.winRight < -1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, WIN_RIGHT_NAME.c_str(),
                                                      std::to_string(qfaInfo.winRight).c_str(),
                                                      "The value of win_right must be >= -1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckSingleParaMaskMode(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaAttnMask(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaWindowParams(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMaskModeQuantMode(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (qfaInfo.opParamInfo.quantMode == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }
    int64_t quantModeVal = *qfaInfo.opParamInfo.quantMode;
    if (quantModeVal == QUANT_MODE_MXFP8) { // MxFP8
        OP_CHECK_IF(qfaInfo.maskMode == static_cast<uint32_t>(MaskMode::BAND),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, MASK_MODE_NAME.c_str(), std::to_string(qfaInfo.maskMode).c_str(),
                        "MxFP8 only supports mask_mode 0 (NO_MASK) and 3 (CAUSAL), BAND(4) is not supported"),
                    return ge::GRAPH_FAILED);
    } else if (quantModeVal == QUANT_MODE_MXFP4) { // MxFP4
        OP_CHECK_IF(qfaInfo.maskMode != static_cast<uint32_t>(MaskMode::NO_MASK),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, MASK_MODE_NAME.c_str(),
                        std::to_string(qfaInfo.maskMode).c_str(),
                        "MxFP4 only supports mask_mode 0 (NO_MASK)"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckMaskModeQuantMode(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMaskModeAttnMaskConsistency(const QuantFlashAttnTilingInfo &qfaInfo)
{
    bool attnMaskExists =
        (qfaInfo.opParamInfo.attnMask.tensor != nullptr && qfaInfo.opParamInfo.attnMask.desc != nullptr);
    if (qfaInfo.maskMode == static_cast<uint32_t>(MaskMode::NO_MASK)) {
        OP_CHECK_IF(attnMaskExists,
                    OP_LOGE(qfaInfo.opName, "When mask_mode is 0 (NO_MASK), attn_mask should not be provided."),
                    return ge::GRAPH_FAILED);
    } else {
        // mask_mode 为 3 (CAUSAL) 或 4 (BAND) 时，必须传入 attn_mask
        OP_CHECK_IF(!attnMaskExists, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, ATTEN_MASK_NAME.c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckMaskModeAttnMaskConsistency(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling

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
 * \file mask_checker.cpp
 * \brief Checker for mask_mode, attn_mask, win_left, win_right ( Mask参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "mask_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

ge::graphStatus MaskChecker::CheckSingleParaMaskMode(const QfaTilingInfo &qfaInfo)
{
    //  data_type 支持 INT32；支持输入为 0/3/4
    // qfaInfo.maskMode 为已解析的 int64_t（默认 0）
    const std::vector<int64_t> supportedMaskModes = {static_cast<int64_t>(MaskMode::NO_MASK),
                                                     static_cast<int64_t>(MaskMode::CAUSAL),
                                                     static_cast<int64_t>(MaskMode::SLIDING_WINDOW)};
    OP_CHECK_IF(
        std::find(supportedMaskModes.begin(), supportedMaskModes.end(), qfaInfo.maskMode) == supportedMaskModes.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, MASK_MODE_NAME.c_str(),
                                              std::to_string(qfaInfo.maskMode).c_str(),
                                              "The value of mask_mode must be in {0, 3, 4}"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaAttnMask(const QfaTilingInfo &qfaInfo)
{
    //  tensor_type 支持 INT8；tensor_shape 为 (2048, 2048)
    // attn_mask 为可选输入，未传入时跳过（存在性校验负责）
    const gert::Tensor *attnMaskTensor = qfaInfo.opParamInfo.attnMask.tensor;
    const gert::CompileTimeTensorDesc *attnMaskDesc = qfaInfo.opParamInfo.attnMask.desc;
    if (attnMaskTensor == nullptr || attnMaskDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(attnMaskDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, ATTN_MASK_NAME.c_str(),
                                          DataTypeToSerialString(attnMaskDesc->GetDataType()).c_str(), "INT8"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(attnMaskDesc, ATTN_MASK_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // shape dim 必须为 2，且 shape 为 (2048, 2048)
    const gert::Shape &shape = attnMaskTensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, ATTN_MASK_NAME.c_str(),
                                             (std::to_string(shape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    const int64_t ATTN_MASK_DIM0 = 2048;
    const int64_t ATTN_MASK_DIM1 = 2048;
    OP_CHECK_IF(shape.GetDim(0) != ATTN_MASK_DIM0 || shape.GetDim(1) != ATTN_MASK_DIM1,
                OP_LOGE_FOR_INVALID_SHAPE(
                    qfaInfo.opName, ATTN_MASK_NAME.c_str(),
                    ("[" + std::to_string(shape.GetDim(0)) + ", " + std::to_string(shape.GetDim(1)) + "]").c_str(),
                    ("[" + std::to_string(ATTN_MASK_DIM0) + ", " + std::to_string(ATTN_MASK_DIM1) + "]").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaWindowParams(const QfaTilingInfo &qfaInfo)
{
    //  data_type 支持 INT32；值需要 >= -1
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

ge::graphStatus MaskChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    if (CheckSingleParaMaskMode(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaAttnMask(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaWindowParams(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    // 约束(存在性校验列):
    //   - mask_mode: 可选属性，默认值为 0
    //   - attn_mask: 可选输入
    //   - win_left / win_right: 可选属性，默认值为 -1
    // 以上参数均为可选，未传入时由 parser 填充默认值，此处无需强制存在。
    // mask_mode 与 attn_mask 的关系约束属于一致性校验(CheckMultiPara)。
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — mask_mode 与 attn_mask 的存在性关系校验 (文档"一致性校验"列)
// ============================================================================

ge::graphStatus MaskChecker::CheckMaskModeAttnMaskConsistency(const QfaTilingInfo &qfaInfo)
{
    //
    //   - mask_mode=0 (NO_MASK): 不支持传入 attn_mask
    //   - mask_mode=3 (CAUSAL):  必须传入 attn_mask 矩阵
    //   - mask_mode=4 (SLIDING_WINDOW):  必须传入 attn_mask 矩阵
    bool attnMaskExists =
        (qfaInfo.opParamInfo.attnMask.tensor != nullptr && qfaInfo.opParamInfo.attnMask.desc != nullptr);
    if (qfaInfo.maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        OP_CHECK_IF(
            attnMaskExists,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, ATTN_MASK_NAME.c_str(), "provided",
                                                  "When mask_mode is 0 (NO_MASK), attn_mask should not be provided"),
            return ge::GRAPH_FAILED);
    } else {
        // mask_mode 为 3 (CAUSAL) 或 4 (SLIDING_WINDOW) 时，必须传入 attn_mask
        OP_CHECK_IF(!attnMaskExists, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, ATTN_MASK_NAME.c_str()),
                    return ge::GRAPH_FAILED);
    }

    // 非 maskMode = 4 (SLIDING_WINDOW) 场景下 winLeft 和 winRight 必须为 -1
    if (qfaInfo.maskMode != static_cast<int64_t>(MaskMode::SLIDING_WINDOW)) {
        OP_CHECK_IF(qfaInfo.winLeft != -1 || qfaInfo.winRight != -1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, MASK_MODE_NAME.c_str(), std::to_string(qfaInfo.maskMode).c_str(),
                        ("When mask_mode is not 4 (SLIDING_WINDOW), win_left and win_right must be -1, "
                         "but got win_left=" +
                         std::to_string(qfaInfo.winLeft) + ", win_right=" + std::to_string(qfaInfo.winRight))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    if (CheckMaskModeAttnMaskConsistency(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — MxFP8 场景下 mask_mode 取值约束 (文档"特性交叉校验"列)
// ============================================================================

ge::graphStatus MaskChecker::CheckMaskModeQuantMode(const QfaTilingInfo &qfaInfo)
{
    //  MxFP8/GQA_FP8_FULLQUANT/HIF8 均仅支持 mask_mode 取 0 和 3 (不支持 SLIDING_WINDOW=4)
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 ||
        qfaInfo.quantMode ==
            QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 ||
        qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        OP_CHECK_IF(qfaInfo.maskMode == static_cast<int64_t>(MaskMode::SLIDING_WINDOW),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, MASK_MODE_NAME.c_str(),
                                                          std::to_string(qfaInfo.maskMode).c_str(),
                                                          "Current quant_mode"
                                                          " only supports mask_mode 0 (NO_MASK) and 3 (CAUSAL), "
                                                          "SLIDING_WINDOW(4) is not supported"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeature(const QfaTilingInfo &qfaInfo)
{
    if (CheckMaskModeQuantMode(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling

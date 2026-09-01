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
 * \file mixed_quant_sparse_flash_mla_check_feature.cpp
 * \brief
 */

#include "mixed_quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;
namespace optiling {

constexpr uint32_t QUANT_MODE_1 = 1;
constexpr uint32_t QUANT_MODE_2 = 1;

ge::graphStatus MQSMLATilingCheck::CheckFeatureWinKV() const
{
    OP_CHECK_IF(oriWinLeft_ != -1 && oriWinLeft_ < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ori_win_left", std::to_string(oriWinLeft_).c_str(),
                                                      "Ori_win_left only supports -1 or >=0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(oriWinRight_ != -1 && oriWinRight_ < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ori_win_right", std::to_string(oriWinLeft_).c_str(),
                                                      "Ori_win_right only supports -1 or >=0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeatureAntiquantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "batch_size", std::to_string(bSize_).c_str(),
                                                      "batch_size should be greater than 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == MQSMLALayout::TND),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "T_size of query", std::to_string(qTSize_).c_str(),
                                                      "T_size of query should be greater than 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "kv_head_num", std::to_string(n2Size_).c_str(),
                                                      "kv_head_num only support 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "q_head_num and kv_head_num",
                                                       std::to_string(n1Size_) + " and " + std::to_string(n2Size_),
                                                       "q_head_num must be divisible by kv_head_num"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSize_ != 512, // 512:当前不泛化
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "Head dim of input q", std::to_string(dSize_).c_str(),
                                                      "Head dim of input q only support 512"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSizeV_ != 512, // 512:当前不泛化
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "dSizeV", std::to_string(dSizeV_).c_str(),
                                                      "dSizeV only support 512"),
                return ge::GRAPH_FAILED);

    if (quant_mode_ == QUANT_MODE_1) {
        OP_CHECK_IF(
            dSizeVInput_ != KV_INPUT_DIM_LIMIT_QUANT_MODE_ONE,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                "When quant_mode is 1, dSizeVInput only supports " + std::to_string(KV_INPUT_DIM_LIMIT_QUANT_MODE_ONE) +
                    ", but got " + std::to_string(dSizeVInput_)),
            return ge::GRAPH_FAILED);
    } else if (quant_mode_ == QUANT_MODE_2) {
        OP_CHECK_IF(
            dSizeVInput_ != KV_INPUT_DIM_LIMIT_QUANT_MODE_TWO,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                "When quant_mode is 2, dSizeVInput only supports " + std::to_string(KV_INPUT_DIM_LIMIT_QUANT_MODE_TWO) +
                    ", but got " + std::to_string(dSizeVInput_)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeatureAntiquantLayout() const
{
    const std::vector<std::string> layoutSupportList = {"BSND", "TND"};
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", layoutQuery.c_str(),
                                                      "Layout_q only supports BSND or TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeatureAntiquantDtype() const
{
    OP_CHECK_IF(
        qType_ != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "query", MQSMLADataTypeToSerialString(qType_).c_str(),
                                              "query dtype only support " + MQSMLADataTypeToSerialString(ge::DT_BF16) +
                                                  " and " + MQSMLADataTypeToSerialString(ge::DT_FLOAT16)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeatureAntiquantAttr() const
{
    OP_CHECK_IF(
        *opParamInfo_.quantMode != QUANT_MODE_1 && *opParamInfo_.quantMode != QUANT_MODE_2,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "quant_mode", std::to_string(*opParamInfo_.quantMode).c_str(),
                                              "Quant_mode only support 1 and 2"),
        return ge::GRAPH_FAILED);

    if (*opParamInfo_.quantMode == QUANT_MODE_1 || *opParamInfo_.quantMode == QUANT_MODE_2) {
        OP_CHECK_IF(opParamInfo_.oriKv.desc->GetDataType() == ge::DT_HIFLOAT8, // 前面已校验dtype在fp8 e4m3和hif8之间
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "oriKv", "DT_HIFLOAT8",
                                                          "oriKv dtype only support DT_FLOAT8_E4M3FN"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(*opParamInfo_.ropeHeadDim != 64, // 64:当前不泛化
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "rope_head_dim",
                                                      std::to_string(*opParamInfo_.ropeHeadDim).c_str(),
                                                      "rope_head_dim only support 64"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeatureAntiquantPa() const
{
    if (kvLayout_ != MQSMLALayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                    "When page attention is enabled, cmpBlockSize_(" + std::to_string(oriBlockSize_) +
                        ") should be in range (0, " + std::to_string(MAX_BLOCK_SIZE) + "]"),
                return ge::GRAPH_FAILED);

    if (cmpBlockSize_ != 0) {
        OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cmp_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                        "When page attention is enabled, cmpBlockSize_(" + std::to_string(cmpBlockSize_) +
                            ") should be in range (0, " + std::to_string(MAX_BLOCK_SIZE) + "]"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeatureAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureAntiquantAttr() || ge::GRAPH_SUCCESS != CheckFeatureAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantLayout() || ge::GRAPH_SUCCESS != CheckFeatureAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureWinKV() || ge::GRAPH_SUCCESS != CheckFeatureAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckFeature() const
{
    return CheckFeatureAntiquant();
}

} // namespace optiling

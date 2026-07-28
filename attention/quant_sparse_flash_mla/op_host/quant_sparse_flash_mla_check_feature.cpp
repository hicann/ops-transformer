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
 * \file quant_sparse_flash_mla_check_feature.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;
namespace optiling {

ge::graphStatus QSMLATilingCheck::CheckFeatureWinKV() const { return ge::GRAPH_SUCCESS; }

ge::graphStatus QSMLATilingCheck::CheckFeatureShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "batch_size", std::to_string(bSize_).c_str(),
                                                      "batch_size should be greater than 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == QSMLALayout::TND),
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

    OP_CHECK_IF(dSizeVInput_ != 512, // 512:当前不泛化
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "dSizeVInput", std::to_string(dSizeVInput_).c_str(),
                                                      "dSizeVInput only support 512"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutSupportList = {"BSND", "TND"};
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
                OP_LOGE(opName_, "layoutQuery only support BSND/TND, but got %s", layoutQuery.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "query", QSMLADataTypeToSerialString(qType_).c_str(),
                    "query dtype only support " + QSMLADataTypeToSerialString(ge::DT_HIFLOAT8)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureQuantModeAndDtype() const
{
    // quant_mode=1: q, ori_kv, cmp_kv are all HIFLOAT8
    OP_CHECK_IF(*opParamInfo_.quantMode != 1,
                OP_LOGE(opName_, "quant_mode only support 1, but got %ld", *opParamInfo_.quantMode),
                return ge::GRAPH_FAILED);

    // quant_mode=1 下，Q、ori_kv 和 cmp_kv 数据类型都必须是 HIFLOAT8
    OP_CHECK_IF(qType_ != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "query", QSMLADataTypeToSerialString(qType_).c_str(),
                    "query dtype only support " + QSMLADataTypeToSerialString(ge::DT_HIFLOAT8)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.desc->GetDataType() != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "oriKv", QSMLADataTypeToSerialString(opParamInfo_.oriKv.desc->GetDataType()).c_str(),
                    "oriKv dtype only support " + QSMLADataTypeToSerialString(ge::DT_HIFLOAT8)),
                return ge::GRAPH_FAILED);

    if (opParamInfo_.cmpKv.desc != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpKv.desc->GetDataType() != ge::DT_HIFLOAT8,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cmpKv", QSMLADataTypeToSerialString(opParamInfo_.cmpKv.desc->GetDataType()).c_str(),
                        "cmpKv dtype only support " + QSMLADataTypeToSerialString(ge::DT_HIFLOAT8)),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeaturePa() const
{
    if (kvLayout_ != QSMLALayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
                OP_LOGE(opName_, "when page attention is enabled, oriBlockSize_(%u) should be in range (0, %u].",
                        oriBlockSize_, MAX_BLOCK_SIZE),
                return ge::GRAPH_FAILED);

    if (cmpBlockSize_ != 0) {
        OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
                    OP_LOGE(opName_, "when page attention is enabled, cmpBlockSize_(%u) should be in range (0, %u].",
                            cmpBlockSize_, MAX_BLOCK_SIZE),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureQuantModeAndDtype() || ge::GRAPH_SUCCESS != CheckFeatureShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() || ge::GRAPH_SUCCESS != CheckFeatureDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureWinKV() || ge::GRAPH_SUCCESS != CheckFeaturePa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

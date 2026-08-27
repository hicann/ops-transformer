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
 * \file common_checker.cpp
 * \brief Common checker for layout, shape, dtype parameters ( 公共参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "common_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;
using namespace Ops::Base;

// ============================================================================
// Layout — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaLayout(const QfaTilingInfo &qfaInfo)
{
    const std::vector<QfaLayout> supportedQLayouts = {QfaLayout::BSND, QfaLayout::BNSD, QfaLayout::TND, QfaLayout::NTD};
    const std::vector<QfaLayout> supportedKvLayouts = {QfaLayout::BSND,    QfaLayout::BNSD,    QfaLayout::TND,
                                                       QfaLayout::PA_BBND, QfaLayout::PA_BNBD, QfaLayout::PA_NZ};
    const std::vector<QfaLayout> supportedOutLayouts = {QfaLayout::BSND, QfaLayout::BNSD, QfaLayout::TND};

    if (std::find(supportedQLayouts.begin(), supportedQLayouts.end(), qfaInfo.qLayout) == supportedQLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "layout_q",
                                              QfaLayoutToSerialString(qfaInfo.qLayout).c_str(),
                                              "The value of layout_q must be in BSND/BNSD/TND/NTD");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((qfaInfo.kvLayout == QfaLayout::PA_NZ) && (qfaInfo.qkHeadDim == 72),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qfaInfo.opName, "layout_kv", QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
                    "When quant_mode is MxFP8 and qkHeadDim is 72, layout_kv cannot be PA_NZ"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), qfaInfo.kvLayout) == supportedKvLayouts.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "layout_kv",
                                              QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
                                              "The value of layout_kv can only be BSND/BNSD/TND/PA_BBND/PA_BNBD/PA_NZ"),
        return ge::GRAPH_FAILED);

    if (std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), qfaInfo.outLayout) ==
        supportedOutLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "layout_out",
                                              QfaLayoutToSerialString(qfaInfo.outLayout).c_str(),
                                              "The value of layout_out must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Dtype — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaDtype(const QfaTilingInfo &qfaInfo)
{
    const std::vector<ge::DataType> supportedQuantDtypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8};
    const auto checkQuantDtype = [&](const gert::CompileTimeTensorDesc *desc,
                                     const std::string &name) -> ge::graphStatus {
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(
            std::find(supportedQuantDtypes.begin(), supportedQuantDtypes.end(), desc->GetDataType()) ==
                supportedQuantDtypes.end(),
            OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, name.c_str(), DataTypeToSerialString(desc->GetDataType()).c_str(),
                                      "FLOAT8_E4M3FN/HIFLOAT8"),
            return ge::GRAPH_FAILED);
        if (CheckFormatSupport(desc, name) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    if (checkQuantDtype(qfaInfo.opParamInfo.query.desc, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        checkQuantDtype(qfaInfo.opParamInfo.key.desc, KEY_NAME) != ge::GRAPH_SUCCESS ||
        checkQuantDtype(qfaInfo.opParamInfo.value.desc, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // attn_out: data_type 仅支持 BFLOAT16
    const gert::CompileTimeTensorDesc *attnOutDesc = qfaInfo.opParamInfo.attnOut.desc;
    if (attnOutDesc != nullptr) {
        OP_CHECK_IF(attnOutDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, ATTN_OUT_NAME.c_str(),
                                              DataTypeToSerialString(attnOutDesc->GetDataType()).c_str(), "BFLOAT16"),
                    return ge::GRAPH_FAILED);
        if (CheckFormatSupport(attnOutDesc, ATTN_OUT_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ShapeDim — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaShapeDim(const QfaTilingInfo &qfaInfo)
{
    const std::vector<uint32_t> supportedKvDims = {DIM_NUM_3, DIM_NUM_4, DIM_NUM_5};
    const std::vector<uint32_t> supportedQOutDims = {DIM_NUM_3, DIM_NUM_4};

    // q: shape dim 支持 3D/4D
    const gert::StorageShape *queryShape = qfaInfo.opParamInfo.query.shape;
    if (queryShape != nullptr) {
        uint32_t queryDimNum = queryShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            std::find(supportedQOutDims.begin(), supportedQOutDims.end(), queryDimNum) == supportedQOutDims.end(),
            OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, QUERY_NAME.c_str(),
                                         (std::to_string(queryDimNum) + "D").c_str(), "3D/4D"),
            return ge::GRAPH_FAILED);
    }

    // k: shape dim 支持 3、4、5
    const gert::StorageShape *keyShape = qfaInfo.opParamInfo.key.shape;
    if (keyShape != nullptr) {
        uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(std::find(supportedKvDims.begin(), supportedKvDims.end(), keyDimNum) == supportedKvDims.end(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, KEY_NAME.c_str(),
                                                 (std::to_string(keyDimNum) + "D").c_str(), "3D/4D/5D"),
                    return ge::GRAPH_FAILED);
    }

    // v: shape dim 支持 3、4、5
    const gert::StorageShape *valueShape = qfaInfo.opParamInfo.value.shape;
    if (valueShape != nullptr) {
        uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(std::find(supportedKvDims.begin(), supportedKvDims.end(), valueDimNum) == supportedKvDims.end(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, VALUE_NAME.c_str(),
                                                 (std::to_string(valueDimNum) + "D").c_str(), "3D/4D/5D"),
                    return ge::GRAPH_FAILED);
    }

    // attn_out: shape dim 支持 3D/4D
    const gert::StorageShape *attnOutShape = qfaInfo.opParamInfo.attnOut.shape;
    if (attnOutShape != nullptr) {
        uint32_t attnOutDimNum = attnOutShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            std::find(supportedQOutDims.begin(), supportedQOutDims.end(), attnOutDimNum) == supportedQOutDims.end(),
            OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, ATTN_OUT_NAME.c_str(),
                                         (std::to_string(attnOutDimNum) + "D").c_str(), "3D/4D"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Attr — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaSoftmaxScale(const QfaTilingInfo &qfaInfo)
{
    //  softmax_scale 必须大于 0
    OP_CHECK_IF(qfaInfo.softmaxScale <= 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, SOFTMAX_SCALE_NAME.c_str(),
                                                      std::to_string(qfaInfo.softmaxScale).c_str(),
                                                      "softmax_scale must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Dtype consistency
// ============================================================================

ge::graphStatus CommonChecker::CheckQuantDataType(const QfaTilingInfo &qfaInfo)
{
    // q/k/v dtype 必须为 FLOAT8_E4M3FN，attn_out dtype 必须为 BFLOAT16
    // (单参数校验阶段已完成，此处复用基类 CheckDtypeSupport 以覆盖 DTYPE_SUPPORT_MAP)
    if (CheckDtypeSupport(qfaInfo.opParamInfo.query.desc, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(qfaInfo.opParamInfo.key.desc, KEY_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(qfaInfo.opParamInfo.value.desc, VALUE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(qfaInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeConsistency(const QfaTilingInfo &qfaInfo)
{
    //  q、k、v 的数据类型必须相同
    const gert::CompileTimeTensorDesc *queryDesc = qfaInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc *keyDesc = qfaInfo.opParamInfo.key.desc;
    const gert::CompileTimeTensorDesc *valueDesc = qfaInfo.opParamInfo.value.desc;

    ge::DataType queryDtype = queryDesc->GetDataType();

    OP_CHECK_IF(keyDesc != nullptr && keyDesc->GetDataType() != queryDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(qfaInfo.opName, KEY_NAME.c_str(),
                                                      DataTypeToSerialString(keyDesc->GetDataType()).c_str(),
                                                      ("The dtype of key must be the same as dtype(" +
                                                       std::string(DataTypeToSerialString(queryDtype)) + ") of query")
                                                          .c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(valueDesc != nullptr && valueDesc->GetDataType() != queryDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(qfaInfo.opName, VALUE_NAME.c_str(),
                                                      DataTypeToSerialString(valueDesc->GetDataType()).c_str(),
                                                      ("The dtype of value must be the same as dtype(" +
                                                       std::string(DataTypeToSerialString(queryDtype)) + ") of query")
                                                          .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// HeadNum
// ============================================================================

ge::graphStatus CommonChecker::CheckHeadNum(const QfaTilingInfo &qfaInfo)
{
    //  Q_N % KV_N == 0 且 Q_N / KV_N > 0
    OP_CHECK_IF(
        qfaInfo.n1Size <= 0 || qfaInfo.n2Size <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(qfaInfo.opName, "query and key",
                                               (ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                                                ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "N of query and key must be greater than 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qfaInfo.n1Size < qfaInfo.n2Size,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    qfaInfo.opName, "num_heads, num_key_value_heads",
                    (std::to_string(qfaInfo.n1Size) + ", " + std::to_string(qfaInfo.n2Size)).c_str(),
                    "The value of num_heads must be greater than or equal to num_key_value_heads"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        qfaInfo.n1Size % qfaInfo.n2Size != 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(qfaInfo.opName, "query and key",
                                               (ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                                                ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "N of query must be an integer multiple of the same axis of key"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Axis
// ============================================================================

ge::graphStatus CommonChecker::CheckAxis(const QfaTilingInfo &qfaInfo)
{
    // 约束(特性交叉校验):
    //   65536 > B > 0；Q_S ≥ 0；KV_S ≥ 0；Q_T ≥ 0、KV_T ≥ 0；D 仅支持 64 或 128
    if (qfaInfo.bSize >= B_LIMIT || qfaInfo.bSize <= 0) {
        std::string reason = "The value of B must be within the range (0, " + std::to_string(B_LIMIT) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis B", std::to_string(qfaInfo.bSize).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (qfaInfo.qLayout == QfaLayout::TND) {
        OP_CHECK_IF(
            qfaInfo.qTSize < 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qfaInfo.opName, QUERY_NAME.c_str(),
                                                  ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                                                  "T of query must be greater than or equal to 0"),
            return ge::GRAPH_FAILED);
    }
    if (qfaInfo.kvLayout == QfaLayout::TND) {
        OP_CHECK_IF(
            qfaInfo.kTSize < 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qfaInfo.opName, KEY_NAME.c_str(),
                                                  ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()).c_str(),
                                                  "T of key/value must be greater than or equal to 0"),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        qfaInfo.s1Size < 0 && qfaInfo.qLayout != QfaLayout::TND && qfaInfo.qLayout != QfaLayout::NTD,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            qfaInfo.opName, QUERY_NAME.c_str(), ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
            "When layout of query is not TND or NTD, S of query must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qfaInfo.s2Size < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis KV_S", std::to_string(qfaInfo.s2Size).c_str(),
                                              "The value of axis KV_S must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);

    const std::vector<int64_t> supportedHeadDims = {64, 72, 128, 256};
    OP_CHECK_IF(CheckValueSupport(qfaInfo.qkHeadDim, supportedHeadDims) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    qfaInfo.opName, "axis D of query and key", std::to_string(qfaInfo.qkHeadDim).c_str(),
                    "The value of axis D of query and key can only be 64/72/128/256"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckValueSupport(qfaInfo.vHeadDim, supportedHeadDims) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis D of value",
                                                      std::to_string(qfaInfo.vHeadDim).c_str(),
                                                      "The value of axis D of value can only be 64/72/128/256"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// SinglePara — combined
// ============================================================================

ge::graphStatus CommonChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    if (CheckSingleParaLayout(qfaInfo) != ge::GRAPH_SUCCESS || CheckSingleParaDtype(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaShapeDim(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaSoftmaxScale(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus CommonChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    // q/k/v/attn_out 为必选参数，desc 和 shape 均不能为空
    OP_CHECK_IF(qfaInfo.opParamInfo.query.desc == nullptr || qfaInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, QUERY_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.key.desc == nullptr || qfaInfo.opParamInfo.key.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, KEY_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.value.desc == nullptr || qfaInfo.opParamInfo.value.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, VALUE_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.attnOut.desc == nullptr || qfaInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, ATTN_OUT_NAME.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — combined (文档"特性交叉校验"列: 轴校验 + HeadNum)
// ============================================================================

ge::graphStatus CommonChecker::CheckFeature(const QfaTilingInfo &qfaInfo)
{
    if (CheckAxis(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckHeadNum(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — combined (文档公共参数组"一致性校验"列: q/k/v dtype 一致)
// 注: layout 匹配和 q/k/v/attn_out shape 校验属于量化场景特性交叉校验,
//     已移至 QuantChecker::CheckFeature。
// ============================================================================

ge::graphStatus CommonChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    if (CheckQuantDataType(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeConsistency(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling

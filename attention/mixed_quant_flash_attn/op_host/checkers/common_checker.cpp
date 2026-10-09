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
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../mqfa_fa_tiling_info.h"
#include "mqfa_common_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;
using namespace Ops::Base;

// ============================================================================
// Layout - SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaLayout(const FaTilingInfo& faInfo)
{
    const std::vector<FaLayout> supportedQLayouts = {FaLayout::BNSD, FaLayout::BSND, FaLayout::TND};
    const std::vector<FaLayout> supportedKvLayouts = {FaLayout::PA_NZ, FaLayout::PA_BNBD, FaLayout::PA_BBND};
    const std::vector<FaLayout> supportedOutLayouts = {FaLayout::BNSD, FaLayout::BSND, FaLayout::TND};

    if (std::find(supportedQLayouts.begin(), supportedQLayouts.end(), faInfo.qLayout) == supportedQLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_q", LayoutToSerialString(faInfo.qLayout).c_str(),
                                              "The value of layout_q must be in BNSD/BSND/TND");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) == supportedKvLayouts.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            faInfo.opName, "layout_kv", LayoutToSerialString(faInfo.kvLayout).c_str(),
            "The value of layout_kv(layout of K/V) can only be PA_NZ for MixedQuantFlashAttn"),
        return ge::GRAPH_FAILED);

    if (std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
        supportedOutLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_attn_out",
                                              LayoutToSerialString(faInfo.outLayout).c_str(),
                                              "The value of layout_attn_out must be in BNSD/BSND/TND");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Attr - SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    if (CheckSingleParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Shape dimension checks (single-para: only dim count, no layout dependency)
    const auto& qShape = faInfo.opParamInfo.query.shape->GetStorageShape();
    OP_CHECK_IF(
        qShape.GetDimNum() < INPUT_Q_SHAPE_MIN_DIMS || qShape.GetDimNum() > INPUT_Q_SHAPE_MAX_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "query", std::to_string(qShape.GetDimNum()).c_str(), "3D or 4D"),
        return ge::GRAPH_FAILED);

    const auto& kShape = faInfo.opParamInfo.key.shape->GetStorageShape();
    OP_CHECK_IF(
        kShape.GetDimNum() < INPUT_KV_SHAPE_MIN_DIMS || kShape.GetDimNum() > INPUT_KV_SHAPE_MAX_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "key", std::to_string(kShape.GetDimNum()).c_str(), "3D to 5D"),
        return ge::GRAPH_FAILED);

    const auto& vShape = faInfo.opParamInfo.value.shape->GetStorageShape();
    OP_CHECK_IF(
        vShape.GetDimNum() < INPUT_KV_SHAPE_MIN_DIMS || vShape.GetDimNum() > INPUT_KV_SHAPE_MAX_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "value", std::to_string(vShape.GetDimNum()).c_str(), "3D to 5D"),
        return ge::GRAPH_FAILED);

    const auto& attnOutShape = faInfo.opParamInfo.attnOut.shape->GetStorageShape();
    OP_CHECK_IF(attnOutShape.GetDimNum() < INPUT_Q_SHAPE_MIN_DIMS || attnOutShape.GetDimNum() > INPUT_Q_SHAPE_MAX_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "attn_out",
                                             std::to_string(attnOutShape.GetDimNum()).c_str(), "3D or 4D"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus CommonChecker::CheckParaExistence(const FaTilingInfo& faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.query.desc == nullptr || faInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "query"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.key.desc == nullptr || faInfo.opParamInfo.key.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "key"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.value.desc == nullptr || faInfo.opParamInfo.value.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "value"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.attnOut.desc == nullptr || faInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "attn_out"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Dtype
// ============================================================================

ge::graphStatus CommonChecker::CheckMixedQuantDataType(const FaTilingInfo& faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.kDescale.desc, K_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.vDescale.desc, V_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.kDescale.desc, K_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.vDescale.desc, V_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeConsistency(const FaTilingInfo& faInfo)
{
    const gert::CompileTimeTensorDesc* queryDesc = faInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc* attnOutDesc = faInfo.opParamInfo.attnOut.desc;
    const gert::CompileTimeTensorDesc* kDescaleDesc = faInfo.opParamInfo.kDescale.desc;
    const gert::CompileTimeTensorDesc* vDescaleDesc = faInfo.opParamInfo.vDescale.desc;

    ge::DataType queryDtype = queryDesc->GetDataType();

    if (attnOutDesc != nullptr) {
        if (attnOutDesc->GetDataType() != queryDtype) {
            std::string reason =
                "The dtype of attn_out must be the same as dtype(" + ToString(queryDtype) + ") of query";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "attn_out",
                                                  ToString(attnOutDesc->GetDataType()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (kDescaleDesc != nullptr && vDescaleDesc != nullptr) {
        if (kDescaleDesc->GetDataType() != vDescaleDesc->GetDataType()) {
            std::string reason = "The dtype of v_descale must be the same as dtype(" +
                                 ToString(kDescaleDesc->GetDataType()) + ") of k_descale";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "v_descale",
                                                  ToString(vDescaleDesc->GetDataType()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// HeadNum
// ============================================================================

ge::graphStatus CommonChecker::CheckHeadNum(const FaTilingInfo& faInfo)
{
    if ((faInfo.n1Size < 0) || (faInfo.n2Size < 0)) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.key.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and key", shapeStr.c_str(),
                                               "N of query and key must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(faInfo.n1Size < faInfo.n2Size,
                OP_LOGE(faInfo.opName, "numHeads(%ld) should be greater than or equal to numKeyValueHeads(%ld)!",
                        faInfo.n1Size, faInfo.n2Size),
                return ge::GRAPH_FAILED);

    if (faInfo.n1Size % faInfo.n2Size != 0) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.key.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and key", shapeStr.c_str(),
                                               "N of query must be an integer multiple of the same axis of key");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Axis
// ============================================================================

ge::graphStatus CommonChecker::CheckAxis(const FaTilingInfo& faInfo)
{
    if (faInfo.bSize >= B_LIMIT || faInfo.bSize <= 0) {
        std::string reason = "The value of B must be within the range (0, " + std::to_string(B_LIMIT) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis B", std::to_string(faInfo.bSize).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(faInfo.qTSize <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        faInfo.opName, "query", ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                        "T of query must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    if (faInfo.kvLayout == FaLayout::TND) {
        OP_CHECK_IF(faInfo.kTSize <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        faInfo.opName, "key", ToString(faInfo.opParamInfo.key.shape->GetStorageShape()).c_str(),
                        "T of key/value must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(faInfo.n1Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "query", ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                    "N of query must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.n2Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "key",
                                                      ToString(faInfo.opParamInfo.key.shape->GetStorageShape()).c_str(),
                                                      "N of key/value must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s1Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "query", ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                    "S of query must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s2Size <= 0,
                OP_LOGE(faInfo.opName, "The axis KV_S must be greater than 0, the current is %ld.", faInfo.s2Size),
                return ge::GRAPH_FAILED);

    const std::vector<int64_t> supportedHeadDims = {128};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.qkHeadDim, supportedHeadDims),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(faInfo.opName, "axis D of query and key",
                                                       std::to_string(faInfo.qkHeadDim).c_str(),
                                                       "The value of axis D of query and key can only be 128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.vHeadDim, supportedHeadDims),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis D of value", std::to_string(faInfo.vHeadDim).c_str(),
                                              "The value of axis D of value can only be 128"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Layout - MultiPara constraint table
// ============================================================================

struct LayoutConstraintConfig {
    std::vector<FaLayout> supportedKvLayouts;
    std::vector<FaLayout> supportedOutLayouts;
};

static const std::map<FaLayout, LayoutConstraintConfig> LAYOUT_CONSTRAINT_TABLE = {
    {FaLayout::BNSD, {{FaLayout::PA_NZ}, {FaLayout::BNSD, FaLayout::BSND}}},
    {FaLayout::BSND, {{FaLayout::PA_NZ}, {FaLayout::BSND}}},
    {FaLayout::TND, {{FaLayout::PA_NZ}, {FaLayout::TND}}},
};

ge::graphStatus CommonChecker::CheckMultiParaLayout(const FaTilingInfo& faInfo)
{
    auto it = LAYOUT_CONSTRAINT_TABLE.find(faInfo.qLayout);
    OP_CHECK_IF(it == LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE(faInfo.opName, "layout_q %s is not supported", LayoutToSerialString(faInfo.qLayout).c_str()),
                return ge::GRAPH_FAILED);

    const auto& config = it->second;
    const std::string qLayoutStr = LayoutToSerialString(faInfo.qLayout);

    OP_CHECK_IF(std::find(config.supportedOutLayouts.begin(), config.supportedOutLayouts.end(), faInfo.outLayout) ==
                    config.supportedOutLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_attn_out must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.outLayout).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Shape Compare
// ============================================================================

void CommonChecker::SetFaShapeCompare(const FaTilingInfo& faInfo)
{
    queryShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.query.shape->GetStorageShape(),
                                                            faInfo.qLayout, QUERY_NAME, faInfo.opName);
    keyShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.key.shape->GetStorageShape(),
                                                          faInfo.kvLayout, KEY_NAME, faInfo.opName);
    valueShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.value.shape->GetStorageShape(),
                                                            faInfo.kvLayout, VALUE_NAME, faInfo.opName);
    attnOutShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.attnOut.shape->GetStorageShape(),
                                                              faInfo.outLayout, ATTN_OUT_NAME, faInfo.opName);
}

ge::graphStatus CommonChecker::CheckQueryShape(const FaTilingInfo& faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus CommonChecker::CheckKVShapeForContinuous(const FaTilingInfo& faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s2Size);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.kTSize);

    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKVShapeForPageAttention(const FaTilingInfo& faInfo) const
{
    ge::DataType kvDtype = faInfo.opParamInfo.key.desc->GetDataType();
    uint32_t kvBlockElemNum = (32 * 8) / FABaseChecker::GetTypeSize(kvDtype);

    if (faInfo.blockSize % kvBlockElemNum != 0) {
        std::string reason = "The value of block_size must be a multiple of " + std::to_string(kvBlockElemNum) +
                             " (32 / sizeof(kv_dtype))";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "block_size", std::to_string(faInfo.blockSize).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    FaTilingShapeCompareParam shapeParams;
    shapeParams.Bn = static_cast<int64_t>(faInfo.totalBlockNum);
    shapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    shapeParams.Bs = static_cast<int64_t>(faInfo.blockSize);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.D0 = static_cast<int64_t>(kvBlockElemNum);

    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKVShape(const FaTilingInfo& faInfo) const
{
    if (faInfo.kvLayout == FaLayout::PA_NZ || faInfo.kvLayout == FaLayout::PA_BBND ||
        faInfo.kvLayout == FaLayout::PA_BNBD) {
        return CheckKVShapeForPageAttention(faInfo);
    }

    std::string reason = "layout_kv: " + LayoutToSerialString(faInfo.kvLayout) +
                         " is not supported for MixedQuantFlashAttn, only PA_NZ is supported";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_kv", LayoutToSerialString(faInfo.kvLayout).c_str(),
                                          reason.c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus CommonChecker::CheckAttnOutShape(const FaTilingInfo& faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    if (attnOutShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckShapeConsistency(const FaTilingInfo& faInfo)
{
    SetFaShapeCompare(faInfo);
    if (ge::GRAPH_SUCCESS != CheckQueryShape(faInfo) || ge::GRAPH_SUCCESS != CheckKVShape(faInfo) ||
        ge::GRAPH_SUCCESS != CheckAttnOutShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara ¡ª combined
// ============================================================================

ge::graphStatus CommonChecker::CheckMultiPara(const FaTilingInfo& faInfo)
{
    if (CheckMultiParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMixedQuantDataType(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAxis(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckHeadNum(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling

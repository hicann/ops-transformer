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
 * \file moe_re_routing_tiling_base.cpp
 * \brief
 */
#include <iostream>
#include "log/log.h"
#include "moe_re_routing_tiling_base.h"

namespace optiling {

constexpr int64_t IN_TOKEN_INDEX = 0;
constexpr int64_t IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX = 1;
constexpr int64_t IN_TOKEN_SCALE_INDEX = 2;
constexpr int64_t IN_EXPERT_TOPK_WEIGHT_INDEX = 3;
constexpr int64_t OUTPUT_PERMUTE_TOKENS_INDEX = 0;
constexpr int64_t OUT_SCALE_INDEX = 1;
constexpr int64_t OUT_PERMUTE_TOKEN_IDX_IDNEX = 2;
constexpr int64_t OUTPUT_EXPERT_TOKEN_NUM_INDEX = 3;
constexpr int64_t OUT_PERMUTE_TOPK_WEIGHT_INDEX = 4;
constexpr int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 0;
constexpr int64_t ATTR_IDX_TYPE_INDEX = 1;
constexpr int64_t DIM_SIZE_TWO = 2;
constexpr int64_t DIM_SIZE_THREE = 3;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr size_t DIM_INDEX_0 = 0;
constexpr size_t DIM_INDEX_1 = 1;
constexpr size_t DIM_INDEX_2 = 2;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

static const std::set<ge::DataType> TOKENS_DTYPE = {ge::DT_INT8,          ge::DT_FLOAT16,     ge::DT_BF16,
                                                    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8,
                                                    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2};
static const std::set<ge::DataType> EXPERT_TOKEN_NUM_DTYPE = {ge::DT_INT32, ge::DT_INT64};
static const std::set<ge::DataType> SCALE_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};

static std::tuple<int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context, const int64_t index = 0,
    const bool isOptional = false)
{
    const gert::StorageShape *shapePtr = isOptional ? context->GetOptionalInputShape(index) :
                                                      context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), ("input " + std::to_string(index) + " shape").c_str(), "nullptr",
                    "Input shape should not be null."),
                return std::make_tuple(0, 0));
    OP_CHECK_IF(
        shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context->GetNodeName(), ("input " + std::to_string(index)).c_str(),
            std::to_string(shapePtr->GetStorageShape().GetDimNum()).c_str(), "2"),
        return std::make_tuple(0, 0));
    return std::make_tuple(shapePtr->GetStorageShape().GetDim(0), shapePtr->GetStorageShape().GetDim(1));
}

static std::tuple<int64_t, int64_t, int64_t> GetShapeTupleN(const gert::TilingContext *context, const int64_t index = 0,
    const bool isOptional = false)
{
    const gert::StorageShape *shapePtr = isOptional ? context->GetOptionalInputShape(index) :
                                                      context->GetInputShape(index);
    OP_CHECK_IF(
        shapePtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), ("input " + std::to_string(index) + " shape").c_str(), "nullptr",
            "Input shape should not be null."),
        return std::make_tuple(0, 0, 0));
    auto dimNum = shapePtr->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != DIM_3,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context->GetNodeName(), ("input " + std::to_string(index)).c_str(), std::to_string(dimNum).c_str(), "3"),
        return std::make_tuple(0, 0, 0));
    return std::make_tuple(shapePtr->GetStorageShape().GetDim(DIM_INDEX_0),
                           shapePtr->GetStorageShape().GetDim(DIM_INDEX_1),
                           shapePtr->GetStorageShape().GetDim(DIM_INDEX_2));
}

ge::graphStatus MoeReRoutingTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MoeReRoutingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "compileInfo", "nullptr", "Compile info should not be null."),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        int64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ((uint64_t &)ubSize));
        ubSize_ = ubSize;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    OP_CHECK_IF((coreNum_ <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "coreNum", std::to_string(coreNum_).c_str(),
                    "Failed to get valid core num."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "ubSize", std::to_string(ubSize_).c_str(),
                    "Failed to get valid ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckNullptr() const
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(IN_TOKEN_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(IN_TOKEN_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_PERMUTE_TOKENS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUT_PERMUTE_TOKEN_IDX_IDNEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUM_INDEX));
    if (hasTopkWeight_) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputShape(IN_EXPERT_TOPK_WEIGHT_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputDesc(IN_EXPERT_TOPK_WEIGHT_INDEX));
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUT_PERMUTE_TOPK_WEIGHT_INDEX));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckDtypeAndAttr() const
{
    OP_CHECK_IF(
        std::find(TOKENS_DTYPE.begin(), TOKENS_DTYPE.end(), tokenDtype_) == TOKENS_DTYPE.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "tokens", ge::TypeUtils::DataTypeToSerialString(tokenDtype_).c_str(),
            "Supported dtypes: INT8, BF16, F16, FP8_E4M3FN, FP8_E5M2, HIFLOAT8, FP4_E2M1, FP4_E1M2."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(std::find(EXPERT_TOKEN_NUM_DTYPE.begin(), EXPERT_TOKEN_NUM_DTYPE.end(), expertDtype_) ==
                    EXPERT_TOKEN_NUM_DTYPE.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "expert_token_num_per_rank",
                    ge::TypeUtils::DataTypeToSerialString(expertDtype_).c_str(),
                    "Supported dtypes: INT32, INT64."),
                return ge::GRAPH_FAILED);
    if (tokenDtype_ == ge::DT_FLOAT8_E4M3FN || tokenDtype_ == ge::DT_FLOAT8_E5M2) {
        OP_CHECK_IF((scaleDtype_ != ge::DT_FLOAT8_E8M0 && scaleDtype_ != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "per_token_scales",
                        ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str(),
                        "When tokens dtype is FP8, scale dtype should be FLOAT8_E8M0 or FLOAT."),
                    return ge::GRAPH_FAILED);
    }
    if (tokenDtype_ == ge::DT_HIFLOAT8 && hasScale_) {
        OP_CHECK_IF(scaleDtype_ != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "per_token_scales",
                        ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str(),
                        "When tokens dtype is HIFLOAT8, scale dtype should be FLOAT."),
                    return ge::GRAPH_FAILED);
    }
    if (tokenDtype_ == ge::DT_FLOAT4_E2M1 || tokenDtype_ == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF((scaleDtype_ != ge::DT_FLOAT8_E8M0 && scaleDtype_ != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "per_token_scales",
                        ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str(),
                        "When tokens dtype is FP4, scale dtype should be FLOAT8_E8M0 or FLOAT."),
                    return ge::GRAPH_FAILED);
    }
    if (hasScale_) {
        auto outputScaleType = context_->GetOutputDesc(OUT_SCALE_INDEX)->GetDataType();
        OP_CHECK_IF(outputScaleType != scaleDtype_,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        context_->GetNodeName(), "permute_per_token_scales",
                        ge::TypeUtils::DataTypeToSerialString(outputScaleType).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    }
    auto outputTokenType = context_->GetOutputDesc(OUTPUT_PERMUTE_TOKENS_INDEX)->GetDataType();
    OP_CHECK_IF(outputTokenType != tokenDtype_,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context_->GetNodeName(), "permute_tokens",
                    ge::TypeUtils::DataTypeToSerialString(outputTokenType).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(tokenDtype_).c_str()),
                return ge::GRAPH_FAILED);

    auto outputTokenIdxType = context_->GetOutputDesc(OUT_PERMUTE_TOKEN_IDX_IDNEX)->GetDataType();
    OP_CHECK_IF(outputTokenIdxType != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context_->GetNodeName(), "permute_tokens_idx",
                    ge::TypeUtils::DataTypeToSerialString(outputTokenIdxType).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(ge::DT_INT32).c_str()),
                return ge::GRAPH_FAILED);

    auto outputTokenNumType = context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUM_INDEX)->GetDataType();
    OP_CHECK_IF(outputTokenNumType != expertDtype_,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context_->GetNodeName(), "expert_token_num",
                    ge::TypeUtils::DataTypeToSerialString(outputTokenNumType).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(expertDtype_).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((idxType_ != 0 && idxType_ != 1),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "idxType", std::to_string(idxType_).c_str(),
                                          "0 or 1"),
                return ge::GRAPH_FAILED);
    if (hasTopkWeight_) {
        auto outputTopkWeightDesc = context_->GetOutputDesc(OUT_PERMUTE_TOPK_WEIGHT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputTopkWeightDesc);
        auto outputTopkWeightType = outputTopkWeightDesc->GetDataType();
        OP_CHECK_IF(outputTopkWeightType != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "permute_topk_weight should be DT_FLOAT, actual %s.",
                            ge::TypeUtils::DataTypeToSerialString(outputTopkWeightType).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckOutputShape() const
{
    const gert::StorageShape *outTokenShapePtr = context_->GetOutputShape(OUTPUT_PERMUTE_TOKENS_INDEX);
    OP_CHECK_IF(outTokenShapePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "permute_tokens shape", "nullptr", "Output shape should not be null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outTokenShapePtr->GetStorageShape() != context_->GetInputShape(IN_TOKEN_INDEX)->GetStorageShape(),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "tokens and permute_tokens",
                    (Ops::Base::ToString(context_->GetInputShape(IN_TOKEN_INDEX)->GetStorageShape()) + " and " +
                     Ops::Base::ToString(outTokenShapePtr->GetStorageShape())).c_str(),
                    "The output permute_tokens shape should be the same as input tokens shape."),
                return ge::GRAPH_FAILED);
    if (hasScale_) {
        const gert::StorageShape *outScaleShapePtr = context_->GetOutputShape(OUT_SCALE_INDEX);
        OP_CHECK_IF(outScaleShapePtr == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "permute_per_token_scales shape", "nullptr",
                        "Output shape should not be null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(outScaleShapePtr->GetStorageShape() !=
                        context_->GetOptionalInputShape(IN_TOKEN_SCALE_INDEX)->GetStorageShape(),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "per_token_scales and permute_per_token_scales",
                        (Ops::Base::ToString(context_->GetOptionalInputShape(IN_TOKEN_SCALE_INDEX)->GetStorageShape())
                         + " and " + Ops::Base::ToString(outScaleShapePtr->GetStorageShape())).c_str(),
                        "The output permute_per_token_scales shape should be the same as per_token_scales."),
                    return ge::GRAPH_FAILED);
    }
    const gert::StorageShape *tokenIdxShapePtr = context_->GetOutputShape(OUT_PERMUTE_TOKEN_IDX_IDNEX);
    OP_CHECK_IF(tokenIdxShapePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "permute_tokens_idx shape", "nullptr",
                    "Output shape should not be null."),
                return ge::GRAPH_FAILED);
    auto tokenIdxShape = tokenIdxShapePtr->GetStorageShape();
    OP_CHECK_IF(tokenIdxShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    context_->GetNodeName(), "permute_tokens_idx",
                    std::to_string(tokenIdxShape.GetDimNum()).c_str(), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenIdxShape.GetDim(0) != tokenSum_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "permute_tokens_idx", Ops::Base::ToString(tokenIdxShape).c_str(),
                    ("Dim 0 of permute_tokens_idx should be equal to tokenSum, tokenSum is " +
                     std::to_string(tokenSum_)).c_str()),
                return ge::GRAPH_FAILED);
    const gert::StorageShape *tokenNumShapePtr = context_->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUM_INDEX);
    OP_CHECK_IF(tokenNumShapePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "expert_token_num shape", "nullptr",
                    "Output shape should not be null."),
                return ge::GRAPH_FAILED);
    auto tokenNumShape = tokenNumShapePtr->GetStorageShape();
    OP_CHECK_IF(tokenNumShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    context_->GetNodeName(), "expert_token_num",
                    std::to_string(tokenNumShape.GetDimNum()).c_str(), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenNumShape.GetDim(0) != expertNum_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "expert_token_num", Ops::Base::ToString(tokenNumShape).c_str(),
                    ("Dim 0 of expert_token_num should be equal to expertNum, expertNum is " +
                     std::to_string(expertNum_)).c_str()),
                return ge::GRAPH_FAILED);
    if (hasTopkWeight_) {
        const gert::StorageShape *outTopkWeightShapePtr = context_->GetOutputShape(OUT_PERMUTE_TOPK_WEIGHT_INDEX);
        OP_CHECK_IF(outTopkWeightShapePtr == nullptr, OP_LOGE(context_->GetNodeName(),
                    "outTopkWeightShapePtr is nullptr."), return ge::GRAPH_FAILED);
        const gert::StorageShape *inTopkWeightShapePtr =
            context_->GetOptionalInputShape(IN_EXPERT_TOPK_WEIGHT_INDEX);
        OP_CHECK_IF(inTopkWeightShapePtr == nullptr, OP_LOGE(context_->GetNodeName(),
                    "inTopkWeightShapePtr is nullptr."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(outTopkWeightShapePtr->GetStorageShape() != inTopkWeightShapePtr->GetStorageShape(),
                    OP_LOGE(context_->GetNodeName(), "expert_topk_weight shape must same with permute_topk_weight."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckParam()
{
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "dtype and attr", "check failed",
                    "Please check input and output dtypes, scales and attrs."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenSum_ < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "tokenSum", std::to_string(tokenSum_).c_str(),
                    "tokenSum should be greater than or equal to 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenSize_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "tokenSize", std::to_string(tokenSize_).c_str(),
                    "tokenSize should be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rankNums_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "rankNums", std::to_string(rankNums_).c_str(),
                    "rankNums should be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertNum_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "expertNum", std::to_string(expertNum_).c_str(),
                    "expertNum should be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShape() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "output shape", "check failed",
                    "Please check output shapes against input shapes and derived sizes."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "context", "nullptr", "Context should not be null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "input/output desc or shape", "nullptr",
                    "Required input/output desc and shape should not be null."),
                return ge::GRAPH_FAILED);
    auto tokenShapeTuple = GetShapeTuple(context_, IN_TOKEN_INDEX);
    auto expertShapeTuple = GetShapeTuple(context_, IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX);
    tokenSum_ = std::get<0>(tokenShapeTuple);
    tokenSize_ = std::get<1>(tokenShapeTuple);
    rankNums_ = std::get<0>(expertShapeTuple);
    expertNum_ = std::get<1>(expertShapeTuple);
    tokenDtype_ = context_->GetInputDesc(IN_TOKEN_INDEX)->GetDataType();
    expertDtype_ = context_->GetInputDesc(IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX)->GetDataType();
    auto scalePtr = context_->GetOptionalInputDesc(IN_TOKEN_SCALE_INDEX);
    if (scalePtr != nullptr) {
        auto scaleShapePtr = context_->GetOptionalInputShape(IN_TOKEN_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
        auto &scaleShape = scaleShapePtr->GetStorageShape();
        auto scaleDimNum = scaleShape.GetDimNum();
        if (scaleDimNum == DIM_1) {
            scaleSize_ = 1;
        } else if (scaleDimNum == DIM_2) {
            auto scaleShapeTuple = GetShapeTuple(context_, IN_TOKEN_SCALE_INDEX, true);
            scaleSize_ = std::get<1>(scaleShapeTuple);
            auto scaleSum = std::get<0>(scaleShapeTuple);
            OP_CHECK_IF(scaleSum != tokenSum_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "per_token_scales",
                            Ops::Base::ToString(scaleShape).c_str(),
                            ("Dim 0 of per_token_scales should be equal to tokenSum, tokenSum is " +
                             std::to_string(tokenSum_)).c_str()),
                        return ge::GRAPH_FAILED);
        } else if (scaleDimNum == DIM_3) {
            if (tokenDtype_ != ge::DT_FLOAT8_E4M3FN && tokenDtype_ != ge::DT_FLOAT8_E5M2) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "tokens", ge::TypeUtils::DataTypeToSerialString(tokenDtype_).c_str(),
                    "3D per_token_scales requires token dtype DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2.");
                return ge::GRAPH_FAILED;
            }
            auto scaleShapeTuple3D = GetShapeTupleN(context_, IN_TOKEN_SCALE_INDEX, true);
            auto scaleSum0 = std::get<DIM_INDEX_0>(scaleShapeTuple3D);
            auto scaleDim1 = std::get<DIM_INDEX_1>(scaleShapeTuple3D);
            auto scaleDim2 = std::get<DIM_INDEX_2>(scaleShapeTuple3D);
            OP_CHECK_IF(scaleSum0 != tokenSum_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "per_token_scales",
                    Ops::Base::ToString(scaleShape).c_str(),
                    ("Dim 0 of per_token_scales should be equal to tokenSum, tokenSum is " +
                     std::to_string(tokenSum_)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(scaleDim2 != DIM_SIZE_TWO,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "per_token_scales", Ops::Base::ToString(scaleShape).c_str(),
                    "Dim 2 of 3D per_token_scales should be 2."),
                return ge::GRAPH_FAILED);
            scaleSize_ = scaleDim1 * scaleDim2;
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                context_->GetNodeName(), "per_token_scales", std::to_string(scaleDimNum).c_str(), "1, 2 or 3");
            return ge::GRAPH_FAILED;
        }
        scaleDtype_ = scalePtr->GetDataType();
        hasScale_ = true;
    }
    hasTopkWeight_ = false;

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto expertTokenNumType = attrs->GetInt(ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokenNumType);
    OP_CHECK_IF(*expertTokenNumType != 1,
                OP_LOGE_WITH_INVALID_ATTR(
                    context_->GetNodeName(), "expertTokenNumType",
                    std::to_string(*expertTokenNumType).c_str(), "1"),
                return ge::GRAPH_FAILED);

    auto idxType = attrs->GetInt(ATTR_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxType);
    idxType_ = static_cast<int64_t>(*idxType);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

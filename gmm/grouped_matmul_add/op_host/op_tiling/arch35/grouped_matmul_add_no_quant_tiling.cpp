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
 * \file grouped_matmul_add_no_quant_tiling.cpp
 * \brief
 */
#include "grouped_matmul_add_no_quant_tiling.h"
#include "gmm/common/op_host/log_format_util.h"

namespace optiling {

bool GroupedMatmulAddNoQuantTiling::SetTiling(gert::TilingContext* context)
{
    auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr."), return false);
    usedCoreNum_ = compileInfoPtr->aicNum;
    OP_CHECK_IF(!Init(context), OP_LOGE(context->GetNodeName(), "Init failed"), return false);
    OP_CHECK_IF(!CalMatMulTiling(context, compileInfoPtr),
                OP_LOGE(context->GetNodeName(), "calculate matmul-tiling failed"), return false);
    SetGMMTiling();
    SetMatMulTiling();
    SetTilingKey(context);
    OP_CHECK_IF(!SaveTilingData(context, tilingData_), OP_LOGE(context->GetNodeName(), "set custom param failed"),
                return false);
    PrintTilingResult(context);
    return true;
}

bool GroupedMatmulAddNoQuantTiling::Init(const gert::TilingContext* context)
{
    OP_CHECK_IF(!GetAttrs(context), OP_LOGE(context->GetNodeName(), "Unable to calculate matmul-tiling"), return false);

    auto xTensor = context->GetDynamicInputTensor(INDEX_X, 0);
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();
    xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());
    for (uint32_t i = 0; i < xDimNum_; i++) {
        OP_LOGI(context->GetNodeName(), "x DIM[%u] is %lu", i, static_cast<uint64_t>(xShape.GetDim(i)));
    }
    auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, 0);
    OP_CHECK_IF(wTensor == nullptr, OP_LOGE(context->GetNodeName(), "wTensor is nullptr."), return false);
    gert::Shape wShape = wTensor->GetOriginShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    for (uint32_t i = 0; i < xDimNum_; i++) {
        OP_LOGI(context->GetNodeName(), "w DIM[%u] is %lu", i, static_cast<uint64_t>(wShape.GetDim(i)));
    }
    OP_CHECK_IF(wDimNum != MIN_DIM || xDimNum_ != MIN_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "x, weight",
                    Ops::Transformer::Gmm::FormatString("%u, %u", xDimNum_, wDimNum).c_str(),
                    Ops::Transformer::Gmm::FormatString("The shape dims of %s must be %d", "x, weight", 2).c_str()),
                return false);
    xKDim_ = transposeX_ ? 0U : xDimNum_ - 1U;
    weightNDim_ = transposeWeight_ ? wDimNum - DIM_TWO : wDimNum - DIM_ONE;

    OP_CHECK_IF(context->GetDynamicInputTensor(INDEX_WEIGHT, 1) != nullptr ||
                    context->GetDynamicInputTensor(INDEX_X, 1) != nullptr,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x, weight", "",
                    Ops::Transformer::Gmm::FormatString("The tensor nums of %s must be %s", "x, weight", "1").c_str()),
                return false);
    if (groupType_ == SPLIT_K) {
        return SplitKSingleXSingleWeightSingleY(context, xShape, wShape);
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        context->GetNodeName(), "groupType", Ops::Transformer::Gmm::FormatString("%d", groupType_).c_str(),
        Ops::Transformer::Gmm::FormatString("The value of %s must be in %s", "groupType", "{2}").c_str());
    return false;
}

bool GroupedMatmulAddNoQuantTiling::GetAttrs(const gert::TilingContext* context)
{
    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr."),
                return false); // check attr is not null
    const bool* transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_W);
    const bool* transposeXPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_X);
    const int32_t* groupTypePtr = attr->GetAttrPointer<int32_t>(ATTR_IDX_GROUPTYPE);
    const uint32_t* groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_IDX_GROUP_LIST_TYPE);
    transposeWeight_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    transposeX_ = transposeXPtr != nullptr ? *transposeXPtr : false;
    groupType_ = groupTypePtr != nullptr ? *groupTypePtr : SPLIT_K;
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    OP_CHECK_IF(groupType_ != SPLIT_K || !transposeX_ || transposeWeight_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "groupType, transposeX, transposeWeight",
                    Ops::Transformer::Gmm::FormatString("%d, %s, %s", groupType_, transposeX_ ? "TRUE" : "FALSE",
                                                        transposeWeight_ ? "TRUE" : "FALSE")
                        .c_str(),
                    Ops::Transformer::Gmm::FormatString(
                        "The value of %s must be %s, the value of %s must be %s, the value of %s must be %s",
                        "groupType", "2", "transposeX", "TRUE", "transposeWeight", "FALSE")
                        .c_str()),
                return false);

    auto xDesc = context->GetDynamicInputDesc(INDEX_X, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "xDesc is nullptr."), return false);
    xDType_ = xDesc->GetDataType();
    auto xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));

    auto w0Desc = context->GetDynamicInputDesc(INDEX_WEIGHT, 0);
    OP_CHECK_IF(w0Desc == nullptr, OP_LOGE(context->GetNodeName(), "w0Desc is nullptr."), return false);
    weightDtype_ = w0Desc->GetDataType();
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
    OP_CHECK_IF(xFormat != ge::FORMAT_ND || wFormat != ge::Format::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                    context->GetNodeName(), "x, weight",
                    Ops::Transformer::Gmm::FormatString("%s, %s", (xFormat != ge::FORMAT_ND) ? "not ND" : "ND",
                                                        (wFormat != ge::Format::FORMAT_ND) ? "not ND" : "ND")
                        .c_str(),
                    Ops::Transformer::Gmm::FormatString("The formats of %s must be %s", "x, weight", "ND").c_str()),
                return false);
    return true;
}

bool GroupedMatmulAddNoQuantTiling::CalMatMulTiling(const gert::TilingContext* context,
                                                    const GMMCompileInfo* compileInfoPtr)
{
    OP_CHECK_IF(groupNum_ < 1U || groupNum_ > MAX_TENSOR,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "groupNum", Ops::Transformer::Gmm::FormatString("%u", groupNum_).c_str(),
                    Ops::Transformer::Gmm::FormatString("The value of %s must be within the range [%d, %d]", "groupNum",
                                                        1, 1024)
                        .c_str()),
                return false);
    CalSplitKBasicBlockTiling(compileInfoPtr);
    mL1_ = baseM_;
    nL1_ = baseN_;
    dbL0C_ = 1UL;
    return true;
}

void GroupedMatmulAddNoQuantTiling::SetGMMTiling()
{
    tilingData_.gmmAddParams.groupNum = groupNum_;
    tilingData_.gmmAddParams.groupListType = groupListType_;
    tilingData_.gmmAddParams.mTailCnt = static_cast<uint32_t>(mTailCnt_);
    tilingData_.gmmAddParams.nTailCnt = static_cast<uint32_t>(nTailCnt_);
}

void GroupedMatmulAddNoQuantTiling::SetMatMulTiling()
{
    FillMmTilingData(tilingData_.mmTilingData);
}

void GroupedMatmulAddNoQuantTiling::SetTilingKey(gert::TilingContext* context) const
{
    uint8_t gmmTrans = static_cast<uint8_t>(transposeX_) | (static_cast<uint8_t>(transposeWeight_) << 1);
    constexpr uint64_t TILINGKEYOFFSETGMM = 10000900009000090000UL; // means base tiling template
    uint64_t tilingKey = TILINGKEYOFFSETGMM + gmmTrans;
    OP_LOGI(context->GetNodeName(), "GMM Tiling key: gmmTrans %u, tilingkey: %lu", gmmTrans, tilingKey);
    context->SetTilingKey(tilingKey);
}

void GroupedMatmulAddNoQuantTiling::PrintTilingResult(const gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "GMM Tiling result: groupNum: %u, groupListType: %u, mTailCnt: %u, nTailCnt: %u",
            tilingData_.gmmAddParams.groupNum, tilingData_.gmmAddParams.groupListType,
            tilingData_.gmmAddParams.mTailCnt, tilingData_.gmmAddParams.nTailCnt);

    OP_LOGI(context->GetNodeName(),
            "GMM MatMul Tiling result: M: %d, N: %d, Ka: %d, mL1: %d, nL1: %d, kL1: %d, baseM: %d, baseN: %d,"
            " baseK: %d, stepKa: %d, stepKb: %d, dbL0C: %d",
            tilingData_.mmTilingData.M, tilingData_.mmTilingData.N, tilingData_.mmTilingData.Ka,
            tilingData_.mmTilingData.mL1, tilingData_.mmTilingData.nL1, tilingData_.mmTilingData.kL1,
            tilingData_.mmTilingData.baseM, tilingData_.mmTilingData.baseN, tilingData_.mmTilingData.baseK,
            tilingData_.mmTilingData.stepKa, tilingData_.mmTilingData.stepKb, tilingData_.mmTilingData.dbL0C);
}

} // namespace optiling

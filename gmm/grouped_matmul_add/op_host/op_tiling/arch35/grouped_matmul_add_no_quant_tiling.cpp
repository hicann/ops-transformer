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

bool GroupedMatmulAddNoQuantTiling::SetTiling(gert::TilingContext *context)
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
    OP_CHECK_IF(!SetCustomParam(context), OP_LOGE(context->GetNodeName(), "set custom param failed"), return false);
    PrintTilingResult(context);
    return true;
}

bool GroupedMatmulAddNoQuantTiling::Init(const gert::TilingContext *context)
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

bool GroupedMatmulAddNoQuantTiling::GetAttrs(const gert::TilingContext *context)
{
    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr."),
                return false); // check attr is not null
    const bool *transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_W);
    const bool *transposeXPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_X);
    const int32_t *groupTypePtr = attr->GetAttrPointer<int32_t>(ATTR_IDX_GROUPTYPE);
    const uint32_t *groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_IDX_GROUP_LIST_TYPE);
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

void GroupedMatmulAddNoQuantTiling::FormulateBasicBlock(const GMMCompileInfo *compileInfoPtr, uint32_t remainCoreNum)
{
    uint64_t mCnt = CeilDiv(m_, baseM_);
    uint64_t nCnt = CeilDiv(n_, baseN_);
    if (mCnt * nCnt >= remainCoreNum) {
        return;
    }
    if (mCnt <= nCnt) {
        baseM_ = CeilAlign(CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        mCnt = CeilDiv(m_, baseM_);
        nCnt = remainCoreNum / mCnt;
        baseN_ = CeilAlign(CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
    } else {
        baseN_ = CeilAlign(CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        nCnt = CeilDiv(n_, baseN_);
        mCnt = remainCoreNum / nCnt;
        baseM_ = CeilAlign(CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
    }

    while (baseN_ >= baseM_ * NUM_TWO && nCnt < remainCoreNum / NUM_TWO) {
        nCnt = nCnt * NUM_TWO;
        mCnt = remainCoreNum / nCnt;
        baseM_ = CeilAlign(CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        baseN_ = CeilAlign(CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        mCnt = CeilDiv(m_, baseM_);
        nCnt = CeilDiv(n_, baseN_);
    }

    while (baseM_ >= baseN_ * NUM_TWO && mCnt < remainCoreNum / NUM_TWO) {
        mCnt = mCnt * NUM_TWO;
        nCnt = remainCoreNum / mCnt;
        baseM_ = CeilAlign(CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        baseN_ = CeilAlign(CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        mCnt = CeilDiv(m_, baseM_);
        nCnt = CeilDiv(n_, baseN_);
    }
    mCnt = CeilDiv(m_, baseM_);
    nCnt = CeilDiv(n_, baseN_);
    uint64_t kValueAlign = CeilAlign(k_, BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax = FloorAlign(static_cast<uint64_t>(compileInfoPtr->l0ASize / DB_SIZE /
                                                          GetSizeByDataType(xDType_) / std::max(baseM_, baseN_)),
                                    BASIC_BLOCK_SIZE_16);
    baseK_ = std::min(kValueAlign, kValueMax);
    usedCoreNum_ = std::min(static_cast<uint32_t>(mCnt * nCnt * groupNum_), compileInfoPtr->aicNum);
}

void GroupedMatmulAddNoQuantTiling::CalAswtL1Tiling(const GMMCompileInfo *compileInfoPtr)
{
    uint64_t totalL1Size = compileInfoPtr->l1Size;
    depthA1_ = totalL1Size / NUM_TWO / baseM_ / baseK_ / GetSizeByDataType(xDType_);      // 2: half of l1
    depthB1_ = totalL1Size / NUM_TWO / baseN_ / baseK_ / GetSizeByDataType(weightDtype_); // 2: half of l1

    uint64_t depthASize = depthA1_ * baseM_ * baseK_ * GetSizeByDataType(xDType_);
    uint64_t depthBSize = depthB1_ * baseN_ * baseK_ * GetSizeByDataType(weightDtype_);
    if (depthASize + depthBSize > totalL1Size) {
        if (baseM_ > baseN_) {
            depthB1_ = std::max(depthB1_ / NUM_TWO, 1UL); // 2: adjust depthB for l1 buffer
        } else {
            depthA1_ = std::max(depthA1_ / NUM_TWO, 1UL); // 2: adjust depthA for l1 buffer
        }
    }
    stepKb_ = std::max(depthB1_ / DB_SIZE, 1UL);
    stepKa_ = std::max(depthA1_ / DB_SIZE, 1UL);
    if (stepKa_ >= stepKb_) {
        stepKa_ = stepKa_ / stepKb_ * stepKb_;
    } else {
        stepKb_ = stepKb_ / stepKa_ * stepKa_;
    }
    depthA1_ = stepKa_ * DB_SIZE; // depth % (stepKa * stepM) == 0
    depthB1_ = stepKb_ * DB_SIZE; // depth % (stepKb * stepN) == 0
    return;
}

void GroupedMatmulAddNoQuantTiling::CalcTailBasicBlock(const GMMCompileInfo *compileInfoPtr)
{
    uint64_t mCnt = CeilDiv(m_, baseM_);
    uint64_t nCnt = CeilDiv(n_, baseN_);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfoPtr->aicNum ? 0UL : mnCnt % compileInfoPtr->aicNum;

    if (tailCnt == 0UL) {
        return;
    }
    while ((mTailCnt_ + 1UL) * nTailCnt_ * tailCnt <= compileInfoPtr->aicNum) {
        mTailCnt_ += 1UL;
        if (mTailCnt_ * (nTailCnt_ + 1UL) * tailCnt <= compileInfoPtr->aicNum) {
            nTailCnt_ += 1UL;
        }
    }
}

bool GroupedMatmulAddNoQuantTiling::CalMatMulTiling(const gert::TilingContext *context,
                                                    const GMMCompileInfo *compileInfoPtr)
{
    OP_CHECK_IF(groupNum_ < 1U || groupNum_ > MAX_TENSOR,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "groupNum", Ops::Transformer::Gmm::FormatString("%u", groupNum_).c_str(),
                    Ops::Transformer::Gmm::FormatString("The value of %s must be within the range [%d, %d]", "groupNum",
                                                        1, 1024)
                        .c_str()),
                return false);
    uint32_t remainCoreNum = std::max(1U, compileInfoPtr->aicNum / groupNum_);
    FormulateBasicBlock(compileInfoPtr, remainCoreNum);
    if (groupNum_ == 1U) {
        CalcTailBasicBlock(compileInfoPtr);
        CalAswtL1Tiling(compileInfoPtr);
    }
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
    tilingData_.mmTilingData.isBias = 0;
    tilingData_.mmTilingData.M = m_;
    tilingData_.mmTilingData.N = n_;
    tilingData_.mmTilingData.Ka = k_;
    tilingData_.mmTilingData.Kb = k_;
    tilingData_.mmTilingData.singleCoreM = m_;
    tilingData_.mmTilingData.singleCoreN = baseN_;
    tilingData_.mmTilingData.singleCoreK = k_;
    tilingData_.mmTilingData.dbL0A = DB_SIZE;
    tilingData_.mmTilingData.dbL0B = DB_SIZE;
    tilingData_.mmTilingData.dbL0C = 1;
    tilingData_.mmTilingData.baseM = baseM_;
    tilingData_.mmTilingData.baseN = baseN_;
    tilingData_.mmTilingData.baseK = baseK_;
    tilingData_.mmTilingData.stepKa = stepKa_;
    tilingData_.mmTilingData.stepKb = stepKb_;
    tilingData_.mmTilingData.depthA1 = depthA1_;
    tilingData_.mmTilingData.depthB1 = depthB1_;
    tilingData_.mmTilingData.stepM = 1;
    tilingData_.mmTilingData.stepN = 1;
    tilingData_.mmTilingData.usedCoreNum = usedCoreNum_;
}

/** @brief split K single-single-single
 */
bool GroupedMatmulAddNoQuantTiling::SplitKSingleXSingleWeightSingleY(const gert::TilingContext *context,
                                                                     const gert::Shape xShape, const gert::Shape wShape)
{
    auto groupListTensor = context->GetDynamicInputTensor(INDEX_GROUPLIST, 0);
    OP_CHECK_IF(groupListTensor == nullptr, OP_LOGE(context->GetNodeName(), "groupListTensor is nullptr"),
                return false);
    gert::Shape groupListShape = groupListTensor->GetStorageShape();
    groupNum_ = static_cast<uint32_t>(groupListShape.GetDim(0)); // 0: the first dim of groupList is groupNum
    m_ = static_cast<uint64_t>(xShape.GetDim(1));
    k_ = static_cast<uint64_t>(xShape.GetDim(xKDim_));
    n_ = static_cast<uint64_t>(wShape.GetDim(weightNDim_));
    return true;
}

bool GroupedMatmulAddNoQuantTiling::SetCustomParam(gert::TilingContext *context)
{
    size_t *workspaces = context->GetWorkspaceSizes(1); // get second variable
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "workspaces is nullptr."),
                return false); // check workspaces is not null
    workspaces[0] = 32U;       // 32: default workspace size

    context->SetBlockDim(usedCoreNum_);
    OP_CHECK_IF(context->GetRawTilingData() == nullptr, OP_LOGE(context->GetNodeName(), "RawTilingData is nullptr."),
                return false);
    errno_t ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return false;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return true;
}

void GroupedMatmulAddNoQuantTiling::SetTilingKey(gert::TilingContext *context) const
{
    uint8_t gmmTrans = static_cast<uint8_t>(transposeX_) | (static_cast<uint8_t>(transposeWeight_) << 1);
    constexpr uint64_t TILINGKEYOFFSETGMM = 10000900009000090000UL; // means base tiling template
    uint64_t tilingKey = TILINGKEYOFFSETGMM + gmmTrans;
    OP_LOGI(context->GetNodeName(), "GMM Tiling key: gmmTrans %u, tilingkey: %lu", gmmTrans, tilingKey);
    context->SetTilingKey(tilingKey);
}

void GroupedMatmulAddNoQuantTiling::PrintTilingResult(const gert::TilingContext *context)
{
    OP_LOGI(context->GetNodeName(), "GMM Tiling result: groupNum: %u, groupListType: %u, mTailCnt: %u, nTailCnt: %u",
            tilingData_.gmmAddParams.groupNum, tilingData_.gmmAddParams.groupListType,
            tilingData_.gmmAddParams.mTailCnt, tilingData_.gmmAddParams.nTailCnt);

    OP_LOGI(context->GetNodeName(),
            "GMM MatMul Tiling result: usedCoreNum: %d, baseM: %d, baseN: %d, baseK: %d, stepKa: %d,"
            "stepKb: %d, depthA1: %d, depthB1: %d",
            tilingData_.mmTilingData.usedCoreNum, tilingData_.mmTilingData.baseM, tilingData_.mmTilingData.baseN,
            tilingData_.mmTilingData.baseK, tilingData_.mmTilingData.stepKa, tilingData_.mmTilingData.stepKb,
            tilingData_.mmTilingData.depthA1, tilingData_.mmTilingData.depthB1);
}

} // namespace optiling

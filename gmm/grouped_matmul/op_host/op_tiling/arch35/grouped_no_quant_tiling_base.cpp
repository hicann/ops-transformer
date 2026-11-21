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
 * \file grouped_no_quant_tiling_base.cpp
 * \brief GMM/GMM-ADD arch35 no-quant tiling 公共算法实现（归一自两侧原重复代码，行为保持不变）。
 */
#include "grouped_no_quant_tiling_base.h"

#include <algorithm>

namespace optiling {
/** @brief split K single-single-single(s-s-s)，groupNum 取自 groupList 首维
 */
bool GroupedNoQuantTilingBase::SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context,
                                                                const gert::Shape xShape, const gert::Shape wShape)
{
    auto groupListTensor = context->GetDynamicInputTensor(groupListIndex_, 0);
    OP_CHECK_IF(groupListTensor == nullptr, OP_LOGE(context->GetNodeName(), "groupListTensor is nullptr"),
                return false);
    gert::Shape groupListShape = groupListTensor->GetStorageShape();
    groupNum_ = static_cast<uint32_t>(groupListShape.GetDim(0)); // 0: the first dim of groupList is groupNum
    m_ = static_cast<uint64_t>(xShape.GetDim(1));
    k_ = static_cast<uint64_t>(xShape.GetDim(xKDim_));
    n_ = static_cast<uint64_t>(wShape.GetDim(weightNDim_) * nzFactor_);
    return true;
}

/** @brief split K single-multi-multi(s-m-m)
 */
bool GroupedNoQuantTilingBase::SplitKSingleXSeparatedWeight(const gert::TilingContext* context,
                                                            const gert::Shape xShape)
{
    int64_t m = xShape.GetDim(1);
    int64_t k = xShape.GetDim(xKDim_);
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, i);
        if (wTensor == nullptr) {
            break;
        }
        auto wTensorShape = wTensor->GetOriginShape();
        groupNum_ += 1U;
        int64_t n = wTensorShape.GetDim(weightNDim_) * nzFactor_;
        n_ = std::max(n_, static_cast<uint64_t>(n));
    }
    m_ = static_cast<uint64_t>(m);
    k_ = static_cast<uint64_t>(k);
    groupType_ = NO_SPLIT;
    return true;
}

/** @brief split M：single-single-single(s-s-s)
 */
bool GroupedNoQuantTilingBase::SplitMSingleXSingleWeightSingleY(const gert::Shape xShape, const gert::Shape wShape)
{
    groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
    int64_t m = xShape.GetDim(0);
    int64_t k = xShape.GetDim(xKDim_);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    m_ = static_cast<uint64_t>(m);
    k_ = static_cast<uint64_t>(k);
    n_ = static_cast<uint64_t>(n);
    kZero_ = kZero_ || (k == 0);
    weightNoL2Cache_ = true;
    return true;
}

/** @brief split M：single-multi-single(s-m-s)/single-multi-multi(s-m-m), share the same function.
 */
bool GroupedNoQuantTilingBase::SplitMSingleXSeparatedWeight(const gert::TilingContext* context,
                                                            const gert::Shape xShape)
{
    int64_t m = xShape.GetDim(0);
    int64_t k = xShape.GetDim(xKDim_);
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, i);
        if (wTensor == nullptr) {
            break;
        } // when x has multi tensors, xTensor is allowed to be empty
        auto wShape = wTensor->GetOriginShape();

        groupNum_ += 1U;
        int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
        n_ = std::max(n_, static_cast<uint64_t>(n));
    }
    m_ = static_cast<uint64_t>(m);
    k_ = static_cast<uint64_t>(k);
    kZero_ = kZero_ || (k == 0);
    weightNoL2Cache_ = IsSingleY();
    return true;
}

/** @brief split M：multi-multi-single(m-m-s); no split: multi-multi-multi(m-m-m), share the same function
 */
bool GroupedNoQuantTilingBase::SeparatedXSeparatedWeight(const gert::TilingContext* context)
{
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, i);
        auto xTensor = context->GetDynamicInputTensor(INDEX_X, i);
        if (wTensor == nullptr || xTensor == nullptr) {
            break;
        }
        auto wShape = wTensor->GetOriginShape();
        auto xShape = xTensor->GetStorageShape();
        groupNum_ += 1U;
        int64_t m = xShape.GetDim(0);
        int64_t k = xShape.GetDim(xKDim_);
        kZero_ = kZero_ || (k == 0);
        int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
        m_ = std::max(m_, static_cast<uint64_t>(m));
        k_ = std::max(k_, static_cast<uint64_t>(k));
        n_ = std::max(n_, static_cast<uint64_t>(n));
    }
    groupType_ = NO_SPLIT;
    return true;
}

/** @brief split M : multi-single-multi(m-s-m), split K : multi-single-multi(m-s-m), share the same function
 */
bool GroupedNoQuantTilingBase::SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape wShape)
{
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto xTensor = context->GetDynamicInputTensor(INDEX_X, i);
        if (xTensor == nullptr) {
            break;
        } // when x has multi tensors, xTensor is allowed to be empty
        auto xShape = xTensor->GetStorageShape();
        groupNum_ += 1U;
        int64_t m = xShape.GetDim(0);
        int64_t k = xShape.GetDim(xKDim_);
        kZero_ = kZero_ || (k == 0);
        m_ = std::max(m_, static_cast<uint64_t>(m));
        k_ = std::max(k_, static_cast<uint64_t>(k));
    }
    n_ = static_cast<uint64_t>(n);
    groupType_ = NO_SPLIT;
    return true;
}

void GroupedNoQuantTilingBase::FormulateBasicBlock(const GMMCompileInfo* compileInfoPtr, uint32_t remainCoreNum)
{
    uint64_t mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
    uint64_t nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    if (mCnt * nCnt >= remainCoreNum) {
        return;
    }
    if (mCnt <= nCnt) {
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
        nCnt = remainCoreNum / mCnt;
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
    } else {
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
        mCnt = remainCoreNum / nCnt;
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
    }

    while (baseN_ >= baseM_ * NUM_TWO && nCnt < remainCoreNum / NUM_TWO) {
        nCnt = nCnt * NUM_TWO;
        mCnt = remainCoreNum / nCnt;
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
        nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    }

    while (baseM_ >= baseN_ * NUM_TWO && mCnt < remainCoreNum / NUM_TWO) {
        mCnt = mCnt * NUM_TWO;
        nCnt = remainCoreNum / mCnt;
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
        nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    }
    mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
    nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    uint64_t kValueAlign = GroupedMatmul::CeilAlign(k_, BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax =
        GroupedMatmul::FloorAlign(static_cast<uint64_t>(compileInfoPtr->l0ASize / DB_SIZE / GetSizeByDataType(xDType_) /
                                                        std::max(baseM_, baseN_)),
                                  BASIC_BLOCK_SIZE_16);
    baseK_ = std::min(kValueAlign, kValueMax);
    usedCoreNum_ = std::min(static_cast<uint32_t>(mCnt * nCnt * groupNum_), compileInfoPtr->aicNum);
}

void GroupedNoQuantTilingBase::CalAswtL1Tiling(const GMMCompileInfo* compileInfoPtr)
{
    uint64_t totalL1Size = compileInfoPtr->l1Size;
    uint64_t reserveBTSize = GetBiasReserveSize();
    depthA1_ = totalL1Size / NUM_TWO / baseM_ / baseK_ / GetSizeByDataType(xDType_);      // 2: half of l1
    depthB1_ = totalL1Size / NUM_TWO / baseN_ / baseK_ / GetSizeByDataType(weightDtype_); // 2: half of l1

    uint64_t depthASize = depthA1_ * baseM_ * baseK_ * GetSizeByDataType(xDType_);
    uint64_t depthBSize = depthB1_ * baseN_ * baseK_ * GetSizeByDataType(weightDtype_);
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (baseM_ <= baseN_) {
            depthA1_ = std::max(depthA1_ / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
        } else {
            depthB1_ = std::max(depthB1_ / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
        }
    }
    stepKa_ = std::max(depthA1_ / DB_SIZE, 1UL);
    stepKb_ = std::max(depthB1_ / DB_SIZE, 1UL);
    if (stepKa_ >= stepKb_) {
        stepKa_ = stepKa_ / stepKb_ * stepKb_;
    } else {
        stepKb_ = stepKb_ / stepKa_ * stepKa_;
    }
    depthA1_ = stepKa_ * DB_SIZE; // depth % (stepKa * stepM) == 0
    depthB1_ = stepKb_ * DB_SIZE; // depth % (stepKb * stepN) == 0
    return;
}

void GroupedNoQuantTilingBase::CalcTailBasicBlock(const GMMCompileInfo* compileInfoPtr)
{
    uint64_t mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
    uint64_t nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfoPtr->aicNum ? 0UL : mnCnt % compileInfoPtr->aicNum;

    if (tailCnt != 0UL) {
        while ((mTailCnt_ + 1UL) * nTailCnt_ * tailCnt <= compileInfoPtr->aicNum) {
            mTailCnt_ += 1UL;
            if (mTailCnt_ * (nTailCnt_ + 1UL) * tailCnt <= compileInfoPtr->aicNum) {
                nTailCnt_ += 1UL;
            }
        }
    }
}

void GroupedNoQuantTilingBase::CalSplitKBasicBlockTiling(const GMMCompileInfo* compileInfoPtr)
{
    uint32_t remainCoreNum = std::max(1U, compileInfoPtr->aicNum / groupNum_);
    FormulateBasicBlock(compileInfoPtr, remainCoreNum);
    if (groupNum_ == 1U) {
        CalcTailBasicBlock(compileInfoPtr);
        CalAswtL1Tiling(compileInfoPtr);
    }
}

void GroupedNoQuantTilingBase::SetL0cDbTiling(const GMMCompileInfo* compileInfoPtr)
{
    // k分组/单组场景m/n固定，host直接判断：整块(baseM×baseN, fp32累加)超L0C半区且m超baseM
    constexpr uint64_t accDtypeSize = 4UL; // fp32累加
    const uint64_t kL1 = baseK_ * std::max(std::min(stepKa_, stepKb_), 1UL);
    const bool enableL0cDb = (xDType_ == ge::DT_FLOAT16 || xDType_ == ge::DT_BF16) && m_ > baseM_ &&
                             baseM_ * baseN_ * accDtypeSize > compileInfoPtr->l0CSize / DB_SIZE && k_ > kL1 &&
                             n_ >= baseN_;
    if (enableL0cDb) {
        EnableL0cDbTiling();
        return;
    }
    mL1_ = baseM_;
    nL1_ = baseN_;
}

void GroupedNoQuantTilingBase::EnableL0cDbTiling()
{
    // 使能l0cdb：mL1/nL1记录原始baseM/baseN，baseN减半作为L0块下发，kernel侧block按L0块
    // 对L1 tile切分并对L0C乒乓分段。减半必须向上16对齐
    mL1_ = baseM_;
    nL1_ = baseN_;
    baseN_ = std::max(GroupedMatmul::CeilAlign(baseN_ / DB_SIZE, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
}

} // namespace optiling

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
 * \file non_zero_tiling_arch35.cpp
 * \brief
 */
#pragma once
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "non_zero_tiling_arch35.h"
using namespace Ops::Base;

namespace optiling {
namespace BSA_ARC35 {
constexpr size_t ATTR_TRANSPOSE_IDX = 0;
constexpr size_t ATTR_DTYPE_IDX = 1;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 1;
constexpr size_t SHAPE_DIM_0 = 0;
constexpr size_t SHAPE_DIM_1 = 1;
constexpr size_t SHAPE_DIM_2 = 2;
constexpr size_t SHAPE_DIM_3 = 3;
constexpr size_t SHAPE_DIM_4 = 4;
constexpr size_t SHAPE_DIM_5 = 5;
constexpr size_t SHAPE_DIM_6 = 6;
constexpr size_t SHAPE_DIM_7 = 7;
constexpr size_t SHAPE_DIM_8 = 8;
constexpr size_t DIV_NUM = 2;
constexpr int64_t SHAPE_DIM_MAX = 7;
constexpr int64_t WORKSPACE_SIZE_OFFSET = 9216L; // 72 * 128
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024 + WORKSPACE_SIZE_OFFSET;
constexpr int64_t TMP_UB_SIZE_SMALL = 128;
constexpr int64_t TMP_UB_SIZE_BIG = 2304L;    // 72 * 32
constexpr int64_t TMP_UB_SIZE_BIGMASK = 9216; // 72 * 128
constexpr int64_t ALIGN_UB_SIZE = 256;
constexpr int64_t ALIGN_UB_32 = 32;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t ALIGN_NUM_8 = 8;
constexpr int64_t QUICK_DIV_NUM_32 = 32;
constexpr int64_t MASK_UB_DIV_NUM = 8;
constexpr int64_t B8_BYTES = 1;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t B64_BYTES = 8;
constexpr int64_t UB_REG_SIZE = 32;
constexpr int64_t VLEN_SIZE = 256;
constexpr int64_t UNPACK_NUM = 8;
constexpr int64_t VSQZ_UB_SIZE = 4;
constexpr int64_t NUM_1 = 1;
constexpr int64_t NUM_2 = 2;
constexpr int64_t NUM_4 = 4;
constexpr int64_t NUM_8 = 8;
constexpr int64_t NUM_32 = 32;
constexpr int64_t NUM_64 = 64;
constexpr int64_t NUM_128 = 128;
constexpr int64_t NUM_256 = 256;
constexpr int64_t HALF_PARAM = 2;

constexpr int64_t TILING_KEY_BIG_MASK_1_DIM = 20001;

class NonZeroAscendCTilingImpl {
public:
    explicit NonZeroAscendCTilingImpl(gert::TilingContext* context)
        : context_(context) {};
    ge::graphStatus DoTiling(int64_t coreNum, int64_t ubSize, int64_t vRegSize, mNonZeroTiling& tiling_data);

private:
    ge::graphStatus CheckInputOutputSize();
    void MatchTilingStrategyAndCalcUbSizeInfo(int64_t inputDataSize);
    void CalcMaskUbSize(int64_t inputDtypeSize);
    void CalcMaskLoopNum(int64_t numInput);
    void FillTilingData(mNonZeroTiling& tiling_data);
    void PrintTilingData();
    void CalcUbSizeInfoSmallMask();
    void CalcUbSizeInfoBigMask();
    void CalcMulInDim(std::vector<int64_t> shape, int64_t dimSize);
    ge::graphStatus CalcQuickDivParams();

private:
    int64_t inputDims_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t numPerCore_ = 0;
    int64_t numTailCore_ = 0;
    int64_t ubFactorNum_ = 0;
    int64_t loopNumPerCore_ = 0;
    int64_t loopTailPerCore_ = 0;
    int64_t loopNumTailCore_ = 0;
    int64_t loopTailTailCore_ = 0;

    int64_t needTranspose_ = 0;
    int64_t tilingKey_ = 0;

    int64_t offsetInt32Trans_ = 0;
    int64_t offsetInt64_ = 0;
    int64_t maskLoopNum_ = 0;

    int64_t loopNumO_ = 0;
    int64_t beforeNumO_ = 0;
    int64_t loopTailO_ = 0;
    int64_t loopNumTo_ = 0;
    int64_t loopTailTo_ = 0;
    int64_t maskLoopNumO_ = 0;

    int64_t xInputSize_ = 0;
    int64_t maskSize_ = 0;

    int64_t ubSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t inputUbSize_ = 0;
    int64_t maskUbSize_ = 0;
    int64_t intputDtypeSize_ = 0;
    int64_t outputDtypeSize_ = 0;

    int64_t dimSize_ = 0;
    int64_t vRegSize_ = 0;
    int64_t ubAlignNum_ = 0;
    int64_t maskUbSizeThres_ = 0; // 模版切分阈值
    int64_t innerSize_ = 1;

    int64_t mulInDimRList_[SHAPE_DIM_MAX] = {0, 0, 0, 0, 0, 0, 0};
    int64_t quickDivRKList_[SHAPE_DIM_MAX] = {0, 0, 0, 0, 0, 0, 0};
    int64_t quickDivRMList_[SHAPE_DIM_MAX] = {0, 0, 0, 0, 0, 0, 0};
    ge::DataType inputDtype_ = ge::DataType::DT_MAX;
    ge::DataType outputDtype_ = ge::DataType::DT_MAX;
    gert::Shape xShape_;
    gert::Shape yShape_;
    gert::TilingContext* context_ = nullptr;
};

void NonZeroAscendCTilingImpl::CalcMaskUbSize(int64_t inputDtypeSize)
{
    maskUbSize_ = CeilDiv(numPerCore_, vRegSize_ / inputDtypeSize) * (UNPACK_NUM / inputDtypeSize) * UB_REG_SIZE;
}

void NonZeroAscendCTilingImpl::MatchTilingStrategyAndCalcUbSizeInfo(int64_t inputDataSize)
{
    numPerCore_ = CeilDiv(inputDataSize, coreNum_);
    CalcMaskUbSize(intputDtypeSize_);
    tilingKey_ = TILING_KEY_BIG_MASK_1_DIM;
    numPerCore_ = inputDataSize / coreNum_;
    realCoreNum_ = inputDataSize >= coreNum_ ? coreNum_ : inputDataSize;
    numTailCore_ = inputDataSize - realCoreNum_ * numPerCore_;
    CalcUbSizeInfoBigMask();
}

void NonZeroAscendCTilingImpl::FillTilingData(mNonZeroTiling& tilingData)
{
    tilingData.set_inputDims(inputDims_);
    tilingData.set_realCoreNum(realCoreNum_);
    tilingData.set_numPerCore(numPerCore_);
    tilingData.set_numTailCore(numTailCore_);
    tilingData.set_ubFactorNum(ubFactorNum_);
    tilingData.set_loopNumPerCore(loopNumPerCore_);
    tilingData.set_loopTailPerCore(loopTailPerCore_);
    tilingData.set_loopNumTailCore(loopNumTailCore_);
    tilingData.set_loopTailTailCore(loopTailTailCore_);

    tilingData.set_needTranspose(needTranspose_);
    tilingData.set_tilingKey(tilingKey_);
    tilingData.set_offsetInt32Trans(offsetInt32Trans_);
    tilingData.set_offsetInt64(offsetInt64_);
    tilingData.set_maskLoopNum(maskLoopNum_);

    tilingData.set_loopNumO(loopNumO_);
    tilingData.set_beforeNumO(beforeNumO_);
    tilingData.set_loopTailO(loopTailO_);
    tilingData.set_loopNumTo(loopNumTo_);
    tilingData.set_loopTailTo(loopTailTo_);
    tilingData.set_maskLoopNumO(maskLoopNumO_);

    tilingData.set_xInputSize(inputUbSize_);
    tilingData.set_maskSize(maskUbSize_);

    tilingData.set_mulInDimRList(mulInDimRList_);
    tilingData.set_quickDivRKList(quickDivRKList_);
    tilingData.set_quickDivRMList(quickDivRMList_);
}

void NonZeroAscendCTilingImpl::CalcMulInDim(std::vector<int64_t> shape, int64_t dimSize)
{
    int64_t tmpProduct = 1;
    std::vector<int64_t> innerMul(SHAPE_DIM_MAX, 1);
    for (int64_t i = dimSize - 1; i > 0; i--) {
        if (i == dimSize - 1) {
            innerMul[i - 1] = shape[i];
            tmpProduct = innerMul[i - 1];
            continue;
        }
        innerMul[i - 1] = shape[i] * tmpProduct;
        tmpProduct = innerMul[i - 1];
    }
    for (int64_t idx = 0; idx < SHAPE_DIM_MAX; idx++) {
        mulInDimRList_[idx] = innerMul[idx];
    }
}

ge::graphStatus NonZeroAscendCTilingImpl::CalcQuickDivParams()
{
    // calc quick div params rk and rm
    for (int64_t i = 0; i < SHAPE_DIM_MAX; i++) {
        uint64_t c = mulInDimRList_[i];
        OP_CHECK_IF(c <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "c", std::to_string(c).c_str(),
                                                          "divisor c must be greater than 0"),
                    return ge::GRAPH_FAILED);
        quickDivRKList_[i] = std::ceil(std::log2(c));
        quickDivRMList_[i] =
            std::ceil(std::exp2(quickDivRKList_[i] + QUICK_DIV_NUM_32) / c) - std::exp2(QUICK_DIV_NUM_32);
    }
    return ge::GRAPH_SUCCESS;
}

static std::vector<int64_t> GetXShape(const gert::Shape& shape)
{
    int32_t dim_num = static_cast<int32_t>(shape.GetDimNum());
    int64_t sum = 1;
    for (int32_t i = 0; i < dim_num; i++) {
        sum *= shape.GetDim(i);
    }
    return {sum};
}

void NonZeroAscendCTilingImpl::CalcMaskLoopNum(int64_t numInput)
{
    ubFactorNum_ = FloorDiv(numInput, (vRegSize_ / intputDtypeSize_)) * (vRegSize_ / intputDtypeSize_);
    maskLoopNum_ =
        (ubFactorNum_ / (vRegSize_ / intputDtypeSize_) * (UNPACK_NUM / intputDtypeSize_) * UB_REG_SIZE) / UNPACK_NUM;
}

void NonZeroAscendCTilingImpl::CalcUbSizeInfoSmallMask()
{
    inputUbSize_ = FloorDiv(((ubSize_ - TMP_UB_SIZE_BIG - maskUbSize_) / DIV_NUM), static_cast<uint64_t>(ALIGN_UB_32)) *
                   ALIGN_UB_32;
    int64_t numInput = inputUbSize_ / intputDtypeSize_;

    CalcMaskLoopNum(numInput);
    loopNumPerCore_ = numPerCore_ / ubFactorNum_;
    if (numPerCore_ % ubFactorNum_ == 0) {
        loopNumPerCore_ = loopNumPerCore_ - 1;
        CalcMaskUbSize(intputDtypeSize_);
    }
    loopTailPerCore_ = numPerCore_ - loopNumPerCore_ * ubFactorNum_;
    loopNumTailCore_ = numTailCore_ / ubFactorNum_;
    if (numTailCore_ > 0 && numTailCore_ % ubFactorNum_ == 0) {
        loopNumTailCore_ = loopNumTailCore_ - 1;
    }
    loopTailTailCore_ = numTailCore_ - loopNumTailCore_ * ubFactorNum_;
    inputUbSize_ = FloorDiv(((ubSize_ - TMP_UB_SIZE_BIG - maskUbSize_) / DIV_NUM), static_cast<uint64_t>(ALIGN_UB_32)) *
                   ALIGN_UB_32;
    int32_t inputDimsNew = inputDims_;
    // 输出是int32且需要转置场景
    if ((inputDims_ != 1) && outputDtypeSize_ == B32_BYTES && (needTranspose_ == 1)) {
        inputDimsNew = inputDims_ + 1;
    }

    int64_t numInputO = inputUbSize_ / outputDtypeSize_;
    beforeNumO_ = numInputO / inputDimsNew;
    beforeNumO_ = (beforeNumO_ / NUM_64) * NUM_64; // repeat对齐

    loopNumO_ = numPerCore_ / beforeNumO_;
    loopTailO_ = numPerCore_ - loopNumO_ * beforeNumO_;
    loopNumTo_ = numTailCore_ / beforeNumO_;
    loopTailTo_ = numTailCore_ - beforeNumO_ * loopNumTo_;
    maskLoopNumO_ = (beforeNumO_ / NUM_64) * NUM_8;

    int64_t TmpBeforeNumO = beforeNumO_;
    if (loopNumO_ == 0) {
        TmpBeforeNumO = CeilDiv(numPerCore_, ALIGN_NUM_8) * ALIGN_NUM_8;
        inputUbSize_ = TmpBeforeNumO * inputDimsNew * outputDtypeSize_;
    }

    if ((inputDims_ != 1) && outputDtypeSize_ == B64_BYTES) {
        offsetInt64_ = TmpBeforeNumO * inputDims_;
        offsetInt32Trans_ = 0;
    } else if ((inputDims_ != 1) && (outputDtypeSize_ == B32_BYTES) && (needTranspose_ == 1)) {
        offsetInt64_ = 0;
        offsetInt32Trans_ = TmpBeforeNumO * inputDims_;
    } else if (inputDims_ == 1 && outputDtypeSize_ == B64_BYTES) {
        offsetInt64_ = 0;
        offsetInt32Trans_ = TmpBeforeNumO;
    } else {
        offsetInt64_ = 0;
        offsetInt32Trans_ = 0;
    }
}

void NonZeroAscendCTilingImpl::CalcUbSizeInfoBigMask()
{
    ubFactorNum_ = (ubSize_ - TMP_UB_SIZE_BIGMASK) / static_cast<int64_t>(DIV_NUM) /
                   (intputDtypeSize_ + VSQZ_UB_SIZE + inputDims_ * outputDtypeSize_);
    // 按照256Byte向下对齐
    ubAlignNum_ = ALIGN_UB_SIZE / intputDtypeSize_;
    ubFactorNum_ = FloorDiv(ubFactorNum_, ubAlignNum_) * ubAlignNum_;

    loopNumPerCore_ = numPerCore_ / ubFactorNum_;
    loopNumTailCore_ = numPerCore_ % ubFactorNum_;
}

ge::graphStatus NonZeroAscendCTilingImpl::DoTiling(int64_t coreNum, int64_t ubSize, int64_t vRegSize,
                                                   mNonZeroTiling& tiling_data)
{
    OP_LOGD(context_->GetNodeName(), "Enter NonZeroAscendCTilingImpl init.");
    coreNum_ = coreNum;
    ubSize_ = ubSize;
    vRegSize_ = vRegSize;
    OP_CHECK_IF(
        coreNum_ <= 0 || ubSize_ <= 0 || vRegSize_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "coreNum, ubSize, vRegSize",
            (std::to_string(coreNum_) + ", " + std::to_string(ubSize_) + ", " + std::to_string(vRegSize_)).c_str(),
            "value must be greater than 0"),
        return ge::GRAPH_FAILED);
    needTranspose_ = 0;
    intputDtypeSize_ = sizeof(uint8_t);
    outputDtypeSize_ = sizeof(int32_t);
    // 获取blockMask，并展开成1维
    auto blockSparseMaskShape = context_->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSparseMaskShape);
    xShape_ = EnsureNotScalar(blockSparseMaskShape->GetStorageShape());
    const std::vector<int64_t> xRunShape = GetXShape(xShape_);
    inputDims_ = static_cast<int64_t>(1);
    dimSize_ = xRunShape.size();
    // calculate shape multiplier in the R dimension
    CalcMulInDim(xRunShape, dimSize_);
    // calculate and verify quick division parameters
    if (CalcQuickDivParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xShape_.GetDimNum(); i++) {
        innerSize_ *= xShape_.GetDim(i);
    }
    maskUbSizeThres_ = (ubSize_ - TMP_UB_SIZE_BIGMASK) / HALF_PARAM;

    // match tiling strategy and calc Ub size info
    MatchTilingStrategyAndCalcUbSizeInfo(innerSize_);

    // fill data
    FillTilingData(tiling_data);
    return ge::GRAPH_SUCCESS;
}
} // namespace BSA_ARC35
} // namespace optiling

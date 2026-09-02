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
 * \file inplace_partial_rotary_mul_tiling.cpp
 * \brief
 */

#include "inplace_partial_rotary_mul_tiling.h"
#include <graph/utils/type_utils.h>

namespace optiling {
using namespace Ops::Base;
constexpr int64_t TILING_KEY_FLOAT16 = 0;
constexpr int64_t TILING_KEY_BFLOAT16 = 10;
constexpr int64_t TILING_KEY_FLOAT32 = 20;
constexpr int64_t TILING_KEY_BFLOAT16_FLOAT32_MIXED = 30;
constexpr int64_t TILING_KEY_FLOAT16_FLOAT32_MIXED = 40;
constexpr int64_t TILING_KEY_UNPAD = 0;
constexpr int64_t TILING_KEY_PAD = 1;
constexpr int64_t TILING_KEY_SPLIT_S = 0;
constexpr int64_t TILING_KEY_SPLIT_BS = 100;
constexpr int64_t TILING_KEY_SPLIT_BSN = 200;
constexpr int64_t TILING_KEY_NOOP = 1; // sliceLength = 0, no computation
constexpr int64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr int64_t FP32_DTYPE_SIZE = 4;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t REPEAT_FP32 = 64;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t INTERLEAVE_MODE_COEF = 2;
constexpr int64_t TBUF_SIZE = 0;
constexpr int64_t ALIGN_32 = 8;
constexpr int64_t ALIGN_16 = 16;
constexpr int64_t IO_NUM = 3; // sin、cos -> tri
constexpr int64_t BASE_KEY = 2000;
constexpr int64_t CONST_4 = 4;
constexpr int64_t CONST_2 = 2;

class InplacePartialRotaryMulTiling {
public:
    explicit InplacePartialRotaryMulTiling(gert::TilingContext *context)
        : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInput();
    ge::graphStatus CalTilingData();
    ge::graphStatus TilingSplitN(int64_t numHeads, int64_t headDimAlign, int64_t ubSize, ge::DataType dataDtype);
    ge::graphStatus TilingSplitB(int64_t batchSize, int64_t numHeads, int64_t headDimAlign, int64_t ubSize,
                                 ge::DataType dataDtype);
    ge::graphStatus TilingSplitS();
    ge::graphStatus TilingSplit();
    ge::graphStatus TilingSplitNMixed(int64_t numHeads, int64_t headDimAlign, int64_t ubSize);
    ge::graphStatus TilingSplitBMixed(int64_t batchSize, int64_t numHeads, int64_t headDimAlign, int64_t ubSize);
    ge::graphStatus TilingSplitSMixed();
    ge::graphStatus TilingSplitMixed();
    int64_t GetMixedTilingKeyOffset() const;
    ge::graphStatus InitSplitTilingData(int64_t &batchSizeOut, int64_t &seqLenOut, int64_t &numHeadsOut);
    void FillTilingData();
    void PrintTilingData() const;
    void PrintInfo();
    ge::graphStatus DoTilingMixedPrecision();
    ge::graphStatus DoTilingRegular();
    void CalcUbCalcSMixed(int64_t coreCalcNum, int64_t coreCalcTail, int64_t totalSeq1Size, int64_t bufferSize,
                          int64_t ioUbSize);

private:
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t dtypeX = 0;
    int64_t repeatNum_ = 0;
    bool isBrc_ = true;
    int64_t dim0_ = 0;
    int64_t dim1_ = 0;
    int64_t dim2_ = 0;
    int64_t end_ = 0;
    int64_t tilingKey_ = TILING_KEY_NOOP;
    bool isAlign_ = false;
    bool isSpecail_ = false;
    int64_t oneBlockSize_ = 0;
    int64_t dtypeSize_ = 2;
    int64_t xdim0_ = 0;
    int64_t xdim1_ = 0;
    int64_t xdim2_ = 0;
    int64_t xdim3_ = 0;
    int64_t r1dim0_ = 0;
    int64_t r1dim1_ = 0;
    int64_t r1dim2_ = 0;
    int64_t r1dim3_ = 0;
    ge::DataType xDtype_ = ge::DT_FLOAT16;
    ge::DataType cosDtype_ = ge::DT_FLOAT16;
    bool isMixedPrecision_ = false;

    // tiingdata
    int64_t usedCoreNum_ = 0;
    int64_t numHead_ = 0;
    int64_t headDim_ = 0;
    int64_t allHeadDim_ = 0;
    int64_t coreTUbLoopTime_ = 0;
    int64_t coreBUbLoopTime_ = 0;
    int64_t coreTUbLoopTail_ = 0;
    int64_t coreBUbLoopTail_ = 0;
    int64_t ubFactor_ = 0;
    int64_t start_ = 0;
    int64_t blockFactor_ = 0;
    gert::TilingContext *context_ = nullptr;
    InplacePartialRopeRegbaseTilingData tilingData_;
};
int64_t GetCeilInt(int64_t value1, int64_t value2)
{
    if (value2 == 0)
        return value2;
    return (value1 + value2 - 1) / value2;
}

int64_t GetDiv(int64_t value1, int64_t value2)
{
    if (value2 == 0)
        return value2;
    return value1 / value2;
}

int64_t GetDivRem(int64_t value1, int64_t value2)
{
    if (value2 == 0)
        return value2;
    return value1 % value2;
}
void InplacePartialRotaryMulTiling::FillTilingData()
{
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_allHeadDim(allHeadDim_);
    tilingData_.set_coreTUbLoopTime(coreTUbLoopTime_);
    tilingData_.set_coreBUbLoopTime(coreBUbLoopTime_);
    tilingData_.set_coreTUbLoopTail(coreTUbLoopTail_);
    tilingData_.set_coreBUbLoopTail(coreBUbLoopTail_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_start(start_);
    tilingData_.set_blockFactor(blockFactor_);
}
void InplacePartialRotaryMulTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(), "InplacePartialRotaryMulTiling  begin print.");
    OP_LOGI(context_->GetNodeName(), "usedCoreNum = %ld.", usedCoreNum_);
    OP_LOGI(context_->GetNodeName(), "numHead_ = %ld.", numHead_);
    OP_LOGI(context_->GetNodeName(), "headDim_ = %ld.", headDim_);
    OP_LOGI(context_->GetNodeName(), "allHeadDim_ = %ld.", allHeadDim_);
    OP_LOGI(context_->GetNodeName(), "coreTUbLoopTime_ = %ld.", coreTUbLoopTime_);
    OP_LOGI(context_->GetNodeName(), "coreBUbLoopTime_ = %ld.", coreBUbLoopTime_);
    OP_LOGI(context_->GetNodeName(), "coreTUbLoopTail_ = %ld.", coreTUbLoopTail_);
    OP_LOGI(context_->GetNodeName(), "coreBUbLoopTail_ = %ld.", coreBUbLoopTail_);
    OP_LOGI(context_->GetNodeName(), "ubFactor = %ld.", ubFactor_);
    OP_LOGI(context_->GetNodeName(), "start_ = %ld.", start_);
    OP_LOGI(context_->GetNodeName(), "blockFactor_ = %ld.", blockFactor_);
    OP_LOGI(context_->GetNodeName(), "tilingKey = %ld.", tilingKey_);
}
void InplacePartialRotaryMulTiling::PrintInfo()
{
    OP_LOGI(context_->GetNodeName(), "usedCoreNum = %ld.", tilingData_.get_usedCoreNum());
    OP_LOGI(context_->GetNodeName(), "start = %ld.", tilingData_.get_start());
    OP_LOGI(context_->GetNodeName(), "allHeadDim = %ld.", tilingData_.get_allHeadDim());
    OP_LOGI(context_->GetNodeName(), " batchSize=%ld.", tilingData_.get_batchSize());
    OP_LOGI(context_->GetNodeName(), " seqLen=%ld.", tilingData_.get_seqLen());
    OP_LOGI(context_->GetNodeName(), " numHeads=%ld.", tilingData_.get_numHeads());
    OP_LOGI(context_->GetNodeName(), " headDim=%ld.", tilingData_.get_headDim());
    OP_LOGI(context_->GetNodeName(), " frontCoreNum=%ld.", tilingData_.get_frontCoreNum());
    OP_LOGI(context_->GetNodeName(), " tailCoreNum=%ld.", tilingData_.get_tailCoreNum());
    OP_LOGI(context_->GetNodeName(), " coreCalcNum=%ld.", tilingData_.get_coreCalcNum());
    OP_LOGI(context_->GetNodeName(), " coreCalcTail=%ld.", tilingData_.get_coreCalcTail());
    OP_LOGI(context_->GetNodeName(), " ubCalcNum=%ld.", tilingData_.get_ubCalcNum());
    OP_LOGI(context_->GetNodeName(), " ubCalcLoop=%ld.", tilingData_.get_ubCalcLoop());
    OP_LOGI(context_->GetNodeName(), " ubCalcTail=%ld.", tilingData_.get_ubCalcTail());
    OP_LOGI(context_->GetNodeName(), " ubCalcTailNum=%ld.", tilingData_.get_ubCalcTailNum());
    OP_LOGI(context_->GetNodeName(), " ubCalcTailLoop=%ld.", tilingData_.get_ubCalcTailLoop());
    OP_LOGI(context_->GetNodeName(), " ubCalcTailTail=%ld.", tilingData_.get_ubCalcTailTail());
    OP_LOGI(context_->GetNodeName(), " ubCalcBNum=%ld.", tilingData_.get_ubCalcBNum());
    OP_LOGI(context_->GetNodeName(), " ubCalcBLoop=%ld.", tilingData_.get_ubCalcBLoop());
    OP_LOGI(context_->GetNodeName(), " ubCalcBTail=%ld.", tilingData_.get_ubCalcBTail());
    OP_LOGI(context_->GetNodeName(), " ubCalcNNum=%ld.", tilingData_.get_ubCalcNNum());
    OP_LOGI(context_->GetNodeName(), " ubCalcNLoop=%ld.", tilingData_.get_ubCalcNLoop());
    OP_LOGI(context_->GetNodeName(), " ubCalcNTail=%ld.", tilingData_.get_ubCalcNTail());
    OP_LOGI(context_->GetNodeName(), "tilingKey = %ld.", tilingKey_);
}
ge::graphStatus InplacePartialRotaryMulTiling::CalTilingData()
{
    OP_CHECK_IF(!isSpecail_, OP_LOGI("Tiling4InplacePartialRotaryMul", "not special"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!isAlign_,
                OP_LOGI("Tiling4InplacePartialRotaryMul", "d = %ld is not aligned to %ld", xdim3_, oneBlockSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xdim3_ > REPEAT_FP32,
                OP_LOGI("Tiling4InplacePartialRotaryMul", "D = %ld exceeds repeat limit %ld", xdim3_, REPEAT_FP32),
                return ge::GRAPH_FAILED);
    int64_t ubNum = ubSize_ / sizeof(float);
    int64_t last = ubNum - dim1_ * dim2_;
    int64_t preCoreNumFactor = (dim0_ + coreNum_ - 1) / coreNum_;
    usedCoreNum_ = (dim0_ + preCoreNumFactor - 1) / preCoreNumFactor;
    int64_t tailCoreNum = dim0_ - preCoreNumFactor * (usedCoreNum_ - 1);
    blockFactor_ = preCoreNumFactor;
    ubFactor_ = last / (CONST_4 * dim1_ * dim2_ + CONST_4 * dim2_);
    if (ubFactor_ > preCoreNumFactor) {
        ubFactor_ = preCoreNumFactor;
    }

    OP_LOGI(context_->GetNodeName(), "ubFactor_ = %ld.", ubFactor_);
    OP_CHECK_IF(ubFactor_ <= 0,
                OP_LOGI("Tiling4InplacePartialRotaryMul",
                        "ubFactor = %ld is non-positive, input size is too large, not supported", ubFactor_),
                return ge::GRAPH_FAILED);
    coreBUbLoopTime_ = (preCoreNumFactor + ubFactor_ - 1) / ubFactor_;
    coreBUbLoopTail_ = preCoreNumFactor % ubFactor_;
    if (coreBUbLoopTail_ == 0) {
        coreBUbLoopTail_ = ubFactor_;
    }
    coreTUbLoopTime_ = (tailCoreNum + ubFactor_ - 1) / ubFactor_;
    coreTUbLoopTail_ = tailCoreNum % ubFactor_;
    if (coreTUbLoopTail_ == 0) {
        coreTUbLoopTail_ = ubFactor_;
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::Init()
{
    OP_LOGI(context_->GetNodeName(), "Tiling4InplacePartialRotaryMul Init running.");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_, "Tiling context is null"), return ge::GRAPH_FAILED);
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "Tiling platformInfo is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_->GetNodeName(), "coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "ubSize must be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "coreNum_ is %ld, ubSize_ %ld ", coreNum_, ubSize_);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::CheckInput()
{
    auto xInput = context_->GetInputShape(0);
    auto inputR1 = context_->GetInputShape(1);
    auto inputR2 = context_->GetInputShape(2);
    OP_CHECK_IF(xInput == nullptr || inputR1 == nullptr || inputR2 == nullptr,
                OP_LOGE(context_->GetNodeName(), "get input nullptr."), return ge::GRAPH_FAILED);
    auto yOutput = context_->GetOutputShape(0);
    OP_CHECK_IF(yOutput == nullptr, OP_LOGE(context_->GetNodeName(), "get output nullptr."), return ge::GRAPH_FAILED);
    auto xDesc = context_->GetInputDesc(0);
    auto cosDesc = context_->GetInputDesc(1);
    auto sinDesc = context_->GetInputDesc(2);
    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_IF(xDesc == nullptr || cosDesc == nullptr || sinDesc == nullptr || yDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "get input or output desc nullptr."), return ge::GRAPH_FAILED);
    xDtype_ = xDesc->GetDataType();
    cosDtype_ = cosDesc->GetDataType();
    auto sinDtype = sinDesc->GetDataType();
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(!(xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "x dtype only support float16, bfloat16 or float32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosDtype_ != sinDtype, OP_LOGE(context_->GetNodeName(), "cos and sin dtype must be same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(cosDtype_ == ge::DT_FLOAT16 || cosDtype_ == ge::DT_BF16 || cosDtype_ == ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "cos/sin dtype only support float16, bfloat16 or float32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDtype != xDtype_, OP_LOGE(context_->GetNodeName(), "y dtype must be same as x dtype."),
                return ge::GRAPH_FAILED);
    bool isSamePrecision = xDtype_ == cosDtype_;
    isMixedPrecision_ = (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && cosDtype_ == ge::DT_FLOAT;
    OP_CHECK_IF(!isSamePrecision && !isMixedPrecision_,
                OP_LOGE(context_->GetNodeName(),
                        "dtype only supports same precision or x/y float16/bfloat16 with cos/sin float32."),
                return ge::GRAPH_FAILED);
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        dtypeSize_ = FP16_BF16_DTYPE_SIZE;
        oneBlockSize_ = ALIGN_16;
    } else {
        dtypeSize_ = FP32_DTYPE_SIZE;
        oneBlockSize_ = ALIGN_32;
    }

    gert::Shape xShape = xInput->GetStorageShape();
    int64_t dimNum = xShape.GetDimNum();
    gert::Shape inputR1Shape = inputR1->GetStorageShape();
    gert::Shape inputR2Shape = inputR2->GetStorageShape();
    int64_t dimNumR1 = inputR1Shape.GetDimNum();
    int64_t dimNumR2 = inputR2Shape.GetDimNum();
    auto inputShape = xInput->GetStorageShape();
    if (dimNum != CONST_4 || dimNumR1 != CONST_4 || dimNumR2 != CONST_4) {
        std::string dimNumMsg =
            std::to_string(dimNum) + ", " + std::to_string(dimNumR1) + " and " + std::to_string(dimNumR2);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x, cos and sin", dimNumMsg.c_str(),
                                                  "The shape dims of input x, cos and sin should all be 4D");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < CONST_4; i++) {
        int64_t r1dim = inputR1Shape.GetDim(i);
        int64_t r2dim = inputR2Shape.GetDim(i);
        if (r1dim != r2dim) {
            std::string shapeMsg = ToString(inputR1Shape) + " and " + ToString(inputR2Shape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                                   "The shapes of input cos and input sin should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    dim0_ = xShape.GetDim(0);
    xdim0_ = dim0_;
    xdim1_ = xShape.GetDim(1);
    xdim2_ = xShape.GetDim(2);
    allHeadDim_ = xShape.GetDim(dimNum - 1);
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr"), return ge::GRAPH_FAILED);
    const int64_t *modePtr = attrs->GetAttrPointer<int64_t>(0);
    int64_t mode = (modePtr == nullptr) ? 0 : *modePtr;
    OP_CHECK_IF(
        mode != 1,
        OP_LOGE(context_->GetNodeName(), "mode = %ld is not supported, only mode = 1 (interleave) is supported", mode),
        return ge::GRAPH_FAILED);
    // 获取slice范围：空属性时默认 [0, allHeadDim_]，与A5行为一致
    auto sliceListAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    if (sliceListAttr == nullptr || sliceListAttr->GetSize() == 0) {
        start_ = 0;
        end_ = allHeadDim_;
    } else {
        auto sliceData = static_cast<const int64_t *>(sliceListAttr->GetData());
        OP_CHECK_IF(sliceListAttr->GetSize() != 2,
                    OP_LOGE(context_->GetNodeName(), "slice_attr must have exactly 2 elements, got %zu",
                            sliceListAttr->GetSize()),
                    return ge::GRAPH_FAILED);
        start_ = sliceData[0];
        end_ = sliceData[1];
    }
    OP_LOGI(context_->GetNodeName(), "start_ %ld, end_ %ld", start_, end_);

    headDim_ = end_ - start_;
    xdim3_ = headDim_;
    // 空切片（sliceStart==sliceEnd）：无需对x做操作，跳过后续cos/sin校验
    if (headDim_ == 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (headDim_ < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                              ("sliceLength=" + std::to_string(headDim_)).c_str(),
                                              "Slice length must be >= 0");
        return ge::GRAPH_FAILED;
    }
    if (allHeadDim_ > D_LIMIT) {
        std::string reasonMsg = "The D axis of input x can not be greater than " + std::to_string(D_LIMIT) +
                                ", where D refers to the last dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                              ("D=" + std::to_string(allHeadDim_)).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (allHeadDim_ % INTERLEAVE_MODE_COEF != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", ("D=" + std::to_string(allHeadDim_)).c_str(),
            "D must be multiples of 2 in interleave mode, where D refers to the last dim");
        return ge::GRAPH_FAILED;
    }
    if (headDim_ % INTERLEAVE_MODE_COEF != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                              ("sliceLength=" + std::to_string(headDim_)).c_str(),
                                              "sliceLength must be multiples of 2 in interleave mode, "
                                              "where sliceLength refers to partialSlice[1] - partialSlice[0]");
        return ge::GRAPH_FAILED;
    }
    if (start_ < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", ("sliceStart=" + std::to_string(start_)).c_str(), "sliceStart must be >= 0");
        return ge::GRAPH_FAILED;
    }
    if (end_ < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                              ("sliceEnd=" + std::to_string(end_)).c_str(), "sliceEnd must be >= 0");
        return ge::GRAPH_FAILED;
    }
    if (end_ > allHeadDim_) {
        std::string reasonMsg =
            "sliceEnd must be <= D(" + std::to_string(allHeadDim_) + "), where D refers to the last dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                              ("sliceEnd=" + std::to_string(end_)).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    r1dim3_ = inputR1Shape.GetDim(3);
    r1dim0_ = inputR1Shape.GetDim(0);
    r1dim1_ = inputR1Shape.GetDim(1);
    r1dim2_ = inputR1Shape.GetDim(2);
    int64_t r1Dim2 = inputR1Shape.GetDim(2);
    int64_t xDim1 = xShape.GetDim(1);
    int64_t xDim2 = xShape.GetDim(2);
    if (headDim_ != r1dim3_) {
        std::string shapeMsg = "x D=" + std::to_string(headDim_) + ", cos/sin D=" + std::to_string(r1dim3_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x and cos", shapeMsg.c_str(),
            "The D axes of input x and input cos should be the same, where D refers to the last dim");
        return ge::GRAPH_FAILED;
    }
    if (r1dim0_ != dim0_) {
        std::string shapeMsg = "x dim0=" + std::to_string(dim0_) + ", cos dim0=" + std::to_string(r1dim0_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and cos", shapeMsg.c_str(),
                                               "The first dimensions of input x and input cos should be the same");
        return ge::GRAPH_FAILED;
    }
    if (r1dim2_ != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos",
                                              ("dim2=" + std::to_string(r1dim2_)).c_str(),
                                              "The third dimension of input cos should be 1");
        return ge::GRAPH_FAILED;
    }
    dim2_ = headDim_;
    dim1_ = xShape.GetDim(1) * xShape.GetDim(2);
    numHead_ = dim1_;
    if (r1dim1_ == 1 && r1dim2_ == 1 && (xdim1_ == 1 || xdim2_ == 1)) {
        tilingKey_ = 1;
        isSpecail_ = true;
        if (r1dim1_ == dim1_) {
            isBrc_ = false;
            tilingKey_ = tilingKey_ + 1;
        }
    }
    if (xdim3_ % oneBlockSize_ == 0) {
        isAlign_ = true;
    }
    OP_LOGI(context_->GetNodeName(), "isSpecail_ %d", isSpecail_);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitN(int64_t numHeads, int64_t headDimAlign, int64_t ubSize,
                                                            ge::DataType dataDtype)
{
    const int64_t bufferSize = ubSize - 0;
    int64_t totalHeadNum1Size = headDimAlign * IO_NUM * dtypeSize_ + headDimAlign * INT32_DTYPE_SIZE;
    if (xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT16) {
        totalHeadNum1Size += headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
    }
    uint32_t ubCalcNNum{1};
    uint32_t ubCalcNLoop{numHeads};
    uint32_t ubCalcNTail{0};
    if (bufferSize < totalHeadNum1Size) {
        std::string reasonMsg =
            "The D dimension of the input shape is too large to fit in UB (bufferSize=" + std::to_string(bufferSize) +
            ", required=" + std::to_string(totalHeadNum1Size) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", reasonMsg.c_str(),
                                              "The D dimension of input x is too large");
        return ge::GRAPH_FAILED;
    }
    ubCalcNNum = GetDiv(bufferSize, totalHeadNum1Size);
    ubCalcNLoop = GetCeilInt(numHeads, ubCalcNNum);
    ubCalcNTail = GetDivRem(numHeads, ubCalcNNum) != 0 ? numHeads - (ubCalcNLoop - 1) * ubCalcNNum : 0;
    tilingData_.set_ubCalcNNum(ubCalcNNum);
    tilingData_.set_ubCalcNLoop(ubCalcNLoop);
    tilingData_.set_ubCalcNTail(ubCalcNTail);
    tilingKey_ += TILING_KEY_SPLIT_BSN;
    context_->SetTilingKey(tilingKey_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitB(int64_t batchSize, int64_t numHeads, int64_t headDimAlign,
                                                            int64_t ubSize, ge::DataType dataDtype)
{
    const int64_t tBufferSize = numHeads * headDimAlign * FP32_DTYPE_SIZE;
    const int64_t bufferSize = ubSize - tBufferSize;
    int64_t totalBatch1Size = numHeads * headDimAlign * IO_NUM * dtypeSize_;

    if (xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT16) {
        totalBatch1Size += numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
    }
    int64_t ubCalcBNum{1};
    int64_t ubCalcBLoop{batchSize};
    int64_t ubCalcBTail{0};
    if (ubSize < tBufferSize || bufferSize < totalBatch1Size) {
        OP_CHECK_IF(TilingSplitN(numHeads, headDimAlign, ubSize, xDtype_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "TilingSplitN fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ubCalcBNum = GetDiv(bufferSize, totalBatch1Size);
    ubCalcBLoop = GetCeilInt(batchSize, ubCalcBNum);
    ubCalcBTail = GetDivRem(batchSize, ubCalcBNum) != 0 ? batchSize - (ubCalcBLoop - 1) * ubCalcBNum : 0;

    tilingData_.set_ubCalcBNum(ubCalcBNum);
    tilingData_.set_ubCalcBLoop(ubCalcBLoop);
    tilingData_.set_ubCalcBTail(ubCalcBTail);

    tilingKey_ += TILING_KEY_SPLIT_S;
    context_->SetTilingKey(tilingKey_);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitS()
{
    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLen();
    int64_t numHeads = tilingData_.get_numHeads();
    int64_t headDim = tilingData_.get_headDim();
    // block split
    int64_t frontCoreNum = GetDivRem(seqLen, coreNum_) != 0 ? GetDivRem(seqLen, coreNum_) : coreNum_;
    int64_t tailCoreNum = seqLen <= coreNum_ ? 0 : coreNum_ - frontCoreNum;
    usedCoreNum_ = frontCoreNum + tailCoreNum;
    int64_t coreCalcNum = GetCeilInt(seqLen, coreNum_);
    int64_t coreCalcTail = GetDiv(seqLen, coreNum_);
    tilingData_.set_frontCoreNum(frontCoreNum);
    tilingData_.set_tailCoreNum(tailCoreNum);
    tilingData_.set_coreCalcNum(coreCalcNum);
    tilingData_.set_coreCalcTail(coreCalcTail);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);
    int64_t headDimAlign = 0;
    if (isAlign_) {
        headDimAlign = headDim;
    } else {
        headDimAlign = GetCeilInt(headDim, oneBlockSize_) * oneBlockSize_;
        tilingKey_ += 1;
    }
    // ub split
    int64_t tBufferSize = numHeads * headDimAlign * FP32_DTYPE_SIZE;
    int64_t bufferSize = ubSize_ - tBufferSize;
    int64_t ioUbSize = batchSize * coreCalcNum * numHeads * headDimAlign * IO_NUM * dtypeSize_;
    int64_t totalSeq1Size = batchSize * numHeads * headDimAlign * IO_NUM * dtypeSize_;
    if (xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT16) {
        ioUbSize += batchSize * coreCalcNum * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
        totalSeq1Size += batchSize * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
    }
    if (tBufferSize >= ubSize_) {
        OP_CHECK_IF(TilingSplitN(numHeads, headDimAlign, ubSize_, xDtype_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "TilingSplitN fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (ubSize_ < tBufferSize || bufferSize < totalSeq1Size) {
        OP_CHECK_IF(TilingSplitB(batchSize, numHeads, headDimAlign, ubSize_, xDtype_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "TilingSplitB fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    context_->SetTilingKey(tilingKey_);
    int64_t ubCalcNum;
    int64_t ubCalcLoop;
    int64_t ubCalcTail;
    if (bufferSize < ioUbSize) {
        ubCalcNum = GetDiv(bufferSize, totalSeq1Size);
        ubCalcLoop = GetCeilInt(coreCalcNum, ubCalcNum);
        ubCalcTail = GetDivRem(coreCalcNum, ubCalcNum) != 0 ? coreCalcNum - (ubCalcLoop - 1) * ubCalcNum : 0;
    } else {
        ubCalcNum = coreCalcNum;
        ubCalcLoop = 1;
        ubCalcTail = 0;
    }
    tilingData_.set_ubCalcNum(ubCalcNum);
    tilingData_.set_ubCalcLoop(ubCalcLoop);
    tilingData_.set_ubCalcTail(ubCalcTail);
    // ub split for tail core
    int64_t ubCalcTailNum{0};
    int64_t ubCalcTailLoop{0};
    int64_t ubCalcTailTail{0};
    if (coreCalcTail != 0) {
        ioUbSize = batchSize * coreCalcTail * numHeads * headDimAlign * IO_NUM * dtypeSize_;
        totalSeq1Size = batchSize * numHeads * headDimAlign * IO_NUM * dtypeSize_;
        if (xDtype_ == ge::DT_BF16 || xDtype_ == ge::DT_FLOAT16) {
            ioUbSize += batchSize * coreCalcNum * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
            totalSeq1Size += batchSize * numHeads * headDimAlign * FP32_DTYPE_SIZE * IO_NUM;
        }
        if (bufferSize < ioUbSize) {
            ubCalcTailNum = GetDiv(bufferSize, totalSeq1Size);
            ubCalcTailLoop = GetCeilInt(coreCalcTail, ubCalcTailNum);
            ubCalcTailTail =
                GetDivRem(coreCalcTail, ubCalcTailNum) != 0 ? coreCalcTail - (ubCalcTailLoop - 1) * ubCalcTailNum : 0;
        } else {
            ubCalcTailNum = coreCalcTail;
            ubCalcTailLoop = 1;
            ubCalcTailTail = 0;
        }
    }
    tilingData_.set_ubCalcTailNum(ubCalcTailNum);
    tilingData_.set_ubCalcTailLoop(ubCalcTailLoop);
    tilingData_.set_ubCalcTailTail(ubCalcTailTail);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplit()
{
    int64_t batchSizeOut{1};
    int64_t seqLenOut{1};
    int64_t numHeadsOut{1};
    OP_CHECK_IF(InitSplitTilingData(batchSizeOut, seqLenOut, numHeadsOut) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "InitSplitTilingData fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(TilingSplitS() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingSplitS fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t InplacePartialRotaryMulTiling::GetMixedTilingKeyOffset() const
{
    if (xDtype_ == ge::DT_BF16) {
        return TILING_KEY_BFLOAT16_FLOAT32_MIXED;
    }
    return TILING_KEY_FLOAT16_FLOAT32_MIXED;
}

ge::graphStatus InplacePartialRotaryMulTiling::InitSplitTilingData(int64_t &batchSizeOut, int64_t &seqLenOut,
                                                                   int64_t &numHeadsOut)
{
    if (r1dim1_ == 1 && r1dim2_ == 1 && xdim0_ == r1dim0_) {
        seqLenOut = r1dim0_;
        numHeadsOut = xdim1_ * xdim2_; // SBND -> 1S(BN)D -> 1SND
    } else if (r1dim0_ == 1 && r1dim2_ == 1 && xdim1_ == r1dim1_) {
        seqLenOut = r1dim1_;
        batchSizeOut = xdim0_; // BSND
        numHeadsOut = xdim2_;
    } else if (r1dim0_ == 1 && r1dim1_ == 1 && xdim2_ == r1dim2_) {
        seqLenOut = r1dim2_;
        batchSizeOut = xdim0_ * xdim1_; // BNSD -> (BN)S1D -> BS1D
    } else if (xdim0_ == r1dim0_ && xdim1_ == r1dim1_) {
        batchSizeOut = 1;
        seqLenOut = r1dim0_ * r1dim1_;
        numHeadsOut = xdim2_; // 1,BS,N,D cons/sin 1,BS, 1,D
    } else {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x, cos and sin", "layout not supported",
            "The shapes of input x, cos and sin do not satisfy any supported layout pattern");
        return ge::GRAPH_FAILED;
    }
    if (batchSizeOut != 1) {
        OP_LOGE(context_->GetNodeName(), "batchSizeOut must be 1, got %ld", batchSizeOut);
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_batchSize(batchSizeOut);
    tilingData_.set_seqLen(seqLenOut);
    tilingData_.set_numHeads(numHeadsOut);
    tilingData_.set_headDim(xdim3_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitNMixed(int64_t numHeads, int64_t headDimAlign, int64_t ubSize)
{
    const int64_t bufferSize = ubSize - TBUF_SIZE;
    int64_t totalHeadNum1Size = headDimAlign * (FP16_BF16_DTYPE_SIZE * CONST_2 + FP32_DTYPE_SIZE +
                                                FP32_DTYPE_SIZE * CONST_2 + INT32_DTYPE_SIZE);
    int64_t ubCalcNNum{1};
    int64_t ubCalcNLoop{numHeads};
    int64_t ubCalcNTail{0};
    OP_CHECK_IF(
        bufferSize < totalHeadNum1Size,
        OP_LOGE(context_->GetNodeName(), "The D dimension of the input shape is too large for mixed precision."),
        return ge::GRAPH_FAILED);
    ubCalcNNum = GetDiv(bufferSize, totalHeadNum1Size);
    ubCalcNLoop = GetCeilInt(numHeads, ubCalcNNum);
    ubCalcNTail = GetDivRem(numHeads, ubCalcNNum) != 0 ? numHeads - (ubCalcNLoop - 1) * ubCalcNNum : 0;
    tilingData_.set_ubCalcNNum(ubCalcNNum);
    tilingData_.set_ubCalcNLoop(ubCalcNLoop);
    tilingData_.set_ubCalcNTail(ubCalcNTail);
    tilingKey_ += TILING_KEY_SPLIT_BSN;
    context_->SetTilingKey(tilingKey_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitBMixed(int64_t batchSize, int64_t numHeads,
                                                                 int64_t headDimAlign, int64_t ubSize)
{
    const int64_t gatherOffsetSize = numHeads * headDimAlign * INT32_DTYPE_SIZE;
    const int64_t bufferSize = ubSize - gatherOffsetSize;
    int64_t totalBatch1Size =
        numHeads * headDimAlign * (FP16_BF16_DTYPE_SIZE * CONST_2 + FP32_DTYPE_SIZE + FP32_DTYPE_SIZE * CONST_2);
    int64_t ubCalcBNum{1};
    int64_t ubCalcBLoop{batchSize};
    int64_t ubCalcBTail{0};
    if (ubSize < gatherOffsetSize || bufferSize < totalBatch1Size) {
        OP_CHECK_IF(TilingSplitNMixed(numHeads, headDimAlign, ubSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "TilingSplitNMixed fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ubCalcBNum = GetDiv(bufferSize, totalBatch1Size);
    ubCalcBLoop = GetCeilInt(batchSize, ubCalcBNum);
    ubCalcBTail = GetDivRem(batchSize, ubCalcBNum) != 0 ? batchSize - (ubCalcBLoop - 1) * ubCalcBNum : 0;
    tilingData_.set_ubCalcBNum(ubCalcBNum);
    tilingData_.set_ubCalcBLoop(ubCalcBLoop);
    tilingData_.set_ubCalcBTail(ubCalcBTail);
    tilingKey_ += TILING_KEY_SPLIT_BS;
    context_->SetTilingKey(tilingKey_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitSMixed()
{
    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLen();
    int64_t numHeads = tilingData_.get_numHeads();
    int64_t headDim = tilingData_.get_headDim();

    int64_t frontCoreNum = GetDivRem(seqLen, coreNum_) != 0 ? GetDivRem(seqLen, coreNum_) : coreNum_;
    int64_t tailCoreNum = seqLen <= coreNum_ ? 0 : coreNum_ - frontCoreNum;
    usedCoreNum_ = frontCoreNum + tailCoreNum;
    int64_t coreCalcNum = GetCeilInt(seqLen, coreNum_);
    int64_t coreCalcTail = GetDiv(seqLen, coreNum_);
    tilingData_.set_frontCoreNum(frontCoreNum);
    tilingData_.set_tailCoreNum(tailCoreNum);
    tilingData_.set_coreCalcNum(coreCalcNum);
    tilingData_.set_coreCalcTail(coreCalcTail);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    int64_t headDimAlign = 0;
    if (isAlign_) {
        headDimAlign = headDim;
    } else {
        headDimAlign = GetCeilInt(headDim, oneBlockSize_) * oneBlockSize_;
        tilingKey_ += TILING_KEY_PAD;
    }

    int64_t gatherOffsetSize = numHeads * headDimAlign * INT32_DTYPE_SIZE;
    int64_t bufferSize = ubSize_ - gatherOffsetSize;
    int64_t totalSeq1Size = batchSize * numHeads * headDimAlign *
                            (FP16_BF16_DTYPE_SIZE * CONST_2 + FP32_DTYPE_SIZE + FP32_DTYPE_SIZE * CONST_2);
    int64_t ioUbSize = coreCalcNum * totalSeq1Size;
    if (gatherOffsetSize >= ubSize_) {
        OP_CHECK_IF(TilingSplitNMixed(numHeads, headDimAlign, ubSize_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "TilingSplitNMixed fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (ubSize_ < gatherOffsetSize || bufferSize < totalSeq1Size) {
        OP_CHECK_IF(TilingSplitBMixed(batchSize, numHeads, headDimAlign, ubSize_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "TilingSplitBMixed fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    context_->SetTilingKey(tilingKey_);
    CalcUbCalcSMixed(coreCalcNum, coreCalcTail, totalSeq1Size, bufferSize, ioUbSize);
    return ge::GRAPH_SUCCESS;
}

void InplacePartialRotaryMulTiling::CalcUbCalcSMixed(int64_t coreCalcNum, int64_t coreCalcTail, int64_t totalSeq1Size,
                                                     int64_t bufferSize, int64_t ioUbSize)
{
    int64_t ubCalcNum;
    int64_t ubCalcLoop;
    int64_t ubCalcTail;
    if (bufferSize < ioUbSize) {
        ubCalcNum = GetDiv(bufferSize, totalSeq1Size);
        ubCalcLoop = GetCeilInt(coreCalcNum, ubCalcNum);
        ubCalcTail = GetDivRem(coreCalcNum, ubCalcNum) != 0 ? coreCalcNum - (ubCalcLoop - 1) * ubCalcNum : 0;
    } else {
        ubCalcNum = coreCalcNum;
        ubCalcLoop = 1;
        ubCalcTail = 0;
    }
    tilingData_.set_ubCalcNum(ubCalcNum);
    tilingData_.set_ubCalcLoop(ubCalcLoop);
    tilingData_.set_ubCalcTail(ubCalcTail);

    int64_t ubCalcTailNum{0};
    int64_t ubCalcTailLoop{0};
    int64_t ubCalcTailTail{0};
    if (coreCalcTail != 0) {
        ioUbSize = coreCalcTail * totalSeq1Size;
        if (bufferSize < ioUbSize) {
            ubCalcTailNum = GetDiv(bufferSize, totalSeq1Size);
            ubCalcTailLoop = GetCeilInt(coreCalcTail, ubCalcTailNum);
            ubCalcTailTail =
                GetDivRem(coreCalcTail, ubCalcTailNum) != 0 ? coreCalcTail - (ubCalcTailLoop - 1) * ubCalcTailNum : 0;
        } else {
            ubCalcTailNum = coreCalcTail;
            ubCalcTailLoop = 1;
            ubCalcTailTail = 0;
        }
    }
    tilingData_.set_ubCalcTailNum(ubCalcTailNum);
    tilingData_.set_ubCalcTailLoop(ubCalcTailLoop);
    tilingData_.set_ubCalcTailTail(ubCalcTailTail);
}

ge::graphStatus InplacePartialRotaryMulTiling::TilingSplitMixed()
{
    int64_t batchSizeOut{1};
    int64_t seqLenOut{1};
    int64_t numHeadsOut{1};
    OP_CHECK_IF(InitSplitTilingData(batchSizeOut, seqLenOut, numHeadsOut) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "InitSplitTilingData fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(TilingSplitSMixed() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingSplitSMixed fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InplacePartialRotaryMulTiling::DoTiling()
{
    OP_LOGI(context_->GetNodeName(), "Enter InplacePartialRotaryMulTiling DoTiling");
    OP_CHECK_IF(CheckInput() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInput failed"),
                return ge::GRAPH_FAILED);
    // No-op: empty slice (sliceStart==sliceEnd), nothing to compute
    if (headDim_ == 0) {
        tilingKey_ = TILING_KEY_NOOP;
        FillTilingData();
        context_->SetBlockDim(1);
        context_->SetTilingKey(tilingKey_);
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        workspaces[0] = static_cast<size_t>(16 * 1024 * 1024);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        OP_LOGI(context_->GetNodeName(), "InplacePartialRotaryMulTiling no-op: sliceStart==sliceEnd, return directly");
        return ge::GRAPH_SUCCESS;
    }
    if (isMixedPrecision_) {
        return DoTilingMixedPrecision();
    }
    return DoTilingRegular();
}

ge::graphStatus InplacePartialRotaryMulTiling::DoTilingMixedPrecision()
{
    tilingKey_ = BASE_KEY + GetMixedTilingKeyOffset();
    tilingData_.set_allHeadDim(allHeadDim_);
    tilingData_.set_start(start_);
    OP_CHECK_IF(TilingSplitMixed() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingSplitMixed fail."),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "[mixed tilingKey]: %ld", tilingKey_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = static_cast<size_t>(16) * 1024 * 1024;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;
    PrintInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulTiling::DoTilingRegular()
{
    if (CalTilingData() == ge::GRAPH_SUCCESS) {
        FillTilingData();
        PrintTilingData();
        context_->SetBlockDim(usedCoreNum_);
        context_->SetTilingKey(tilingKey_);

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        workspaces[0] = static_cast<size_t>(16 * 1024 * 1024);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    } else {
        // 开始走原始的71的逻辑
        tilingKey_ = BASE_KEY;
        if (xDtype_ == ge::DT_FLOAT16) {
            tilingKey_ += TILING_KEY_SPLIT_S;
        } else if (xDtype_ == ge::DT_BF16) {
            tilingKey_ += TILING_KEY_BFLOAT16;
        } else if (xDtype_ == ge::DT_FLOAT) {
            tilingKey_ += TILING_KEY_FLOAT32;
            dtypeSize_ = FP32_DTYPE_SIZE;
        }
        tilingData_.set_allHeadDim(allHeadDim_);
        tilingData_.set_start(start_);
        OP_CHECK_IF(TilingSplit() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingSplit fail."),
                    return ge::GRAPH_FAILED);

        OP_LOGI(context_->GetNodeName(), "[tilingKey]: %ld", tilingKey_);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        size_t usrWorkspaceSize = 0;
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
        currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

        PrintInfo();
        return ge::GRAPH_SUCCESS;
    }
};
ge::graphStatus Tiling4InplacePartialRotaryMul(gert::TilingContext *context)
{
    InplacePartialRotaryMulTiling tilingImpl = InplacePartialRotaryMulTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4InplacePartialRotaryMul init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4InplacePartialRotaryMul do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "end Tiling4InplacePartialRotaryMul");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

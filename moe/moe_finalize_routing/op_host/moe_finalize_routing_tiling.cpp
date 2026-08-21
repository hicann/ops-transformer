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
 * \file moe_finalize_routing_tiling.cc
 * \brief
 */
#include "moe_finalize_routing_tiling.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {

static const int64_t WORKSPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024);
static const int64_t ONE_BLK_SIZE = 32;
static const size_t TIMES = 2;
static const size_t TRIPLE_BUFFER = 3; // 当数据类型为BF16时转为float类型计算，部分输入所需内存为原来的3倍
static const size_t DOUBLE_BUFFER = 2;       // 启动double buffer
static const int64_t H_SIZE_PER_SLICE = 256; // 切H后，每一块的H大小
static const int64_t INT_NUM_OF_BYTES = 4;
static const int64_t UNROLL_TIMES_WITH_K2 = 2;
static const int64_t UNROLL_TIMES_WITH_K4 = 4;
static const int64_t BOUND_K = 256;
static const int64_t ALIGNED_H = 512;
static const int64_t NETWORKSIZE2 = 5120;

static const size_t INDEX_IN_EXPAND_PERMUTED_ROWS = 0;
static const size_t INDEX_IN_SKIP1 = 1;
static const size_t INDEX_IN_SKIP2 = 2;
static const size_t INDEX_IN_BIAS = 3;
static const size_t INDEX_IN_SCALES = 4;
static const size_t INDEX_IN_EXPANDED_SRC_TO_DST_ROW = 5;
static const size_t INDEX_IN_EXPERT_FOR_SOURCE_ROW = 6;
static const size_t INPUT_NUM = 7;
static const size_t SHAPE_SIZE = 2;
static const int64_t STRIDE_BLOCK_LIMIT = static_cast<int64_t>(numeric_limits<uint32_t>::max() / sizeof(int32_t));

inline static int64_t AlignParam(const int64_t param, const int32_t typeSize)
{
    if (typeSize == 0) {
        return 0;
    }
    return ((param * typeSize + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE) / typeSize;
}

ge::graphStatus MoeFinalizeRoutingTiling::Init(gert::TilingContext *context)
{
    context_ = context;
    OP_CHECK_IF(CheckParamsShape() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeFinalizeRoutingTiling check shape fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetPlatformInfo() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeFinalizeRoutingTiling get platform info fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetParamInfo() != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeFinalizeRoutingTiling get input param info fail."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTiling::SetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTiling::SetParamInfo()
{
    // 获取bias的Shape
    auto biasInput = context_->GetInputShape(INDEX_IN_BIAS - offset_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasInput);
    auto biasInputShape = biasInput->GetStorageShape();
    biasRowNum_ = biasInputShape.GetDim(0);

    // 获取scales的Shape
    auto scalesInput = context_->GetInputShape(INDEX_IN_SCALES - offset_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scalesInput);
    auto scalesInputShape = scalesInput->GetStorageShape();
    k_ = scalesInputShape.GetDim(1);

    // 获取skip1的Shape
    auto skip1Input = context_->GetInputShape(INDEX_IN_SKIP1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, skip1Input);
    auto skip1InputShape = skip1Input->GetStorageShape();
    totalRowNum_ = skip1InputShape.GetDim(0);
    h_ = skip1InputShape.GetDim(1);

    // 获取skip1的输入数据类型
    auto skip1InputDesc = context_->GetInputDesc(INDEX_IN_SKIP1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, skip1InputDesc);
    dataType_ = skip1InputDesc->GetDataType();
    inputDataTypeSize_ = ge::GetSizeByDataType(dataType_);

    // skip2是否为空，为空时不需要buffer分配
    auto skip2Input = context_->GetOptionalInputShape(INDEX_IN_SKIP2);
    skip2IsNull_ = skip2Input == nullptr || skip2Input->GetStorageShape().GetShapeSize() == 0 ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTiling::CheckParamsShape()
{
    auto expandedXShapePtr = context_->GetInputShape(INDEX_IN_EXPAND_PERMUTED_ROWS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShapePtr);
    auto expandedXShape = expandedXShapePtr->GetStorageShape();

    auto x1ShapePtr = context_->GetInputShape(INDEX_IN_SKIP1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1ShapePtr);
    auto x1Shape = x1ShapePtr->GetStorageShape();

    auto x2Shape = context_->GetOptionalInputShape(INDEX_IN_SKIP2);
    offset_ = (context_->GetComputeNodeInputNum() == INPUT_NUM) ? 0 : 1;
    auto biasShapePtr = context_->GetInputShape(INDEX_IN_BIAS - offset_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasShapePtr);
    auto biasShape = biasShapePtr->GetStorageShape();

    auto scalesShapePtr = context_->GetInputShape(INDEX_IN_SCALES - offset_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scalesShapePtr);
    auto scalesShape = scalesShapePtr->GetStorageShape();

    auto expandedRowIdxShapePtr = context_->GetInputShape(INDEX_IN_EXPANDED_SRC_TO_DST_ROW - offset_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr);
    auto expandedRowIdxShape = expandedRowIdxShapePtr->GetStorageShape();

    auto expandedExpertIdxShapePtr = context_->GetInputShape(INDEX_IN_EXPERT_FOR_SOURCE_ROW - offset_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedExpertIdxShapePtr);
    auto expandedExpertIdxShape = expandedExpertIdxShapePtr->GetStorageShape();

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    // check dim num
    OP_CHECK_IF(expandedXShape.GetDimNum() != SHAPE_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The expanded_x of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(x1Shape.GetDimNum() != SHAPE_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The x1 of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(x2Shape != nullptr && x2Shape->GetStorageShape().GetDimNum() != SHAPE_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The x2 of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(biasShape.GetDimNum() != SHAPE_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The bias of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(scalesShape.GetDimNum() != SHAPE_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The scales of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        expandedRowIdxShape.GetDimNum() != 1,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The expanded_row_idx of input should be 1D tensor."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        expandedExpertIdxShape.GetDimNum() != SHAPE_SIZE,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The expanded_expert_idx of input should be 2D tensor."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(yShape.GetDimNum() != SHAPE_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The y should be 2D tensor."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(x1Shape.GetDim(0) != scalesShape.GetDim(0) || x1Shape.GetDim(0) != expandedExpertIdxShape.GetDim(0) ||
                    x1Shape.GetDim(0) != yShape.GetDim(0) ||
                    (x2Shape != nullptr && x1Shape.GetDim(0) != x2Shape->GetStorageShape().GetDim(0)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The dim 0 of x1, x2, scales,"
                                                                     " expanded_expert_idx and out should be same."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(x1Shape.GetDim(1) != expandedXShape.GetDim(1) || x1Shape.GetDim(1) != biasShape.GetDim(1) ||
                    x1Shape.GetDim(1) != yShape.GetDim(1) ||
                    (x2Shape != nullptr && x1Shape.GetDim(1) != x2Shape->GetStorageShape().GetDim(1)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The dim 1 of expanded_x,"
                                                                     " x1, x2, bias and out should be same."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(scalesShape.GetDim(1) != expandedExpertIdxShape.GetDim(1),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "the dim 1 of scales and expanded_expert_idx should be same."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(expandedRowIdxShape.GetDim(0) != expandedXShape.GetDim(0),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "The dim 0 of expanded_x and"
                                                                     " expanded_row_idx should be same."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(biasShape.GetDim(0) == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "E can not be 0."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((scalesShape.GetDim(1) == 0) || (expandedExpertIdxShape.GetDim(1) == 0),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "K can not be 0."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingTiling::GetTilingData(MoeFinalizeRoutingTilingData &tilingData) const
{
    tilingData.set_totalCoreNum(totalCoreNum_);
    tilingData.set_usedCoreNum(usedCoreNum_);
    tilingData.set_skip2IsNull(skip2IsNull_);
    tilingData.set_biasRowNum(biasRowNum_);
    tilingData.set_totalRowNum(totalRowNum_);
    tilingData.set_H(h_);
    tilingData.set_normalH(normalH_);
    tilingData.set_unnormalH(unnormalH_);
    tilingData.set_hSliceNum(hSliceNum_);
    tilingData.set_normalK(normalK_);
    tilingData.set_unnormalK(unnormalK_);
    tilingData.set_kSliceNum(kSliceNum_);
    tilingData.set_K(k_);
    tilingData.set_normalCoreHandleNum(normalCoreHandleNum_);
    tilingData.set_normalCoreLoopNum(normalCoreLoopNum_);
    tilingData.set_normalCoreHandleNumPerLoop(normalCoreHandleNumPerLoop_);
    tilingData.set_normalCoreHandleNumTailLoop(normalCoreHandleNumTailLoop_);
    tilingData.set_tailCoreHandleNum(tailCoreHandleNum_);
    tilingData.set_tailCoreLoopNum(tailCoreLoopNum_);
    tilingData.set_tailCoreHandleNumPerLoop(tailCoreHandleNumPerLoop_);
    tilingData.set_tailCoreHandleNumTailLoop(tailCoreHandleNumTailLoop_);
}

void MoeFinalizeRoutingTiling::CutH()
{
    // int32类型，k按照32字节对齐后的列数
    int64_t alignIntK = AlignParam(k_, INT_NUM_OF_BYTES);
    // 输入数据类型，k按照32字节对齐后的列数
    int64_t alignK = AlignParam(k_, inputDataTypeSize_);
    // 如果K大于256，直接从gm中取值，不需要占用buffer
    if (k_ > BOUND_K) {
        alignIntK = BOUND_K;
        alignK = BOUND_K;
        normalK_ = BOUND_K;
        kSliceNum_ = (k_ + normalK_ - 1) / normalK_;
        unnormalK_ = k_ - k_ / normalK_ * normalK_;
        unnormalK_ = unnormalK_ == 0 ? normalK_ : unnormalK_;
    }
    int64_t skip2IsNeedBuffer = skip2IsNull_ == 1 ? 0 : 1;
    int64_t unrollTimes = UNROLL_TIMES_WITH_K2;
    if (k_ == UNROLL_TIMES_WITH_K4) {
        unrollTimes = UNROLL_TIMES_WITH_K4;
    }
    if (dataType_ != ge::DT_BF16) {
        normalH_ = (ubSize_ - alignIntK * INT_NUM_OF_BYTES - alignK * inputDataTypeSize_) /
                   (1 + skip2IsNeedBuffer + 1 + unrollTimes + unrollTimes + 1) / ONE_BLK_SIZE * ONE_BLK_SIZE /
                   inputDataTypeSize_;
    } else {
        normalH_ = (ubSize_ - alignIntK * INT_NUM_OF_BYTES - alignK * static_cast<int64_t>(inputDataTypeSize_)) /
                   (static_cast<int64_t>(TRIPLE_BUFFER) + skip2IsNeedBuffer * static_cast<int64_t>(TRIPLE_BUFFER) + 1 +
                    (unrollTimes + unrollTimes) * static_cast<int64_t>(TRIPLE_BUFFER)) /
                   ONE_BLK_SIZE * ONE_BLK_SIZE / static_cast<int64_t>(inputDataTypeSize_);
    }
    // H切分后，切了多少次
    hSliceNum_ = (h_ + normalH_ - 1) / normalH_ - 1;
    // H切分后，最后一个块的列数
    unnormalH_ = h_ - h_ / normalH_ * normalH_;
    unnormalH_ = unnormalH_ == 0 ? normalH_ : unnormalH_;
    normalCoreLoopNum_ = normalCoreHandleNum_ * (hSliceNum_ + 1);

    // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数 (每次loop处理的行数，尽量用满inputUbSize)
    normalCoreHandleNumPerLoop_ = 1;
    // 非尾核，每个核，尾Loop需要处理的skip1行数
    normalCoreHandleNumTailLoop_ = 1;
    // 尾核，非尾Loop需要处理的skip1行数
    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    // 尾核，非尾loop需要的循环次数
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ? 0 : tailCoreHandleNum_ * (hSliceNum_ + 1);
    // 尾核，尾Loop需要处理的skip1行数
    tailCoreHandleNumTailLoop_ = tailCoreHandleNumPerLoop_;
}

ge::graphStatus MoeFinalizeRoutingTiling::LoadHKAndCalcTiling()
{
    // int32类型，k按照32字节对齐后的列数
    int64_t alignIntK = AlignParam(k_, INT_NUM_OF_BYTES);
    // 输入数据类型，k按照32字节对齐后的列数
    int64_t alignK = AlignParam(k_, inputDataTypeSize_);

    // 输入数据类型，h按照32字节对齐后的列数
    int64_t alignH = AlignParam(h_, inputDataTypeSize_);
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1) ? 0 : 1;
    // bias和expandedPermutedRows所占的内存大小总和
    int64_t biasAndPermutedRowsBytesSum =
        (alignH + alignH) * static_cast<int64_t>(inputDataTypeSize_) * static_cast<int64_t>(DOUBLE_BUFFER);
    // skip1,skip2,out在单行的情况下所占的内存大小总和
    int64_t skip1Skip2OutBytesSumPerRow = (alignH + alignH * skip2IsNeedBuffer + alignH + alignH) * inputDataTypeSize_;
    if (dataType_ == ge::DT_BF16) {
        biasAndPermutedRowsBytesSum *= static_cast<int64_t>(TRIPLE_BUFFER);
        skip1Skip2OutBytesSumPerRow = (alignH * static_cast<int64_t>(TRIPLE_BUFFER) +
                                       alignH * skip2IsNeedBuffer * static_cast<int64_t>(TRIPLE_BUFFER) + alignH) *
                                      static_cast<int64_t>(inputDataTypeSize_);
    }

    // 每个loop可以处理的行数，尽可能将内存占满
    int64_t normalCoreHandleNumPerLoop = static_cast<int64_t>(-1);
    // 如果K大于256，将对K进行切分，最大使用256个数的buffer
    if (k_ > BOUND_K) {
        alignIntK = BOUND_K;
        alignK = BOUND_K;
        normalK_ = BOUND_K;
        kSliceNum_ = (k_ + normalK_ - 1) / normalK_;
        unnormalK_ = k_ - k_ / normalK_ * normalK_;
        unnormalK_ = unnormalK_ == 0 ? normalK_ : unnormalK_;
        normalCoreHandleNumPerLoop =
            (ubSize_ - biasAndPermutedRowsBytesSum - alignIntK * INT_NUM_OF_BYTES - alignK * inputDataTypeSize_) /
            skip1Skip2OutBytesSumPerRow;
    } else {
        normalCoreHandleNumPerLoop =
            (ubSize_ - biasAndPermutedRowsBytesSum) /
            (alignIntK * INT_NUM_OF_BYTES + alignK * inputDataTypeSize_ + skip1Skip2OutBytesSumPerRow);
    }
    if (normalCoreHandleNumPerLoop <= 0) {
        return ge::GRAPH_FAILED; // 计算失败，需要切分H
    }
    // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数 (每次loop处理的行数，尽量用满inputUbSize)
    normalCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop, normalCoreHandleNum_);
    // 非尾核，每个核的loop次数
    if (normalCoreHandleNumPerLoop_ == 0) {
        return ge::GRAPH_FAILED; // 计算失败，需要切分H
    }
    normalCoreLoopNum_ = (normalCoreHandleNum_ + normalCoreHandleNumPerLoop_ - 1) / normalCoreHandleNumPerLoop_;
    // 非尾核，每个核，尾Loop需要处理的skip1行数
    normalCoreHandleNumTailLoop_ = normalCoreHandleNum_ - (normalCoreLoopNum_ - 1) * normalCoreHandleNumPerLoop_;
    // 尾核，非尾Loop需要处理的skip1行数
    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    // 尾核，非尾loop需要的循环次数
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ?
                           0 :
                           (tailCoreHandleNum_ + tailCoreHandleNumPerLoop_ - 1) / tailCoreHandleNumPerLoop_;
    // 尾核，尾Loop需要处理的skip1行数
    tailCoreHandleNumTailLoop_ = tailCoreHandleNum_ - (tailCoreLoopNum_ - 1) * tailCoreHandleNumPerLoop_;
    normalH_ = h_;
    unnormalH_ = h_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTiling::LoadBiasAndCalcTiling()
{
    if (k_ > BOUND_K || totalRowNum_ > STRIDE_BLOCK_LIMIT || normalCoreHandleNum_ > STRIDE_BLOCK_LIMIT) {
        return ge::GRAPH_FAILED;
    }
    // int32类型，k按照32字节对齐后的列数
    int64_t alignIntK = AlignParam(k_, INT_NUM_OF_BYTES);
    // 输入数据类型，k按照32字节对齐后的列数
    int64_t alignK = AlignParam(k_, inputDataTypeSize_);

    // 输入数据类型，h按照32字节对齐后的列数
    int64_t alignH = AlignParam(h_, inputDataTypeSize_);
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1) ? 0 : 1;
    // bias和expandedPermutedRows,tmpbuff所占的内存大小总和
    int64_t biasAndPermutedRowsBytesSum =
        (alignH + alignH) * static_cast<int64_t>(inputDataTypeSize_) * static_cast<int64_t>(DOUBLE_BUFFER) +
        alignH * static_cast<int64_t>(inputDataTypeSize_);
    // skip1,skip2,out在单行的情况下所占的内存大小总和
    int64_t skip1Skip2OutBytesSumPerRow = (alignH + alignH * skip2IsNeedBuffer + alignH) * inputDataTypeSize_;

    int64_t expandedSrcToDstRowSum = AlignParam(normalCoreHandleNum_, INT_NUM_OF_BYTES) * k_ * INT_NUM_OF_BYTES;

    if (dataType_ == ge::DT_BF16) {
        biasAndPermutedRowsBytesSum = (alignH + alignH) * static_cast<int64_t>(inputDataTypeSize_) *
                                      static_cast<int64_t>(DOUBLE_BUFFER) * static_cast<int64_t>(TRIPLE_BUFFER);
        skip1Skip2OutBytesSumPerRow = (alignH * static_cast<int64_t>(TRIPLE_BUFFER) +
                                       alignH * skip2IsNeedBuffer * static_cast<int64_t>(TRIPLE_BUFFER) + alignH) *
                                      static_cast<int64_t>(inputDataTypeSize_);
    }

    int64_t normalCoreHandleNumPerLoop =
        (ubSize_ - biasAndPermutedRowsBytesSum - expandedSrcToDstRowSum) /
        (skip1Skip2OutBytesSumPerRow + alignIntK * INT_NUM_OF_BYTES + alignK * inputDataTypeSize_);

    if (normalCoreHandleNumPerLoop <= 0) {
        isCanLoadAllBias_ = false;
        return ge::GRAPH_FAILED;
    }
    // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数 (每次loop处理的行数，尽量用满inputUbSize)
    normalCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop, normalCoreHandleNum_);
    // 非尾核，每个核的loop次数
    if (normalCoreHandleNumPerLoop_ == 0) {
        return ge::GRAPH_FAILED;
    }
    normalCoreLoopNum_ = (normalCoreHandleNum_ + normalCoreHandleNumPerLoop_ - 1) / normalCoreHandleNumPerLoop_;
    // 非尾核，每个核，尾Loop需要处理的skip1行数
    normalCoreHandleNumTailLoop_ = normalCoreHandleNum_ - (normalCoreLoopNum_ - 1) * normalCoreHandleNumPerLoop_;
    // 尾核，非尾Loop需要处理的skip1行数
    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    // 尾核，非尾loop需要的循环次数
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ?
                           0 :
                           (tailCoreHandleNum_ + tailCoreHandleNumPerLoop_ - 1) / tailCoreHandleNumPerLoop_;
    // 尾核，尾Loop需要处理的skip1行数
    tailCoreHandleNumTailLoop_ = tailCoreHandleNum_ - (tailCoreLoopNum_ - 1) * tailCoreHandleNumPerLoop_;
    normalH_ = h_;
    unnormalH_ = h_;
    isCanLoadAllBias_ = true;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MoeFinalizeRoutingTiling::OptimizedCutH()
{
    // 对H切分一半并且按照512字节对齐
    normalH_ = (h_ * static_cast<int64_t>(inputDataTypeSize_) / static_cast<int64_t>(TIMES) + ALIGNED_H - 1) /
               ALIGNED_H * ALIGNED_H / static_cast<int64_t>(inputDataTypeSize_);
    hSliceNum_ = (h_ + normalH_ - 1) / normalH_ - 1;

    unnormalH_ = h_ - h_ / normalH_ * normalH_;
    unnormalH_ = unnormalH_ == 0 ? normalH_ : unnormalH_;

    int64_t alignIntK = AlignParam(k_, INT_NUM_OF_BYTES);
    int64_t alignK = AlignParam(k_, inputDataTypeSize_);

    int64_t alignH = normalH_;
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1) ? 0 : 1;

    int64_t biasAndPermutedRowsBytesSum =
        (alignH + alignH) * inputDataTypeSize_ * UNROLL_TIMES_WITH_K2 + alignH * inputDataTypeSize_;

    int64_t skip1Skip2OutBytesSumPerRow = (alignH + alignH * skip2IsNeedBuffer + alignH) * inputDataTypeSize_;

    int64_t expandedSrcToDstRowSum = AlignParam(normalCoreHandleNum_, INT_NUM_OF_BYTES) * k_ * INT_NUM_OF_BYTES;

    int64_t normalCoreHandleNumPerLoop =
        (ubSize_ - biasAndPermutedRowsBytesSum - expandedSrcToDstRowSum) /
        (skip1Skip2OutBytesSumPerRow * static_cast<int64_t>(DOUBLE_BUFFER) +
         alignIntK * INT_NUM_OF_BYTES * static_cast<int64_t>(DOUBLE_BUFFER) +
         alignK * static_cast<int64_t>(inputDataTypeSize_) * static_cast<int64_t>(DOUBLE_BUFFER));

    if (normalCoreHandleNumPerLoop <= 0) {
        isOptimizedCutH_ = false;
        return ge::GRAPH_FAILED;
    }

    normalCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop, normalCoreHandleNum_);
    if (normalCoreHandleNumPerLoop_ == 0) {
        return ge::GRAPH_FAILED;
    }
    normalCoreLoopNum_ = (normalCoreHandleNum_ + normalCoreHandleNumPerLoop_ - 1) / normalCoreHandleNumPerLoop_;

    normalCoreHandleNumTailLoop_ = normalCoreHandleNum_ - (normalCoreLoopNum_ - 1) * normalCoreHandleNumPerLoop_;

    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ?
                           0 :
                           (tailCoreHandleNum_ + tailCoreHandleNumPerLoop_ - 1) / tailCoreHandleNumPerLoop_;
    tailCoreHandleNumTailLoop_ = tailCoreHandleNum_ - (tailCoreLoopNum_ - 1) * tailCoreHandleNumPerLoop_;

    isOptimizedCutH_ = true;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MoeFinalizeRoutingTiling::CalcTilingData()
{
    // 非尾核，每个核处理的skip1行数
    normalCoreHandleNum_ = (totalRowNum_ + totalCoreNum_ - 1) / totalCoreNum_;
    // 使用的核数
    usedCoreNum_ = min((totalRowNum_ + normalCoreHandleNum_ - 1) / normalCoreHandleNum_, totalCoreNum_);
    // 尾核, 需要处理的skip1行数
    tailCoreHandleNum_ = totalRowNum_ - (usedCoreNum_ - 1) * normalCoreHandleNum_;
    if ((h_ == NETWORKSIZE2) && (dataType_ == ge::DT_FLOAT) && (k_ % TIMES == 0)) {
        if (OptimizedCutH() == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }

    if (LoadBiasAndCalcTiling() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    // 开启double buffer并全载H和K，计算tilingData
    if (LoadHKAndCalcTiling() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGI("Tiling4MoeFinalizeRouting", "CalcTilingData load all h failed, will cut h.");
    // 切分H并计算tilingData
    isCanLoadH_ = false;
    CutH();
    return ge::GRAPH_SUCCESS;
}
int64_t MoeFinalizeRoutingTiling::GetAllBiasTilingKey() const
{
    if (dataType_ == ge::DT_FLOAT) {
        return DTYPE_FLOAT_DB_ALL_BIAS;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        return DTYPE_FLOAT16_DB_ALL_BIAS;
    }
    return DTYPE_BF16_ALL_BIAS;
}

int64_t MoeFinalizeRoutingTiling::GetTilingKey() const
{
    if (isOptimizedCutH_) {
        return DTYPE_FLOAT_CUTH_NETWORK;
    }
    if (isCanLoadAllBias_) {
        return GetAllBiasTilingKey();
    }
    if (k_ > BOUND_K) {
        if (dataType_ == ge::DT_FLOAT) {
            return DTYPE_FLOAT_BIG_K;
        }
        if (dataType_ == ge::DT_FLOAT16) {
            return DTYPE_FLOAT16_BIG_K;
        }
        return DTYPE_BF16_BIG_K;
    }
    if (isCanLoadH_) {
        if (dataType_ == ge::DT_FLOAT) {
            return DTYPE_FLOAT_DB;
        }
        if (dataType_ == ge::DT_FLOAT16) {
            return DTYPE_FLOAT16_DB;
        }
        return DTYPE_BF16;
    }
    if (k_ == UNROLL_TIMES_WITH_K2) {
        if (dataType_ == ge::DT_FLOAT) {
            return DTYPE_FLOAT_CUTH_K2;
        }
        if (dataType_ == ge::DT_FLOAT16) {
            return DTYPE_FLOAT16_CUTH_K2;
        }
        return DTYPE_BF16_CUTH_K2;
    } else if (k_ == UNROLL_TIMES_WITH_K4) {
        if (dataType_ == ge::DT_FLOAT) {
            return DTYPE_FLOAT_CUTH_K4;
        }
        if (dataType_ == ge::DT_FLOAT16) {
            return DTYPE_FLOAT16_CUTH_K4;
        }
        return DTYPE_BF16_CUTH_K4;
    }
    if (dataType_ == ge::DT_FLOAT) {
        return DTYPE_FLOAT_CUTH;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        return DTYPE_FLOAT16_CUTH;
    }
    return DTYPE_BF16_CUTH;
}

inline static ge::graphStatus MoeFinalizeRoutingSetTilingData(gert::TilingContext *context,
                                                              MoeFinalizeRoutingTilingData &tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(MoeFinalizeRoutingTilingData &tilingData)
{
    OP_LOGI("MoeFinalizeRouting", "totalCoreNum: %ld", tilingData.get_totalCoreNum());
    OP_LOGI("MoeFinalizeRouting", "usedCoreNum: %ld", tilingData.get_usedCoreNum());
    OP_LOGI("MoeFinalizeRouting", "skip2IsNull: %ld", tilingData.get_skip2IsNull());
    OP_LOGI("MoeFinalizeRouting", "biasRowNum: %ld", tilingData.get_biasRowNum());
    OP_LOGI("MoeFinalizeRouting", "totalRowNum: %ld", tilingData.get_totalRowNum());
    OP_LOGI("MoeFinalizeRouting", "H: %ld", tilingData.get_H());
    OP_LOGI("MoeFinalizeRouting", "normalH: %ld", tilingData.get_normalH());
    OP_LOGI("MoeFinalizeRouting", "unnormalH: %ld", tilingData.get_unnormalH());
    OP_LOGI("MoeFinalizeRouting", "hSliceNum: %ld", tilingData.get_hSliceNum());
    OP_LOGI("MoeFinalizeRouting", "normalK: %ld", tilingData.get_normalK());
    OP_LOGI("MoeFinalizeRouting", "unnormalK: %ld", tilingData.get_unnormalK());
    OP_LOGI("MoeFinalizeRouting", "kSliceNum: %ld", tilingData.get_kSliceNum());
    OP_LOGI("MoeFinalizeRouting", "K: %ld", tilingData.get_K());
    OP_LOGI("MoeFinalizeRouting", "normalCoreHandleNum: %ld", tilingData.get_normalCoreHandleNum());
    OP_LOGI("MoeFinalizeRouting", "normalCoreLoopNum: %ld", tilingData.get_normalCoreLoopNum());
    OP_LOGI("MoeFinalizeRouting", "normalCoreHandleNumPerLoop: %ld", tilingData.get_normalCoreHandleNumPerLoop());
    OP_LOGI("MoeFinalizeRouting", "normalCoreHandleNumTailLoop: %ld", tilingData.get_normalCoreHandleNumTailLoop());
    OP_LOGI("MoeFinalizeRouting", "tailCoreHandleNum: %ld", tilingData.get_tailCoreHandleNum());
    OP_LOGI("MoeFinalizeRouting", "tailCoreLoopNum: %ld", tilingData.get_tailCoreLoopNum());
    OP_LOGI("MoeFinalizeRouting", "tailCoreHandleNumPerLoop: %ld", tilingData.get_tailCoreHandleNumPerLoop());
    OP_LOGI("MoeFinalizeRouting", "tailCoreHandleNumTailLoop: %ld", tilingData.get_tailCoreHandleNumTailLoop());
}

ge::graphStatus Tiling4MoeFinalizeRouting(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "[MoeFinalizeRouting] Tiling4MoeFinalizeRouting running begin");
    MoeFinalizeRoutingTiling tilingOp;
    OP_CHECK_IF(tilingOp.Init(context) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeFinalizeRoutingTiling init fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingOp.CalcTilingData() != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeFinalizeRoutingTiling calc tilingData fail."),
                return ge::GRAPH_FAILED);
    MoeFinalizeRoutingTilingData tilingData;
    tilingOp.GetTilingData(tilingData);
    OP_CHECK_IF(
        MoeFinalizeRoutingSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "MoeFinalizeRoutingSetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingOp.GetTilingKey());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMoeFinalizeRouting(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeFinalizeRouting)
    .Tiling(Tiling4MoeFinalizeRouting)
    .TilingParse<MoeFinalizeRoutingCompileInfo>(TilingPrepareForMoeFinalizeRouting);

} // namespace optiling

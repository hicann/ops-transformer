/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_half_grad.cc
 * \brief
 */
#include "rope_half_grad_tiling.h"
#include "rotary_position_embedding_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace {
constexpr uint64_t INDEX_GRAD = 0;
constexpr uint64_t INDEX_COS = 1;
constexpr uint64_t INDEX_SIN = 2;
constexpr uint64_t INDEX_X = 3;
constexpr uint64_t DIM_ZERO = 0;
constexpr uint64_t DIM_ONE = 1;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t DIM_THREE = 3;
constexpr uint64_t DIM_FOUR = 4;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DTYPE_SIZE_FP16 = 2;
constexpr int64_t DTYPE_SIZE_FP32 = 4;
constexpr int64_t RESERVE_UB_SIZE = 20 * 1024;
constexpr int64_t IN_OUT_TENSOR_NUM = 7;
constexpr int64_t TENSOR_NUM = 5;
constexpr int64_t TENSOR_CONV_NUM = 9;
constexpr int64_t IN_OUT_TENSOR_SINGLE_NUM = 4;
constexpr int64_t TENSOR_SINGLE_NUM = 3;
constexpr int64_t TENSOR_CONV_SINGLE_NUM = 7;
constexpr uint64_t MASK_NUM_FP32 = 64;
constexpr uint64_t CHUNK_SIZE = 2;
constexpr int64_t D_THRESHOLD = 896;

constexpr uint64_t DTYPE_KEY_FP16 = 0;
constexpr uint64_t DTYPE_KEY_FP32 = 1;
constexpr uint64_t DTYPE_KEY_BF16 = 2;
constexpr uint64_t LAYOUT_KEY_UNDEFINED = -100;
constexpr uint64_t LAYOUT_KEY_BSND = 0;
constexpr uint64_t LAYOUT_KEY_BNSD = 10;
constexpr uint64_t LAYOUT_KEY_SBND = 20;
constexpr uint64_t SHAPE_KEY_ALIGN = 0;
constexpr uint64_t SHAPE_KEY_PAD = 100;
constexpr uint64_t NEED_BACKWARD_KEY_TRUE = 0;
constexpr uint64_t NEED_BACKWARD_KEY_FALSE = 1000;

optiling::RotaryPositionEmbeddingGradTilingData tiling;

int64_t GetCeilInt(int64_t value1, int64_t value2)
{
    OP_CHECK_IF(
        value2 == 0, OP_LOGE("RopeHalfGrad", "In the GetCeilInt function, the divisor is 0"),
        return value1);
    return static_cast<int64_t>((value1 + value2 - 1) / value2);
}

int64_t GetGcdForRotary(int64_t value1, int64_t value2)
{
    OP_CHECK_IF(
        value2 == 0,
        OP_LOGE("RopeHalfGrad", "In the GetGcdForRotary function, the divisor is 0"),
        return value1);
    int64_t remainder = value1 % value2;
    if (remainder == 0) {
        return value2;
    } else {
        return GetGcdForRotary(value2, remainder);
    }
}

int64_t GetLcmForRotary(int64_t value1, int64_t value2)
{
    int64_t temp = value1 * value2;
    OP_CHECK_IF(
        value2 == 0,
        OP_LOGE("RopeHalfGrad", "In the GetLcmForRotary function, the divisor is 0"),
        return temp);
    temp = temp / GetGcdForRotary(value1, value2);
    return temp;
}

int64_t GetDtypeSize(const gert::TilingContext* context)
{
    int64_t dtypeSize = DTYPE_SIZE_FP16;
    auto indexGrad = context->GetInputDesc(INDEX_GRAD);
    OP_CHECK_IF(
        indexGrad == nullptr,
        OP_LOGE(context->GetNodeName(), "[GetDtypeSize] indexGrad is null."),
        return ge::GRAPH_FAILED);
    auto inputDtype = indexGrad->GetDataType();
    if (inputDtype == ge::DT_FLOAT) {
        dtypeSize = DTYPE_SIZE_FP32;
    }
    return dtypeSize;
}
} // namespace

namespace optiling {
static void RopeHalfGradPrintParam(const gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "layout = %ld", tiling.ropeHalfGradParams.get_layout());
    OP_LOGD(context->GetNodeName(), "xShapeSize = %ld", tiling.ropeHalfGradParams.get_xShapeSize());
    OP_LOGD(context->GetNodeName(), "cosShapeSize = %ld", tiling.ropeHalfGradParams.get_cosShapeSize());
    OP_LOGD(context->GetNodeName(), "dimB = %ld", tiling.ropeHalfGradParams.get_dimB());
    OP_LOGD(context->GetNodeName(), "dimS = %ld", tiling.ropeHalfGradParams.get_dimS());
    OP_LOGD(context->GetNodeName(), "dimN = %ld", tiling.ropeHalfGradParams.get_dimN());
    OP_LOGD(context->GetNodeName(), "dimD = %ld", tiling.ropeHalfGradParams.get_dimD());
    OP_LOGD(context->GetNodeName(), "cosDimB = %ld", tiling.ropeHalfGradParams.get_cosDimB());
    OP_LOGD(context->GetNodeName(), "cosDimN = %ld", tiling.ropeHalfGradParams.get_cosDimN());
    OP_LOGD(context->GetNodeName(), "halfDimDAlignNum = %ld", tiling.ropeHalfGradParams.get_halfDimDAlignNum());

    OP_LOGD(context->GetNodeName(), "coreData = %ld", tiling.ropeHalfGradParams.get_coreData());
    OP_LOGD(context->GetNodeName(), "coreLast = %ld", tiling.ropeHalfGradParams.get_coreLast());
    OP_LOGD(context->GetNodeName(), "copyLoop = %ld", tiling.ropeHalfGradParams.get_copyLoop());
    OP_LOGD(context->GetNodeName(), "copyTail = %ld", tiling.ropeHalfGradParams.get_copyTail());
    OP_LOGD(context->GetNodeName(), "lastCopyLoop = %ld", tiling.ropeHalfGradParams.get_lastCopyLoop());
    OP_LOGD(context->GetNodeName(), "lastCopyTail = %ld", tiling.ropeHalfGradParams.get_lastCopyTail());
    OP_LOGD(context->GetNodeName(), "alignUbSize = %ld", tiling.ropeHalfGradParams.get_alignUbSize());
    OP_LOGD(context->GetNodeName(), "calcUbSize = %ld", tiling.ropeHalfGradParams.get_calcUbSize());
    OP_LOGD(context->GetNodeName(), "coreUsed = %ld", tiling.ropeHalfGradParams.get_coreUsed());
    OP_LOGD(context->GetNodeName(), "coreNum = %ld", tiling.ropeHalfGradParams.get_coreNum());

    OP_LOGD(context->GetNodeName(), "firstReduce = %ld", tiling.ropeHalfGradParams.get_firstReduce());
    OP_LOGD(context->GetNodeName(), "secondReduce = %ld", tiling.ropeHalfGradParams.get_secondReduce());
    OP_LOGD(context->GetNodeName(), "ubLoopGap = %ld", tiling.ropeHalfGradParams.get_ubLoopGap());
    OP_LOGD(context->GetNodeName(), "blockLenInner = %ld", tiling.ropeHalfGradParams.get_blockLenInner());
    OP_LOGD(context->GetNodeName(), "strideInner = %ld", tiling.ropeHalfGradParams.get_strideInner());
    OP_LOGD(context->GetNodeName(), "blockLenPadInner = %ld", tiling.ropeHalfGradParams.get_blockLenPadInner());
    OP_LOGD(context->GetNodeName(), "stridePadInner = %ld", tiling.ropeHalfGradParams.get_stridePadInner());
}

static ge::graphStatus RopeHalfGradShapeDimCheck(const gert::TilingContext* context)
{
    const auto xStorage = context->GetInputShape(INDEX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const auto xShape = xStorage->GetStorageShape();
    const auto cosStorage = context->GetInputShape(INDEX_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosStorage);
    const auto cosShape = cosStorage->GetStorageShape();
    const auto sinStorage = context->GetInputShape(INDEX_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinStorage);
    const auto sinShape = sinStorage->GetStorageShape();
    const uint64_t xDimNum = xShape.GetDimNum();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    if (xDimNum != DIM_FOUR || cosShape.GetDimNum() != DIM_FOUR) {
        OP_LOGE(context->GetNodeName(), "only support 4-d input.");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != sinShape) {
        OP_LOGE(context->GetNodeName(), "cos shape and sin shape are not equal, do not support.");
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(xDimNum - 1) != cosShape.GetDim(xDimNum - 1)) {
        OP_LOGE(context->GetNodeName(), "x and cos shape in dimension D are not equal, do not support.");
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(xDimNum - 1) > D_THRESHOLD) {
        OP_LOGE(context->GetNodeName(), "dimension D should be less than %ld.", D_THRESHOLD);
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(xDimNum - 1) % CHUNK_SIZE != 0) {
        OP_LOGE(context->GetNodeName(), "dimension D is not a multiple of 2, do not support.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static uint64_t RopeHalfGradSelectLayout(
    const gert::TilingContext* context, const std::vector<int64_t> xShapePerDim,
    const std::vector<int64_t> cosShapePerDim)
{
    uint64_t layout = LAYOUT_KEY_UNDEFINED;
    int64_t xDim0 = xShapePerDim[DIM_ZERO];
    int64_t xDim1 = xShapePerDim[DIM_ONE];
    int64_t xDim2 = xShapePerDim[DIM_TWO];
    int64_t cosDim0 = cosShapePerDim[DIM_ZERO];
    int64_t cosDim1 = cosShapePerDim[DIM_ONE];
    int64_t cosDim2 = cosShapePerDim[DIM_TWO];

    bool xBSNDcos1S1D = xDim0 != 1 && cosDim0 == 1 && xDim1 == cosDim1 && cosDim2 == 1;
    bool xBSNDcosBS1D = xDim0 == cosDim0 && xDim1 == cosDim1 && cosDim2 == 1 && xDim0 < xDim1;
    bool xBNSDcos11SD = xDim0 != 1 && cosDim0 == 1 && cosDim1 == 1 && xDim2 == cosDim2;
    bool xBNSDcosB1SD = xDim0 == cosDim0 && cosDim1 == 1 && xDim2 == cosDim2 && xDim0 < xDim2;
    bool xSBNDcosS11D = xDim0 == cosDim0 && xDim1 != 1 && cosDim1 == 1 && cosDim2 == 1;
    bool xSBNDcosSB1D = xDim0 == cosDim0 && xDim1 == cosDim1 && cosDim2 == 1;

    if (xShapePerDim == cosShapePerDim) {
        // non-broadcast
        layout = LAYOUT_KEY_BSND;
    } else if (xBSNDcos1S1D) {
        // BSND and 1S1D
        layout = LAYOUT_KEY_BSND;
    } else if (xBSNDcosBS1D) {
        // BSND and BS1D
        layout = LAYOUT_KEY_BSND;
    } else if (xBNSDcos11SD) {
        // BNSD and 11SD
        layout = LAYOUT_KEY_BNSD;
    } else if (xBNSDcosB1SD) {
        // BNSD and B1SD
        layout = LAYOUT_KEY_BNSD;
    } else if (xSBNDcosS11D) {
        // SBND and S11D
        layout = LAYOUT_KEY_SBND;
    } else if (xSBNDcosSB1D) {
        // SBND and SB1D
        layout = LAYOUT_KEY_SBND;
    } else {
        OP_LOGE(
            context->GetNodeName(), "invalid shape! x shape: [%ld, %ld, %ld, %ld] and cos shape: [%ld, %ld, %ld, %ld]",
            xDim0, xDim1, xDim2, xShapePerDim[DIM_THREE], cosDim0, cosDim1, cosDim2, xShapePerDim[DIM_THREE]);
    }

    return layout;
}

static bool SetLayoutTiling(
    const gert::TilingContext* context, const std::vector<int64_t> xShapePerDim,
    const std::vector<int64_t> cosShapePerDim)
{
    uint64_t layout = RopeHalfGradSelectLayout(context, xShapePerDim, cosShapePerDim);
    uint64_t dimD = xShapePerDim[DIM_THREE];
    uint64_t dimB, dimS, dimN, cosDimB, cosDimN;
    uint64_t dtypeSize = GetDtypeSize(context);
    uint64_t dataEachBlock = BLOCK_SIZE / dtypeSize;
    uint64_t halfDimDAlignNum = GetCeilInt(dimD / CHUNK_SIZE, dataEachBlock) * dataEachBlock;

    switch (layout) {
        case LAYOUT_KEY_BSND:
            dimB = xShapePerDim[DIM_ZERO];
            dimS = xShapePerDim[DIM_ONE];
            dimN = xShapePerDim[DIM_TWO];
            cosDimB = cosShapePerDim[DIM_ZERO];
            cosDimN = cosShapePerDim[DIM_TWO];
            break;
        case LAYOUT_KEY_BNSD:
            dimB = xShapePerDim[DIM_ZERO];
            dimS = xShapePerDim[DIM_TWO];
            dimN = xShapePerDim[DIM_ONE];
            cosDimB = cosShapePerDim[DIM_ZERO];
            cosDimN = cosShapePerDim[DIM_ONE];
            break;
        case LAYOUT_KEY_SBND:
            dimB = xShapePerDim[DIM_ONE];
            dimS = xShapePerDim[DIM_ZERO];
            dimN = xShapePerDim[DIM_TWO];
            cosDimB = cosShapePerDim[DIM_ONE];
            cosDimN = cosShapePerDim[DIM_TWO];
            break;
        default:
            OP_LOGE(context->GetNodeName(), "layout only support BSND, BNSD, SBND.");
            return false;
    }

    tiling.ropeHalfGradParams.set_layout(layout);
    tiling.ropeHalfGradParams.set_dimB(dimB);
    tiling.ropeHalfGradParams.set_dimS(dimS);
    tiling.ropeHalfGradParams.set_dimN(dimN);
    tiling.ropeHalfGradParams.set_dimD(dimD);
    tiling.ropeHalfGradParams.set_cosDimB(cosDimB);
    tiling.ropeHalfGradParams.set_cosDimN(cosDimN);
    tiling.ropeHalfGradParams.set_halfDimDAlignNum(halfDimDAlignNum);
    return true;
}

static bool RopeHalfGradInitShapeInfo(const gert::TilingContext* context)
{
    const auto xStorage = context->GetInputShape(INDEX_GRAD);
    OP_CHECK_IF(
        xStorage == nullptr,
        OP_LOGE(context->GetNodeName(), "[RopeHalfGradInitShapeInfo] xStorage is null."),
        return ge::GRAPH_FAILED);
    const auto xShape = xStorage->GetStorageShape();
    const uint64_t xShapeSize = static_cast<uint64_t>(xShape.GetShapeSize());
    const auto cosStorage = context->GetInputShape(INDEX_COS);
    OP_CHECK_IF(
        cosStorage == nullptr, OP_LOGE(context->GetNodeName(), "cosStorage is null."),
        return false);
    const auto cosShape = cosStorage->GetStorageShape();
    const uint64_t cosShapeSize = static_cast<uint64_t>(cosShape.GetShapeSize());
    tiling.ropeHalfGradParams.set_xShapeSize(xShapeSize);
    tiling.ropeHalfGradParams.set_cosShapeSize(cosShapeSize);

    vector<int64_t> xShapePerDim(DIM_FOUR, 0);
    vector<int64_t> cosShapePerDim(DIM_FOUR, 0);
    for (int64_t i = DIM_FOUR - 1; i >= 0; i--) {
        xShapePerDim[i] = xShape.GetDim(i);
        cosShapePerDim[i] = cosShape.GetDim(i);
    }

    OP_CHECK_IF(
        !SetLayoutTiling(context, xShapePerDim, cosShapePerDim),
        OP_LOGE(context->GetNodeName(), "[SetLayoutTiling] SetLayoutTiling failed."),
        return false);

    return true;
}

static bool RopeHalfGradSetTiling(const gert::TilingContext* context, int64_t availableUbSize, uint64_t maxCoreNum)
{
    const auto xStorage = context->GetInputShape(INDEX_GRAD);
    OP_CHECK_IF(
        xStorage == nullptr, OP_LOGE(context->GetNodeName(), "xStorage is null."),
        return false);
    const auto xShape = xStorage->GetStorageShape();
    uint64_t dimD = xShape.GetDim(DIM_THREE);
    uint64_t halfDimDAlignNum = tiling.ropeHalfGradParams.get_halfDimDAlignNum();
    uint64_t reserveAlignNum = dimD;
    uint64_t alignUbSize = availableUbSize;
    uint64_t calcUbSize = availableUbSize;
    uint64_t ubPerReserveNum = 1;

    if (dimD % MASK_NUM_FP32 != 0) {
        reserveAlignNum = GetLcmForRotary(halfDimDAlignNum * CHUNK_SIZE, MASK_NUM_FP32);
        ubPerReserveNum = reserveAlignNum / (halfDimDAlignNum * CHUNK_SIZE);
    }
    OP_CHECK_IF(
        reserveAlignNum == 0, OP_LOGE("RopeHalfGrad", "reserveAlignNum is 0"), return false);

    alignUbSize = availableUbSize / reserveAlignNum * reserveAlignNum;
    calcUbSize = availableUbSize / reserveAlignNum * ubPerReserveNum * dimD;
    OP_CHECK_IF(
        halfDimDAlignNum * 2 > alignUbSize || alignUbSize == 0 || calcUbSize == 0,
        OP_LOGE(
            "RopeHalfGrad", "reserveAlignNum = %lu too large, aicore do not support.", halfDimDAlignNum * 2),
        return false);

    uint64_t taskNum = tiling.ropeHalfGradParams.get_cosShapeSize();
    if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_BNSD && tiling.ropeHalfGradParams.get_cosDimB() != 1 &&
        tiling.ropeHalfGradParams.get_cosDimN() == 1) {
        taskNum = tiling.ropeHalfGradParams.get_dimS() * tiling.ropeHalfGradParams.get_dimD();
    }
    uint64_t coreData = GetCeilInt(taskNum, maxCoreNum);
    coreData = GetCeilInt(coreData, dimD) * dimD;
    OP_CHECK_IF(coreData == 0, OP_LOGE("RopeHalfGrad", "coreData is 0"), return false);
    uint64_t coreUsed = GetCeilInt(taskNum, coreData);
    uint64_t coreLast = taskNum % coreData != 0 ? taskNum % coreData : coreData;
    uint64_t copyLoop = coreData / calcUbSize;
    uint64_t copyTail = coreData % calcUbSize;
    uint64_t lastCopyLoop = coreLast / calcUbSize;
    uint64_t lastCopyTail = coreLast % calcUbSize;

    tiling.ropeHalfGradParams.set_coreData(coreData);
    tiling.ropeHalfGradParams.set_coreLast(coreLast);
    tiling.ropeHalfGradParams.set_copyLoop(copyLoop);
    tiling.ropeHalfGradParams.set_copyTail(copyTail);
    tiling.ropeHalfGradParams.set_lastCopyLoop(lastCopyLoop);
    tiling.ropeHalfGradParams.set_lastCopyTail(lastCopyTail);
    tiling.ropeHalfGradParams.set_alignUbSize(alignUbSize);
    tiling.ropeHalfGradParams.set_calcUbSize(calcUbSize);
    tiling.ropeHalfGradParams.set_coreUsed(coreUsed);
    return true;
}

static bool RopeHalfGradInitSplitInfo(const gert::TilingContext* context)
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    tiling.ropeHalfGradParams.set_coreNum(maxCoreNum);

    uint64_t maxUbSize;
    auto indexGrad = context->GetInputDesc(INDEX_GRAD);
    OP_CHECK_IF(
        indexGrad == nullptr,
        OP_LOGE(context->GetNodeName(), "[RopeHalfGradInitSplitInfo] indexGrad is null."),
        return false);
    auto inputDtype = indexGrad->GetDataType();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    int64_t availableUbSize = static_cast<int64_t>(maxUbSize) - RESERVE_UB_SIZE;
    if (context->GetInputShape(INDEX_X) != nullptr) {
        if (inputDtype == ge::DT_FLOAT) {
            availableUbSize = availableUbSize / (DTYPE_SIZE_FP32 * IN_OUT_TENSOR_NUM + DTYPE_SIZE_FP32 * TENSOR_NUM);
        } else {
            availableUbSize =
                availableUbSize / (DTYPE_SIZE_FP16 * IN_OUT_TENSOR_NUM + DTYPE_SIZE_FP32 * TENSOR_CONV_NUM);
        }
    } else {
        if (inputDtype == ge::DT_FLOAT) {
            availableUbSize =
                availableUbSize / (DTYPE_SIZE_FP32 * IN_OUT_TENSOR_SINGLE_NUM + DTYPE_SIZE_FP32 * TENSOR_SINGLE_NUM);
        } else {
            availableUbSize = availableUbSize /
                              (DTYPE_SIZE_FP16 * IN_OUT_TENSOR_SINGLE_NUM + DTYPE_SIZE_FP32 * TENSOR_CONV_SINGLE_NUM);
        }
    }

    OP_CHECK_IF(
        !RopeHalfGradSetTiling(context, availableUbSize, maxCoreNum),
        OP_LOGE(context->GetNodeName(), "RopeHalfGradSetTiling failed."), return false);

    return true;
}

static bool RopeHalfGradInitParamInfo(const gert::TilingContext* context)
{
    uint64_t dimB = tiling.ropeHalfGradParams.get_dimB();
    uint64_t dimS = tiling.ropeHalfGradParams.get_dimS();
    uint64_t dimN = tiling.ropeHalfGradParams.get_dimN();
    uint64_t dimD = tiling.ropeHalfGradParams.get_dimD();
    uint64_t cosDimB = tiling.ropeHalfGradParams.get_cosDimB();
    uint64_t cosDimN = tiling.ropeHalfGradParams.get_cosDimN();
    uint64_t firstReduce, secondReduce, ubLoopGap;
    uint64_t blockLenInner, strideInner, blockLenPadInner, stridePadInner;

    uint64_t dtypeSize = GetDtypeSize(context);
    uint64_t dataEachBlock = BLOCK_SIZE / dtypeSize;
    blockLenPadInner = dimD / CHUNK_SIZE * dtypeSize;

    if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_BSND) {
        firstReduce = dimB / cosDimB;
        secondReduce = dimN / cosDimN;
        ubLoopGap = cosDimB * dimS * dimN * dimD;
        blockLenInner = GetCeilInt(dimD, dataEachBlock);
        strideInner = GetCeilInt(dimN / cosDimN * dimD - dimD, dataEachBlock);
        stridePadInner = (dimN / cosDimN * dimD - dimD / CHUNK_SIZE) * dtypeSize;
    } else if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_BNSD) {
        firstReduce = dimB / cosDimB * dimN / cosDimN;
        secondReduce = 1;
        ubLoopGap = cosDimB * cosDimN * dimS * dimD;
        blockLenInner = 1;
        strideInner = 0;
        stridePadInner = dimD / CHUNK_SIZE * dtypeSize;
    } else if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_SBND) {
        firstReduce = 1;
        secondReduce = dimB / cosDimB * dimN / cosDimN;
        ubLoopGap = 1;
        blockLenInner = GetCeilInt(dimD, dataEachBlock);
        strideInner = GetCeilInt(dimB / cosDimB * dimN / cosDimN * dimD - dimD, dataEachBlock);
        stridePadInner = (dimB / cosDimB * dimN / cosDimN * dimD - dimD / CHUNK_SIZE) * dtypeSize;
    } else {
        OP_LOGE(context->GetNodeName(), "layout only support BSND, BNSD, SBND.");
        return false;
    }

    tiling.ropeHalfGradParams.set_firstReduce(firstReduce);
    tiling.ropeHalfGradParams.set_secondReduce(secondReduce);
    tiling.ropeHalfGradParams.set_ubLoopGap(ubLoopGap);
    tiling.ropeHalfGradParams.set_blockLenInner(blockLenInner);
    tiling.ropeHalfGradParams.set_strideInner(strideInner);
    tiling.ropeHalfGradParams.set_blockLenPadInner(blockLenPadInner);
    tiling.ropeHalfGradParams.set_stridePadInner(stridePadInner);

    return true;
}

static uint32_t RopeHalfGradSelectTilingKey(const gert::TilingContext* context)
{
    auto indexGrad = context->GetInputDesc(INDEX_GRAD);
    OP_CHECK_IF(
        indexGrad == nullptr,
        OP_LOGE(context->GetNodeName(), "[RopeHalfGradSelectTilingKey] indexGrad is null."),
        return 0);
    auto inputDtype = indexGrad->GetDataType();
    uint32_t inputDtypeKey = 0;
    uint32_t layoutKey = 0;
    uint32_t shapeAlignKey = 0;
    uint32_t needBackwardKey = 0;

    if (inputDtype == ge::DT_FLOAT16) {
        inputDtypeKey = DTYPE_KEY_FP16;
    } else if (inputDtype == ge::DT_FLOAT) {
        inputDtypeKey = DTYPE_KEY_FP32;
    } else if (inputDtype == ge::DT_BF16) {
        inputDtypeKey = DTYPE_KEY_BF16;
    }
    if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_BSND) {
        layoutKey = LAYOUT_KEY_BSND;
    } else if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_BNSD) {
        layoutKey = LAYOUT_KEY_BNSD;
    } else if (tiling.ropeHalfGradParams.get_layout() == LAYOUT_KEY_SBND) {
        layoutKey = LAYOUT_KEY_SBND;
    }
    if (tiling.ropeHalfGradParams.get_dimD() % MASK_NUM_FP32 == 0) {
        shapeAlignKey = SHAPE_KEY_ALIGN;
    } else {
        shapeAlignKey = SHAPE_KEY_PAD;
    }
    if (context->GetInputShape(INDEX_X) != nullptr) {
        needBackwardKey = NEED_BACKWARD_KEY_TRUE;
    } else {
        needBackwardKey = NEED_BACKWARD_KEY_FALSE;
    }
    uint32_t tilingKey = inputDtypeKey + layoutKey + shapeAlignKey + needBackwardKey;
    return tilingKey;
}

ge::graphStatus RopeRotateHalfGradTlingClass::DoOpTiling()
{
    OP_LOGD("RopeHalfGradTiling tiling start");

    OP_CHECK_IF(
        RopeHalfGradShapeDimCheck(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "RopeHalfGradShapeDimCheck not support."),
        return ge::GRAPH_FAILED);
    OP_LOGD("RopeHalfGradTiling RopeHalfGradShapeDimCheck");
    OP_CHECK_IF(
        !RopeHalfGradInitShapeInfo(context_),
        OP_LOGE(context_->GetNodeName(), "RopeHalfGradInitShapeInfo not support."),
        return ge::GRAPH_FAILED);
    OP_LOGD("RopeHalfGradTiling RopeHalfGradInitShapeInfo");
    OP_CHECK_IF(
        !RopeHalfGradInitSplitInfo(context_),
        OP_LOGE(context_->GetNodeName(), "RopeHalfGradInitSplitInfo not support."),
        return ge::GRAPH_FAILED);
    OP_LOGD("RopeHalfGradTiling RopeHalfGradInitSplitInfo");
    OP_CHECK_IF(
        !RopeHalfGradInitParamInfo(context_),
        OP_LOGE(context_->GetNodeName(), "RopeHalfGradInitParamInfo not support."),
        return ge::GRAPH_FAILED);
    OP_LOGD("RopeHalfGradTiling RopeHalfGradInitParamInfo");

    RopeHalfGradPrintParam(context_);

    context_->SetBlockDim(tiling.ropeHalfGradParams.get_coreUsed());

    uint32_t tilingKey = RopeHalfGradSelectTilingKey(context_);
    context_->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD("RopeHalfGradTiling tiling end");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
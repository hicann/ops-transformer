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
 * \file rope_interleaved_grad.cc
 * \brief
 */
#include "rope_interleaved_grad_tiling.h"
#include "rotary_position_embedding_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace {
constexpr uint64_t INPUT_GRAD_IDX = 0;
constexpr uint64_t INPUT_COS_IDX = 1;
constexpr uint64_t INPUT_SIN_IDX = 2;
constexpr uint64_t INPUT_X_IDX = 3;
constexpr uint64_t INPUT_DIM_NUM = 4;
constexpr uint64_t TND_INPUT_DIM_NUM = 3;
constexpr uint64_t NEED_BACKWARD_ATTR_IDX = 1;
constexpr uint64_t LAYOUT_ATTR_IDX = 1;
constexpr uint64_t INPUT_DIM_0 = 0;
constexpr uint64_t INPUT_DIM_1 = 1;
constexpr uint64_t INPUT_DIM_2 = 2;
constexpr uint64_t INPUT_DIM_3 = 3;
constexpr uint64_t LAYOUT_BSND = 0;
constexpr uint64_t LAYOUT_BNSD = 1;
constexpr uint64_t LAYOUT_SBND = 2;
constexpr uint64_t TILING_KEY_FLOAT16 = 0;
constexpr uint64_t TILING_KEY_BFLOAT16 = 10;
constexpr uint64_t TILING_KEY_FLOAT32 = 20;
constexpr uint64_t TILING_KEY_SMALL = 0;
constexpr uint64_t TILING_KEY_LARGE = 100;
constexpr uint64_t TILING_KEY_NONEEDBACKWARD = 0;
constexpr uint64_t TILING_KEY_NEEDBACKWARD = 1000;
constexpr uint64_t SIZE_FLOAT16 = 2;
constexpr uint64_t SIZE_BFLOAT16 = 2;
constexpr uint64_t SIZE_FLOAT32 = 4;
constexpr uint64_t BASE_TILING_KEY = 20000;
constexpr uint64_t ALIGN_FP32_BLOCK = 8;
constexpr uint64_t ALIGN_FP16_BLOCK = 16;
constexpr uint64_t INPUT_OUTPUT_NUM = 7;
constexpr uint64_t CALC_NUM = 3;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t FP32_DIVIDE_FP16 = 2;
constexpr uint64_t FP16_EXTRA = 2;
constexpr uint64_t FP32_EXTRA = 1;
constexpr uint64_t MAX_BN = 1000;

// other
constexpr uint64_t EXTRA_FP16_BF16_BUFFER_NUM = 8;
constexpr uint64_t BASE_BUFFER_NUM = 14 + 3;
constexpr uint64_t ONE_BLOCK = 32;
constexpr uint64_t ONE_KB = 1024;
constexpr uint64_t RESERVE_NUM = 4;

uint64_t dtypeSize = 2;
uint64_t batchSize;
uint64_t seqLen;
uint64_t numHeads;
uint64_t headDim;
uint64_t alignHeadDim;
uint64_t padHeadDim;

uint64_t bufferSize;
uint64_t wholeBufferNum;
uint64_t maxElementNum;
uint64_t seqFrontLen;
uint64_t seqTailLen;
uint64_t layout;
uint64_t wholeBufferBytes;
uint64_t tilingKey;
bool isTndLayout = false;
uint64_t GetDiv(uint64_t value1, uint64_t value2)
{
    if (value2 == 0)
        return value2;
    return value1 / value2;
}

uint64_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    if (value2 == 0)
        return value2;
    return (value1 + value2 - 1) / value2;
}

uint64_t GetDivRem(uint64_t value1, uint64_t value2)
{
    if (value2 == 0)
        return value2;
    return value1 % value2;
}
} // namespace

namespace optiling {
using namespace Ops::Base;

RotaryPositionEmbeddingGradTilingData tiling;

static void PrintInfo(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), " batchSize=%ld.", tiling.ropeInterleavedGradParams.get_batchSize());
    OP_LOGD(context->GetNodeName(), " seqLen=%ld.", tiling.ropeInterleavedGradParams.get_seqLen());
    OP_LOGD(context->GetNodeName(), " numHeads=%ld.", tiling.ropeInterleavedGradParams.get_numHeads());
    OP_LOGD(context->GetNodeName(), " headDim=%ld.", tiling.ropeInterleavedGradParams.get_headDim());
    OP_LOGD(context->GetNodeName(), " alignHeadDim=%ld.", tiling.ropeInterleavedGradParams.get_alignHeadDim());
    OP_LOGD(context->GetNodeName(), " padHeadDim=%ld.", tiling.ropeInterleavedGradParams.get_padHeadDim());
    OP_LOGD(context->GetNodeName(), " frontCoreNum=%ld.", tiling.ropeInterleavedGradParams.get_frontCoreNum());
    OP_LOGD(context->GetNodeName(), " tailCoreNum=%ld.", tiling.ropeInterleavedGradParams.get_tailCoreNum());
    OP_LOGD(context->GetNodeName(), " seqFrontLen=%ld.", tiling.ropeInterleavedGradParams.get_seqFrontLen());
    OP_LOGD(context->GetNodeName(), " seqTailLen=%ld.", tiling.ropeInterleavedGradParams.get_seqTailLen());

    OP_LOGD(context->GetNodeName(), " seqFrontCalcNum=%ld.", tiling.ropeInterleavedGradParams.get_seqFrontCalcNum());
    OP_LOGD(context->GetNodeName(), " seqFrontCalcLoop=%ld.", tiling.ropeInterleavedGradParams.get_seqFrontCalcLoop());
    OP_LOGD(context->GetNodeName(), " seqFrontCalcTail=%ld.", tiling.ropeInterleavedGradParams.get_seqFrontCalcTail());
    OP_LOGD(context->GetNodeName(), " seqTailCalcNum=%ld.", tiling.ropeInterleavedGradParams.get_seqTailCalcNum());
    OP_LOGD(context->GetNodeName(), " seqTailCalcLoop=%ld.", tiling.ropeInterleavedGradParams.get_seqTailCalcLoop());
    OP_LOGD(context->GetNodeName(), " seqTailCalcTail=%ld.", tiling.ropeInterleavedGradParams.get_seqTailCalcTail());
    OP_LOGD(context->GetNodeName(), " numHeadsLength=%ld.", tiling.ropeInterleavedGradParams.get_numHeadsLength());
    OP_LOGD(context->GetNodeName(), " numHeadsLoop=%ld.", tiling.ropeInterleavedGradParams.get_numHeadsLoop());
    OP_LOGD(context->GetNodeName(), " numHeadsTail=%ld.", tiling.ropeInterleavedGradParams.get_numHeadsTail());
    OP_LOGD(context->GetNodeName(), " batchNumHeadsLength=%ld.",
            tiling.ropeInterleavedGradParams.get_batchNumHeadsLength());
    OP_LOGD(context->GetNodeName(), " batchNumHeadsLoop=%ld.",
            tiling.ropeInterleavedGradParams.get_batchNumHeadsLoop());
    OP_LOGD(context->GetNodeName(), " batchNumHeadsTail=%ld.",
            tiling.ropeInterleavedGradParams.get_batchNumHeadsTail());
    OP_LOGD(context->GetNodeName(), " layout=%ld.", tiling.ropeInterleavedGradParams.get_layout());
}

static void UbTilingCalcForTail()
{
    uint64_t seqTailCalcNum = 0, seqTailCalcLoop = 0, seqTailCalcTail = 0;
    if (seqTailLen != 0) {
        if (maxElementNum >= seqTailLen * batchSize * numHeads * alignHeadDim) {
            seqTailCalcNum = seqTailLen;
            seqTailCalcLoop = 1;
            seqTailCalcTail = 0;
        } else if (maxElementNum >= batchSize * numHeads * alignHeadDim) {
            seqTailCalcNum = GetDiv(bufferSize, (batchSize * numHeads * alignHeadDim * wholeBufferNum * dtypeSize));
            seqTailCalcLoop = GetCeilInt(seqTailLen, seqTailCalcNum);
            seqTailCalcTail =
                GetDivRem(seqTailLen, seqTailCalcNum) != 0 ? seqTailLen - (seqTailCalcLoop - 1) * seqTailCalcNum : 0;
        }
    }
    tiling.ropeInterleavedGradParams.set_seqTailCalcNum(seqTailCalcNum);
    tiling.ropeInterleavedGradParams.set_seqTailCalcLoop(seqTailCalcLoop);
    tiling.ropeInterleavedGradParams.set_seqTailCalcTail(seqTailCalcTail);
}

static void UbTilingCalc()
{
    uint64_t seqFrontCalcNum = 0, seqFrontCalcLoop = 0, seqFrontCalcTail = 0;
    uint64_t numHeadsLength = 0, numHeadsLoop = 0, numHeadsTail = 0;
    uint64_t batchNumHeadsLength = 0, batchNumHeadsLoop = 0, batchNumHeadsTail = 0;
    if (maxElementNum >= seqFrontLen * batchSize * numHeads * alignHeadDim) { // full bsnd
        seqFrontCalcNum = seqFrontLen;
        seqFrontCalcLoop = 1;
        seqFrontCalcTail = 0;
    } else if (maxElementNum >= batchSize * numHeads * alignHeadDim) { // full bnd and split s
        seqFrontCalcNum = GetDiv(bufferSize, (batchSize * numHeads * alignHeadDim * wholeBufferNum * dtypeSize));
        seqFrontCalcLoop = GetCeilInt(seqFrontLen, seqFrontCalcNum);
        seqFrontCalcTail =
            GetDivRem(seqFrontLen, seqFrontCalcNum) != 0 ? seqFrontLen - (seqFrontCalcLoop - 1) * seqFrontCalcNum : 0;
    } else if (maxElementNum >= alignHeadDim) { // full d
        if (layout == 0) {
            numHeadsLength = GetDiv(bufferSize, (alignHeadDim * wholeBufferNum * dtypeSize));
            numHeadsLoop = GetCeilInt(numHeads, numHeadsLength);
            numHeadsTail =
                GetDivRem(numHeads, numHeadsLength) != 0 ? numHeads - (numHeadsLoop - 1) * numHeadsLength : 0;
        } else {
            batchNumHeadsLength = GetDiv(bufferSize, (alignHeadDim * wholeBufferNum * dtypeSize));
            batchNumHeadsLoop = GetCeilInt(batchSize * numHeads, batchNumHeadsLength);
            batchNumHeadsTail = GetDivRem(batchSize * numHeads, batchNumHeadsLength) != 0 ?
                                    (batchSize * numHeads) - (batchNumHeadsLoop - 1) * batchNumHeadsLength :
                                    0;
        }
    }
    if (numHeadsLoop == 0 && batchNumHeadsLoop == 0) {
        tilingKey += TILING_KEY_SMALL;
    } else {
        tilingKey += TILING_KEY_LARGE;
    }
    tiling.ropeInterleavedGradParams.set_seqFrontCalcNum(seqFrontCalcNum);
    tiling.ropeInterleavedGradParams.set_seqFrontCalcLoop(seqFrontCalcLoop);
    tiling.ropeInterleavedGradParams.set_seqFrontCalcTail(seqFrontCalcTail);

    tiling.ropeInterleavedGradParams.set_numHeadsLength(numHeadsLength);
    tiling.ropeInterleavedGradParams.set_numHeadsLoop(numHeadsLoop);
    tiling.ropeInterleavedGradParams.set_numHeadsTail(numHeadsTail);

    tiling.ropeInterleavedGradParams.set_batchNumHeadsLength(batchNumHeadsLength);
    tiling.ropeInterleavedGradParams.set_batchNumHeadsLoop(batchNumHeadsLoop);
    tiling.ropeInterleavedGradParams.set_batchNumHeadsTail(batchNumHeadsTail);

    tiling.ropeInterleavedGradParams.set_layout(layout);
}

static void SetMaxElementNum(ge::DataType dataDtype, uint64_t reserveBufferSize, uint64_t ubSize)
{
    bufferSize = ubSize - reserveBufferSize;
    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
        wholeBufferNum =
            (INPUT_OUTPUT_NUM + CALC_NUM) * FP32_DIVIDE_FP16 + INPUT_OUTPUT_NUM * DOUBLE_BUFFER + FP16_EXTRA;
    } else {
        wholeBufferNum = INPUT_OUTPUT_NUM * DOUBLE_BUFFER + CALC_NUM;
    }

    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
        if (layout == LAYOUT_BSND || layout == LAYOUT_SBND) {
            wholeBufferNum += FP16_EXTRA;
        }
        wholeBufferBytes = wholeBufferNum * SIZE_FLOAT16;
    } else {
        if (layout == LAYOUT_BSND || layout == LAYOUT_SBND) {
            wholeBufferNum += FP32_EXTRA;
        }
        wholeBufferBytes = wholeBufferNum * SIZE_FLOAT32;
    }

    maxElementNum = GetDiv(bufferSize, wholeBufferBytes);
}

ge::graphStatus RopeCheckInputShape(gert::TilingContext *context, const gert::StorageShape *xShape,
                                    const gert::StorageShape *cosShape, const gert::StorageShape *sinShape)
{
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "[RopeCheckInputShape] xShape is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosShape == nullptr, OP_LOGE(context->GetNodeName(), "[RopeCheckInputShape] cosShape is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinShape == nullptr, OP_LOGE(context->GetNodeName(), "[RopeCheckInputShape] sinShape is null."),
                return ge::GRAPH_FAILED);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t cosDimNum = cosShape->GetStorageShape().GetDimNum();
    size_t sinDimNum = sinShape->GetStorageShape().GetDimNum();

    uint64_t inputDimNum = INPUT_DIM_NUM;
    uint64_t headDimIndex = INPUT_DIM_3;

    isTndLayout = (xDimNum == TND_INPUT_DIM_NUM);
    if (isTndLayout) {
        OP_LOGD(context->GetNodeName(), "Enter TND layout.");
        inputDimNum = TND_INPUT_DIM_NUM;
        headDimIndex = INPUT_DIM_2;
    }

    if (xDimNum != inputDimNum || cosDimNum != inputDimNum || sinDimNum != inputDimNum) {
        std::string dimNumMsg =
            std::to_string(xDimNum) + ", " + std::to_string(cosDimNum) + " and " + std::to_string(sinDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "dy, cos and sin", dimNumMsg.c_str(),
            "The numbers of dimensions of input dy, cos and sin should all be 3D or 4D");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < xDimNum; ++i) {
        if (cosShape->GetStorageShape().GetDim(i) != sinShape->GetStorageShape().GetDim(i)) {
            std::string shapeMsg =
                ToString(cosShape->GetStorageShape()) + " and " + ToString(sinShape->GetStorageShape());
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                                   "The shapes of input cos and sin should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    uint32_t xHeadDim = xShape->GetStorageShape().GetDim(headDimIndex);
    uint32_t cosHeadDim = cosShape->GetStorageShape().GetDim(headDimIndex);
    uint32_t sinHeadDim = sinShape->GetStorageShape().GetDim(headDimIndex);
    if ((xHeadDim != cosHeadDim) && (xHeadDim != sinHeadDim)) {
        std::string shapeMsg = ToString(xShape->GetStorageShape()) + ", " + ToString(cosShape->GetStorageShape()) +
                               " and " + ToString(sinShape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "dy, cos and sin", shapeMsg.c_str(),
            "The D axis of input dy, cos and sin should be the same, where D refers to the last dim");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeCheckOptInputShape(gert::TilingContext *context)
{
    auto xOptionalInput = context->GetOptionalInputDesc(INPUT_X_IDX);
    auto xOptionalShape = context->GetOptionalInputShape(INPUT_X_IDX);
    if (xOptionalInput != nullptr && xOptionalShape != nullptr) {
        auto dyShape = context->GetInputShape(INPUT_GRAD_IDX);
        auto dyStorageShape = dyShape->GetStorageShape();
        auto xOptionalStorageShape = xOptionalShape->GetStorageShape();
        if (xOptionalStorageShape != dyStorageShape) {
            std::string shapeMsg = ToString(xOptionalStorageShape) + " and " + ToString(dyStorageShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and dy", shapeMsg.c_str(),
                                                   "The shapes of input x and input dy should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSplitS(gert::TilingContext *context, uint64_t coreNum, uint64_t ubSize)
{
    batchSize = tiling.ropeInterleavedGradParams.get_batchSize();
    seqLen = tiling.ropeInterleavedGradParams.get_seqLen();
    numHeads = tiling.ropeInterleavedGradParams.get_numHeads();
    headDim = tiling.ropeInterleavedGradParams.get_headDim();
    auto inputGradIdx = context->GetInputDesc(INPUT_GRAD_IDX);
    OP_CHECK_IF(inputGradIdx == nullptr, OP_LOGE(context->GetNodeName(), "[TilingSplitS] inputGradIdx is null."),
                return ge::GRAPH_FAILED);
    auto dataDtype = inputGradIdx->GetDataType();
    if (dataDtype == ge::DT_FLOAT16 || dataDtype == ge::DT_BF16) {
        alignHeadDim = GetCeilInt(headDim, ALIGN_FP16_BLOCK) * ALIGN_FP16_BLOCK;
    } else {
        alignHeadDim = GetCeilInt(headDim, ALIGN_FP32_BLOCK) * ALIGN_FP32_BLOCK;
    }
    padHeadDim = alignHeadDim - headDim;
    tiling.ropeInterleavedGradParams.set_alignHeadDim(alignHeadDim);
    tiling.ropeInterleavedGradParams.set_padHeadDim(padHeadDim);

    uint64_t frontCoreNum = seqLen % coreNum != 0 ? seqLen % coreNum : coreNum;
    uint64_t tailCoreNum = seqLen < coreNum ? 0 : coreNum - frontCoreNum;
    uint64_t numBlocks = frontCoreNum + tailCoreNum;

    seqFrontLen = GetCeilInt(seqLen, coreNum);
    seqTailLen = GetDiv(seqLen, coreNum);

    tiling.ropeInterleavedGradParams.set_frontCoreNum(frontCoreNum);
    tiling.ropeInterleavedGradParams.set_tailCoreNum(tailCoreNum);
    tiling.ropeInterleavedGradParams.set_seqFrontLen(seqFrontLen);
    tiling.ropeInterleavedGradParams.set_seqTailLen(seqTailLen);

    context->SetBlockDim(numBlocks);

    uint64_t reserveBufferSize = ONE_BLOCK;
    if (ubSize < reserveBufferSize) {
        OP_LOGE(context->GetNodeName(), "Because the size of the shape D is too large, it exceeds the ub range!");
        return ge::GRAPH_FAILED;
    }

    SetMaxElementNum(dataDtype, reserveBufferSize, ubSize);
    UbTilingCalc();
    UbTilingCalcForTail();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingLayoutSplit(gert::TilingContext *context, const gert::StorageShape *xShape,
                                  const gert::StorageShape *cosShape)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "[TilingLayoutSplit] coreNum is zero."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t indexOffset = isTndLayout ? 1UL : 0UL;
    uint64_t xShape0 = isTndLayout ? 1UL : xShape->GetStorageShape().GetDim(INPUT_DIM_0);
    uint64_t xShape1 = xShape->GetStorageShape().GetDim(INPUT_DIM_1 - indexOffset);
    uint64_t xShape2 = xShape->GetStorageShape().GetDim(INPUT_DIM_2 - indexOffset);
    uint64_t xHeadDim = xShape->GetStorageShape().GetDim(INPUT_DIM_3 - indexOffset);
    uint64_t cosShape0 = isTndLayout ? 1UL : cosShape->GetStorageShape().GetDim(INPUT_DIM_0);
    uint64_t cosShape1 = cosShape->GetStorageShape().GetDim(INPUT_DIM_1 - indexOffset);
    uint64_t cosShape2 = cosShape->GetStorageShape().GetDim(INPUT_DIM_2 - indexOffset);
    if (cosShape0 == 1 && cosShape2 == 1 && xShape1 == cosShape1) {
        // BSND
        tiling.ropeInterleavedGradParams.set_batchSize(xShape0);
        tiling.ropeInterleavedGradParams.set_seqLen(xShape1);
        tiling.ropeInterleavedGradParams.set_numHeads(xShape2);
        tiling.ropeInterleavedGradParams.set_headDim(xHeadDim);
        layout = LAYOUT_BSND;
    } else if (cosShape0 == 1 && cosShape1 == 1 && xShape2 == cosShape2) {
        // BNSD
        tiling.ropeInterleavedGradParams.set_batchSize(xShape0);
        tiling.ropeInterleavedGradParams.set_seqLen(xShape2);
        tiling.ropeInterleavedGradParams.set_numHeads(xShape1);
        tiling.ropeInterleavedGradParams.set_headDim(xHeadDim);
        layout = LAYOUT_BNSD;
    } else if (cosShape1 == 1 && cosShape2 == 1 && xShape0 == cosShape0) {
        // SBND
        tiling.ropeInterleavedGradParams.set_batchSize(xShape1);
        tiling.ropeInterleavedGradParams.set_seqLen(xShape0);
        tiling.ropeInterleavedGradParams.set_numHeads(xShape2);
        tiling.ropeInterleavedGradParams.set_headDim(xHeadDim);
        layout = LAYOUT_SBND;
    } else {
        OP_LOGE(context->GetNodeName(), "The shape of the input x, cos and sin is not supported.");
        return ge::GRAPH_FAILED;
    }
    if (tiling.ropeInterleavedGradParams.get_batchSize() * tiling.ropeInterleavedGradParams.get_numHeads() >= MAX_BN) {
        OP_LOGE(context->GetNodeName(), "B * N should be smaller than 1000.");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(INPUT_X_IDX) != nullptr) {
        tilingKey += TILING_KEY_NEEDBACKWARD;
    } else {
        tilingKey += TILING_KEY_NONEEDBACKWARD;
    }
    return TilingSplitS(context, coreNum, ubSize);
}

ge::graphStatus RopeInterLeavedGradTlingClass::DoOpTiling()
{
    const gert::StorageShape *xShape = context_->GetInputShape(INPUT_GRAD_IDX);
    const gert::StorageShape *cosShape = context_->GetInputShape(INPUT_COS_IDX);
    const gert::StorageShape *sinShape = context_->GetInputShape(INPUT_SIN_IDX);

    OP_CHECK_IF(RopeCheckInputShape(context_, xShape, cosShape, sinShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "RopeCheckInputShape fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(RopeCheckOptInputShape(context_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "RopeCheckOptInputShape fail."), return ge::GRAPH_FAILED);

    tilingKey = BASE_TILING_KEY;
    auto dataDtype = context_->GetInputDesc(INPUT_GRAD_IDX)->GetDataType();
    if (dataDtype == ge::DT_FLOAT16) {
        dtypeSize = SIZE_FLOAT16;
        tilingKey += TILING_KEY_FLOAT16;
    } else if (dataDtype == ge::DT_BF16) {
        dtypeSize = SIZE_BFLOAT16;
        tilingKey += TILING_KEY_BFLOAT16;
    } else if (dataDtype == ge::DT_FLOAT) {
        dtypeSize = SIZE_FLOAT32;
        tilingKey += TILING_KEY_FLOAT32;
    } else {
        std::string dataDtypeStr = ToString(dataDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", dataDtypeStr.c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(TilingLayoutSplit(context_, xShape, cosShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "TilingSplitS fail."), return ge::GRAPH_FAILED);

    context_->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t usr_workspace_size = 0;
    size_t sys_work_space_size = 16 * 1024 * 1024;
    size_t *current_workspace = context_->GetWorkspaceSizes(1);
    current_workspace[0] = usr_workspace_size + sys_work_space_size;
    PrintInfo(context_);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

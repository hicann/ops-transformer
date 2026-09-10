/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef W8A8_GSA_ARCH22_KERNEL_UTILS
#define W8A8_GSA_ARCH22_KERNEL_UTILS

#include "../attn_infra/generic_block_sparse_attention_base_defs.hpp"
#include "catlass/arch/arch.hpp"

#include "../attn_infra/gemm/block/generic_block_sparse_attention_block_mmad.hpp"

#include "../attn_infra/arch/generic_block_sparse_attention_cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "../attn_infra/epilogue/block/generic_block_sparse_attention_block_epilogue.hpp"
#include "../attn_infra/epilogue/generic_block_sparse_attention_dispatch_policy.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
namespace GsaKernelArch22 {

// Regular (non-quantized) kernel params.
// Inputs order: q, k, v, sparseBlockIdx, sparseBlockCount, metaData,
//               cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable
struct GsaKernelParamsArch22 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR sparseBlockIdx;
    GM_ADDR sparseBlockCount;
    GM_ADDR metaData;
    GM_ADDR cuSeqLengths;
    GM_ADDR cuSeqLengthsKv;
    GM_ADDR sequsedQ;
    GM_ADDR sequsedKv;
    GM_ADDR blockTable;
    GM_ADDR qDequantScale; // W8A8: combined KV antiquant scale (K first, V second)
    GM_ADDR kDequantScale; // W8A8: independent K antiquant scale
    GM_ADDR vDequantScale; // W8A8: independent V antiquant scale
    GM_ADDR o;
    GM_ADDR softmaxLse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    __aicore__ inline GsaKernelParamsArch22()
        : qDequantScale(nullptr),
          kDequantScale(nullptr),
          vDequantScale(nullptr)
    {}

    // Regular mode (no dequantScale).
    __aicore__ inline GsaKernelParamsArch22(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR sparseBlockIdx_,
                                            GM_ADDR sparseBlockCount_, GM_ADDR metaData_, GM_ADDR cuSeqLengths_,
                                            GM_ADDR cuSeqLengthsKv_, GM_ADDR sequsedQ_, GM_ADDR sequsedKv_,
                                            GM_ADDR blockTable_, GM_ADDR o_, GM_ADDR softmaxLse_, GM_ADDR workSpace_,
                                            GM_ADDR tiling_)
        : q(q_),
          k(k_),
          v(v_),
          sparseBlockIdx(sparseBlockIdx_),
          sparseBlockCount(sparseBlockCount_),
          metaData(metaData_),
          cuSeqLengths(cuSeqLengths_),
          cuSeqLengthsKv(cuSeqLengthsKv_),
          sequsedQ(sequsedQ_),
          sequsedKv(sequsedKv_),
          blockTable(blockTable_),
          qDequantScale(nullptr),
          kDequantScale(nullptr),
          vDequantScale(nullptr),
          o(o_),
          softmaxLse(softmaxLse_),
          workSpace(workSpace_),
          tiling(tiling_)
    {}

    // W8A8 antiquant mode (with dequantScale).
    __aicore__ inline GsaKernelParamsArch22(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR sparseBlockIdx_,
                                            GM_ADDR sparseBlockCount_, GM_ADDR metaData_, GM_ADDR cuSeqLengths_,
                                            GM_ADDR cuSeqLengthsKv_, GM_ADDR sequsedQ_, GM_ADDR sequsedKv_,
                                            GM_ADDR blockTable_, GM_ADDR qDequantScale_, GM_ADDR kDequantScale_,
                                            GM_ADDR vDequantScale_, GM_ADDR o_, GM_ADDR softmaxLse_, GM_ADDR workSpace_,
                                            GM_ADDR tiling_)
        : q(q_),
          k(k_),
          v(v_),
          sparseBlockIdx(sparseBlockIdx_),
          sparseBlockCount(sparseBlockCount_),
          metaData(metaData_),
          cuSeqLengths(cuSeqLengths_),
          cuSeqLengthsKv(cuSeqLengthsKv_),
          sequsedQ(sequsedQ_),
          sequsedKv(sequsedKv_),
          blockTable(blockTable_),
          qDequantScale(qDequantScale_),
          kDequantScale(kDequantScale_),
          vDequantScale(vDequantScale_),
          o(o_),
          softmaxLse(softmaxLse_),
          workSpace(workSpace_),
          tiling(tiling_)
    {}
};

// W8A8 pseudo-quantization helpers (ported from sparse_attention_score arch22).
static constexpr float antiquantExpandCoeff = 254.0f;
static constexpr float antiqCoeff1 = 127.0f;
static constexpr float antiqCoeff2 = 1.0f / 127.0f;
static constexpr uint32_t BYTE_BLOCK = 32;
static constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);         // = 8
static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float); // = 64
static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

__aicore__ inline void AbsRowMax(AscendC::LocalTensor<float> &tmpAMaxRes, AscendC::LocalTensor<float> &srcUb,
                                 AscendC::LocalTensor<float> tmpAUb, AscendC::LocalTensor<float> tmpRowMaxUb,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    AscendC::Abs(tmpAUb, srcUb, dealRowCount * columnCount);
    AscendC::PipeBarrier<PIPE_V>();
    // arch22 has no RowMax API; use WholeReduceMax for per-row max.
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    AscendC::BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        AscendC::Max(tmpAUb, tmpAUb, tmpAUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < loopCount; j++) {
            AscendC::Max(tmpAUb[j * dtypeMask], tmpAUb[j * dtypeMask], tmpAUb[(j + blockCount) * dtypeMask], dtypeMask,
                         dealRowCount, repeatParamsMax);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t dataBlockNumPerRow = columnCount / FP32_BLOCK_ELEMENT_NUM;
    AscendC::WholeReduceMax<float, false>(tmpRowMaxUb, tmpAUb,
                                          (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount,
                                          1, 1, dataBlockNumPerRow, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(tmpAMaxRes, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});
}

__aicore__ inline void VecMulMat(AscendC::LocalTensor<float> dstUb, AscendC::LocalTensor<float> src0Ub,
                                 AscendC::LocalTensor<float> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
                                 uint32_t actualColumnCount)
{
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            AscendC::Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            AscendC::Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            AscendC::Mul(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

template <typename T>
__aicore__ inline void RowMuls(AscendC::LocalTensor<T> dstUb, AscendC::LocalTensor<T> src0Ub,
                               AscendC::LocalTensor<T> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;
    if constexpr (std::is_same<T, half>::value) {
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    }
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                AscendC::Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            AscendC::BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src0RepStride = 8;
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstRepStride = 8;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                AscendC::Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum],
                             repeatElementNum, dLoop, columnRepeatParams);
            }
        }
        if (dRemain > 0) {
            AscendC::Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain,
                         dealRowCount, repeatParams);
        }
    } else {
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            AscendC::Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                         dLoop, repeatParams);
            if (dRemain > 0) {
                AscendC::Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                             src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain,
                             1, repeatParams);
            }
        }
    }
}

__aicore__ inline void RowDivs(AscendC::LocalTensor<float> dstUb, AscendC::LocalTensor<float> src0Ub,
                               AscendC::LocalTensor<float> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;
    AscendC::BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (dLoop <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            AscendC::Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        AscendC::BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8;
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            AscendC::Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, dLoop,
                         columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        AscendC::Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount,
                     repeatParamsDiv);
    }
}

__aicore__ inline void AntiquantAIterExpand(AscendC::GlobalTensor<int8_t> dstGm, AscendC::LocalTensor<float> &tmpA1,
                                            AscendC::LocalTensor<float> &tmpA2, AscendC::LocalTensor<half> &aResOutUb,
                                            AscendC::LocalTensor<int8_t> &aResOutUbI8, uint32_t calcSize, bool isFirst,
                                            uint64_t outOffset, uint32_t eventId, bool deferMte3Wait)
{
    if (!isFirst) {
        AscendC::Sub(tmpA1, tmpA1, tmpA2, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmpA1, tmpA1, antiquantExpandCoeff, calcSize);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Cast(tmpA2, tmpA1, AscendC::RoundMode::CAST_ROUND, calcSize);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(aResOutUb, tmpA2, AscendC::RoundMode::CAST_ROUND, calcSize);
    AscendC::PipeBarrier<PIPE_V>();
    aResOutUbI8 = aResOutUb.template ReinterpretCast<int8_t>();
    aResOutUbI8.SetSize(aResOutUb.GetSize());
    AscendC::Cast(aResOutUbI8, aResOutUb, AscendC::RoundMode::CAST_ROUND, calcSize);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::DataCopy(dstGm[outOffset], aResOutUbI8, calcSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    if (!deferMte3Wait) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
}

__aicore__ inline void AntiquantMatmulPreProcess(
    AscendC::GlobalTensor<int8_t> dstGm, AscendC::LocalTensor<float> aMaxResUb, AscendC::LocalTensor<float> srcUb,
    AscendC::LocalTensor<float> tmpAFloorUb, AscendC::LocalTensor<float> tmpRowMaxUb,
    AscendC::LocalTensor<half> aResOutUb, AscendC::LocalTensor<int8_t> aResOutUbI8,
    AscendC::LocalTensor<half> aResOutUbAlt, AscendC::LocalTensor<int8_t> aResOutUbI8Alt, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, uint32_t groupSize, uint32_t msdIterNum,
    bool enableMte3DoubleBuffer)
{
    uint32_t step = groupSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;
    AscendC::LocalTensor<float> tmpAMaxRes = aMaxResUb[startRow * FP32_BLOCK_ELEMENT_NUM];
    AbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, tmpRowMaxUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(tmpAFloorUb, antiqCoeff1, dealRowCount * FP32_BLOCK_ELEMENT_NUM);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * FP32_BLOCK_ELEMENT_NUM);
    AscendC::PipeBarrier<PIPE_V>();
    RowMuls(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < msdIterNum; i++) {
        const uint32_t slot = i & 1U;
        const uint32_t eventId = slot == 0U ? EVENT_ID7 : EVENT_ID6;
        if (enableMte3DoubleBuffer && i >= 2U) {
            const uint32_t releaseEventId = ((i - 2U) & 1U) == 0U ? EVENT_ID7 : EVENT_ID6;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(releaseEventId);
        }
        if (slot == 0U) {
            AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, aResOutUb, aResOutUbI8, calcSize, (i == 0),
                                 step * i + baseOffset, eventId, enableMte3DoubleBuffer);
        } else {
            AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, aResOutUbAlt, aResOutUbI8Alt, calcSize, (i == 0),
                                 step * i + baseOffset, eventId, enableMte3DoubleBuffer);
        }
    }
    if (enableMte3DoubleBuffer) {
        const uint32_t firstPending = msdIterNum > 2U ? msdIterNum - 2U : 0U;
        for (uint32_t i = firstPending; i < msdIterNum; i++) {
            const uint32_t eventId = (i & 1U) == 0U ? EVENT_ID7 : EVENT_ID6;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
    }
}

__aicore__ inline void AntiquantMatmulResCombine(AscendC::LocalTensor<float> bmmResUb,
                                                 AscendC::GlobalTensor<int32_t> srcGm,
                                                 AscendC::LocalTensor<int32_t> tmpCInt,
                                                 AscendC::LocalTensor<float> tmpCFp, uint32_t startRow,
                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount, uint32_t groupSize, uint32_t msdIterNum)
{
    uint32_t step = groupSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;
    float scale = 1;
    uint32_t offset = baseOffset;
    for (uint32_t i = 0; i < msdIterNum; i++) {
        AscendC::DataCopy(tmpCInt, srcGm[offset], copySize);
        if (i == 0) {
            AscendC::Cast(bmmResUb, tmpCInt, AscendC::RoundMode::CAST_NONE, copySize);
        } else {
            tmpCFp = tmpCInt.template ReinterpretCast<float>();
            tmpCFp.SetSize(tmpCInt.GetSize());
            AscendC::Cast(tmpCFp, tmpCInt, AscendC::RoundMode::CAST_NONE, copySize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmpCFp, tmpCFp, scale, copySize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(bmmResUb, bmmResUb, tmpCFp, copySize);
        }
        offset += step;
        scale = scale / antiquantExpandCoeff;
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(bmmResUb, bmmResUb, antiqCoeff2, copySize);
}

__aicore__ inline void AntiquantSoftmaxResPreProcess(
    AscendC::GlobalTensor<int8_t> dstGm, AscendC::LocalTensor<float> srcUb, AscendC::LocalTensor<float> tmpAFloorUb,
    AscendC::LocalTensor<half> aResOutUb, AscendC::LocalTensor<int8_t> aResOutUbI8, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, uint32_t groupSize, uint32_t msdIterNum)
{
    uint32_t step = groupSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;
    AscendC::Muls(srcUb, srcUb, antiqCoeff1, calcSize);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUb, tmpAFloorUb, aResOutUb, aResOutUbI8, calcSize, (i == 0),
                             step * i + baseOffset, EVENT_ID7, false);
    }
}

template <typename Q_T>
__aicore__ inline void CopyAntiqQuery(AscendC::LocalTensor<float> &queryCastUb, AscendC::LocalTensor<Q_T> &inputUb,
                                      AscendC::GlobalTensor<Q_T> queryGm, uint64_t qOffset, uint32_t dealRowCount,
                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T);
    AscendC::DataCopyExtParams copyInParams;
    AscendC::DataCopyPadExtParams<Q_T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = actualColumnCount * sizeof(Q_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (columnCount - actualColumnCount) / qTypeElementSize;
    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % qTypeElementSize;
    copyInPadParams.paddingValue = 0;
    AscendC::DataCopyPad(inputUb, queryGm[qOffset], copyInParams, copyInPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
    AscendC::Cast(queryCastUb, inputUb, AscendC::RoundMode::CAST_NONE, dealRowCount * columnCount);
}

__aicore__ inline void CopyAntiquantScalePerToken(AscendC::LocalTensor<float> &scaleUb,
                                                  AscendC::GlobalTensor<float> srcGm, uint64_t offset,
                                                  uint32_t actualColumnCount)
{
    AscendC::DataCopyExtParams copyInParams;
    AscendC::DataCopyPadExtParams<float> copyInPadParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = actualColumnCount * sizeof(float);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;
    AscendC::DataCopyPad(scaleUb, srcGm[offset], copyInParams, copyInPadParams);
}

} // namespace GsaKernelArch22

#endif // GSA_ARCH22_KERNEL_UTILS

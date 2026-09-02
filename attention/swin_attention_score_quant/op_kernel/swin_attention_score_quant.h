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
 * \file swin_attention_score_quant.h
 * \brief
 */
#ifndef SWIN_ATTENTION_SCORE_QUANT_H
#define SWIN_ATTENTION_SCORE_QUANT_H

#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"

const uint32_t BLOCK_SIZE_32 = 32;
const uint32_t BLOCK_SIZE_16 = 16;
const uint32_t SWIN_BUFFER_NUM = 1;
const uint32_t BLOCK_NUM_PER_FRACTAL = 16;
const uint32_t BLOCK_NUM_PER_VEC = 8;
const uint32_t CONST_2 = 2;
const uint32_t CONST_8 = 8;
const uint32_t CONST_16 = 16;
const uint32_t CONST_64 = 64;
const uint32_t FRACTAL_SIZE = 512;
const uint32_t BANK_CONFLICT_BLOCK_NUM = 32;
const uint32_t CRITICAL_S_DIM = 970;
constexpr MatmulConfig CFG_ENVECND2NZ = GetNormalConfig(false, false, true);

__aicore__ inline uint32_t RoundUp(uint32_t num, uint32_t align)
{
    if (align > 0) {
        return (num + align - 1) / align * align;
    }
    return 0;
}

__aicore__ inline uint32_t DivUp(uint32_t num, uint32_t align)
{
    if (align > 0) {
        return (num + align - 1) / align;
    }
    return 0;
}

template <typename DTYPE_IN, typename DTYPE_OUT, typename DTYPE_SCALE, typename DTYPE_BIAS, typename DTYPE_MASK,
          bool HAS_BIAS1>
class SwinAttentionScoreQuant;

template <typename DTYPE>
__aicore__ inline int CopyNDBlock(AscendC::LocalTensor<DTYPE> &transTensor, const AscendC::GlobalTensor<DTYPE> &src,
                                  int64_t srcOffset, const int height, const int width, const int gCol,
                                  const bool isBankConflict)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    int calcWidth = DivUp(width, c0Size);

    if (gCol % c0Size > 0) {
        int blockLen = calcWidth;
        int dstOffset = 0;
        int bankConflictPadSize = isBankConflict ? (BLOCK_SIZE_32 / sizeof(DTYPE)) : 0;
        for (int i = 0; i < height; i++) {
            AscendC::DataCopy(transTensor[dstOffset], src[srcOffset], {1, static_cast<uint16_t>(blockLen), 0, 0});
            dstOffset += (RoundUp(width, c0Size) + bankConflictPadSize);
            srcOffset += gCol;
        }
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(enQueEvtID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(enQueEvtID);
    } else {
        int srcStride = (gCol - width) * sizeof(DTYPE) / BLOCK_SIZE_32;
        int blockLen = DivUp(width * sizeof(DTYPE), BLOCK_SIZE_32);
        uint16_t dstStride = isBankConflict ? 1 : 0;
        AscendC::DataCopy(transTensor, src[srcOffset],
                          {static_cast<uint16_t>(height), static_cast<uint16_t>(blockLen),
                           static_cast<uint16_t>(srcStride), dstStride});
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(enQueEvtID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(enQueEvtID);
    }
    return calcWidth;
}

template <typename DTYPE>
__aicore__ inline void NDPadZeros(AscendC::LocalTensor<DTYPE> &dst, const int height, const int calcWidth,
                                  const int width)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    int tail = width % c0Size;
    constexpr int maxSrcBlkStride0 = 64;
    constexpr int maxSrcBlkStride1 = 32;

    if (tail > 0) {
        auto offset = width / c0Size * c0Size;
        uint64_t mask[2];
        uint16_t mask_tail = ~((1 << tail) - 1);
        uint64_t masktail = mask_tail;
        mask[0] =
            masktail + (masktail << 16) + (masktail << 32) + (masktail << 48); // 将masktail左移16、32、48位填充mask
        mask[1] = mask[0];
        if (masktail != 0) {
            if (calcWidth >= maxSrcBlkStride0) {
                AscendC::Duplicate(dst[offset], (DTYPE)0, mask, DivUp(height, 2), calcWidth,
                                   2 * calcWidth); // 每次处理2个block的数据
            } else if (calcWidth >= maxSrcBlkStride1) {
                AscendC::Duplicate(dst[offset], (DTYPE)0, mask, DivUp(height, 4), calcWidth,
                                   4 * calcWidth); // 每次处理4个block的数据
            } else {
                AscendC::Duplicate(dst[offset], (DTYPE)0, mask, DivUp(height, BLOCK_NUM_PER_VEC), calcWidth,
                                   BLOCK_NUM_PER_VEC * calcWidth);
            }
        }
    }
    int tailHigh = height % BLOCK_NUM_PER_FRACTAL;
    if (tailHigh > 0) {
        auto dstOffset = height * calcWidth * c0Size;
        AscendC::Duplicate(dst[dstOffset], (DTYPE)0, (BLOCK_NUM_PER_FRACTAL - tailHigh) * calcWidth * c0Size);
    }
}

template <typename DTYPE>
__aicore__ inline void NDTrans2NZ(AscendC::LocalTensor<DTYPE> &dst, AscendC::LocalTensor<DTYPE> &src,
                                  const int calcHigh, const int calcWidth, const bool isBankConflict)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    const int cubeNum = FRACTAL_SIZE / sizeof(DTYPE);
    struct AscendC::UnaryRepeatParams intriParams;
    uint64_t mask[2] = {uint64_t(-1), uint64_t(-1)};
    int32_t padBlock = 1;

    int actualWidth = isBankConflict ? calcWidth + padBlock : calcWidth;
    intriParams.dstBlkStride = 1;
    intriParams.srcBlkStride = actualWidth;
    intriParams.dstRepStride = intriParams.dstBlkStride * BLOCK_NUM_PER_VEC;
    intriParams.srcRepStride = intriParams.srcBlkStride * BLOCK_NUM_PER_VEC;
    int dstOffset = 0;
    int srcOffset = 0;
    constexpr int maxSrcBlkStride0 = 64;
    constexpr int maxSrcBlkStride1 = 32;
    if (intriParams.srcBlkStride >= maxSrcBlkStride0) {
        mask[0] = 0xffffffff;
        mask[1] = 0;
        intriParams.dstRepStride = intriParams.dstBlkStride * 2; // vec每次处理2个block的数据
        intriParams.srcRepStride = intriParams.srcBlkStride * 2; // vec每次处理2个block的数据

        for (int i = 0; i < calcWidth; i++) {
            dstOffset = i * calcHigh * cubeNum;
            srcOffset = i * c0Size;
            Muls(dst[dstOffset], src[srcOffset], (DTYPE)1, mask, 8 * calcHigh, intriParams); // 每个calcHigh repeat 8次
        }
    } else if (intriParams.srcBlkStride >= maxSrcBlkStride1) {
        mask[1] = 0;
        intriParams.dstRepStride = intriParams.dstBlkStride * 4; // vec每次处理4个block的数据
        intriParams.srcRepStride = intriParams.srcBlkStride * 4; // vec每次处理4个block的数据

        for (int i = 0; i < calcWidth; i++) {
            dstOffset = i * calcHigh * cubeNum;
            srcOffset = i * c0Size;
            Muls(dst[dstOffset], src[srcOffset], (DTYPE)1, mask, 4 * calcHigh, intriParams); // 每个calcHigh repeat 4次
        }
    } else {
        for (int i = 0; i < calcWidth; i++) {
            dstOffset = i * calcHigh * cubeNum;
            srcOffset = i * c0Size;
            Muls(dst[dstOffset], src[srcOffset], (DTYPE)1, mask, 2 * calcHigh, intriParams); // 每个calcHigh repeat 2次
        }
    }
}

template <typename DTYPE>
__aicore__ inline void CopyND2NZ(AscendC::LocalTensor<DTYPE> &dst, const AscendC::GlobalTensor<DTYPE> &src,
                                 AscendC::LocalTensor<DTYPE> &transTensor, const int row, const int col,
                                 const int height, const int width, const int gCol)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    auto srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
    bool isBankConflict = false;

    int calcHigh = DivUp(height, BLOCK_NUM_PER_FRACTAL);
    int calcWidth = CopyNDBlock(transTensor, src, srcOffset, height, width, gCol, isBankConflict);
    int padWidth = isBankConflict ? calcWidth + 1 : calcWidth;
    int size = calcHigh * padWidth * BLOCK_NUM_PER_FRACTAL * c0Size;

    transTensor.SetSize(size);
    dst.SetSize(size);
    NDPadZeros(transTensor, height, padWidth, width);
    NDTrans2NZ(dst, transTensor, calcHigh, calcWidth, isBankConflict);
}

template <typename DTYPE>
__aicore__ inline void CopyND2NZOnTheFly(const AscendC::LocalTensor<DTYPE> &dst,
                                         const AscendC::GlobalTensor<DTYPE> &src, const int row, const int col,
                                         const int height, const int width, const int gCol)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    int calcWidth = width / c0Size;
    int64_t dstOffset = 0;
    int64_t srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
    int calcWidthExr = DivUp(width, c0Size);
    int calcHeightExr = DivUp(height, BLOCK_NUM_PER_FRACTAL);
    if (height % BLOCK_NUM_PER_FRACTAL != 0) {
        int64_t repeat = calcWidthExr * calcHeightExr;
        AscendC::LocalTensor<int16_t> tmp = dst.template ReinterpretCast<int16_t>();
        AscendC::InitConstValueParams<int16_t> initConstValueParams;
        initConstValueParams.repeatTimes = (uint16_t)repeat;
        initConstValueParams.initValue = 0;
        InitConstValue(tmp, initConstValueParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    int srcGap = gCol * sizeof(DTYPE) / BLOCK_SIZE_32 - 1;
    if (gCol % c0Size != 0 || srcGap >= UINT16_MAX) {
        int64_t oriSrcOffset = srcOffset;
        int64_t oriDstOffset = dstOffset;
        for (int i = 0; i < calcWidth; i++) {
            for (int j = 0; j < height; j++) {
                AscendC::DataCopy(dst[dstOffset], src[srcOffset], {1, 1, 0, 0});
                dstOffset += c0Size;
                srcOffset += gCol;
            }
            srcOffset = oriSrcOffset + (i + 1) * c0Size;
            dstOffset = oriDstOffset + (i + 1) * calcHeightExr * BLOCK_NUM_PER_FRACTAL * c0Size;
        }
    } else {
        for (int i = 0; i < calcWidth; i++) {
            AscendC::DataCopy(dst[dstOffset], src[srcOffset],
                              {static_cast<uint16_t>(height), 1, static_cast<uint16_t>(srcGap), 0});
            dstOffset += calcHeightExr * BLOCK_NUM_PER_FRACTAL * c0Size;
            srcOffset += c0Size;
        }
    }
    event_t eventIDMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE1));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventIDMte2ToMte1);
    event_t eventIDMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE1_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventIDMte1ToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventIDMte1ToMte2);
}

template <typename DTYPE>
__aicore__ inline void TransDataBMatrix(const AscendC::LocalTensor<DTYPE> &dst, const AscendC::LocalTensor<DTYPE> &src,
                                        int height, int width)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    int iterK = DivUp(height, c0Size);
    int iterN = DivUp(width, c0Size);
    int calcWidth = iterN * c0Size;
    int tailWidth = (width % c0Size) > CONST_16 ? 0 : width % CONST_16;
    AscendC::TransDataTo5HDParams params;
    params.repeatTimes = iterK;
    params.dstRepStride = (iterK == 1) ? 0 : calcWidth;
    params.srcRepStride = (iterK == 1) ? 0 : calcWidth;
    int dstHighHalfOffset = CONST_16 * c0Size;
    int srcHighHalfOffset = CONST_16 * calcWidth;
    iterN = (tailWidth > 0) ? iterN - 1 : iterN;
    uint64_t dstLocalList[CONST_16];
    uint64_t srcLocalList[CONST_16];
    int dstOffset = 0;
    int srcOffset = 0;
    for (int curN = 0; curN < iterN; curN++) {
        int dstListOffset = 0;
        int srcListOffset = 0;
        for (int i = 0; i < CONST_16; i++) {
            dstLocalList[i] = (uint64_t)(dst[dstOffset + dstListOffset].GetPhyAddr());
            srcLocalList[i] = (uint64_t)(src[srcOffset + srcListOffset].GetPhyAddr());
            dstListOffset += c0Size;
            srcListOffset += calcWidth;
        }
        params.dstHighHalf = false;
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<DTYPE>(dstLocalList, srcLocalList, params);
        AscendC::PipeBarrier<PIPE_V>();
        srcListOffset = 0;
        for (int i = 0; i < CONST_16; i++) {
            srcLocalList[i] = (uint64_t)(src[srcOffset + srcListOffset + srcHighHalfOffset].GetPhyAddr());
            srcListOffset += calcWidth;
        }
        params.dstHighHalf = true;
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<DTYPE>(dstLocalList, srcLocalList, params);
        AscendC::PipeBarrier<PIPE_V>();
        dstListOffset = 0;
        srcListOffset = 0;
        for (int i = 0; i < CONST_16; i++) {
            dstLocalList[i] = (uint64_t)(dst[dstOffset + dstListOffset + dstHighHalfOffset].GetPhyAddr());
            srcLocalList[i] = (uint64_t)(src[srcOffset + srcListOffset].GetPhyAddr());
            dstListOffset += c0Size;
            srcListOffset += calcWidth;
        }
        params.dstHighHalf = false;
        params.srcHighHalf = true;
        AscendC::TransDataTo5HD<DTYPE>(dstLocalList, srcLocalList, params);
        AscendC::PipeBarrier<PIPE_V>();
        srcListOffset = 0;
        for (int i = 0; i < CONST_16; i++) {
            srcLocalList[i] = (uint64_t)(src[srcOffset + srcListOffset + srcHighHalfOffset].GetPhyAddr());
            srcListOffset += calcWidth;
        }
        params.dstHighHalf = true;
        params.srcHighHalf = true;
        AscendC::TransDataTo5HD<DTYPE>(dstLocalList, srcLocalList, params);
        AscendC::PipeBarrier<PIPE_V>();
        dstOffset += c0Size * c0Size;
        srcOffset += c0Size;
    }
    if (tailWidth > 0) {
        dstOffset = iterN * c0Size * c0Size;
        srcOffset = iterN * c0Size;
        int dstListTailOffset = 0;
        int srcListTailOffset = 0;
        params.dstRepStride = iterK == 1 ? 0 : CONST_16;
        for (int i = 0; i < CONST_16; i++) {
            dstLocalList[i] = (uint64_t)(dst[dstOffset + dstListTailOffset].GetPhyAddr());
            srcLocalList[i] = (uint64_t)(src[srcOffset + srcListTailOffset].GetPhyAddr());
            dstListTailOffset += c0Size;
            srcListTailOffset += calcWidth;
        }
        params.dstHighHalf = false;
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<DTYPE>(dstLocalList, srcLocalList, params);
        AscendC::PipeBarrier<PIPE_V>();
        srcListTailOffset = 0;
        for (int i = 0; i < CONST_16; i++) {
            srcLocalList[i] = (uint64_t)(src[srcOffset + srcListTailOffset + srcHighHalfOffset].GetPhyAddr());
            srcListTailOffset += calcWidth;
        }
        params.dstHighHalf = true;
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<DTYPE>(dstLocalList, srcLocalList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename DTYPE>
__aicore__ inline void CopyNZ2NZ(const AscendC::LocalTensor<DTYPE> &dst, const AscendC::LocalTensor<DTYPE> &src,
                                 const int row, const int col, const int height, const int width, const int gRow)
{
    const int c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
    int srcOffset = row * c0Size + col * gRow;
    auto alignHeight = RoundUp(height, BLOCK_NUM_PER_FRACTAL);
    int blockLen = alignHeight * c0Size * sizeof(DTYPE) / BLOCK_SIZE_32;
    int srcStride = (gRow - alignHeight) * (c0Size * sizeof(DTYPE) / BLOCK_SIZE_32);

    if (srcStride >= UINT16_MAX) {
        for (int i = 0; i < width / c0Size; i++) {
            AscendC::DataCopy(dst[i * alignHeight * c0Size], src[srcOffset + i * gRow * c0Size],
                              {1, static_cast<uint16_t>(blockLen), 0, 0});
        }
    } else {
        AscendC::DataCopy(dst, src[srcOffset],
                          {static_cast<uint16_t>(DivUp(width, c0Size)), static_cast<uint16_t>(blockLen),
                           static_cast<uint16_t>(srcStride), 0});
    }
}
#endif

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
 * \file swin_transformer_ln_qkv_quant_base.h
 * \brief
 */

#ifndef SWIN_TRANSFORMER_LN_QKV_QUANT_BASE_H
#define SWIN_TRANSFORMER_LN_QKV_QUANT_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"

constexpr int32_t BLOCK_UINT = 32;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_NUM_PER_FRACTAL = 16;
using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::CUBE_MAX_SIZE;
using AscendC::Muls;
using AscendC::Duplicate;
using AscendC::IsSameType;
using AscendC::BLOCK_CUBE;
using AscendC::PipeBarrier;
using AscendC::DEFAULT_BLK_NUM;
using AscendC::SetVectorMask;
using AscendC::UnaryRepeatParams;
__aicore__ inline uint32_t RoundUp(uint32_t num, uint32_t align)
{
    if (align == 0) {
        return 0;
    }
    return (num + align - 1) / align * align;
}

__aicore__ inline uint32_t DivUp(uint32_t num, uint32_t align)
{
    if (align == 0) {
        return 0;
    }
    return (num + align - 1) / align;
}

template <typename aDType, typename bDType, typename cDType, bool aTrans, bool bTrans, bool isReuseSource = false>
class SwinTransformerLnQkvQuantBase {
public:
    __aicore__ inline SwinTransformerLnQkvQuantBase(){};
    __aicore__ inline void CopyND2NZ(LocalTensor<int8_t> &dst, const GlobalTensor<int8_t> &src,
                                     LocalTensor<int8_t> &transTensor, const int row, const int col, const int height,
                                     const int width, const int gCol);
    __aicore__ inline void NDPadZeros(LocalTensor<bDType> &dst, const int height, const int calcWidth, const int gCol,
                                      const int width, bool isBankConflict);
    __aicore__ inline void NDTrans2NZ(LocalTensor<bDType> &dst, LocalTensor<bDType> &src, const int calcHigh,
                                      const int calcWidth, const bool isBankConflict);
    __aicore__ inline void VecND2NZ(LocalTensor<int8_t> &dst, LocalTensor<int8_t> &src, const int height,
                                    const int width, const int gCol);
};

template <typename aDType, typename bDType, typename cDType, bool aTrans, bool bTrans, bool isReuseSource>
__aicore__ inline void SwinTransformerLnQkvQuantBase<aDType, bDType, cDType, aTrans, bTrans, isReuseSource>::CopyND2NZ(
    LocalTensor<int8_t> &dst, const GlobalTensor<int8_t> &src, LocalTensor<int8_t> &transTensor, const int row,
    const int col, const int height, const int width, const int gCol)
{
    auto srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
    bool isBankConflict = false;
    int calcHigh = DivUp(height, BLOCK_NUM_PER_FRACTAL);
    int calcWidth = DivUp(width, BLOCK_SIZE_32);
    int padWidth = isBankConflict ? calcWidth + 1 : calcWidth;
    int size = calcHigh * padWidth * BLOCK_NUM_PER_FRACTAL * BLOCK_SIZE_16;

    transTensor.SetSize(size);
    (const_cast<LocalTensor<int8_t> &>(dst)).SetSize(size);
    NDPadZeros(transTensor, height, padWidth, gCol, width, isBankConflict);
    PipeBarrier<PIPE_V>();
    NDTrans2NZ(dst, transTensor, calcHigh, calcWidth, isBankConflict);
}

template <typename aDType, typename bDType, typename cDType, bool aTrans, bool bTrans, bool isReuseSource>
__aicore__ inline void SwinTransformerLnQkvQuantBase<aDType, bDType, cDType, aTrans, bTrans, isReuseSource>::VecND2NZ(
    LocalTensor<int8_t> &dst, LocalTensor<int8_t> &src, const int height, const int width, const int gCol)
{
    int calcHigh = DivUp(height, BLOCK_NUM_PER_FRACTAL);
    int calcWidth = DivUp(width, BLOCK_SIZE_32);
    int padWidth = calcWidth;
    int size = calcHigh * padWidth * BLOCK_NUM_PER_FRACTAL * BLOCK_SIZE_32;
    src.SetSize(size);
    (const_cast<LocalTensor<int8_t> &>(dst)).SetSize(size);
    NDPadZeros(src, height, padWidth, gCol, width, false);
    NDTrans2NZ(dst, src, calcHigh, calcWidth, false);
}

template <typename aDType, typename bDType, typename cDType, bool aTrans, bool bTrans, bool isReuseSource>
__aicore__ inline void SwinTransformerLnQkvQuantBase<aDType, bDType, cDType, aTrans, bTrans, isReuseSource>::NDPadZeros(
    LocalTensor<bDType> &dst, const int height, const int calcWidth, const int gCol, const int width,
    bool isBankConflict)
{
    const int C0_SIZE = BLOCK_SIZE_32 / sizeof(int8_t);
    if (gCol % BLOCK_NUM_PER_FRACTAL) {
        int tail = width % C0_SIZE;
        if (tail) {
            auto offset = width / C0_SIZE * C0_SIZE;
            uint64_t mask[2];
            if constexpr (IsSameType<bDType, int8_t>::value) {
                tail = DivUp(tail, 2);
                offset /= 2;
            }
            uint16_t mask_tail = ~((1 << tail) - 1);
            uint64_t masktail = mask_tail;
            mask[0] = masktail + (masktail << 16) + (masktail << 32) + (masktail << 48);
            mask[1] = mask[0];
            int stride = calcWidth * (C0_SIZE * sizeof(bDType) / BLOCK_SIZE_32);
            if (masktail != 0) {
                if constexpr (IsSameType<bDType, int8_t>::value) {
                    LocalTensor<int16_t> tmpTrnasTensor = dst.template ReinterpretCast<int16_t>();
                    if (stride < 32) {
                        Duplicate(tmpTrnasTensor[offset], (int16_t)0, mask, DivUp(height, 8), stride, 8 * stride);
                    } else {
                        for (int32_t i = 0; i < DivUp(height, 8); ++i) {
                            Duplicate(tmpTrnasTensor[offset], (int16_t)0, mask, 1, stride, 0);
                            offset += stride * BLOCK_NUM_PER_FRACTAL;
                        }
                    }
                } else {
                    Duplicate(dst[offset], (bDType)0, mask, DivUp(height, 8), stride, 8 * stride);
                }
                PipeBarrier<PIPE_V>();
            }
        }
    }
    int tailHigh = height % BLOCK_NUM_PER_FRACTAL;
    if (tailHigh) {
        auto dstOffset = height * calcWidth * BLOCK_CUBE;
        if constexpr (IsSameType<bDType, int8_t>::value) {
            LocalTensor<int16_t> tmpDst = dst.template ReinterpretCast<int16_t>();
            Duplicate(tmpDst[dstOffset], (int16_t)0,
                      (BLOCK_NUM_PER_FRACTAL - tailHigh) * calcWidth * BLOCK_NUM_PER_FRACTAL);
        } else {
            Duplicate(dst[dstOffset], (bDType)0,
                      (BLOCK_NUM_PER_FRACTAL - tailHigh) * calcWidth * BLOCK_NUM_PER_FRACTAL);
        }
    }
}

template <typename aDType, typename bDType, typename cDType, bool aTrans, bool bTrans, bool isReuseSource>
__aicore__ inline void SwinTransformerLnQkvQuantBase<aDType, bDType, cDType, aTrans, bTrans, isReuseSource>::NDTrans2NZ(
    LocalTensor<bDType> &dst, LocalTensor<bDType> &src, const int calcHigh, const int calcWidth,
    const bool isBankConflict)
{
    const int C0_SIZE = BLOCK_SIZE_32 / sizeof(bDType);
    if constexpr (IsSameType<bDType, int8_t>::value) {
        struct UnaryRepeatParams intriParams;
        uint64_t mask[2] = {uint64_t(-1), uint64_t(-1)};
        int blkStride = isBankConflict ? calcWidth + 1 : calcWidth;
        intriParams.dstBlkStride = 1;
        intriParams.srcBlkStride = blkStride;
        intriParams.dstRepStride = intriParams.dstBlkStride * DEFAULT_BLK_NUM;
        intriParams.srcRepStride = intriParams.srcBlkStride * DEFAULT_BLK_NUM;
        int dstOffset = 0;
        int srcOffset = 0;
        constexpr int maxSrcBlkStride = BLOCK_SIZE_32;
        LocalTensor<int16_t> tmpSrc = src.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> tmpDst = dst.template ReinterpretCast<int16_t>();
        if (intriParams.srcBlkStride >= maxSrcBlkStride) {
            intriParams.dstBlkStride = 1;
            intriParams.srcBlkStride = 1;
            mask[0] = (1 << BLOCK_NUM_PER_FRACTAL) - 1;
            mask[1] = 0;
            SetVectorMask<int16_t>(mask[1], mask[0]);
            for (int i = 0; i < calcWidth; i++) {
                for (int j = 0; j < calcHigh * BLOCK_NUM_PER_FRACTAL; ++j) {
                    dstOffset = i * calcHigh * CUBE_MAX_SIZE + j * BLOCK_NUM_PER_FRACTAL;
                    srcOffset = j * blkStride * BLOCK_NUM_PER_FRACTAL + i * BLOCK_NUM_PER_FRACTAL;
                    Muls<int16_t, false>(tmpDst[dstOffset], tmpSrc[srcOffset], (int16_t)1, mask, 1, intriParams);
                }
            }
        } else {
            SetVectorMask<int16_t>(mask[1], mask[0]);
            for (int i = 0; i < calcWidth; i++) {
                dstOffset = i * calcHigh * CUBE_MAX_SIZE;
                srcOffset = i * BLOCK_NUM_PER_FRACTAL;
                Muls<int16_t, false>(tmpDst[dstOffset], tmpSrc[srcOffset], (int16_t)1, mask, 2 * calcHigh, intriParams);
            }
        }
    }
}

#endif // SWIN_TRANSFORMER_LN_QKV_QUANT_BASE_H

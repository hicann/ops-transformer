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
 * \file fa_block_vec.h
 * \brief Vector-side compute — static tensor, no TPipe, template block size.
 */
#ifndef FA_BLOCK_VEC_H
#define FA_BLOCK_VEC_H

#ifndef __DAV_C310_CUBE__

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "adv_api/activation/softmax.h"
#include "../vector/vf/vf_softmax.h"
#include "../vector/vf/vf_rescale.h"
#include "../fa_kernel_public.h"
#include "../memcopy/memory_copy.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;

static constexpr uint32_t BLOCK_BYTES = 32;
static constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;

static constexpr uint32_t UB_SOFTMAX_BUF   = 256;
static constexpr uint32_t UB_SOFTMAX_ELEMS = UB_SOFTMAX_BUF / sizeof(float);
static constexpr uint32_t UB_COMMON_BUF    = 512;
static constexpr uint32_t UB_COMMON_ELEMS  = UB_COMMON_BUF / sizeof(uint8_t);

static constexpr uint32_t FLAG_BMM1_READY  = 0;
static constexpr uint32_t FLAG_L1P_READY   = 1;
static constexpr uint32_t FLAG_BMM2_READY  = 2;
static constexpr uint32_t E_VEC1_ID        = 0;
static constexpr uint32_t E_VEC2_ID        = 1;

using IN_T  = bfloat16_t;
using T     = float;
using OUT_T = bfloat16_t;

// ---- ComputeSoftmax: softmax computation ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeSoftmax(
    LocalTensor<float> &bmm1Ub,
    LocalTensor<IN_T>  &stage1Out,
    LocalTensor<float> &sumBuf,
    LocalTensor<float> &maxBuf,
    LocalTensor<float> &expBuf,
    LocalTensor<uint8_t> &commonBuf,
    bool isFirstS2Loop, bool isUpdate,
    float softmaxScale, float negMin,
    uint32_t mActualHalf, uint32_t nActual)
{
    CrossCoreWaitFlag<4, PIPE_V>(FLAG_BMM1_READY);

    LocalTensor<float> sumU = sumBuf;
    LocalTensor<float> maxU = maxBuf;
    LocalTensor<float> expU = expBuf;
    LocalTensor<uint8_t> tmp = commonBuf;
    LocalTensor<uint8_t> dummyMask;

    Mutex::Lock<PIPE_V>(E_VEC1_ID);
    if (!isUpdate) {
        ProcessVec1Vf<float, IN_T, false, M_BASE, N_BASE, EQ_128, false>(
            stage1Out, nullptr, sumU, maxU, bmm1Ub, expU, sumU, maxU, dummyMask,
            tmp, mActualHalf, nActual, softmaxScale, negMin);
    } else {
        ProcessVec1Vf<float, IN_T, true, M_BASE, N_BASE, EQ_128, false>(
            stage1Out, nullptr, sumU, maxU, bmm1Ub, expU, sumU, maxU, dummyMask,
            tmp, mActualHalf, nActual, softmaxScale, negMin);
    }

    if (!isFirstS2Loop) {
        UpdateExpSumAndExpMax<float>(sumU, maxU, expU, sumU, maxU, tmp, mActualHalf);
    }
    Mutex::Unlock<PIPE_V>(E_VEC1_ID);
}

// ---- CopySoftmaxToL1: copy P to L1 ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void CopySoftmaxToL1(
    LocalTensor<IN_T> &stage1Out,
    LocalTensor<IN_T> &l1P,
    uint32_t subBlockIdx, uint32_t mActual, uint32_t nActual)
{
    Mutex::Lock<PIPE_MTE3>(E_VEC1_ID);
    constexpr uint32_t halfM = M_BASE / 2;
    uint32_t nRound = (nActual + 15) / 16;
    DataCopy(l1P[subBlockIdx * (BLOCK_BYTES / sizeof(IN_T)) * ((uint32_t)M_BASE - halfM)],
             stage1Out,
             {static_cast<uint16_t>(nRound), static_cast<uint16_t>(mActual / 2),
              static_cast<uint16_t>(M_BASE / 2 + 1 - halfM),
              static_cast<uint16_t>(M_BASE - halfM)});
    Mutex::Unlock<PIPE_MTE3>(E_VEC1_ID);

    CrossCoreSetFlag<4, PIPE_V>(FLAG_BMM1_READY);
    CrossCoreSetFlag<4, PIPE_MTE3>(FLAG_L1P_READY);
}

// ---- ComputeRescale: flash update + last div + cast ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void ComputeRescale(
    LocalTensor<float> &bmm2Ub,
    LocalTensor<float> &stage2Buf,
    LocalTensor<float> &sumBuf,
    LocalTensor<float> &expBuf,
    bool isFirstS2Loop, bool isLastS2Loop,
    uint32_t mActual)
{
    CrossCoreWaitFlag<4, PIPE_V>(FLAG_BMM2_READY);
    uint32_t halfM = mActual / 2;

    if (isFirstS2Loop) {
        Mutex::Lock<PIPE_V>(E_VEC2_ID);
        DataCopy(stage2Buf, bmm2Ub, halfM * (uint32_t)D_SIZE);
    } else {
        if (!isLastS2Loop) {
            FlashUpdate<float, IN_T, OUT_T, D_SIZE, false>(
                stage2Buf, bmm2Ub, stage2Buf, expBuf, stage2Buf,
                halfM, (uint32_t)D_SIZE, 1.0f, 1.0f);
        } else {
            FlashUpdateLast<float, IN_T, OUT_T, D_SIZE, false>(
                stage2Buf, bmm2Ub, stage2Buf, expBuf, stage2Buf, sumBuf,
                halfM, (uint32_t)D_SIZE, 1.0f, 1.0f);
        }
    }
    CrossCoreSetFlag<4, PIPE_V>(FLAG_BMM2_READY);
}

// ---- CopyAttnOutToGm: output copy to GM ----
template <uint32_t M_BASE, uint32_t N_BASE, uint32_t D_SIZE>
__aicore__ inline void CopyAttnOutToGm(
    LocalTensor<float> &stage2Buf, LocalTensor<OUT_T> &castResult,
    FaGmTensor<OUT_T, GmFormat::BSND> &outGmTensor,
    LocalTensor<float> &sumBuf,
    uint32_t bIdx, uint32_t n1Head, uint32_t s1BlockBase,
    uint32_t subBlockIdx, bool isFirstS2Loop, uint32_t mActual)
{
    uint32_t halfM = mActual / 2;
    if (isFirstS2Loop) {
        LastDiv<float, IN_T, OUT_T, (uint32_t)D_SIZE>(stage2Buf, stage2Buf, sumBuf, halfM, (uint32_t)D_SIZE, 0.0f);
    }

    Cast(castResult, stage2Buf, RoundMode::CAST_ROUND, halfM * (uint32_t)D_SIZE);
    Mutex::Unlock<PIPE_V>(E_VEC2_ID);
    Mutex::Lock<PIPE_MTE3>(E_VEC2_ID);

    FaUbTensor<OUT_T> ubTensor{.tensor = castResult,
                               .rowCount = halfM,
                               .colCount = (uint32_t)D_SIZE};
    CopyAttnOutBN1UbToGm<OUT_T>(outGmTensor, ubTensor, bIdx, n1Head,
                                s1BlockBase + subBlockIdx * halfM);

    Mutex::Unlock<PIPE_MTE3>(E_VEC2_ID);
}

// ---- Main Vector entry ----
template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__aicore__ inline void VectorFunc(__gm__ uint8_t *query, __gm__ uint8_t *key,
                              __gm__ uint8_t *value, __gm__ uint8_t *attentionOut,
                              __gm__ uint8_t *workspace, __gm__ uint8_t *tiling,
                              uint32_t blockDim, float softmaxScale,
                              uint32_t B, uint32_t N1, uint32_t N2, uint32_t S1, uint32_t S2)
{
    static constexpr uint32_t UB_BMM1_ELEMS   = M_BASE / 2 * N_BASE;
    static constexpr uint32_t UB_BMM2_ELEMS   = M_BASE / 2 * D_SIZE;
    static constexpr uint32_t UB_STAGE1_ELEMS = (M_BASE / 2 + 2) * N_BASE;
    static constexpr uint32_t UB_STAGE2_ELEMS = M_BASE / 2 * D_SIZE;
    static constexpr uint32_t L1_P_ELEMS      = M_BASE * N_BASE;

    uint32_t bn1Total = B * N1;
    uint32_t bn1PerCore = (bn1Total + blockDim - 1) / blockDim;
    uint32_t aivIdx = GetBlockIdx();
    uint32_t aicIdx = aivIdx / 2;
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t bn1Start = aicIdx * bn1PerCore;
    uint32_t bn1End = bn1Start + bn1PerCore;
    if (bn1End > bn1Total) { bn1End = bn1Total; }

    uint32_t numS1Blocks = (S1 + (uint32_t)M_BASE - 1) / (uint32_t)M_BASE;
    uint32_t numS2Blocks = (S2 + (uint32_t)N_BASE - 1) / (uint32_t)N_BASE;

    // === GM output tensor setup ===
    FaGmTensor<OUT_T, GmFormat::BSND> outGmTensor((__gm__ OUT_T *)attentionOut, B, N1, S1, (uint32_t)D_SIZE);

    // === UB buffers (shared — must match Cube addresses) ===
    uint32_t ubAddr = 0;
    LocalTensor<float> ubBmm1 = LocalTensor<float>(TPosition::VECIN, ubAddr, UB_BMM1_ELEMS);
    ubAddr += UB_BMM1_ELEMS * sizeof(float);
    LocalTensor<float> ubBmm2 = LocalTensor<float>(TPosition::VECIN, ubAddr, UB_BMM2_ELEMS);

    // === L1 P buffers (shared — must match Cube addresses) ===
    uint32_t l1Addr = 0;
    LocalTensor<IN_T> l1P = LocalTensor<IN_T>(TPosition::A1, l1Addr, L1_P_ELEMS);

    // === UB AIV-private buffers ===
    LocalTensor<float> softmaxSum(TPosition::VECIN, ubAddr += UB_BMM2_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS);
    LocalTensor<float> softmaxMax(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS);
    LocalTensor<float> softmaxExp(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_SOFTMAX_ELEMS);
    LocalTensor<uint8_t> commonBuf(TPosition::VECIN, ubAddr += UB_SOFTMAX_ELEMS * sizeof(float), UB_COMMON_ELEMS);
    LocalTensor<IN_T> stage1Buf(TPosition::VECIN, ubAddr += UB_COMMON_ELEMS * sizeof(uint8_t), UB_STAGE1_ELEMS);
    LocalTensor<float> stage2Buf(TPosition::VECIN, ubAddr += UB_STAGE1_ELEMS * sizeof(IN_T), UB_STAGE2_ELEMS);
    LocalTensor<IN_T> castBuf(TPosition::VECIN, ubAddr, UB_STAGE2_ELEMS);

    // === Cross-core init ===
    CrossCoreSetFlag<4, PIPE_V>(FLAG_BMM1_READY);
    CrossCoreSetFlag<4, PIPE_V>(FLAG_BMM2_READY);

    // === Main loop ===
    uint32_t tmpNeg = NEGATIVE_MIN_VALUE_FP32;
    float negMin = *((float *)&tmpNeg);

    for (uint32_t bn1 = bn1Start; bn1 < bn1End; bn1++) {
        uint32_t bIdx   = bn1 / N1;
        uint32_t n1Head = bn1 % N1;

        for (uint32_t s1Block = 0; s1Block < numS1Blocks; s1Block++) {
            uint32_t s1BlockBase = s1Block * M_BASE;
            uint32_t mActual = s1BlockBase + M_BASE > S1 ? S1 - s1BlockBase : (uint32_t)M_BASE;

            for (uint32_t s2Block = 0; s2Block < numS2Blocks; s2Block++) {
                uint32_t s2BlockBase = s2Block * N_BASE;
                uint32_t nActual = s2BlockBase + N_BASE > S2 ? S2 - s2BlockBase : (uint32_t)N_BASE;
                bool isFirstS2Loop = (s2Block == 0);
                bool isLastS2Loop = (s2Block + 1 == numS2Blocks);

                ComputeSoftmax<M_BASE, N_BASE, D_SIZE>(ubBmm1, stage1Buf,
                               softmaxSum, softmaxMax, softmaxExp, commonBuf,
                               isFirstS2Loop, !isFirstS2Loop,
                               softmaxScale, negMin, mActual / 2, nActual);

                CopySoftmaxToL1<M_BASE, N_BASE, D_SIZE>(stage1Buf, l1P,
                                subBlockIdx, mActual, nActual);

                ComputeRescale<M_BASE, N_BASE, D_SIZE>(ubBmm2, stage2Buf,
                               softmaxSum, softmaxExp, isFirstS2Loop, isLastS2Loop, mActual);

                if (isLastS2Loop) {
                    CopyAttnOutToGm<M_BASE, N_BASE, D_SIZE>(stage2Buf, castBuf, outGmTensor,
                                    softmaxSum, bIdx, n1Head, s1BlockBase,
                                    subBlockIdx, isFirstS2Loop, mActual);
                }
            }
        }
    }
}
#endif // __DAV_C310_CUBE__
#endif // FA_BLOCK_VEC_H
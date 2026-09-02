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
 * \file rotate_half_rope_fullLoad_xd.h
 * \brief
 */

#ifndef ROTATE_HALF_ROPE_FULLLOAD_XD_H
#define ROTATE_HALF_ROPE_FULLLOAD_XD_H

#include "rotate_half_base.h"

namespace RotateHalfN {
using namespace AscendC;

template <typename OriT, typename CmpT>
class RotateHalfRopeFullLoadXd : public RotateHalfBase<OriT, CmpT> {
public:
    __aicore__ inline RotateHalfRopeFullLoadXd(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                const RotaryPositionEmbeddingTilingData &tilingData);
    __aicore__ inline void Process();

private:
    // ---- Helper methods ----
    __aicore__ inline void CopyInRope(LocalTensor<CmpT> &cosFp32Out, LocalTensor<CmpT> &sinFp32Out);
    __aicore__ inline void CopyInX();
    __aicore__ inline void BroadcastMul(LocalTensor<CmpT> &xFp32, LocalTensor<CmpT> &rotL, LocalTensor<CmpT> &cosFp32,
                                        LocalTensor<CmpT> &sinFp32);
    __aicore__ inline void CopyOut(LocalTensor<CmpT> &result);

    // ---- Data members (FullLoadXD-specific, base class provides the rest) ----
    TPipe pipe;

    GlobalTensor<OriT> xGm;
    GlobalTensor<OriT> cosGm;
    GlobalTensor<OriT> sinGm;
    GlobalTensor<OriT> yGm;

    TQue<QuePosition::VECIN, SINGLE_BUFFER> inQueueX;
    TQue<QuePosition::VECIN, SINGLE_BUFFER> inQueueCos;
    TQue<QuePosition::VECIN, SINGLE_BUFFER> inQueueSin;
    TQue<QuePosition::VECOUT, SINGLE_BUFFER> outQueueY;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> cosBufFp32;
    TBuf<TPosition::VECCALC> sinBufFp32;
};

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfRopeFullLoadXd<OriT, CmpT>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                  const RotaryPositionEmbeddingTilingData &tilingData)
{
    this->FullLoadXDInit(tilingData);

    xGm.SetGlobalBuffer((__gm__ OriT *)x + this->xGmBase);
    yGm.SetGlobalBuffer((__gm__ OriT *)y + this->xGmBase);
    cosGm.SetGlobalBuffer((__gm__ OriT *)cos + this->rGmBase);
    sinGm.SetGlobalBuffer((__gm__ OriT *)sin + this->rGmBase);

    // LocalTensors aligned to 32 bytes
    if constexpr (!IsSameType<OriT, CmpT>::value) {
        // MTE2 staging queues only needed when Cast between OriT and CmpT
        pipe.InitBuffer(inQueueX, SINGLE_BUFFER, this->xCountDPad * sizeof(OriT));
        pipe.InitBuffer(inQueueCos, SINGLE_BUFFER, this->rCountDPad * sizeof(OriT));
        pipe.InitBuffer(inQueueSin, SINGLE_BUFFER, this->rCountDPad * sizeof(OriT));
    }
    // outQueueY holds rotL (swap(x)) and final output
    pipe.InitBuffer(outQueueY, SINGLE_BUFFER, this->xCountDPad * sizeof(CmpT));

    pipe.InitBuffer(xBufFp32, this->xCountDPad * sizeof(CmpT));

    // Broadcast optimization: when isThreeOneDim && D > 64, pre-broadcast cos/sin
    // to axisX3 rows so BroadcastMul can use element-wise Mul instead of per-line loop.
    uint64_t ropeBufSize;
    if (this->isThreeOneDim && this->dPadLength > MASK_MAX) {
        ropeBufSize = this->axisLenX3 * this->dPadLength * sizeof(CmpT);
    } else {
        ropeBufSize = this->rCountDPad * sizeof(CmpT);
    }
    pipe.InitBuffer(cosBufFp32, ropeBufSize);
    pipe.InitBuffer(sinBufFp32, ropeBufSize);
}

// ============================================================================
// CopyInRope: load cos/sin → cosBufFp32 / sinBufFp32
// ============================================================================
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfRopeFullLoadXd<OriT, CmpT>::CopyInRope(LocalTensor<CmpT> &cosFp32Out,
                                                                        LocalTensor<CmpT> &sinFp32Out)
{
    if constexpr (IsSameType<OriT, CmpT>::value) {
        // FP32 path — DataCopy directly to TBuf, then wait MTE2→V
        if (this->isAligned) {
            DataCopy(cosFp32Out, cosGm[0], static_cast<uint32_t>(this->rCount));
            DataCopy(sinFp32Out, sinGm[0], static_cast<uint32_t>(this->rCount));
        } else {
            DataCopyExtParams copyP{static_cast<uint16_t>(2 * this->ropeEleNum), this->halfDBytes, 0, 0, 0};
            DataCopyPad(cosFp32Out, cosGm[0], copyP, this->noPadParams);
            DataCopyPad(sinFp32Out, sinGm[0], copyP, this->noPadParams);
        }
    } else {
        LocalTensor<OriT> cosLocal = inQueueCos.AllocTensor<OriT>();
        LocalTensor<OriT> sinLocal = inQueueSin.AllocTensor<OriT>();
        if (this->isAligned) {
            DataCopy(cosLocal, cosGm[0], static_cast<uint32_t>(this->rCount));
            DataCopy(sinLocal, sinGm[0], static_cast<uint32_t>(this->rCount));
        } else {
            DataCopyExtParams copyP{static_cast<uint16_t>(2 * this->ropeEleNum), this->halfDBytes, 0, 0, 0};
            DataCopyPad(cosLocal, cosGm[0], copyP, this->noPadParams);
            DataCopyPad(sinLocal, sinGm[0], copyP, this->noPadParams);
        }
        inQueueCos.EnQue(cosLocal);
        inQueueSin.EnQue(sinLocal);

        LocalTensor<OriT> tempCos = inQueueCos.DeQue<OriT>();
        LocalTensor<OriT> tempSin = inQueueSin.DeQue<OriT>();
        Cast(cosFp32Out, tempCos, RoundMode::CAST_NONE, static_cast<uint32_t>(this->rCountDPad));
        Cast(sinFp32Out, tempSin, RoundMode::CAST_NONE, static_cast<uint32_t>(this->rCountDPad));
        inQueueCos.FreeTensor(tempCos);
        inQueueSin.FreeTensor(tempSin);
    }
}

// ============================================================================
// CopyInX: load x → xBufFp32
// ============================================================================
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfRopeFullLoadXd<OriT, CmpT>::CopyInX()
{
    LocalTensor<CmpT> xFp32 = xBufFp32.Get<CmpT>();

    if constexpr (IsSameType<OriT, CmpT>::value) {
        if (this->isAligned) {
            DataCopy(xFp32, xGm[0], static_cast<uint32_t>(this->xCount));
        } else {
            DataCopyExtParams copyP{static_cast<uint16_t>(2 * this->ubLoop * this->axisLenX3), this->halfDBytes, 0, 0,
                                    0};
            DataCopyPad(xFp32, xGm[0], copyP, this->noPadParams);
        }
    } else {
        LocalTensor<OriT> xLocal = inQueueX.AllocTensor<OriT>();
        if (this->isAligned) {
            DataCopy(xLocal, xGm[0], static_cast<uint32_t>(this->xCount));
        } else {
            DataCopyExtParams copyP{static_cast<uint16_t>(2 * this->ubLoop * this->axisLenX3), this->halfDBytes, 0, 0,
                                    0};
            DataCopyPad(xLocal, xGm[0], copyP, this->noPadParams);
        }
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<OriT>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(this->xCountDPad));
        inQueueX.FreeTensor(xLocal);
    }
}

// ============================================================================
// Perform broadcast multiplication: x *= cos and rotL *= sin
// ============================================================================
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfRopeFullLoadXd<OriT, CmpT>::BroadcastMul(LocalTensor<CmpT> &xFp32,
                                                                          LocalTensor<CmpT> &rotL,
                                                                          LocalTensor<CmpT> &cosFp32,
                                                                          LocalTensor<CmpT> &sinFp32)
{
    const uint64_t N = this->axisLenX3;
    const uint32_t count = static_cast<uint32_t>(this->dPadLength);
    const uint32_t ubLoop = static_cast<uint32_t>(this->ubLoop);

    if (N == 1 && !this->isThreeOneDim) {
        // Total rows match x, element-wise Mul handles all
        Mul(xFp32, xFp32, cosFp32, static_cast<uint32_t>(this->xCountDPad));
        Mul(rotL, rotL, sinFp32, static_cast<uint32_t>(this->xCountDPad));
        return;
    }

    const uint8_t repStride = static_cast<uint8_t>(this->dPadLength * sizeof(CmpT) / BYTE_OF_BLOCK);
    const BinaryRepeatParams repeatParams = {1, 1, 1, repStride, repStride, 0};

    if (count <= MASK_MAX) {
        if (N == 1) {
            // D <= 64: repeat broadcast, optimal performance
            uint32_t repeatLoop = ubLoop / REPEAT_MAX;
            uint8_t repeatLast = static_cast<uint8_t>(ubLoop % REPEAT_MAX);
            for (uint32_t i = 0; i < repeatLoop; i++) {
                uint64_t offset = static_cast<uint64_t>(i) * REPEAT_MAX * this->dPadLength;
                Mul(xFp32[offset], xFp32[offset], cosFp32, count, REPEAT_MAX, repeatParams);
                Mul(rotL[offset], rotL[offset], sinFp32, count, REPEAT_MAX, repeatParams);
            }
            if (repeatLast > 0) {
                uint64_t offset = static_cast<uint64_t>(repeatLoop) * REPEAT_MAX * this->dPadLength;
                Mul(xFp32[offset], xFp32[offset], cosFp32, count, repeatLast, repeatParams);
                Mul(rotL[offset], rotL[offset], sinFp32, count, repeatLast, repeatParams);
            }
        } else {
            // D <= 64: repeat unroll over N dimension
            for (uint32_t g = 0; g < ubLoop; g++) {
                const uint64_t gXOff = static_cast<uint64_t>(g) * N * this->dPadLength;
                const uint64_t gROff = static_cast<uint64_t>(this->isThreeOneDim ? NUM_ZERO : g) * this->dPadLength;
                uint64_t nRemain = N;
                uint64_t nOff = NUM_ZERO;
                while (nRemain > 0) {
                    uint64_t chunkN = (nRemain > REPEAT_MAX) ? REPEAT_MAX : nRemain;
                    uint64_t xChunkOff = gXOff + nOff * this->dPadLength;
                    Mul(xFp32[xChunkOff], xFp32[xChunkOff], cosFp32[gROff], count, static_cast<uint8_t>(chunkN),
                        repeatParams);
                    Mul(rotL[xChunkOff], rotL[xChunkOff], sinFp32[gROff], count, static_cast<uint8_t>(chunkN),
                        repeatParams);
                    nRemain -= chunkN;
                    nOff += chunkN;
                }
            }
        }
    } else {
        // D > 64: per-line loop over N dimension
        if (this->isThreeOneDim) {
            // cos/sin pre-broadcast to axisX3 rows, element-wise Mul over full group
            const uint32_t groupSize = static_cast<uint32_t>(N * count);
            for (uint32_t g = 0; g < ubLoop; g++) {
                const uint64_t groupOff = static_cast<uint64_t>(g) * N * count;
                Mul(xFp32[groupOff], xFp32[groupOff], cosFp32, groupSize);
                Mul(rotL[groupOff], rotL[groupOff], sinFp32, groupSize);
            }
        } else {
            for (uint32_t g = 0; g < ubLoop; g++) {
                const uint64_t rotBaseOff = static_cast<uint64_t>(g) * count;
                const uint64_t groupOff = static_cast<uint64_t>(g) * N * count;
                for (uint32_t i = 0; i < N; i++) {
                    const uint64_t xOff = groupOff + static_cast<uint64_t>(i) * count;
                    Mul(xFp32[xOff], xFp32[xOff], cosFp32[rotBaseOff], count);
                    Mul(rotL[xOff], rotL[xOff], sinFp32[rotBaseOff], count);
                }
            }
        }
    }
}

// ============================================================================
// CopyOut: result (CmpT) → Cast to OriT → GM
// ============================================================================
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfRopeFullLoadXd<OriT, CmpT>::CopyOut(LocalTensor<CmpT> &result)
{
    if constexpr (IsSameType<OriT, CmpT>::value) {
        // FP32 path — write directly from outQueueY
        LocalTensor<OriT> outLocal = result.template ReinterpretCast<OriT>();
        outQueueY.EnQue(result);
        result = outQueueY.DeQue<CmpT>();
        outLocal = result.template ReinterpretCast<OriT>();
        if (this->isAligned) {
            DataCopy(yGm[0], outLocal, static_cast<uint32_t>(this->xCount));
        } else {
            DataCopyExtParams copyP{static_cast<uint16_t>(2 * this->ubLoop * this->axisLenX3), this->halfDBytes, 0, 0,
                                    0};
            DataCopyPad(yGm[0], outLocal, copyP);
        }
        outQueueY.FreeTensor(result);
    } else {
        // BF16/FP16 path — Cast in-place then write
        LocalTensor<OriT> outOri = result.template ReinterpretCast<OriT>();
        Cast(outOri, result, RoundMode::CAST_RINT, static_cast<uint32_t>(this->xCountDPad));
        outQueueY.EnQue(result);
        result = outQueueY.DeQue<CmpT>();
        outOri = result.template ReinterpretCast<OriT>();
        if (this->isAligned) {
            DataCopy(yGm[0], outOri, static_cast<uint32_t>(this->xCount));
        } else {
            DataCopyExtParams copyP{static_cast<uint16_t>(2 * this->ubLoop * this->axisLenX3), this->halfDBytes, 0, 0,
                                    0};
            DataCopyPad(yGm[0], outOri, copyP);
        }
        outQueueY.FreeTensor(result);
    }
}

// ============================================================================
// Process: main computation flow
// ============================================================================
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfRopeFullLoadXd<OriT, CmpT>::Process()
{
    if (this->ubLoop == 0) {
        return;
    }

    // STEP 1: Load cos/sin → cosBufFp32 / sinBufFp32
    LocalTensor<CmpT> cosFp32 = cosBufFp32.Get<CmpT>();
    LocalTensor<CmpT> sinFp32 = sinBufFp32.Get<CmpT>();
    CopyInRope(cosFp32, sinFp32);

    // STEP 2: Load x → xBufFp32
    CopyInX();

    if constexpr (IsSameType<OriT, CmpT>::value) {
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    // STEP 3: sin_mod = [-sin[:HALF], sin[HALF:]]  (reuse base class SinCompute)
    this->SinCompute(sinFp32, static_cast<uint32_t>(this->ropeEleNum));

    // STEP 4: swap(x) → rotL  (reuse base class XNewCopy)
    LocalTensor<CmpT> xFp32 = xBufFp32.Get<CmpT>();
    LocalTensor<CmpT> rotL = outQueueY.AllocTensor<CmpT>();
    this->XNewCopy(xFp32, rotL, static_cast<uint16_t>(this->ubLoop * this->axisLenX3));

    // STEP 4.5: Pre-broadcast cos/sin from 1 row to axisX3 rows (isThreeOneDim + D > 64)
    if (this->isThreeOneDim && this->dPadLength > MASK_MAX) {
        uint32_t broadcastLines = static_cast<uint32_t>(this->axisLenX3 - 1);
        if (broadcastLines > 0) {
            this->RBroadCast(cosFp32, sinFp32, broadcastLines);
        }
    }

    // STEP 5: xFp32 *= cos, rotL *= sin_mod (broadcast over axisX3)
    BroadcastMul(xFp32, rotL, cosFp32, sinFp32);

    // STEP 6: y = x * cos + swap(x) * sin_mod
    Add(rotL, rotL, xFp32, static_cast<uint32_t>(this->xCountDPad));

    // STEP 7: Cast back to OriT (if needed) and CopyOut
    CopyOut(rotL);
}

} // namespace RotateHalfN
#endif // ROTATE_HALF_ROPE_FULLLOAD_XD_H

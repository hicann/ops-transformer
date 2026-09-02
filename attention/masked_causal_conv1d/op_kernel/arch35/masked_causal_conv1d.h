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
 * \file masked_causal_conv1d.h
 * \brief MaskedCausalConv1d AIV kernel implementation
 *
 * Shape: input [S,B,H] contiguous, weight [3,H], mask [B,S] -> output [S,B,H]
 * Strategy: H×B×S 3D inter-core split, H outer loop, B middle, S inner;
 *           unified ioQueue layout [bUb][sUb+2][ubFactorH]:
 *             rows 0,1        = prefix x[s0-2], x[s0-1]  (loaded fresh from GM each tile)
 *             rows 2..sUb+1   = current x[s0..s0+sUb-1]
 *             each row is ubFactorH elements wide (ubFactorH = k×H_REG, k >= 1)
 *           VF writes y[s] in-place to row s; CopyOut writes rows 0..sCur-1.
 */

#ifndef MASKED_CAUSAL_CONV1D_H
#define MASKED_CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "./vf/masked_conv1d_vf.h"
#include "masked_causal_conv1d_struct.h"

namespace MaskedCausalConv1dNs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    // double buffer for all queues
constexpr uint32_t ALIGN_BYTES = 32; // DataCopy 32-byte alignment unit
constexpr uint32_t PREFIX_NUM = 2;

// ============================================================================
// MaskedCausalConv1d Kernel
// ============================================================================
template <typename T>
class MaskedCausalConv1d {
public:
    __aicore__ inline MaskedCausalConv1d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR y,
                                const MaskedCausalConv1dTilingData *tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MaskedCausalConv1dTilingData *tiling);
    __aicore__ inline void CopyIn(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0,
                                  uint32_t ubFactorHCur);
    __aicore__ inline void Compute(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t ubFactorHCur,
                                   LocalTensor<T> &wBuf, int32_t eventIdToMTE2);
    __aicore__ inline void CopyOut(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0,
                                   uint32_t ubFactorHCur);

    // Helper: align up to align bytes
    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align)
    {
        return (n + align - 1) / align * align;
    }

private:
    // -------------------------------------------------------------------------
    // Tiling parameters
    // -------------------------------------------------------------------------
    uint32_t S_, B_, H_;

    // H-dim inter-core
    uint32_t hCoreCnt_, hMainCnt_, hBlockFactor_, hBlockTailFactor_;
    // B-dim inter-core
    uint32_t bCoreCnt_, bMainCnt_, bBlockFactor_, bBlockTailFactor_;
    // S-dim inter-core
    uint32_t sCoreCnt_, sMainCnt_, sBlockFactor_, sBlockTailFactor_;

    // UB tile
    uint32_t ubFactorH_; // = ubFactorH = k * H_REG
    uint32_t ubFactorB_;
    uint32_t ubFactorS_;

    // Main-core loop params
    uint32_t loopNumH_, ubTailFactorH_;
    uint32_t loopNumB_, ubTailFactorB_;
    uint32_t loopNumS_, ubTailFactorS_;

    // Tail-core loop params
    uint32_t tailBlockLoopNumH_, tailBlockUbTailFactorH_;
    uint32_t tailBlockLoopNumB_, tailBlockUbTailFactorB_;
    uint32_t tailBlockLoopNumS_, tailBlockUbTailFactorS_;

    uint32_t realCoreNum_;
    uint32_t isMaskNone_; // 1 = skip mask application; 0 = apply mask
    uint32_t xSStride_;   // = B_ * H_, pre-computed to avoid repeated multiply in CopyIn/CopyOut

    // Core 3D index (decoded from blockIdx)
    uint32_t hCoreIdx_;
    uint32_t bCoreIdx_;
    uint32_t sCoreIdx_;

    // Current core's assigned ranges
    uint32_t hStart_, hLen_;
    uint32_t bStart_, bLen_;
    uint32_t sStart_, sLen_;

    // Loop params selected for this core (main vs tail)
    uint32_t myLoopNumH_, myUbTailFactorH_;
    uint32_t myLoopNumB_, myUbTailFactorB_;
    uint32_t myLoopNumS_, myUbTailFactorS_;

    // -------------------------------------------------------------------------
    // Pipeline & Global Tensors
    // -------------------------------------------------------------------------
    TPipe pipe_;

    GlobalTensor<T> xGM_;          // [S][B][H] contiguous
    GlobalTensor<T> weightGM_;     // [3][H]
    GlobalTensor<uint8_t> maskGM_; // [B][S] bool (uint8_t)
    GlobalTensor<T> yGM_;          // [S][B][H] contiguous

    // -------------------------------------------------------------------------
    // UB buffers
    // ioQueue    : [bUb][sUb+2][ubFactorH] T, double buffer (ping-pong)
    //              rows 0,1 = prefix x[s0-2], x[s0-1]; rows 2..sUb+1 = x[s0..]
    //              each row: ubFactorH elements wide (ubFactorH = k×H_REG)
    // weightQueue: [3][ubFactorH] T, double buffer, loaded once per S-tile
    // maskRowQueue_: [bUb][sPad] — uint8 DMA landing (step1), then T mask values (step3), double buffer
    // maskCastQueue_: [bUb][sPad] half (step2) → reused as [sPad][ubFactorH] T Broadcast dst (step4), double buffer
    // -------------------------------------------------------------------------
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> ioQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;   // [3][ubFactorH]
    TQue<QuePosition::VECIN, 1> maskRowQueue_;  // double-buffered: uint8 DMA / T mask
    TQue<QuePosition::VECIN, 1> maskCastQueue_; // double-buffered: half Cast / T Broadcast dst
};

// ============================================================================
// ParseTilingData
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::ParseTilingData(const MaskedCausalConv1dTilingData *tiling)
{
    S_ = tiling->S;
    B_ = tiling->B;
    H_ = tiling->H;

    hCoreCnt_ = tiling->hCoreCnt;
    hMainCnt_ = tiling->hMainCnt;
    hBlockFactor_ = tiling->hBlockFactor;
    hBlockTailFactor_ = tiling->hBlockTailFactor;

    bCoreCnt_ = tiling->bCoreCnt;
    bMainCnt_ = tiling->bMainCnt;
    bBlockFactor_ = tiling->bBlockFactor;
    bBlockTailFactor_ = tiling->bBlockTailFactor;

    sCoreCnt_ = tiling->sCoreCnt;
    sMainCnt_ = tiling->sMainCnt;
    sBlockFactor_ = tiling->sBlockFactor;
    sBlockTailFactor_ = tiling->sBlockTailFactor;

    ubFactorH_ = tiling->ubFactorH; // k × H_REG, variable
    ubFactorB_ = tiling->ubFactorB;
    ubFactorS_ = tiling->ubFactorS;

    loopNumH_ = tiling->loopNumH;
    ubTailFactorH_ = tiling->ubTailFactorH;
    loopNumB_ = tiling->loopNumB;
    ubTailFactorB_ = tiling->ubTailFactorB;
    loopNumS_ = tiling->loopNumS;
    ubTailFactorS_ = tiling->ubTailFactorS;

    tailBlockLoopNumH_ = tiling->tailBlockLoopNumH;
    tailBlockUbTailFactorH_ = tiling->tailBlockUbTailFactorH;
    tailBlockLoopNumB_ = tiling->tailBlockLoopNumB;
    tailBlockUbTailFactorB_ = tiling->tailBlockUbTailFactorB;
    tailBlockLoopNumS_ = tiling->tailBlockLoopNumS;
    tailBlockUbTailFactorS_ = tiling->tailBlockUbTailFactorS;

    realCoreNum_ = tiling->realCoreNum;
    isMaskNone_ = tiling->isMaskNone;
    xSStride_ = B_ * H_; // pre-compute once
}

// ============================================================================
// Init
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR y,
                                                   const MaskedCausalConv1dTilingData *tiling)
{
    ParseTilingData(tiling);

    // Decode 3D core index: blockIdx = hCoreIdx + bCoreIdx*hCoreCnt + sCoreIdx*(hCoreCnt*bCoreCnt)
    uint32_t blockIdx = GetBlockIdx();

    hCoreIdx_ = blockIdx - (blockIdx / hCoreCnt_) * hCoreCnt_;
    bCoreIdx_ = (blockIdx / hCoreCnt_) - ((blockIdx / hCoreCnt_) / bCoreCnt_) * bCoreCnt_;
    sCoreIdx_ = blockIdx / (hCoreCnt_ * bCoreCnt_);

    // Compute this core's H range
    if (hCoreIdx_ < hMainCnt_) {
        hStart_ = hCoreIdx_ * hBlockFactor_;
        hLen_ = hBlockFactor_;
    } else {
        hStart_ = hMainCnt_ * hBlockFactor_ + (hCoreIdx_ - hMainCnt_) * hBlockTailFactor_;
        hLen_ = hBlockTailFactor_;
    }

    // Compute this core's B range
    if (bCoreIdx_ < bMainCnt_) {
        bStart_ = bCoreIdx_ * bBlockFactor_;
        bLen_ = bBlockFactor_;
    } else {
        bStart_ = bMainCnt_ * bBlockFactor_ + (bCoreIdx_ - bMainCnt_) * bBlockTailFactor_;
        bLen_ = bBlockTailFactor_;
    }

    // Compute this core's S range
    if (sCoreIdx_ < sMainCnt_) {
        sStart_ = sCoreIdx_ * sBlockFactor_;
        sLen_ = sBlockFactor_;
    } else {
        sStart_ = sMainCnt_ * sBlockFactor_ + (sCoreIdx_ - sMainCnt_) * sBlockTailFactor_;
        sLen_ = sBlockTailFactor_;
    }

    // Select loop params for this core
    bool isHTail = (hCoreIdx_ >= hMainCnt_);
    bool isBTail = (bCoreIdx_ >= bMainCnt_);
    bool isSTail = (sCoreIdx_ >= sMainCnt_);

    myLoopNumH_ = isHTail ? tailBlockLoopNumH_ : loopNumH_;
    myUbTailFactorH_ = isHTail ? tailBlockUbTailFactorH_ : ubTailFactorH_;
    myLoopNumB_ = isBTail ? tailBlockLoopNumB_ : loopNumB_;
    myUbTailFactorB_ = isBTail ? tailBlockUbTailFactorB_ : ubTailFactorB_;
    myLoopNumS_ = isSTail ? tailBlockLoopNumS_ : loopNumS_;
    myUbTailFactorS_ = isSTail ? tailBlockUbTailFactorS_ : ubTailFactorS_;

    // Bind GM — x and y are contiguous [S][B][H]
    xGM_.SetGlobalBuffer((__gm__ T *)x, (uint64_t)S_ * B_ * H_);
    weightGM_.SetGlobalBuffer((__gm__ T *)weight, (uint64_t)3 * H_);
    maskGM_.SetGlobalBuffer((__gm__ uint8_t *)mask, (uint64_t)B_ * S_);
    yGM_.SetGlobalBuffer((__gm__ T *)y, (uint64_t)S_ * B_ * H_);

    // ioQueue: [bUb][sUb+2][ubFactorH] T — 2 prefix rows + sCur data rows, each row ubFactorH wide
    uint32_t ioBufBytes = AlignUp(ubFactorB_ * (ubFactorS_ + PREFIX_NUM) * ubFactorH_ * sizeof(T), ALIGN_BYTES);
    // weightQueue: [3][ubFactorH] T — tap-major layout
    uint32_t weightBufBytes = AlignUp(3 * ubFactorH_ * sizeof(T), ALIGN_BYTES);

    pipe_.InitBuffer(ioQueue_, BUFFER_NUM, ioBufBytes);
    pipe_.InitBuffer(weightQueue_, BUFFER_NUM, weightBufBytes);

    if (isMaskNone_ == 0) {
        uint32_t sPadAligned = AlignUp(ubFactorS_, ALIGN_BYTES);

        // maskRowQueue_: uint8 DMA landing (step1), then T mask values (step3)
        uint32_t maskRowBytes = AlignUp(ubFactorB_ * sPadAligned * static_cast<uint32_t>(sizeof(T)), ALIGN_BYTES);
        pipe_.InitBuffer(maskRowQueue_, BUFFER_NUM, maskRowBytes);

        // maskCastQueue_: half Cast result (step2), then reused as T Broadcast dst (step4)
        uint32_t castHalfBytes = AlignUp(ubFactorB_ * sPadAligned * static_cast<uint32_t>(sizeof(half)), ALIGN_BYTES);
        uint32_t broadcastBytes = AlignUp(sPadAligned * ubFactorH_ * static_cast<uint32_t>(sizeof(T)), ALIGN_BYTES);
        pipe_.InitBuffer(maskCastQueue_, BUFFER_NUM, castHalfBytes > broadcastBytes ? castHalfBytes : broadcastBytes);
    }
}

// ============================================================================
// CopyIn: load [bUb][sCur+2][ubFactorHCur] into ioQueue
//
// x GM layout: [S][B][H] contiguous; S-stride = B*H, B-stride = H.
// UB layout per batch b:
//   row 0:          x[s0-2, b0+b, h0..h0+ubFactorHCur-1]  (prefix-2)
//   row 1:          x[s0-1, b0+b, h0..h0+ubFactorHCur-1]  (prefix-1)
//   rows 2..sCur+1: x[s0..s0+sCur-1, b0+b, h0..h0+ubFactorHCur-1]
//
// Boundary (s0 < 2, only when sCoreIdx=0 && sIter=0):
//   s0 == 0: rows 0,1 = 0; DataCopy sCur   rows from GM[0, b0+b, h0]
//   s0 == 1: row 0   = 0; DataCopy sCur+1 rows from GM[0, b0+b, h0]
//   s0 >= 2: DataCopy all sCur+2 rows from GM[s0-2, b0+b, h0]
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::CopyIn(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur,
                                                     uint32_t h0, uint32_t ubFactorHCur)
{
    int32_t eventIdToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdToMTE3);
    WaitFlag<HardEvent::MTE3_V>(eventIdToMTE3);
    LocalTensor<T> ioBuf = ioQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // x is contiguous [S][B][H]: S-stride = xSStride_, B-stride = H_
    uint32_t rowElems = ubFactorHCur;
    uint32_t bRowStride = (sCur + PREFIX_NUM) * rowElems; // element stride per batch in ioQueue
    uint32_t srcSkipBytes = (xSStride_ - ubFactorHCur) * static_cast<uint32_t>(sizeof(T));
    for (uint32_t b = 0; b < bUbCur; ++b) {
        uint32_t dstBase = b * bRowStride;

        if (s0 >= 2) {
            // Common case: load sCur+2 rows starting at x[s0-2, b0+b, h0]
            uint64_t gmOff = (uint64_t)(s0 - 2) * xSStride_ + (uint64_t)(b0 + b) * H_ + h0;
            DataCopyExtParams xcp{static_cast<uint16_t>(sCur + PREFIX_NUM), static_cast<uint16_t>(rowElems * sizeof(T)),
                                  srcSkipBytes, 0, 0};
            DataCopyPad(ioBuf[dstBase], xGM_[gmOff], xcp, padParams);
        } else if (s0 == 0) {
            // Prefix rows 0,1 = zero; load sCur rows at rows 2..sCur+1
            Duplicate(ioBuf[dstBase], static_cast<T>(0), 2 * rowElems);
            uint64_t gmOff = (uint64_t)(b0 + b) * H_ + h0;
            DataCopyExtParams xcp{static_cast<uint16_t>(sCur), static_cast<uint16_t>(rowElems * sizeof(T)),
                                  srcSkipBytes, 0, 0};
            DataCopyPad(ioBuf[dstBase + PREFIX_NUM * rowElems], xGM_[gmOff], xcp, padParams);
        } else {
            // s0 == 1: row 0 = zero; load sCur+1 rows from GM[0, b0+b, h0] into rows 1..sCur+1
            Duplicate(ioBuf[dstBase], static_cast<T>(0), rowElems);
            uint64_t gmOff = (uint64_t)(b0 + b) * H_ + h0;
            DataCopyExtParams xcp{static_cast<uint16_t>(sCur + 1), static_cast<uint16_t>(rowElems * sizeof(T)),
                                  srcSkipBytes, 0, 0};
            DataCopyPad(ioBuf[dstBase + rowElems], xGM_[gmOff], xcp, padParams);
        }
    }
    ioQueue_.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, T>(ioBuf);
}

// ============================================================================
// Compute: VF unified-buffer convolution, then apply mask (if isMaskNone_==0)
//
// After VF: y[s] is at ioBuf[b*(sCur+2)*ubFactorHCur + s*ubFactorHCur] (row s, ubFactorHCur wide).
//
// Mask pipeline (isMaskNone_==0) — vectorized Broadcast + Mul:
//   Step1: DMA mask rows (uint8) from GM to maskRowBuf
//   Step2: Cast uint8 → half → maskCastBuf
//   Step3: Cast/Copy half → T → maskRowBuf (frees maskCastBuf for Broadcast)
//   Step4: Per-batch Broadcast(maskRowBuf → maskCastBuf) + Mul(ioBuf *= maskCastBuf)
//
// Mask (isMaskNone_==1): skip mask — result written directly to CopyOut.
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Compute(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur,
                                                      uint32_t ubFactorHCur, LocalTensor<T> &wBuf,
                                                      int32_t eventIdToMTE2)
{
    WaitFlag<HardEvent::MTE2_V>(eventIdToMTE2);
    LocalTensor<T> ioBuf = ioQueue_.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, T>();

    // VF convolution across all H_REG chunks (loop inside VF)
    CallMaskedConv1dVF<T>(ioBuf, wBuf, sCur, bUbCur, ubFactorHCur);

    if (isMaskNone_ == 0) {
        // ---- Mask application: 2-step Cast (uint8→half→T) + Broadcast + Mul ----
        uint32_t sPad = AlignUp(sCur, ALIGN_BYTES);

        // Alloc from double-buffered queues
        LocalTensor<T> maskRowBuf = maskRowQueue_.AllocTensor<T>();
        LocalTensor<uint8_t> maskRaw = maskRowBuf.template ReinterpretCast<uint8_t>();

        // Step1: DMA mask rows (uint8) from GM to maskRowBuf
        for (uint32_t b = 0; b < bUbCur; ++b) {
            uint64_t maskGmOff = (uint64_t)(b0 + b) * S_ + s0;
            DataCopyExtParams mcp{1, static_cast<uint16_t>(sCur), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> mpadParams{false, 0, 0, 0};
            DataCopyPad(maskRaw[b * sPad], maskGM_[maskGmOff], mcp, mpadParams);
        }
        maskRowQueue_.EnQue(maskRaw);
        maskRaw = maskRowQueue_.DeQue<uint8_t>();

        // Step2: Cast uint8 → half, store to maskCastBuf
        LocalTensor<T> maskCastBuf = maskCastQueue_.AllocTensor<T>();
        LocalTensor<half> maskHalf = maskCastBuf.template ReinterpretCast<half>();
        for (uint32_t b = 0; b < bUbCur; ++b) {
            Cast(maskHalf[b * sPad], maskRaw[b * sPad], RoundMode::CAST_NONE, sPad);
        }
        PipeBarrier<PIPE_V>();

        // Step3: half → T, store back to maskRowBuf (frees maskCastBuf for Broadcast)
        if constexpr (std::is_same<T, half>::value) {
            for (uint32_t b = 0; b < bUbCur; ++b) {
                Adds(maskRowBuf[b * sPad], maskHalf[b * sPad], static_cast<T>(0), sPad);
            }
        } else {
            for (uint32_t b = 0; b < bUbCur; ++b) {
                Cast(maskRowBuf[b * sPad], maskHalf[b * sPad], RoundMode::CAST_RINT, sPad);
            }
        }
        PipeBarrier<PIPE_V>();

        // Step4: Per-batch Broadcast + Mul (maskCastBuf reused as Broadcast dst)
        for (uint32_t b = 0; b < bUbCur; ++b) {
            uint32_t srcShape[2] = {sPad, 1u};
            uint32_t dstShape[2] = {sPad, ubFactorHCur};
            Broadcast<T, 2, 1>(maskCastBuf, maskRowBuf[b * sPad], dstShape, srcShape);

            uint32_t ioOff = b * (sCur + PREFIX_NUM) * ubFactorHCur;
            Mul(ioBuf[ioOff], ioBuf[ioOff], maskCastBuf, sCur * ubFactorHCur);
            PipeBarrier<PIPE_V>();
        }

        // Free back to queues
        maskCastQueue_.FreeTensor(maskCastBuf);
        maskRowQueue_.FreeTensor(maskRowBuf);
    }
    ioQueue_.EnQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, T>(ioBuf);
}

// ============================================================================
// CopyOut: write y[0..sCur-1] back to GM
//
// y[s] of batch b is at ioBuf[b*(sCur+2)*ubFactorHCur + s*ubFactorHCur] (ubFactorHCur elements wide).
// Output y GM layout: [S][B][H] contiguous; S-stride = B*H, B-stride = H.
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::CopyOut(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur,
                                                      uint32_t h0, uint32_t ubFactorHCur)
{
    LocalTensor<T> ioBuf = ioQueue_.DeQue<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, T>();

    // dstJumpStride: bytes to skip in y GM between consecutive S rows
    uint32_t dstSkipBytes = (xSStride_ - ubFactorHCur) * static_cast<uint32_t>(sizeof(T));
    uint32_t bIoStride = (sCur + PREFIX_NUM) * ubFactorHCur;

    for (uint32_t b = 0; b < bUbCur; ++b) {
        // y[0..sCur-1] are at rows 0..sCur-1 of this batch's region
        uint32_t ioOff = b * bIoStride;

        uint64_t dstOff = (uint64_t)s0 * xSStride_ + (uint64_t)(b0 + b) * H_ + h0;
        DataCopyExtParams ycp{static_cast<uint16_t>(sCur), static_cast<uint16_t>(ubFactorHCur * sizeof(T)), 0,
                              dstSkipBytes, 0};
        DataCopyPad(yGM_[dstOff], ioBuf[ioOff], ycp);
    }
    ioQueue_.FreeTensor(ioBuf);
}

// ============================================================================
// Process: main compute loop  H(outer) -> B(middle) -> S(inner)
//
// Weight is loaded once per S-tile (same data reloaded; enables DMA-compute
// overlap via double-buffered weightQueue_). Weight tile size = ubFactorHCur.
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    // ---- H outer loop ----
    uint32_t h0 = hStart_;
    for (uint32_t hIter = 0; hIter < myLoopNumH_; ++hIter, h0 += ubFactorH_) {
        uint32_t ubFactorHCur = (hIter < myLoopNumH_ - 1) ? ubFactorH_ : myUbTailFactorH_;
        // Load weight once per H-tile: [3][ubFactorHCur] from weightGM_[h0]
        {
            LocalTensor<T> wBufTmp = weightQueue_.AllocTensor<T>();
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            uint32_t wSkipBytes = (H_ - ubFactorHCur) * static_cast<uint32_t>(sizeof(T));
            DataCopyExtParams wcp{3, static_cast<uint16_t>(ubFactorHCur * sizeof(T)), wSkipBytes, 0, 0};
            DataCopyPad(wBufTmp[0], weightGM_[h0], wcp, padParams);
            weightQueue_.EnQue(wBufTmp);
        }
        // DeQue once, hold across all B×S iterations
        LocalTensor<T> wBuf = weightQueue_.DeQue<T>();

        // ---- B middle loop ----
        for (uint32_t bIter = 0; bIter < myLoopNumB_; ++bIter) {
            uint32_t bUbCur = (bIter < myLoopNumB_ - 1) ? ubFactorB_ : myUbTailFactorB_;
            uint32_t b0 = bStart_ + bIter * ubFactorB_;

            // ---- S inner loop ----
            for (uint32_t sIter = 0; sIter < myLoopNumS_; ++sIter) {
                uint32_t sCur = (sIter < myLoopNumS_ - 1) ? ubFactorS_ : myUbTailFactorS_;
                uint32_t s0 = sStart_ + sIter * ubFactorS_;

                // CopyIn loads sCur+2 rows per batch (rows 0,1=prefix; rows 2..=x[s0..])
                CopyIn(s0, sCur, b0, bUbCur, h0, ubFactorHCur);
                int32_t eventIdToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdToMTE2);
                Compute(s0, sCur, b0, bUbCur, ubFactorHCur, wBuf, eventIdToMTE2);
                CopyOut(s0, sCur, b0, bUbCur, h0, ubFactorHCur);
            }
        }
        // Free weight at end of H-tile
        weightQueue_.FreeTensor(wBuf);
    }
}

} // namespace MaskedCausalConv1dNs

#endif // MASKED_CAUSAL_CONV1D_H

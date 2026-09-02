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
 * \file mhc_post_backward_kernel.h
 * \brief
 */

#ifndef MHC_POST_BACKWARD_H
#define MHC_POST_BACKWARD_H
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "mhc_post_backward_tiling_data_arch35.h"

namespace MhcPostBackward {
using namespace AscendC;

// Constants for memory alignment and buffer configuration
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16; // 16 elements = 32 bytes for bf16/fp16
constexpr uint32_t FLOAT32_ALIGN_SIZE = 8;    // 8 elements = 32 bytes for float32

// Queue depth configuration
constexpr uint32_t QUEUE_DEPTH = 1; // Single Buffer depth for all queues

template <typename T, bool IS_HRES>
class MhcPostBackwardKernel {
public:
    __aicore__ inline MhcPostBackwardKernel() {}
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR gradX,
                                GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost, GM_ADDR workspace,
                                const MhcPostBackwardTilingDataArch35 *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInWeights(uint32_t globalItemIdx);
    __aicore__ inline void ComputeInitGrads();
    __aicore__ inline void CopyInTile(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void ComputeTile(uint32_t tileId);
    __aicore__ inline void CopyOutGradHOut(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void CopyOutGradX(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void CopyOutWeightGrads(uint32_t globalItemIdx);
    __aicore__ inline void SetMMParaAndCompute(uint32_t globalItemIdx);
    __aicore__ inline float HierarchicalReduceSum(const LocalTensor<float> &srcLocal, LocalTensor<float> &workLocal,
                                                  uint32_t count);
    __aicore__ inline void ComputeGradHOutTile(LocalTensor<float> gradOutF32UB, LocalTensor<float> hPostLocalUB,
                                               LocalTensor<T> gradHOutTileUB, uint32_t actualTileD);
    __aicore__ inline void ComputeGradXTile(LocalTensor<float> gradOutF32UB, LocalTensor<float> hResLocalUB,
                                            LocalTensor<T> gradXTileUB, uint32_t actualTileD);

private:
    TPipe *pipe_;
    constexpr static AscendC::MicroAPI::CastTrait castB32ToB16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    // Input queues
    TQue<QuePosition::VECIN, QUEUE_DEPTH> gradOutTileQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> hOutTileQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> hPostQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> hResQueue_;

    // Output queues
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> gradHOutTileQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> gradXTileQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> gradHPostQueue_;

    // Intermediate buffers for float computation
    TBuf<QuePosition::VECCALC> gradOutF32Buf_;
    TBuf<QuePosition::VECCALC> hOutF32Buf_;
    TBuf<QuePosition::VECCALC> tempBuf_;
    TBuf<QuePosition::VECCALC> workBuf_;
    // Additional buffers for vectorized accumulation
    TBuf<QuePosition::VECCALC> gradHPostTileSumsBuf_; // n elements

    // Global memory tensors - inputs (bf16/fp16)
    GlobalTensor<T> gradOutputGm_;
    GlobalTensor<T> hOutGm_;
    GlobalTensor<T> xGm_;
    // Global memory tensors - inputs (float32)
    GlobalTensor<float> hPostGm_;
    GlobalTensor<float> hResGm_;

    // Global memory tensors - outputs (bf16/fp16)
    GlobalTensor<T> gradHOutGm_;
    GlobalTensor<T> gradXGm_;
    // Global memory tensors - outputs (float32)
    GlobalTensor<float> gradHPostGm_;
    GlobalTensor<float> gradHResGm_;

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    matmul::MatmulImpl<aType, bType, cType> mm;

    // Tiling parameters
    uint32_t totalItems_;
    uint32_t itemsPerCore_;
    uint32_t remainderItems_;
    uint32_t usedCores_;
    uint32_t S_;
    uint32_t n_;
    uint32_t D_;
    uint32_t tileD_;
    uint32_t nTilesD_;
    uint32_t alignedD_;
    uint32_t lastTileD_;
    uint32_t alignedN_;  // n aligned to 8 for float32 vector ops
    uint32_t alignedNN_; // n*n aligned to 8 for float32 vector ops
    uint32_t usedAic_;
    uint32_t itemsPerAic_;
    uint32_t remainderItemsAic_;
    uint32_t singleCoreK_;

    uint32_t blockIdx_;
    uint32_t myItemCount_;
    uint32_t itemStart_;
};

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::Init(GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes,
                                                               GM_ADDR hOut, GM_ADDR hPost, GM_ADDR gradX,
                                                               GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost,
                                                               GM_ADDR workspace,
                                                               const MhcPostBackwardTilingDataArch35 *tilingData,
                                                               TPipe *tPipe)
{
    pipe_ = tPipe;

    // Get tiling data using direct member access
    totalItems_ = tilingData->totalItems;
    itemsPerCore_ = tilingData->itemsPerCore;
    remainderItems_ = tilingData->remainderItems;
    usedCores_ = tilingData->usedCores;
    S_ = tilingData->S;
    n_ = tilingData->n;
    D_ = tilingData->D;
    tileD_ = tilingData->tileD;
    nTilesD_ = tilingData->nTilesD;
    alignedD_ = tilingData->alignedD;
    lastTileD_ = tilingData->lastTileD;
    usedAic_ = tilingData->matmulTiling.usedCoreNum;
    itemsPerAic_ = tilingData->itemsPerAic;
    remainderItemsAic_ = tilingData->remainderItemsAic;
    singleCoreK_ = tilingData->matmulTiling.singleCoreK;

    alignedN_ = tilingData->alignedN;
    alignedNN_ = tilingData->alignedNN;

    blockIdx_ = GetBlockIdx();

    // Set global memory buffers for inputs (bf16/fp16)
    gradOutputGm_.SetGlobalBuffer((__gm__ T *)gradOutput, totalItems_ * n_ * D_);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut, totalItems_ * D_);
    xGm_.SetGlobalBuffer((__gm__ T *)x, totalItems_ * n_ * D_);
    // Set global memory buffers for inputs (float32)
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost, totalItems_ * n_);
    if constexpr (IS_HRES) {
        hResGm_.SetGlobalBuffer((__gm__ float *)hRes, totalItems_ * n_ * n_);
    }

    // Set global memory buffers for outputs (bf16/fp16)
    gradHOutGm_.SetGlobalBuffer((__gm__ T *)gradHOut, totalItems_ * D_);
    gradXGm_.SetGlobalBuffer((__gm__ T *)gradX, totalItems_ * n_ * D_);
    // Set global memory buffers for outputs (float32)
    gradHPostGm_.SetGlobalBuffer((__gm__ float *)gradHPost, totalItems_ * n_);
    gradHResGm_.SetGlobalBuffer((__gm__ float *)gradHRes, totalItems_ * n_ * n_);

    if ASCEND_IS_AIV {
        // Calculate work distribution with remainder handling
        if (blockIdx_ < remainderItems_) {
            myItemCount_ = itemsPerCore_ + 1;
            itemStart_ = blockIdx_ * (itemsPerCore_ + 1);
        } else {
            myItemCount_ = itemsPerCore_;
            itemStart_ = remainderItems_ * (itemsPerCore_ + 1) + (blockIdx_ - remainderItems_) * itemsPerCore_;
        }

        // Initialize input queues (bf16/fp16 for grad_output, hOut, x)
        pipe_->InitBuffer(gradOutTileQueue_, QUEUE_DEPTH, n_ * tileD_ * sizeof(T));
        pipe_->InitBuffer(hOutTileQueue_, QUEUE_DEPTH, tileD_ * sizeof(T));
        // float32 for hPost, hRes (use aligned sizes for vector ops)
        pipe_->InitBuffer(hPostQueue_, QUEUE_DEPTH, alignedN_ * sizeof(float));
        if constexpr (IS_HRES) {
            pipe_->InitBuffer(hResQueue_, QUEUE_DEPTH, alignedNN_ * sizeof(float));
        }

        // Initialize output queues (bf16/fp16 for gradHOut, gradX)
        pipe_->InitBuffer(gradHOutTileQueue_, QUEUE_DEPTH, tileD_ * sizeof(T));
        pipe_->InitBuffer(gradXTileQueue_, QUEUE_DEPTH, n_ * tileD_ * sizeof(T));
        // float32 for gradHPost, gradHRes (use aligned sizes for vector ops)
        pipe_->InitBuffer(gradHPostQueue_, QUEUE_DEPTH, alignedN_ * sizeof(float));

        // Initialize intermediate buffers for float32 computation
        pipe_->InitBuffer(gradOutF32Buf_, n_ * tileD_ * sizeof(float));
        pipe_->InitBuffer(hOutF32Buf_, tileD_ * sizeof(float));
        pipe_->InitBuffer(tempBuf_, tileD_ * sizeof(float));
        pipe_->InitBuffer(workBuf_, tileD_ * sizeof(float));
        // Initialize buffers for vectorized accumulation (use aligned sizes)
        pipe_->InitBuffer(gradHPostTileSumsBuf_, alignedN_ * sizeof(float));
    }

    if ASCEND_IS_AIC {
        if (blockIdx_ < remainderItemsAic_) {
            myItemCount_ = itemsPerAic_ + 1;
            itemStart_ = blockIdx_ * (itemsPerAic_ + 1);
        } else {
            myItemCount_ = itemsPerAic_;
            itemStart_ = remainderItemsAic_ * (itemsPerAic_ + 1) + (blockIdx_ - remainderItemsAic_) * itemsPerAic_;
        }

        mm.Init(&tilingData->matmulTiling, pipe_);
    }
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::Process()
{
    if ASCEND_IS_AIV {
        if (blockIdx_ >= usedCores_) {
            return;
        }

        for (uint32_t itemIdx = 0; itemIdx < myItemCount_; itemIdx++) {
            uint32_t globalItemIdx = itemStart_ + itemIdx;
            CopyInWeights(globalItemIdx);
            ComputeInitGrads();

            for (uint32_t tileId = 0; tileId < nTilesD_; tileId++) {
                CopyInTile(globalItemIdx, tileId);
                ComputeTile(tileId);
                // Split CopyOut to enable pipeline overlap
                CopyOutGradHOut(globalItemIdx, tileId); // Can overlap with gradX computation
                CopyOutGradX(globalItemIdx, tileId);
            }

            CopyOutWeightGrads(globalItemIdx);
        }
    }
    if ASCEND_IS_AIC {
        if (blockIdx_ >= usedAic_) {
            return;
        }

        for (uint32_t itemIdx = 0; itemIdx < myItemCount_; itemIdx++) {
            uint32_t globalItemIdx = itemStart_ + itemIdx;
            SetMMParaAndCompute(globalItemIdx);
        }
    }
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::CopyInWeights(uint32_t globalItemIdx)
{
    uint32_t hPostOffset = globalItemIdx * n_;

    LocalTensor<float> hPostLocal = hPostQueue_.AllocTensor<float>();
    // Slow path: need padding, zero out first then copy
    uint8_t hPostrightPad = static_cast<uint8_t>(alignedN_ - n_);
    DataCopyPadParams hPostpadParams = {true, 0, hPostrightPad, 0};
    DataCopyPad(hPostLocal, hPostGm_[hPostOffset], {1, static_cast<uint16_t>(n_ * sizeof(float)), 0, 0},
                hPostpadParams);
    hPostQueue_.EnQue(hPostLocal);

    if constexpr (IS_HRES) {
        uint32_t hResOffset = globalItemIdx * n_ * n_;
        LocalTensor<float> hResLocal = hResQueue_.AllocTensor<float>();
        // Slow path: need padding, zero out first then copy
        uint8_t hResrightPad = static_cast<uint8_t>(alignedNN_ - n_ * n_);
        DataCopyPadParams hRespadParams = {true, 0, hResrightPad, 0};
        DataCopyPad(hResLocal, hResGm_[hResOffset], {1, static_cast<uint16_t>(n_ * n_ * sizeof(float)), 0, 0},
                    hRespadParams);
        hResQueue_.EnQue(hResLocal);
    }
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::ComputeInitGrads()
{
    LocalTensor<float> gradHPostLocal = gradHPostQueue_.AllocTensor<float>();

    // Initialize gradHPost and gradHRes to zero
    // Use actual size when aligned, aligned size when not
    Duplicate(gradHPostLocal, 0.0f, alignedN_);

    gradHPostQueue_.EnQue(gradHPostLocal);
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::CopyInTile(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t gradOutBase = globalItemIdx * n_ * D_;
    uint64_t hOutBase = globalItemIdx * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<T> gradOutTileLocal = gradOutTileQueue_.AllocTensor<T>();
    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    // Slow path: multi-tile or non-aligned
    // 数据在 buffer 中连续存放，第 i 行起始位置为 i * actualTileD
    // Load hOut tile
    // 需要 padding 到 actualTileD（已对齐到 16）
    uint8_t rightPad = static_cast<uint8_t>(actualTileD - gmCopyD);
    DataCopyPadParams padParams = {true, 0, rightPad, 0};
    DataCopyPad(hOutTileLocal, hOutGm_[hOutBase + dStart], {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0},
                padParams);
    // Load grad_output and x row by row，连续存放
    for (uint32_t i = 0; i < n_; i++) {
        uint8_t rightPad = static_cast<uint8_t>(actualTileD - gmCopyD);
        DataCopyPadParams padParams = {true, 0, rightPad, 0};
        DataCopyPad(gradOutTileLocal[i * actualTileD], gradOutputGm_[gradOutBase + i * D_ + dStart],
                    {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0}, padParams);
    }

    gradOutTileQueue_.EnQue(gradOutTileLocal);
    hOutTileQueue_.EnQue(hOutTileLocal);
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::ComputeTile(uint32_t tileId)
{
    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;

    LocalTensor<T> gradOutTile = gradOutTileQueue_.DeQue<T>();
    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<float> hPostLocal = hPostQueue_.DeQue<float>();
    LocalTensor<float> gradHPostLocal = gradHPostQueue_.DeQue<float>();

    LocalTensor<T> gradHOutTile = gradHOutTileQueue_.AllocTensor<T>();
    LocalTensor<T> gradXTile = gradXTileQueue_.AllocTensor<T>();
    // Get work buffers
    LocalTensor<float> tempLocal = tempBuf_.Get<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> gradHPostTileSums = gradHPostTileSumsBuf_.Get<float>();

    // Only zero out when not aligned (padding area needs to be zero)
    // When aligned, SetValue will fill all valid positions
    Duplicate(gradHPostTileSums, 0.0f, alignedN_);

    // ===== Common path for bf16 and fp16: Cast to f32 for computation =====
    LocalTensor<float> gradOutF32 = gradOutF32Buf_.Get<float>();
    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    Cast(gradOutF32, gradOutTile, RoundMode::CAST_NONE, n_ * actualTileD);

    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, actualTileD);

    for (uint32_t i = 0; i < n_; i++) {
        Mul(tempLocal, gradOutF32[i * actualTileD], hOutF32, actualTileD);
        float sumHPost = HierarchicalReduceSum(tempLocal, workLocal, actualTileD);
        event_t eventVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVtoS);
        WaitFlag<HardEvent::V_S>(eventVtoS);
        gradHPostTileSums.SetValue(i, sumHPost);
    }
    Add(gradHPostLocal, gradHPostLocal, gradHPostTileSums, alignedN_);
    gradHPostQueue_.EnQue(gradHPostLocal);

    ComputeGradHOutTile(gradOutF32, hPostLocal, gradHOutTile, actualTileD);
    gradHOutTileQueue_.EnQue(gradHOutTile);

    if constexpr (IS_HRES) {
        LocalTensor<float> hResLocal = hResQueue_.DeQue<float>();
        ComputeGradXTile(gradOutF32, hResLocal, gradXTile, actualTileD);
        hResQueue_.EnQue(hResLocal);
    } else {
        DataCopy(gradXTile, gradOutTile, n_ * actualTileD);
    }
    gradXTileQueue_.EnQue(gradXTile);

    hPostQueue_.EnQue(hPostLocal);

    gradOutTileQueue_.FreeTensor(gradOutTile);
    hOutTileQueue_.FreeTensor(hOutTile);
}

template <typename T, bool IS_HRES>
__aicore__ inline float MhcPostBackwardKernel<T, IS_HRES>::HierarchicalReduceSum(const LocalTensor<float> &srcLocal,
                                                                                 LocalTensor<float> &workLocal,
                                                                                 uint32_t count)
{
    constexpr uint32_t BLOCK_SIZE = 64; // 64 float elements = 256 bytes

    // Cast away const for in-place modification
    LocalTensor<float> &src = const_cast<LocalTensor<float> &>(srcLocal);

    // Align count to 8 upfront and zero-pad to avoid alignment issues in reduction loop
    uint32_t alignedCount = ((count + 7) / 8) * 8;
    if (count < alignedCount) {
        Duplicate(src[count], 0.0f, alignedCount - count);
        PipeBarrier<PIPE_V>();
    }

    // For small counts, use direct ReduceSum (no precision issue)
    if (alignedCount <= BLOCK_SIZE) {
        ReduceSum(workLocal, src, workLocal, alignedCount);
        return workLocal.GetValue(0);
    }

    uint32_t currentCount = alignedCount;

    // Hierarchical reduction: repeatedly halve the data in-place
    // Each iteration adds the second half to the first half
    while (currentCount > BLOCK_SIZE) {
        uint32_t halfCount = currentCount / 2;
        // Align halfCount to 8 for vector operations
        uint32_t alignedHalfCount = (halfCount / 8) * 8;

        if (alignedHalfCount >= 8) {
            // Add src[alignedHalfCount : 2*alignedHalfCount] to src[0 : alignedHalfCount]
            Add(src, src, src[alignedHalfCount], alignedHalfCount);
            PipeBarrier<PIPE_V>();

            // Handle remaining elements (from 2*alignedHalfCount to currentCount)
            uint32_t processedCount = alignedHalfCount * 2;
            uint32_t remainderCount = currentCount - processedCount;
            if (remainderCount > 0) {
                // Align remainder to 8
                uint32_t alignedRemainder = ((remainderCount + 7) / 8) * 8;
                // Zero-pad if needed
                if (remainderCount < alignedRemainder) {
                    Duplicate(src[processedCount + remainderCount], 0.0f, alignedRemainder - remainderCount);
                    PipeBarrier<PIPE_V>();
                }
                // Add remainder to the END of the first half (not the beginning)
                // This ensures no element is added twice
                uint32_t remainderDst = alignedHalfCount - alignedRemainder;
                if (remainderDst < alignedHalfCount) {
                    Add(src[remainderDst], src[remainderDst], src[processedCount], alignedRemainder);
                    PipeBarrier<PIPE_V>();
                }
            }

            currentCount = alignedHalfCount;
        } else {
            break;
        }
    }

    // Final reduction using WholeReduceSum
    // currentCount is already aligned to 8 from the loop
    uint32_t finalAlignedCount = ((currentCount + 7) / 8) * 8;
    if (currentCount < finalAlignedCount) {
        Duplicate(src[currentCount], 0.0f, finalAlignedCount - currentCount);
        PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(workLocal, src, finalAlignedCount, 1, 1, 1, finalAlignedCount / 8);
    event_t eventVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVtoS);
    WaitFlag<HardEvent::V_S>(eventVtoS);

    return workLocal.GetValue(0);
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::ComputeGradHOutTile(LocalTensor<float> gradOutF32UB,
                                                                              LocalTensor<float> hPostLocalUB,
                                                                              LocalTensor<T> gradHOutTileUB,
                                                                              uint32_t actualTileD)
{
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (actualTileD + vfLen - 1) / vfLen;
    uint16_t ntimes = n_;

    auto gradOutTileAddr = (__ubuf__ float *)gradOutF32UB.GetPhyAddr();
    auto hPostLocalAddr = (__ubuf__ float *)hPostLocalUB.GetPhyAddr();
    auto gradHOutTileAddr = (__ubuf__ T *)gradHOutTileUB.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t gradOutDealNum = static_cast<uint32_t>(actualTileD);
        AscendC::MicroAPI::RegTensor<float> gradOutF32Reg;
        AscendC::MicroAPI::RegTensor<float> hPostLocalReg;
        AscendC::MicroAPI::RegTensor<float> gradHOutTileReg;
        AscendC::MicroAPI::RegTensor<float> mulResReg;
        AscendC::MicroAPI::RegTensor<T> gradHOutTileB16Reg;
        AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t k = 0; k < repeatTimes; k++) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(gradOutDealNum);
            AscendC::MicroAPI::Duplicate(gradHOutTileReg, 0.0f);
            for (uint16_t i = 0; i < ntimes; i++) {
                AscendC::MicroAPI::LoadAlign(gradOutF32Reg, gradOutTileAddr + i * actualTileD + k * vfLen);
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hPostLocalReg,
                                                                                               hPostLocalAddr + i);
                AscendC::MicroAPI::Mul(mulResReg, gradOutF32Reg, hPostLocalReg, pMask);
                AscendC::MicroAPI::Add(gradHOutTileReg, mulResReg, gradHOutTileReg, pMask);
            }
            AscendC::MicroAPI::Cast<T, float, castB32ToB16>(gradHOutTileB16Reg, gradHOutTileReg, pMask);
            AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(gradHOutTileAddr + k * vfLen,
                                                                                          gradHOutTileB16Reg, pMask);
        }
    }
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::ComputeGradXTile(LocalTensor<float> gradOutF32UB,
                                                                           LocalTensor<float> hResLocalUB,
                                                                           LocalTensor<T> gradXTileUB,
                                                                           uint32_t actualTileD)
{
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (actualTileD + vfLen - 1) / vfLen;
    uint16_t ntimes = n_;

    auto gradOutTileAddr = (__ubuf__ float *)gradOutF32UB.GetPhyAddr();
    auto hResLocalAddr = (__ubuf__ float *)hResLocalUB.GetPhyAddr();
    auto gradXTileAddr = (__ubuf__ T *)gradXTileUB.GetPhyAddr();

    __VEC_SCOPE__
    {
        uint32_t curDealNum = vfLen;
        AscendC::MicroAPI::RegTensor<float> gradOutF32Reg;
        AscendC::MicroAPI::RegTensor<float> hResLocalReg;
        AscendC::MicroAPI::RegTensor<float> gradXTileReg;
        AscendC::MicroAPI::RegTensor<float> mulResReg;
        AscendC::MicroAPI::RegTensor<T> gradXTileB16Reg;
        AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < ntimes; i++) {
            for (uint16_t k = 0; k < repeatTimes; k++) {
                curDealNum = (k == repeatTimes - 1) ? (actualTileD % vfLen == 0 ? vfLen : actualTileD % vfLen) : vfLen;
                pMask = AscendC::MicroAPI::UpdateMask<float>(curDealNum);
                AscendC::MicroAPI::Duplicate(gradXTileReg, 0.0f, pregMain);
                for (uint16_t j = 0; j < ntimes; j++) {
                    AscendC::MicroAPI::LoadAlign(gradOutF32Reg, gradOutTileAddr + j * actualTileD + k * vfLen);
                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        hResLocalReg, hResLocalAddr + i * n_ + j);
                    AscendC::MicroAPI::Mul(mulResReg, gradOutF32Reg, hResLocalReg, pMask);
                    AscendC::MicroAPI::Add(gradXTileReg, mulResReg, gradXTileReg, pMask);
                }
                AscendC::MicroAPI::Cast<T, float, castB32ToB16>(gradXTileB16Reg, gradXTileReg, pMask);
                AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    gradXTileAddr + i * actualTileD + k * vfLen, gradXTileB16Reg, pMask);
            }
        }
    }
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::CopyOutGradHOut(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t hOutBase = globalItemIdx * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<T> gradHOutTile = gradHOutTileQueue_.DeQue<T>();
    DataCopyPad(gradHOutGm_[hOutBase + dStart], gradHOutTile, {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0});

    gradHOutTileQueue_.FreeTensor(gradHOutTile);
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::CopyOutGradX(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t gradXBase = globalItemIdx * n_ * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<T> gradXTile = gradXTileQueue_.DeQue<T>();
    // Copy gradX row by row，从 buffer 的连续位置读取
    for (uint32_t i = 0; i < n_; i++) {
        DataCopyPad(gradXGm_[gradXBase + i * D_ + dStart], gradXTile[i * actualTileD],
                    {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0});
    }

    gradXTileQueue_.FreeTensor(gradXTile);
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::CopyOutWeightGrads(uint32_t globalItemIdx)
{
    uint32_t hPostOffset = globalItemIdx * n_;

    LocalTensor<float> hPostLocal = hPostQueue_.DeQue<float>();
    hPostQueue_.FreeTensor(hPostLocal);
    if constexpr (IS_HRES) {
        LocalTensor<float> hResLocal = hResQueue_.DeQue<float>();
        hResQueue_.FreeTensor(hResLocal);
    }

    LocalTensor<float> gradHPostLocal = gradHPostQueue_.DeQue<float>();
    DataCopyPad(gradHPostGm_[hPostOffset], gradHPostLocal, {1, static_cast<uint16_t>(n_ * sizeof(float)), 0, 0});
    gradHPostQueue_.FreeTensor(gradHPostLocal);
}

template <typename T, bool IS_HRES>
__aicore__ inline void MhcPostBackwardKernel<T, IS_HRES>::SetMMParaAndCompute(uint32_t globalItemIdx)
{
    uint64_t inputBase = globalItemIdx * n_ * D_;
    uint32_t outputBase = globalItemIdx * n_ * n_;
    uint32_t curSingleCoreK = singleCoreK_;

    mm.SetSingleShape(n_, n_, curSingleCoreK);
    mm.SetTensorA(xGm_[inputBase]);
    mm.SetTensorB(gradOutputGm_[inputBase], true);
    mm.IterateAll(gradHResGm_[outputBase]);
    mm.End();
}

} // namespace MhcPostBackward

#endif // MHC_POST_BACKWARD_H

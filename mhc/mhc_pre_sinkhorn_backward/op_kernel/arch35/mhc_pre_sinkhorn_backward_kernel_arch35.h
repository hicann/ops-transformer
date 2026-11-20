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
 * \file mhc_pre_sinkhorn_backward_kernel_arch35.h
 * \brief MhcPre 与 Sinkhorn 反向融合计算
 */

#ifndef __MHC_PRE_SINKHORN_BACKWARD_KERNEL_H_
#define __MHC_PRE_SINKHORN_BACKWARD_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_sinkhorn_backward_base_arch35.h"
#include "mhc_pre_sinkhorn_backward_cube_compute_arch35.h"
#include "mhc_pre_sinkhorn_backward_sinkhorn_arch35.h"

namespace MhcPreSinkhornBackwardNs {

using namespace AscendC;

constexpr uint32_t MHC_PRE_BACKWARD_IMPL_MODE_HF32 = 1U;

template <class T, class P>
class MhcPreSinkhornBackwardKernel {
public:
    __aicore__ inline MhcPreSinkhornBackwardKernel() = default;
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();

    __aicore__ inline void InitGlobalBuffersAndTiling(InitParams initParams);
    __aicore__ inline void InitUBAndAIVBuffers(InitParams initParams);
    __aicore__ inline void InitStage2AndDataCopy();

    // V0/V1 流程 (向量计算)
    __aicore__ inline void ClearReductionOutputs();
    __aicore__ inline void ProcessV0Main();
    __aicore__ inline void ProcessV0MainLoop(uint64_t vecRunTimes, float alphaPre, float alphaPost, float alphaComb,
                                             LocalTensor<P>& sumBuf, LocalTensor<P>& biasGradSumBuf);
    __aicore__ inline void ProcessV0MainAlphaReduce(LocalTensor<P>& sumBuf, uint32_t coreId);
    __aicore__ inline void PreProcessV0(LocalTensor<P>& hPreGradBuf, uint64_t runBSStart, uint64_t runBSEnd);
    __aicore__ inline void PreProcessV0LargeD(LocalTensor<P>& hPreGradBuf, uint64_t runBSStart, uint64_t runBSEnd);
    __aicore__ inline void AllocV0V1Buffers(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers);
    __aicore__ inline void ProcessV0(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers, float alphaPre,
                                     float alphaPost, float alphaComb, LocalTensor<P>& hMixInput);
    __aicore__ inline void ProcessV1(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P>& buffers,
                                     uint64_t vecRuntimesId, LocalTensor<P>& sumBuf, LocalTensor<P>& biasGradSumBuf,
                                     const LocalTensor<P>& hMixInput);
    // 当前行分块的 Sinkhorn 反向计算
    __aicore__ inline void ProcessSinkhorn(uint64_t runBSStart, uint32_t currentDealBsNum, LocalTensor<P>& hResBuf);

    // Cube侧C0/C1流水
    __aicore__ inline void ProcessC0C1Pipeline();
    template <bool TILED_HMIX_GRAD>
    __aicore__ inline __attribute__((always_inline)) void ProcessStage2Tiles();
    template <bool TILED_HMIX_GRAD>
    __aicore__ inline __attribute__((always_inline)) void ProcessC0(uint32_t offsetND, uint32_t currentNDBlock,
                                                                    uint64_t offsetBS, uint32_t currentBSBlock,
                                                                    uint32_t buffId);
    template <bool TILED_HMIX_GRAD>
    __aicore__ inline __attribute__((always_inline)) void ProcessC1(uint32_t offsetND, uint32_t currentNDBlock,
                                                                    uint64_t offsetBS, uint32_t currentBSBlock,
                                                                    uint32_t buffId);

    // Vector侧V2流水
    __aicore__ inline void ProcessV2Pipeline();
    __aicore__ inline void ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart, uint64_t bsEnd,
                                     uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                              uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize,
                                              uint32_t currentN, uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkLoadInvRms(uint64_t bsChunkStart, uint32_t currentChunkSize);
    __aicore__ inline void ProcessV2ChunkLoadHInGrad(uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t currentN, uint64_t bsStart);
    __aicore__ inline void ProcessV2ChunkLoadX(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                               uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkProcessXRsGradMm(uint64_t bsStart, uint64_t bsChunkStart, uint32_t copySizeND,
                                                          uint32_t currentChunkSize, uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkOutputXGrad(uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t chunkNDSize);

    // V3 流程 (向量计算)
    __aicore__ inline void ProcessV3();
    __aicore__ inline void ProcessV3SingleCore();
    __aicore__ inline void ProcessV3AlphaGrad();
    __aicore__ inline void ProcessV3AlphaGradSingleCore();
    __aicore__ inline void ProcessV3BiasGrad();
    __aicore__ inline void ProcessV3BiasGradSingleCore();

private:
    MhcPreSinkhornBackwardCubeCompute cubeCompute_;
    MMConfig mmConfigC0_;
    MMConfig mmConfigC1_;
    uint32_t coreNum_;
    uint64_t totalLength_;
    uint64_t vecDealBSPeCore_;
    uint32_t vecCoreNum_;
    uint64_t dealStartBS_;
    uint64_t dealEndBS_;
    uint32_t usedVecCoreNum_;
    uint32_t D_;
    uint32_t N_;
    uint32_t fusionSize_;
    float hcEps_;

    uint32_t blockIdx_;
    uint32_t nD_;
    uint32_t v1UsedCubeCoreNum_;
    uint32_t cubeDealnDPeCore_;
    uint32_t dealStartND_;
    uint32_t dealEndND_;

    GlobalTensor<T> xGm_;         // 输入 x
    GlobalTensor<P> phiGm_;       // 输入 phi
    GlobalTensor<P> alphaGm_;     // 输入 alpha
    GlobalTensor<P> biasGm_;      // 输入 bias
    GlobalTensor<T> gradHInGm_;   // 输入 grad_h_in
    GlobalTensor<P> gradHPostGm_; // 输入 grad_h_post
    GlobalTensor<P> invRmsGm_;    // 前向结果 inv_rms
    GlobalTensor<P> hMixGm_;      // 前向结果 hc_before_norm
    GlobalTensor<P> hPreGm_;      // 前向结果 h_pre
    GlobalTensor<P> hPostGm_;     // 前向结果 h_post
    GlobalTensor<P> preMixGm_;    // 可选输入 pre_mix（不在场时为占位 tensor，不读取）
    GlobalTensor<P> gradPreGm_;   // 可选输入 grad_pre（不在场时为占位 tensor，不读取）
    GlobalTensor<P> workSpaceGm_;
    WorkspaceBuffer<P> workspaceBuf_; // Workspace 管理

    GlobalTensor<T> xGradGm_;      // 输出 grad_x
    GlobalTensor<P> gradPhiGm_;    // 输出 grad_phi
    GlobalTensor<P> alphaGradGm_;  // 输出 grad_alpha
    GlobalTensor<P> gradBiasGm_;   // 输出 grad_bias
    GlobalTensor<P> gradPreMixGm_; // 可选输出 grad_pre_mix（不在场时为占位 tensor，不写入）

    LocalTensor<T> gradHInUb;    // grad_h_in
    LocalTensor<P> hPreUb;       // h_pre
    LocalTensor<P> invRmsGradUb; // inv_rms_grad
    LocalTensor<P> invRmsUb;     // inv_rms
    LocalTensor<P> xRsGradMmUb;  // x_rs_grad_mm
    LocalTensor<T> xGradUb;      // grad_x，与输入同类型

    // ProcessV2 分块输入缓冲区
    LocalTensor<P> invRmsInBuf_;
    LocalTensor<P> invRmsGradBuf_;
    LocalTensor<T> gradHInInBuf_;
    LocalTensor<T> bf16InputBuf_;
    LocalTensor<P> xRsOutUb_;
    LocalTensor<P> xRsGradMmInBuf_;
    LocalTensor<T> bf16OutputBuf_;

    // ProcessV2 分块计算缓冲区
    LocalTensor<P> xRsFp32Buf_;
    LocalTensor<P> xGradVec3Buf_;
    LocalTensor<P> invRmsUb_;
    LocalTensor<P> hPreUb_;
    LocalTensor<P> paramBias_;
    bool smallM_ = false;
    bool directC0_ = false;
    uint32_t c0UbBufferIndex_ = 0;

    GlobalTensor<P> invRmsGradGm_; // V1 输出的 RMS 梯度系数
    GlobalTensor<P> xRsGradMmGm_;  // C0 输出 x_rs_grad_mm
    GlobalTensor<P> xRsFp32Gm_;    // V2 输出、C1 输入 x_rs

    TPipe* pipe_;
    const MhcPreSinkhornBackwardArch35TilingData* tiling_;
    TQue<QuePosition::VECIN, 0> vecInQueue0_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue1_;
    TQue<QuePosition::VECIN, 1> vecInQueue1_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueSmall_;

    TBuf<TPosition::VECCALC> fp32TBuf_;
    TBuf<TPosition::VECCALC> paramsBuf_;
    TBuf<TPosition::VECCALC> invScratchBuf_;

    MhcPreSinkhornBackwardSinkhornNs::SinkhornBackward sinkhorn_;

    uint32_t hPreMaxBufLen_;
    uint32_t hPostMaxBufLen_;
    uint32_t gradHResBufLen_;
    uint32_t gradHPostOffset_;
    uint32_t hMixOffset_;
    uint32_t hFusionBufLen_;
    uint32_t hFusionOffset_;
    uint32_t globalUbOffset_;
    uint32_t vecDealChunk_;
    float scaleMean_;
    bool hasPreMix_ = false;
};

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::Init(InitParams initParams)
{
    InitGlobalBuffersAndTiling(initParams);
    InitUBAndAIVBuffers(initParams);
    InitStage2AndDataCopy();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::InitGlobalBuffersAndTiling(InitParams initParams)
{
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.x));
    xGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.phi));
    phiGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_PERSISTENT);
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.alpha));
    gradHInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.grad_h_in));
    gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_h_post));
    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.inv_rms));
    hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_mix));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_pre));
    preMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.pre_mix));
    gradPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_pre));
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.bias));
    xGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.grad_x));
    xGradGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_phi));
    alphaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_alpha));
    gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_bias));
    gradPreMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_pre_mix));
    workSpaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.workspace));
    tiling_ = initParams.tilingData;
    smallM_ = tiling_->smallM != 0U;
    hasPreMix_ = tiling_->hasPreMix != 0U;
    mmConfigC0_ = tiling_->mmConfigC0;
    mmConfigC1_ = tiling_->mmConfigC1;
    nD_ = static_cast<uint32_t>(tiling_->nD);
    D_ = static_cast<uint32_t>(tiling_->D);
    N_ = static_cast<uint32_t>(tiling_->N);
    fusionSize_ = static_cast<uint32_t>(tiling_->fusionSize);
    coreNum_ = tiling_->coreNum;
    totalLength_ = tiling_->totalLength;
    hcEps_ = tiling_->hcEps;
    vecCoreNum_ = tiling_->vecCoreNum;
    vecDealBSPeCore_ = tiling_->vecDealBSPeCore;
    usedVecCoreNum_ = tiling_->usedVecCoreNum;
    cubeDealnDPeCore_ = tiling_->cubeDealnDPeCore;
    v1UsedCubeCoreNum_ = tiling_->v1UsedCubeCoreNum;
    scaleMean_ = (nD_ > 0) ? (1.0f / nD_) : 0.0f;
    workspaceBuf_.Init(totalLength_, fusionSize_, vecCoreNum_, coreNum_);
    blockIdx_ = GetBlockIdx();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::InitUBAndAIVBuffers(InitParams initParams)
{
    // 小 M 满足列对齐时直传 UB；大 M 经 GM 解耦 C0/V2，避免 UB 缓冲区反压。
    directC0_ = smallM_ && nD_ % ND_BLOCK_SIZE == 0;
    vecDealChunk_ = N_ > LARGE_N_THRESHOLD ? VEC_DEAL_CHUNK_LARGE_N : VEC_DEAL_CHUNK_SMALL_N;
    hPreMaxBufLen_ = vecDealChunk_ * N_;
    hPostMaxBufLen_ = hPreMaxBufLen_;
    hMixOffset_ = (hPreMaxBufLen_ + hPostMaxBufLen_) * sizeof(P);
    gradHPostOffset_ = hPreMaxBufLen_ * sizeof(P);
    gradHResBufLen_ = vecDealChunk_ * N_ * N_;
    hFusionBufLen_ = hPreMaxBufLen_ + hPostMaxBufLen_ + gradHResBufLen_;
    pipe_ = initParams.tPipeIn;
    if ASCEND_IS_AIC {
        cubeCompute_.Init(pipe_, tiling_, directC0_);
    }
    if ASCEND_IS_AIV {
        // queue0 的固定 UB 起始地址同时供 C0 Fixpipe 直传使用。
        pipe_->InitBuffer(vecInQueue0_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecInQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
        pipe_->InitBuffer(paramsBuf_, CeilAlign<uint32_t>(static_cast<uint32_t>(fusionSize_ * sizeof(P)), 32U) * 2);
        pipe_->InitBuffer(invScratchBuf_, CeilAlign<uint32_t>(vecDealChunk_ * sizeof(P), 32U));
        static_assert(MhcPreSinkhornBackwardSinkhornNs::MaxScratchBytes() <= INOUT_QUEUE_SIZE,
                      "Sinkhorn tile exceeds the input queue buffer");
        sinkhorn_.Init(initParams.grad_h_res, initParams.sum_out, initParams.norm_out, tiling_->sinkhornPlan,
                       totalLength_, static_cast<uint32_t>(tiling_->iterTimes), hcEps_);
        globalUbOffset_ = 0;
        dealStartBS_ = vecDealBSPeCore_ * blockIdx_;
        dealEndBS_ = dealStartBS_ + vecDealBSPeCore_;
        if (dealEndBS_ > totalLength_ || blockIdx_ == usedVecCoreNum_ - 1) {
            dealEndBS_ = totalLength_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::InitStage2AndDataCopy()
{
    if ASCEND_IS_AIC {
        dealStartND_ = cubeDealnDPeCore_ * blockIdx_;
        dealEndND_ = dealStartND_ + cubeDealnDPeCore_;
        if (dealEndND_ > nD_) {
            dealEndND_ = nD_;
        }
    }
    if ASCEND_IS_AIV {
        dealStartND_ = cubeDealnDPeCore_ * uint32_t(blockIdx_ / VEC_DEAL_VECIDX_DIV);
        dealEndND_ = dealStartND_ + cubeDealnDPeCore_;
        if (dealEndND_ > nD_) {
            dealEndND_ = nD_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::Process()
{
    if (totalLength_ == 0) {
        // 空输入时清空输出
        ClearReductionOutputs();
        return;
    }
    if ASCEND_IS_AIC {
        if (blockIdx_ < v1UsedCubeCoreNum_) {
            // 与 V0 并行预取 C0 所需的 Phi。
            cubeCompute_.PrepareFirstC0(mmConfigC0_, phiGm_[dealStartND_], dealEndND_ - dealStartND_, fusionSize_, nD_);
        }
    }
    if ASCEND_IS_AIV {
        ProcessV0Main();
    }

    SyncAll<false>();

    if ASCEND_IS_AIC {
        ProcessC0C1Pipeline();
    }

    if ASCEND_IS_AIV {
        ProcessV2Pipeline();
    }

    if ASCEND_IS_AIV {
        if (usedVecCoreNum_ == 1U) {
            if (blockIdx_ == 0U) {
                ProcessV3SingleCore();
            }
        } else if (GetBlockIdx() == (GetBlockNum() - 1)) {
            ProcessV3();
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::PreProcessV0(LocalTensor<P>& hPreGradBuf,
                                                                        uint64_t runBSStart, uint64_t runBSEnd)
{
    if (D_ > MAX_D_LEN) {
        PreProcessV0LargeD(hPreGradBuf, runBSStart, runBSEnd);
        return;
    }
    uint32_t maxDLen = MAX_D_LEN;
    uint32_t queueCopySize = INOUT_QUEUE_SIZE / sizeof(T);
    uint32_t copySizeD = D_ <= maxDLen ? D_ : maxDLen;
    uint32_t lenN = queueCopySize / copySizeD > N_ ? N_ : queueCopySize / copySizeD;
    AscendC::LocalTensor<T> bf16InputBuf;
    DataCopyParams dataCopyParams;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    for (uint64_t i = runBSStart; i < runBSEnd; i++) {
        uint64_t inputOffset = i * N_ * copySizeD;
        dataCopyParams.blockLen = copySizeD * sizeof(T);
        dataCopyParams.blockCount = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        AscendC::LocalTensor<T> gradHInInput = vecInQueue1_.AllocTensor<T>();
        DataCopyPad(gradHInInput, gradHInGm_[i * copySizeD], dataCopyParams, dataCopyPadParams);
        vecInQueue1_.EnQue(gradHInInput);
        LocalTensor<T> gradHInInputBuf = vecInQueue1_.DeQue<T>();

        for (uint32_t j = 0; j < N_; j += lenN) {
            uint32_t curLenN = j + lenN > N_ ? N_ - j : lenN;
            dataCopyParams.blockLen = copySizeD * sizeof(T);
            dataCopyParams.blockCount = curLenN;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            vecInQueue0_.AllocTensor<T>(bf16InputBuf);
            DataCopyPad(bf16InputBuf, xGm_[inputOffset], dataCopyParams, dataCopyPadParams);
            vecInQueue0_.EnQue(bf16InputBuf);
            vecInQueue0_.DeQue<T>(bf16InputBuf);

            __ubuf__ T* xB16InAddr = (__ubuf__ T*)bf16InputBuf.GetPhyAddr();
            __ubuf__ T* gradHInB16InAddr = (__ubuf__ T*)gradHInInputBuf.GetPhyAddr();
            __ubuf__ P* hPreGradAddr = (__ubuf__ P*)hPreGradBuf.GetPhyAddr();
            VFDoPreProcessV0<T, P, false>(hPreGradAddr, xB16InAddr, gradHInB16InAddr, curLenN, copySizeD, i, j,
                                          runBSStart, N_);

            inputOffset += copySizeD * curLenN;
            vecInQueue0_.FreeTensor(bf16InputBuf);
        }
        vecInQueue1_.FreeTensor(gradHInInputBuf);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::PreProcessV0LargeD(LocalTensor<P>& hPreGradBuf,
                                                                              uint64_t runBSStart, uint64_t runBSEnd)
{
    DataCopyPadExtParams<T> pad = {false, 0, 0, 0};
    for (uint64_t row = runBSStart; row < runBSEnd; ++row) {
        for (uint32_t offsetD = 0; offsetD < D_; offsetD += MAX_D_LEN) {
            const uint32_t currentD = Min(MAX_D_LEN, D_ - offsetD);
            const uint32_t lenN = Min(INOUT_QUEUE_SIZE / (currentD * static_cast<uint32_t>(sizeof(T))), N_);
            DataCopyExtParams copy = {1, static_cast<uint32_t>(currentD * sizeof(T)), 0, 0, 0};
            LocalTensor<T> gradIn = vecInQueue1_.AllocTensor<T>();
            DataCopyPad(gradIn, gradHInGm_[row * D_ + offsetD], copy, pad);
            vecInQueue1_.EnQue(gradIn);
            gradIn = vecInQueue1_.DeQue<T>();
            for (uint32_t offsetN = 0; offsetN < N_; offsetN += lenN) {
                const uint32_t currentN = Min(lenN, N_ - offsetN);
                LocalTensor<T> xIn;
                vecInQueue0_.AllocTensor<T>(xIn);
                copy.blockCount = currentN;
                copy.srcStride = (D_ - currentD) * sizeof(T);
                DataCopyPad(xIn, xGm_[(row * N_ + offsetN) * D_ + offsetD], copy, pad);
                vecInQueue0_.EnQue(xIn);
                vecInQueue0_.DeQue<T>(xIn);
                if (offsetD == 0) {
                    VFDoPreProcessV0<T, P, false>((__ubuf__ P*)hPreGradBuf.GetPhyAddr(), (__ubuf__ T*)xIn.GetPhyAddr(),
                                                  (__ubuf__ T*)gradIn.GetPhyAddr(), currentN, currentD, row, offsetN,
                                                  runBSStart, N_);
                } else {
                    VFDoPreProcessV0<T, P, true>((__ubuf__ P*)hPreGradBuf.GetPhyAddr(), (__ubuf__ T*)xIn.GetPhyAddr(),
                                                 (__ubuf__ T*)gradIn.GetPhyAddr(), currentN, currentD, row, offsetN,
                                                 runBSStart, N_);
                }
                PipeBarrier<PIPE_V>();
                vecInQueue0_.FreeTensor(xIn);
            }
            vecInQueue1_.FreeTensor(gradIn);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ClearReductionOutputs()
{
    if ASCEND_IS_AIC {
        cubeCompute_.End();
    }
    if ASCEND_IS_AIV {
        constexpr uint32_t chunkElements = INOUT_QUEUE_SIZE / sizeof(P);
        LocalTensor<P> zeros = vecOutQueue1_.AllocTensor<P>();
        Duplicate(zeros, static_cast<P>(0), chunkElements);
        vecOutQueue1_.EnQue(zeros);
        zeros = vecOutQueue1_.DeQue<P>();
        const uint64_t phiElements = static_cast<uint64_t>(fusionSize_) * nD_;
        DataCopyExtParams copy = {1, 0, 0, 0, 0};
        for (uint64_t offset = static_cast<uint64_t>(blockIdx_) * chunkElements; offset < phiElements;
             offset += static_cast<uint64_t>(vecCoreNum_) * chunkElements) {
            copy.blockLen =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(chunkElements), phiElements - offset)) * sizeof(P);
            DataCopyPad(gradPhiGm_[offset], zeros, copy);
        }
        if (blockIdx_ == 0) {
            copy.blockLen = ALPHA_GRAD_LAST_DIM_SIZE * sizeof(P);
            DataCopyPad(alphaGradGm_, zeros, copy);
            copy.blockLen = fusionSize_ * sizeof(P);
            DataCopyPad(gradBiasGm_, zeros, copy);
        }
        vecOutQueue1_.FreeTensor(zeros);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV0Main()
{
    if (blockIdx_ >= usedVecCoreNum_) {
        return;
    }
    uint64_t dealBsNum = dealEndBS_ - dealStartBS_;
    uint64_t vecRunTimes = CeilDiv(dealBsNum, static_cast<uint64_t>(vecDealChunk_));
    float alphaPre = alphaGm_.GetValue(0);
    float alphaPost = alphaGm_.GetValue(1);
    float alphaComb = alphaGm_.GetValue(2);
    paramBias_ = paramsBuf_.Get<P>();
    DataCopyPadParams biasPadParams;
    biasPadParams.isPad = false;
    DataCopyParams biasCopyParams;
    biasCopyParams.blockCount = 1;
    biasCopyParams.blockLen = static_cast<uint32_t>(fusionSize_ * sizeof(P));
    biasCopyParams.srcStride = 0;
    biasCopyParams.dstStride = 0;
    DataCopyPad(paramBias_, biasGm_, biasCopyParams, biasPadParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<P> sumBuf = fp32TBuf_.GetWithOffset<P>(fusionSize_, globalUbOffset_);
    globalUbOffset_ += CeilAlign<uint32_t>(fusionSize_, CEIL_ALIGN_DEFAULT / sizeof(P)) * sizeof(P);
    // bias 和部分和均按 32 字节对齐，兼容奇数 N。
    LocalTensor<P> biasGradSumBuf =
        paramsBuf_.Get<P>()[CeilAlign<uint32_t>(fusionSize_, CEIL_ALIGN_DEFAULT / sizeof(P))];
    Duplicate(biasGradSumBuf, 0.0f, fusionSize_);
    PipeBarrier<PIPE_V>();
    ProcessV0MainLoop(vecRunTimes, alphaPre, alphaPost, alphaComb, sumBuf, biasGradSumBuf);
    ProcessV0MainAlphaReduce(sumBuf, GetBlockIdx());
    // 每个 AIV 累计完成后写回一次 bias 部分和。
    DataCopyParams biasOutCopyParams;
    biasOutCopyParams.blockCount = 1;
    biasOutCopyParams.blockLen = static_cast<uint32_t>(fusionSize_ * sizeof(P));
    biasOutCopyParams.srcStride = 0;
    biasOutCopyParams.dstStride = 0;
    if (usedVecCoreNum_ > 1U) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(GetBlockIdx())], biasGradSumBuf, biasOutCopyParams);
    }
    // 跨核同步前等待 hMixGrad 和 RMS 梯度系数的 GM 写入完成。
    PipeBarrier<PIPE_MTE3>();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV0MainLoop(uint64_t vecRunTimes, float alphaPre,
                                                                             float alphaPost, float alphaComb,
                                                                             LocalTensor<P>& sumBuf,
                                                                             LocalTensor<P>& biasGradSumBuf)
{
    for (uint64_t cIdx = 0; cIdx < vecRunTimes; cIdx++) {
        uint64_t runBSStart = cIdx * vecDealChunk_ + dealStartBS_;
        uint64_t runBSEnd = runBSStart + vecDealChunk_;
        if (runBSEnd > dealEndBS_) {
            runBSEnd = dealEndBS_;
        }
        V0V1Buffers<P> buffers;
        AllocV0V1Buffers(runBSStart, runBSEnd, buffers);
        PreProcessV0(buffers.hPreGradBuf, runBSStart, runBSEnd);
        LocalTensor<P> hMixInput;
        // hMix 输入跨 V0/V1 复用，V1 完成后释放队列槽。
        ProcessV0(runBSStart, runBSEnd, buffers, alphaPre, alphaPost, alphaComb, hMixInput);
        ProcessV1(runBSStart, runBSEnd, buffers, cIdx, sumBuf, biasGradSumBuf, hMixInput);
        vecInQueue1_.FreeTensor(hMixInput);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV0MainAlphaReduce(LocalTensor<P>& sumBuf,
                                                                                    uint32_t coreId)
{
    const bool directParams = usedVecCoreNum_ == 1U;
    // 单 AIV 时用 invRms 缓冲区保存 alpha 部分和，供 V3 输出；V2 不覆盖此区域。
    static_assert(VEC_DEAL_CHUNK_LARGE_N >= ALPHA_GRAD_PADDING, "parameter partials exceed invRms scratch");
    LocalTensor<P> alphaOutBuf;
    if (directParams) {
        alphaOutBuf = invScratchBuf_.Get<P>();
    } else {
        alphaOutBuf = vecOutQueue1_.AllocTensor<P>();
    }
    uint32_t srcShape1[] = {1, N_};
    uint32_t srcShape2[] = {1, 2 * N_};
    uint32_t srcShape3[] = {1, fusionSize_};
    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf, sumBuf, srcShape1, true);
    PipeBarrier<PIPE_V>();
    // 显式填零隔离已归约分量，避免 Inf/NaN 乘零污染后续归约。
    Duplicate(sumBuf, ZERO, N_);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf[8], sumBuf, srcShape2, true);
    PipeBarrier<PIPE_V>();
    Duplicate(sumBuf, ZERO, N_ * 2);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::AR, true>(alphaOutBuf[16], sumBuf, srcShape3, true);
    PipeBarrier<PIPE_V>();
    if (directParams) {
        return;
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    vecOutQueue1_.EnQue(alphaOutBuf);
    alphaOutBuf = vecOutQueue1_.DeQue<P>();
    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = ALPHA_GRAD_PADDING * sizeof(P);
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetAlphaGradOffset(coreId)], alphaOutBuf, bsCopyParams);
    vecOutQueue1_.FreeTensor(alphaOutBuf);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::AllocV0V1Buffers(uint64_t runBSStart, uint64_t runBSEnd,
                                                                            V0V1Buffers<P>& buffers)
{
    buffers.stepLength = static_cast<uint32_t>((runBSEnd - runBSStart) * N_);
    uint32_t hMixLength = static_cast<uint32_t>((runBSEnd - runBSStart) * N_ * N_);
    buffers.gatherLength = buffers.stepLength * 2 + hMixLength;
    uint32_t stepOffset1 = CeilAlign(uint32_t(buffers.stepLength * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    uint32_t hpreGradTotlenLen = CeilAlign(buffers.stepLength, uint32_t(CEIL_ALIGN_DEFAULT));
    buffers.hPreGradBuf = fp32TBuf_.GetWithOffset<P>(hpreGradTotlenLen, globalUbOffset_);

    uint32_t ubOffset = globalUbOffset_ + hpreGradTotlenLen * sizeof(P);
    uint32_t fusionOffset = ubOffset;

    // V0 缓冲区
    buffers.hPreBufS1 = fp32TBuf_.GetWithOffset<P>(hPreMaxBufLen_, ubOffset);
    buffers.hPostBufS1 = fp32TBuf_.GetWithOffset<P>(hPostMaxBufLen_, ubOffset + gradHPostOffset_);
    buffers.hResBufS1 = fp32TBuf_.GetWithOffset<P>(gradHResBufLen_, ubOffset + hMixOffset_);
    buffers.hFusionBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    ubOffset += hFusionBufLen_ * sizeof(P);
    fusionOffset = ubOffset;

    buffers.hPreBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    ubOffset += stepOffset1;
    buffers.hPostBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    // AIV02Process 使用整寄存器写入拼接的 dL/dz，为尾块预留冗余空间。
    const uint32_t gatherElems = hFusionBufLen_ + VECTOR_OVERWRITE_SLACK;
    buffers.gatherFusionOutBuf = fp32TBuf_.GetWithOffset<P>(gatherElems, fusionOffset);

    ubOffset += gatherElems * sizeof(P);
    buffers.invRmsBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    const uint32_t calcTmpElems = hFusionBufLen_ + VECTOR_OVERWRITE_SLACK;
    buffers.calcTmpBuf = fp32TBuf_.GetWithOffset<P>(calcTmpElems, ubOffset);
    ubOffset += calcTmpElems * sizeof(P);
    buffers.brcbTmpBuf = fp32TBuf_.GetWithOffset<uint8_t>(FP32_BUF_SIZE - ubOffset, ubOffset);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV0(uint64_t runBSStart, uint64_t runBSEnd,
                                                                     V0V1Buffers<P>& buffers, float alphaPre,
                                                                     float alphaPost, float alphaComb,
                                                                     LocalTensor<P>& hMixInput)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    // premix: V0 预计算的 HPreGrad 直写 grad_pre_mix 输出；
    // 随后用外部 grad_pre 覆盖 hPreGradBuf，sigmoid 反传链的 dPre 改由外部输入提供。
    if (hasPreMix_) {
        dataCopyParams.blockLen = buffers.stepLength * sizeof(P);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
        DataCopyPad(gradPreMixGm_[runBSStart * N_], buffers.hPreGradBuf, dataCopyParams);
        // MTE3 读取 hPreGradBuf 完成后才允许 MTE2 覆写同一缓冲区。
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        DataCopyPad(buffers.hPreGradBuf, gradPreGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }

    auto hMixGradBuf = buffers.calcTmpBuf;
    dataCopyParams.blockLen = buffers.stepLength * sizeof(P);

    uint32_t currentDealBsNum = static_cast<uint32_t>(runBSEnd - runBSStart);

    // 加载 grad_h_post，并重算 post 门控。
    AscendC::LocalTensor<P> gradHPost;
    vecInQueue0_.AllocTensor<P>(gradHPost);
    dataCopyParams.blockLen = buffers.stepLength * sizeof(P);
    DataCopyPad(gradHPost, gradHPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHPost);

    AscendC::LocalTensor<P> hMixBufLocal = vecInQueue1_.AllocTensor<P>();
    DataCopyParams hMixCopyParams;
    hMixCopyParams.blockCount = 1;
    hMixCopyParams.blockLen = static_cast<uint32_t>(currentDealBsNum * fusionSize_ * sizeof(P));
    hMixCopyParams.srcStride = 0;
    hMixCopyParams.dstStride = 0;
    DataCopyPad(hMixBufLocal, hMixGm_[runBSStart * fusionSize_], hMixCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(hMixBufLocal);

    // 当前行块的 inv_rms 跨 V0/V1 复用。
    LocalTensor<P> invRmsBufLocal = invScratchBuf_.Get<P>();
    DataCopyParams invCopyParams;
    invCopyParams.blockCount = 1;
    invCopyParams.blockLen = static_cast<uint32_t>(currentDealBsNum * sizeof(P));
    invCopyParams.srcStride = 0;
    invCopyParams.dstStride = 0;
    DataCopyPad(invRmsBufLocal, invRmsGm_[runBSStart], invCopyParams, dataCopyPadParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);

    vecInQueue0_.DeQue<P>(gradHPost);
    hMixBufLocal = vecInQueue1_.DeQue<P>();

    // 使用 hc_before_norm 和 inv_rms 重算 z_pre。
    VFDoV0PreGrad<P>((__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr(), (__ubuf__ P*)hMixBufLocal.GetPhyAddr(),
                     (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), (__ubuf__ P*)buffers.hPreGradBuf.GetPhyAddr(),
                     (__ubuf__ P*)paramBias_.GetPhyAddr(), alphaPre, currentDealBsNum, fusionSize_, N_);
    VFDoV0PostGrad<P>((__ubuf__ P*)buffers.hPostBufS1.GetPhyAddr(), (__ubuf__ P*)hMixBufLocal.GetPhyAddr(),
                      (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), (__ubuf__ P*)gradHPost.GetPhyAddr(),
                      (__ubuf__ P*)paramBias_.GetPhyAddr() + N_, alphaPost, currentDealBsNum, fusionSize_, N_);
    vecInQueue0_.FreeTensor(gradHPost);

    ProcessSinkhorn(runBSStart, currentDealBsNum, buffers.hResBufS1);

    // Sinkhorn 反向完成后计算 exp/max 的梯度。
    VFDoV0ResGrad<P>((__ubuf__ P*)buffers.hResBufS1.GetPhyAddr(), (__ubuf__ P*)hMixBufLocal.GetPhyAddr(),
                     (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), (__ubuf__ P*)buffers.hResBufS1.GetPhyAddr(),
                     (__ubuf__ P*)paramBias_.GetPhyAddr() + 2 * N_, alphaComb, currentDealBsNum, fusionSize_, N_);
    hMixInput = hMixBufLocal;

    AIV02Process<P>(buffers, hMixGradBuf, currentDealBsNum, alphaPre, alphaPost, alphaComb, N_);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessSinkhorn(uint64_t runBSStart,
                                                                           uint32_t currentDealBsNum,
                                                                           LocalTensor<P>& hResBuf)
{
    sinkhorn_.Process(runBSStart, currentDealBsNum, hResBuf, vecInQueue0_);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV1(uint64_t runBSStart, uint64_t runBSEnd,
                                                                     V0V1Buffers<P>& buffers, uint64_t vecRuntimesId,
                                                                     LocalTensor<P>& sumBuf,
                                                                     LocalTensor<P>& biasGradSumBuf,
                                                                     const LocalTensor<P>& hMixInput)
{
    uint32_t curBSSize = static_cast<uint32_t>(runBSEnd - runBSStart);
    constexpr bool isReuse = false;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    AscendC::LocalTensor<P> fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    AscendC::LocalTensor<P> invRmsGradUb = vecOutQueueSmall_.AllocTensor<P>();
    if (N_ == 8) {
        VFDoV1ProcessBiasGradForN8<P>((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                      (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize, fusionSize_);
    } else if ((fusionSize_ % (CEIL_ALIGN_DEFAULT / sizeof(P))) == 0) {
        VFDoV1ProcessBiasGradForN4N6<P, true>((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                              (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize,
                                              fusionSize_);
    } else {
        VFDoV1ProcessBiasGradForN4N6<P, false>((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                               (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize,
                                               fusionSize_);
    }
    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> BiasOutBuf = vecOutQueue1_.DeQue<P>();

    if (vecRuntimesId != 0) {
        Add(biasGradSumBuf, biasGradSumBuf, BiasOutBuf, fusionSize_);
    } else {
        Adds(biasGradSumBuf, BiasOutBuf, ZERO, fusionSize_);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue1_.FreeTensor(BiasOutBuf);

    auto hMixGradBuf = buffers.calcTmpBuf;
    fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    LocalTensor<P> invRmsBufLocal = invScratchBuf_.Get<P>();
    const uint32_t xRowSumBroadCastDst[2] = {curBSSize, fusionSize_};
    const uint32_t xRowSumBroadCastSrc[2] = {curBSSize, 1};
    BroadCast<float, 2, 1>(buffers.invRmsBuf, invRmsBufLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc,
                           buffers.brcbTmpBuf);
    PipeBarrier<PIPE_V>();

    VFDoV1Process<P>((__ubuf__ P*)fp32OutBuf.GetPhyAddr(), (__ubuf__ P*)hMixGradBuf.GetPhyAddr(),
                     (__ubuf__ P*)buffers.hFusionBuf.GetPhyAddr(), (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(),
                     (__ubuf__ P*)hMixInput.GetPhyAddr(), (__ubuf__ P*)buffers.invRmsBuf.GetPhyAddr(), curBSSize,
                     fusionSize_);

    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> hMixGradOutBuf = vecOutQueue1_.DeQue<P>();
    dataCopyParams.blockLen = curBSSize * fusionSize_ * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(runBSStart)], hMixGradOutBuf, dataCopyParams);
    PipeBarrier<PIPE_V>();

    uint32_t shape[] = {curBSSize, fusionSize_};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(invRmsGradUb, hMixGradBuf, buffers.brcbTmpBuf, shape, true);
    vecOutQueueSmall_.EnQue(invRmsGradUb);
    LocalTensor<P> invRmsGradUbtBuf = vecOutQueueSmall_.DeQue<P>();

    VFDoV1InvRmsCoeff<P>((__ubuf__ P*)invRmsGradUbtBuf.GetPhyAddr(), (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(),
                         curBSSize, scaleMean_);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    dataCopyParams.blockLen = curBSSize * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(runBSStart)], invRmsGradUbtBuf, dataCopyParams);
    if (N_ == 6U && curBSSize > N6_NONREUSE_MAX_ROWS) {
        // N=6 超过 32 行时复用 alpha 乘积缓冲区，满足 RA 临时空间需求。
        ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(hMixGradBuf, buffers.hFusionBuf, buffers.brcbTmpBuf, shape,
                                                             true);
    } else {
        ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(hMixGradBuf, buffers.hFusionBuf, buffers.brcbTmpBuf,
                                                                shape, true);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue1_.FreeTensor(hMixGradOutBuf);
    vecOutQueueSmall_.FreeTensor(invRmsGradUbtBuf);

    if (vecRuntimesId == 0) {
        Adds(sumBuf, hMixGradBuf, ZERO, fusionSize_);
    } else {
        Add(sumBuf, hMixGradBuf, sumBuf, fusionSize_);
    }
    PipeBarrier<PIPE_V>();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessC0C1Pipeline()
{
    if (GetBlockIdx() >= v1UsedCubeCoreNum_) {
        // 空闲 AIC 也需回收初始化时设置的事件。
        cubeCompute_.End();
        return;
    }
    bool enableHf32 = tiling_->implMode == MHC_PRE_BACKWARD_IMPL_MODE_HF32;
    int64_t oriHf32Mode = 0;
    int64_t oriHf32TransMode = 0;
    if ASCEND_IS_AIC {
        if (enableHf32) {
            oriHf32Mode = AscendC::GetCtrlSpr<AscendC::HF32_MODE_BIT, AscendC::HF32_MODE_BIT>();
            oriHf32TransMode = AscendC::GetCtrlSpr<AscendC::HF32_TRANS_MODE_BIT, AscendC::HF32_TRANS_MODE_BIT>();
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
    }

    if (tiling_->cubeCachePlan.hMixGradMode == HMixGradCacheMode::FULL) {
        cubeCompute_.PreloadHMixGrad(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(0)]);
    }

    if (tiling_->cubeCachePlan.hMixGradMode == HMixGradCacheMode::TILED) {
        ProcessStage2Tiles<true>();
    } else {
        ProcessStage2Tiles<false>();
    }
    cubeCompute_.WaitC0UbBuffersReleased();
    cubeCompute_.End();

    if ASCEND_IS_AIC {
        if (enableHf32) {
            AscendC::SetHF32TransMode(oriHf32TransMode);
            AscendC::SetHF32Mode(oriHf32Mode);
        }
    }
}

template <class T, class P>
template <bool TILED_HMIX_GRAD>
__aicore__ inline __attribute__((always_inline)) void MhcPreSinkhornBackwardKernel<T, P>::ProcessStage2Tiles()
{
    uint32_t currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - dealStartND_);
    uint32_t currentBSBlock = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_));
    uint32_t buffId = 0;
    // 预先发射首个 C0 分块，启动 C0/V2/C1 流水。
    ProcessC0<TILED_HMIX_GRAD>(dealStartND_, currentNDBlock, 0, currentBSBlock, buffId);

    // 各缓存模式均按 M 外层、ND 内层遍历，保持 gradPhi 沿 M 递增累加。
    for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
        currentBSBlock = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));
        for (uint32_t offsetND = dealStartND_; offsetND < dealEndND_; offsetND += ND_BLOCK_SIZE) {
            currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - offsetND);
            uint32_t nextND = offsetND + ND_BLOCK_SIZE;
            uint64_t nextBS = offsetBS;
            if (nextND >= dealEndND_) {
                nextND = dealStartND_;
                nextBS += BS_BLOCK_SIZE;
            }
            // 先发射下一 Tile 的 C0，再等待当前 Tile 的 V2 并执行 C1。
            if (nextBS < totalLength_) {
                uint32_t nextRows =
                    static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - nextBS));
                ProcessC0<TILED_HMIX_GRAD>(nextND, Min(ND_BLOCK_SIZE, dealEndND_ - nextND), nextBS, nextRows,
                                           buffId + 1);
            }
            ProcessC1<TILED_HMIX_GRAD>(offsetND, currentNDBlock, offsetBS, currentBSBlock, buffId);
            ++buffId;
        }
    }
}

template <class T, class P>
template <bool TILED_HMIX_GRAD>
__aicore__ inline __attribute__((always_inline)) void MhcPreSinkhornBackwardKernel<T, P>::ProcessC0(
    uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS, uint32_t currentBSBlock, uint32_t buffIdx)
{
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    uint64_t writeOffset = workspaceBuf_.GetXRsGradOffset(GetBlockIdx(), buffIdx);
    if constexpr (TILED_HMIX_GRAD) {
        cubeCompute_.PrepareHMixGradTile(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], currentBSBlock,
                                         offsetBS, offsetND == dealStartND_);
    }
    cubeCompute_.ProcessC0(mmConfigC0_, workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], phiGm_[offsetND],
                           workSpaceGm_[writeOffset], currentBSBlock, currentNDBlock, fusionSize_, nD_, offsetBS,
                           offsetND - dealStartND_);
}

template <class T, class P>
template <bool TILED_HMIX_GRAD>
__aicore__ inline __attribute__((always_inline)) void MhcPreSinkhornBackwardKernel<T, P>::ProcessC1(
    uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS, uint32_t currentBSBlock, uint32_t buffIdx)
{
    AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C1);
    // 两个 AIV 分别写入 ND 的前后半块，C1 读取整个 ND 分块。
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    uint64_t offsetXs = workspaceBuf_.GetXRsOffset(GetBlockIdx(), buffIdx);
    bool enAtomic = (offsetBS != 0);
    cubeCompute_.ProcessC1(mmConfigC1_, workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], workSpaceGm_[offsetXs],
                           gradPhiGm_[offsetND], fusionSize_, currentNDBlock, currentBSBlock, fusionSize_,
                           ND_BLOCK_SIZE, nD_, enAtomic, offsetBS);
    if constexpr (TILED_HMIX_GRAD) {
        if (offsetND + currentNDBlock == dealEndND_) {
            cubeCompute_.ReleaseHMixGradTile(offsetBS);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2Pipeline()
{
    if (blockIdx_ >= 2 * v1UsedCubeCoreNum_) {
        return;
    }
    if (directC0_) {
        CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_V>(C0_UB_BUFFER_FREE_FLAG_BASE);
        CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_V>(C0_UB_BUFFER_FREE_FLAG_BASE + 1);
    }
    // 预留索引表原占用的 UB 空间，保持 V2 缓冲区起始地址和 Bank 布局。
    globalUbOffset_ = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t);

    // 与 AIC 使用相同的 M 外层、ND 内层顺序；buffId 标识同一个逻辑 Tile。
    uint32_t vecIdx = blockIdx_ % VEC_CORE_VECIDX_MOD;
    uint32_t buffId = 0;
    for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
        const uint32_t rows = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));
        for (uint32_t baseND = dealStartND_; baseND < dealEndND_; baseND += ND_BLOCK_SIZE) {
            const uint32_t offsetND = baseND + (ND_BLOCK_SIZE / 2) * vecIdx;
            if (offsetND >= dealEndND_) {
                // 64 列尾块仍由两个 AIV 参与握手，空半块消费 C0 事件后通知 C1。
                for (uint32_t row = 0; row < rows; row += MATMUL_WRITE_OFFSET_M) {
                    CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C0);
                }
                CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
            } else {
                ProcessV2(offsetND, Min(ND_BLOCK_SIZE / 2, dealEndND_ - offsetND), offsetBS, offsetBS + rows, buffId);
                CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
            }
            ++buffId;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2(uint32_t offsetND, uint32_t copySizeND,
                                                                     uint64_t bsStart, uint64_t bsEnd, uint32_t buffId)
{
    // 保留运行时除法，避免常量形状编译将小 D 分块的通道下标错误折叠为 0。
    volatile uint32_t channelOffset = offsetND;
    uint32_t currentN = channelOffset / D_;
    // 按 64 行分块，降低 V2 的 TBuf 峰值占用。
    uint32_t chunkSize = PROCESS_V2_CHUNK_SIZE / NUMBER_TWO;
    uint32_t currentBsSize = static_cast<uint32_t>(bsEnd - bsStart);
    buffId = buffId % VEC_CORE_VECIDX_MOD;
    uint32_t ubOffset = globalUbOffset_;
    uint32_t maxChunkNDSize = chunkSize * copySizeND;

    xRsFp32Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset = CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    xGradVec3Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset = CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    invRmsUb_ = fp32TBuf_.GetWithOffset<P>(currentBsSize, ubOffset);
    ubOffset = CeilAlign(uint32_t(ubOffset + currentBsSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    uint32_t hPreUbLen = CeilAlign(uint32_t(currentBsSize * N_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT)) / sizeof(P);
    hPreUb_ = fp32TBuf_.GetWithOffset<P>(hPreUbLen, ubOffset);
    ubOffset = CeilAlign(uint32_t(ubOffset + currentBsSize * N_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    bsCopyParams.blockLen = currentBsSize * N_ * sizeof(P);
    // premix: xGradVec3 的加权源由 h_pre 切换为 pre_mix。
    GlobalTensor<P>& hPreSrcGm = hasPreMix_ ? preMixGm_ : hPreGm_;
    DataCopyPad(hPreUb_, hPreSrcGm[bsStart * N_], bsCopyParams, dataCopyPadParams);

    // 一次加载整个 BS 分块的 RMS 梯度系数，供各行分块复用。
    ProcessV2ChunkLoadInvRms(bsStart, currentBsSize);
    for (uint64_t bsChunkStart = bsStart; bsChunkStart < bsEnd; bsChunkStart += chunkSize) {
        uint64_t bsChunkEnd = Min(bsChunkStart + chunkSize, bsEnd);
        uint32_t currentChunkSize = static_cast<uint32_t>(bsChunkEnd - bsChunkStart);
        uint32_t chunkNDSize = currentChunkSize * copySizeND;

        // C0 每次输出 128 行，由两个 64 行分块依次消费。
        if ((bsChunkStart - bsStart) % MATMUL_WRITE_OFFSET_M == 0) {
            AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C0);
        }
        ProcessV2ChunkLoop(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, chunkNDSize, currentN,
                           buffId);
        if (directC0_ && ((bsChunkEnd - bsStart) % MATMUL_WRITE_OFFSET_M == 0 || bsChunkEnd == bsEnd)) {
            CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_V>(C0_UB_BUFFER_FREE_FLAG_BASE +
                                                            c0UbBufferIndex_ % DOUBLE_BUFFER);
            ++c0UbBufferIndex_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND,
                                                                              uint64_t bsStart, uint64_t bsChunkStart,
                                                                              uint32_t currentChunkSize,
                                                                              uint32_t chunkNDSize, uint32_t currentN,
                                                                              uint32_t buffId)
{
    ProcessV2ChunkLoadHInGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, currentN, bsStart);

    ProcessV2ChunkLoadX(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, buffId);

    ProcessV2ChunkProcessXRsGradMm(bsStart, bsChunkStart, copySizeND, currentChunkSize, buffId);

    ProcessV2ChunkOutputXGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, chunkNDSize);
}

// 加载 V1 预计算的 RMS 梯度系数
template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkLoadInvRms(uint64_t bsChunkStart,
                                                                                    uint32_t currentChunkSize)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = currentChunkSize * sizeof(P);
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;

    invRmsGradBuf_ = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(invRmsGradBuf_, workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(bsChunkStart)], bsCopyParams,
                dataCopyPadParams);
    vecInQueue1_.EnQue(invRmsGradBuf_);
    LocalTensor<P> invRmsGradUb = vecInQueue1_.DeQue<P>();
    Muls(invRmsUb_, invRmsGradUb, ONE, currentChunkSize);
    PipeBarrier<PIPE_V>();
    vecInQueue1_.FreeTensor(invRmsGradUb);
}

// 加载 hInGrad
template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkLoadHInGrad(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t currentN,
    uint64_t bsStart)
{
    DataCopyPadExtParams<T> dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyExtParams blockCopyParams;
    blockCopyParams.blockCount = currentChunkSize;
    blockCopyParams.blockLen = copySizeND * sizeof(T);
    blockCopyParams.srcStride = (D_ - copySizeND) * sizeof(T);
    blockCopyParams.dstStride = 0;
    if (smallM_) {
        gradHInInBuf_ = vecInQueue1_.AllocTensor<T>();
        DataCopyPad(gradHInInBuf_, gradHInGm_[bsChunkStart * D_ + (offsetND % D_)], blockCopyParams, dataCopyPadParams);
        vecInQueue1_.EnQue(gradHInInBuf_);
        gradHInInBuf_ = vecInQueue1_.DeQue<T>();
    } else {
        vecInQueue0_.AllocTensor<T>(gradHInInBuf_);
        DataCopyPad(gradHInInBuf_, gradHInGm_[bsChunkStart * D_ + (offsetND % D_)], blockCopyParams, dataCopyPadParams);
        vecInQueue0_.EnQue(gradHInInBuf_);
        vecInQueue0_.DeQue<T>(gradHInInBuf_);
    }

    __ubuf__ T* gradHInInBufAddr = (__ubuf__ T*)gradHInInBuf_.GetPhyAddr();
    __ubuf__ P* hPreUbAddr = (__ubuf__ P*)hPreUb_.GetPhyAddr();
    __ubuf__ P* xGradVec3BufAddr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    VFDoV2HInMulHPre<T, P>(xGradVec3BufAddr, gradHInInBufAddr, hPreUbAddr, currentChunkSize, copySizeND, currentN,
                           static_cast<uint32_t>(bsChunkStart - bsStart), N_);
    if (smallM_) {
        vecInQueue1_.FreeTensor(gradHInInBuf_);
    } else {
        vecInQueue0_.FreeTensor(gradHInInBuf_);
    }
}

// 加载 x 并转换为 FP32
template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkLoadX(uint32_t offsetND, uint32_t copySizeND,
                                                                               uint64_t bsStart, uint64_t bsChunkStart,
                                                                               uint32_t currentChunkSize,
                                                                               uint32_t buffId)
{
    DataCopyExtParams xCopyParams;
    DataCopyPadExtParams<T> xCopyPadParams;
    xCopyParams.blockCount = currentChunkSize;
    xCopyParams.blockLen = copySizeND * sizeof(T);
    xCopyParams.srcStride = (nD_ - copySizeND) * sizeof(T);
    xCopyParams.dstStride = 0;

    bf16InputBuf_ = vecInQueue1_.AllocTensor<T>();
    // 先转为 64 位再相乘，避免地址溢出。
    DataCopyPad(bf16InputBuf_, xGm_[bsChunkStart * nD_ + offsetND], xCopyParams, xCopyPadParams);
    vecInQueue1_.EnQue(bf16InputBuf_);
    bf16InputBuf_ = vecInQueue1_.DeQue<T>();

    // 保留 FP32 x 供梯度计算，并输出给 C1。
    __ubuf__ T* bf16InputAddr = (__ubuf__ T*)bf16InputBuf_.GetPhyAddr();
    __ubuf__ P* xRsFp32Addr = (__ubuf__ P*)xRsFp32Buf_.GetPhyAddr();
    xRsOutUb_ = vecOutQueue1_.AllocTensor<P>();
    __ubuf__ P* xRsOutAddr = (__ubuf__ P*)xRsOutUb_.GetPhyAddr();
    VFDoV2CastXToFp32<T, P>(xRsFp32Addr, xRsOutAddr, bf16InputAddr, currentChunkSize, copySizeND);
    vecInQueue1_.FreeTensor(bf16InputBuf_);
    vecOutQueue1_.EnQue(xRsOutUb_);
    xRsOutUb_ = vecOutQueue1_.DeQue<P>();

    // 将 xRs 写入 Workspace
    DataCopyExtParams fp32ExtCopyParams;
    fp32ExtCopyParams.blockCount = currentChunkSize;
    fp32ExtCopyParams.blockLen = copySizeND * sizeof(P);
    fp32ExtCopyParams.srcStride = 0;
    fp32ExtCopyParams.dstStride = (ND_BLOCK_SIZE - copySizeND) * sizeof(P);

    uint64_t offset = workspaceBuf_.GetXRsOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
    uint32_t vecIdx = blockIdx_ % NUMBER_TWO;
    offset += (bsChunkStart - bsStart) * ND_BLOCK_SIZE;
    offset += vecIdx * (ND_BLOCK_SIZE / 2);

    DataCopyPad(workSpaceGm_[offset], xRsOutUb_, fp32ExtCopyParams);
    vecOutQueue1_.FreeTensor(xRsOutUb_);
}

// 计算 xRsGrad
template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkProcessXRsGradMm(
    uint64_t bsStart, uint64_t bsChunkStart, uint32_t copySizeND, uint32_t currentChunkSize, uint32_t buffId)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    uint32_t vecIdx = blockIdx_ % NUMBER_TWO;
    uint64_t offsetXRsGradMm = workspaceBuf_.GetXRsGradOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
    offsetXRsGradMm += (bsChunkStart - bsStart) * ND_BLOCK_SIZE + vecIdx * (ND_BLOCK_SIZE / 2);

    if (directC0_) {
        uint32_t rowInTile = (bsChunkStart - bsStart) % MATMUL_WRITE_OFFSET_M;
        xRsGradMmInBuf_ = LocalTensor<P>(TPosition::VECIN,
                                         C0_UB_BUFFER_OFFSET + (c0UbBufferIndex_ % DOUBLE_BUFFER) * INOUT_QUEUE_SIZE +
                                             rowInTile * (ND_BLOCK_SIZE / 2) * sizeof(P),
                                         currentChunkSize * (ND_BLOCK_SIZE / 2));
    } else {
        vecInQueue0_.AllocTensor<P>(xRsGradMmInBuf_);
        DataCopyParams blockCopyParams;
        blockCopyParams.blockCount = currentChunkSize;
        blockCopyParams.blockLen = copySizeND * sizeof(P);
        blockCopyParams.srcStride = (ND_BLOCK_SIZE - copySizeND) * sizeof(P);
        blockCopyParams.dstStride = 0;
        DataCopyPad(xRsGradMmInBuf_, workSpaceGm_[offsetXRsGradMm], blockCopyParams, dataCopyPadParams);
        vecInQueue0_.EnQue(xRsGradMmInBuf_);
        vecInQueue0_.DeQue<P>(xRsGradMmInBuf_);
    }

    __ubuf__ P* xRsGradMmInBufAddr = (__ubuf__ P*)xRsGradMmInBuf_.GetPhyAddr();
    __ubuf__ P* xGradVec3Addr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    __ubuf__ P* xRsFp32Addr = (__ubuf__ P*)xRsFp32Buf_.GetPhyAddr();
    __ubuf__ P* invRmsAddr = (__ubuf__ P*)invRmsUb_.GetPhyAddr() + (bsChunkStart - bsStart);
    VFDoV2AccumulateXGrad<P>(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, currentChunkSize, copySizeND);
    if (!directC0_) {
        vecInQueue0_.FreeTensor(xRsGradMmInBuf_);
    }

    PipeBarrier<PIPE_V>();
}

// 转换并写回 grad_x
template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkOutputXGrad(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize)
{
    bf16OutputBuf_ = vecOutQueue1_.AllocTensor<T>();
    __ubuf__ T* outputAddr = (__ubuf__ T*)bf16OutputBuf_.GetPhyAddr();
    __ubuf__ P* xGradAddr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    VFDoV2CastXGrad<T, P>(outputAddr, xGradAddr, chunkNDSize);
    PipeBarrier<PIPE_V>();
    vecOutQueue1_.EnQue(bf16OutputBuf_);
    bf16OutputBuf_ = vecOutQueue1_.DeQue<T>();

    DataCopyExtParams bf16ExtCopyParams;
    bf16ExtCopyParams.blockCount = currentChunkSize;
    bf16ExtCopyParams.blockLen = copySizeND * sizeof(T);
    bf16ExtCopyParams.srcStride = 0;
    bf16ExtCopyParams.dstStride = (nD_ - copySizeND) * sizeof(T);
    DataCopyPad(xGradGm_[bsChunkStart * nD_ + offsetND], bf16OutputBuf_, bf16ExtCopyParams);
    vecOutQueue1_.FreeTensor(bf16OutputBuf_);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV3SingleCore()
{
    ProcessV3AlphaGradSingleCore();
    ProcessV3BiasGradSingleCore();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV3AlphaGradSingleCore()
{
    LocalTensor<P> alphaGradOutLocal = vecOutQueueSmall_.AllocTensor<P>();
    VFDoV3ProcessAlphaGradSingleCore<P>((__ubuf__ P*)alphaGradOutLocal.GetPhyAddr(),
                                        (__ubuf__ P*)invScratchBuf_.Get<P>().GetPhyAddr());
    vecOutQueueSmall_.EnQue(alphaGradOutLocal);
    alphaGradOutLocal = vecOutQueueSmall_.DeQue<P>();
    DataCopyParams dataCopyParams = {1, ALPHA_GRAD_LAST_DIM_SIZE * sizeof(P), 0, 0};
    DataCopyPad(alphaGradGm_, alphaGradOutLocal, dataCopyParams);
    vecOutQueueSmall_.FreeTensor(alphaGradOutLocal);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV3BiasGradSingleCore()
{
    LocalTensor<P> biasGradOutLocal = vecOutQueueSmall_.AllocTensor<P>();
    LocalTensor<P> biasGradInLocal =
        paramsBuf_.Get<P>()[CeilAlign<uint32_t>(fusionSize_, CEIL_ALIGN_DEFAULT / sizeof(P))];
    LocalTensor<uint8_t> tmpLocal = fp32TBuf_.GetWithOffset<uint8_t>(hFusionBufLen_ / 4, 0);
    constexpr bool isReuse = false;
    uint32_t biasGradShapeSrc[] = {1U, fusionSize_};
    ReduceSum<P, AscendC::Pattern::Reduce::RA, isReuse>(biasGradOutLocal, biasGradInLocal, tmpLocal, biasGradShapeSrc,
                                                        true);
    vecOutQueueSmall_.EnQue(biasGradOutLocal);
    biasGradOutLocal = vecOutQueueSmall_.DeQue<P>();
    DataCopyParams dataCopyParams = {1, static_cast<uint16_t>(fusionSize_ * sizeof(P)), 0, 0};
    DataCopyPad(gradBiasGm_, biasGradOutLocal, dataCopyParams);
    vecOutQueueSmall_.FreeTensor(biasGradOutLocal);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV3()
{
    ProcessV3AlphaGrad();
    ProcessV3BiasGrad();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV3AlphaGrad()
{
    LocalTensor<P> alphaGradInLocal = vecInQueue1_.AllocTensor<P>();
    LocalTensor<P> alphaGradOutLocal = vecOutQueueSmall_.AllocTensor<P>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = usedVecCoreNum_ * ALPHA_GRAD_PADDING * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyPad(alphaGradInLocal, workSpaceGm_[workspaceBuf_.GetAlphaGradOffset(0)], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(alphaGradInLocal);
    alphaGradInLocal = vecInQueue1_.DeQue<P>();
    VFDoV3ProcessAlphaGrad<P>((__ubuf__ P*)alphaGradOutLocal.GetPhyAddr(), (__ubuf__ P*)alphaGradInLocal.GetPhyAddr(),
                              usedVecCoreNum_);
    vecOutQueueSmall_.EnQue(alphaGradOutLocal);
    alphaGradOutLocal = vecOutQueueSmall_.DeQue<P>();
    PipeBarrier<PIPE_V>();
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = ALPHA_GRAD_LAST_DIM_SIZE * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(alphaGradGm_, alphaGradOutLocal, dataCopyParams);
    vecInQueue1_.FreeTensor(alphaGradInLocal);
    vecOutQueueSmall_.FreeTensor(alphaGradOutLocal);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV3BiasGrad()
{
    LocalTensor<uint8_t> tmpLocal = fp32TBuf_.GetWithOffset<uint8_t>(hFusionBufLen_ / 4, 0);
    constexpr bool isReuse = false;
    LocalTensor<P> biasGradInLocal = vecInQueue1_.AllocTensor<P>();
    LocalTensor<P> biasGradOutLocal = vecOutQueueSmall_.AllocTensor<P>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = fusionSize_ * usedVecCoreNum_ * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyPad(biasGradInLocal, workSpaceGm_[workspaceBuf_.GetBiasGradOffset(0)], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(biasGradInLocal);
    biasGradInLocal = vecInQueue1_.DeQue<P>();
    uint32_t biasGradShapeSrc[] = {usedVecCoreNum_, fusionSize_};
    ReduceSum<P, AscendC::Pattern::Reduce::RA, isReuse>(biasGradOutLocal, biasGradInLocal, tmpLocal, biasGradShapeSrc,
                                                        true);
    vecOutQueueSmall_.EnQue(biasGradOutLocal);
    biasGradOutLocal = vecOutQueueSmall_.DeQue<P>();
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = fusionSize_ * sizeof(P);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(gradBiasGm_[0], biasGradOutLocal, dataCopyParams);
    vecInQueue1_.FreeTensor(biasGradInLocal);
    vecOutQueueSmall_.FreeTensor(biasGradOutLocal);
}

} // namespace MhcPreSinkhornBackwardNs

#endif // __MHC_PRE_SINKHORN_BACKWARD_KERNEL_H_

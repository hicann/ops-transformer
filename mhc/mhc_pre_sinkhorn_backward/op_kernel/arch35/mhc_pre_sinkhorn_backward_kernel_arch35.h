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
#include "mhc_pre_sinkhorn_backward_utils_arch35.h"
#include "mhc_pre_sinkhorn_backward_cube_compute_arch35.h"
#include "mhc_pre_sinkhorn_backward_sinkhorn_arch35.h"

namespace MhcPreSinkhornBackwardNs {

using namespace AscendC;
using namespace MhcPreSinkhornBackwardUtils;

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
    __aicore__ inline void AIV02Process(V0V1Buffers<P>& buffers, LocalTensor<P>& hMixGradBuf, uint64_t currentDealBsNum,
                                        float alphaPre, float alphaPost, float alphaComb);
    __aicore__ inline void AIV21Process(LocalTensor<P>& invRmsInBuf, LocalTensor<P>& invRmsUb,
                                        LocalTensor<P>& invRmsGradUb, uint32_t currentChunkSize);
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

    // 向量化计算函数 (VFDo*)
    template <bool accumulate>
    __aicore__ inline void VFDoPreProcessV0(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr,
                                            __ubuf__ T* gradHInB16InAddr, uint16_t curLenN, uint32_t lenD, uint64_t i,
                                            uint32_t j, uint64_t runBSStart);
    __aicore__ inline void VFDoV0PreGrad(__ubuf__ P* gradPreOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr,
                                         __ubuf__ P* gradHPreAddr, __ubuf__ P* biasPreAddr, float alphaPre,
                                         uint32_t rows, uint32_t fusionSize, uint32_t n);
    __aicore__ inline void VFDoV0PostGrad(__ubuf__ P* gradPostOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr,
                                          __ubuf__ P* gradHPostAddr, __ubuf__ P* biasPostAddr, float alphaPost,
                                          uint32_t rows, uint32_t fusionSize, uint32_t n);
    template <uint32_t N>
    __aicore__ inline void VFDoV0PackedRes(__ubuf__ P* output, __ubuf__ P* hc, __ubuf__ P* inv, __ubuf__ P* upstream,
                                           __ubuf__ P* bias, float alpha, uint32_t rows);
    __aicore__ inline void VFDoV0ResGrad(__ubuf__ P* gradResOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr,
                                         __ubuf__ P* skAddr, __ubuf__ P* biasResAddr, float alphaRes, uint32_t rows,
                                         uint32_t fusionSize, uint32_t n);
    __aicore__ inline void VFDoV0ProcessGradHPost(__ubuf__ P* hPostIn, __ubuf__ P* gradHPostIn,
                                                  __ubuf__ P* gradHPostOut, uint32_t stepLen);
    __aicore__ inline void VFDoV1Process(__ubuf__ P* gradHMix, __ubuf__ P* gradInvRmsOut, __ubuf__ P* gradAlphaOut,
                                         __ubuf__ P* gatherIn, __ubuf__ P* hMixIn, __ubuf__ P* invRmsIn,
                                         uint32_t dealBSSize);
    __aicore__ inline void VFDoV1InvRmsCoeff(__ubuf__ P* invRmsGradAddr, __ubuf__ P* invRmsAddr, uint32_t count,
                                             float scaleMean);
    template <bool aligned>
    __aicore__ inline void VFDoV1ProcessBiasGradForN4N6(__ubuf__ P* outBufDst, __ubuf__ P* gatherFusion,
                                                        uint32_t curBSSize);
    __aicore__ inline void VFDoV1ProcessBiasGradForN8(__ubuf__ P* outBufDst, __ubuf__ P* gatherFusion,
                                                      uint32_t curBSSize);
    __aicore__ inline void VFDoV2HInMulHPre(__ubuf__ P* xGradVec3BufAddr, __ubuf__ T* hInGradInBufAddr,
                                            __ubuf__ P* hPreUbAddr, uint32_t currentChunkSize, uint32_t copySizeND,
                                            uint32_t currentN, uint32_t bsGlobalOffset);

    __aicore__ inline void VFDoV2CastXToFp32(__ubuf__ P* xRsFp32Addr, __ubuf__ P* xRsOutAddr, __ubuf__ T* bf16InputAddr,
                                             uint32_t currentChunkSize, uint32_t copySizeND);

    __aicore__ inline void VFDoV2AccumulateXGrad(__ubuf__ P* xGradVec3Addr, __ubuf__ P* xRsGradMmAddr,
                                                 __ubuf__ P* xRsFp32Addr, __ubuf__ P* invRmsAddr,
                                                 uint32_t currentChunkSize, uint32_t copySizeND);

    __aicore__ inline void VFDoV2CastXGrad(__ubuf__ T* outputAddr, __ubuf__ P* xGradAddr, uint32_t totalElem);

    __aicore__ inline void VFDoV3ProcessAlphaGrad(__ubuf__ P* alphaGradOut, __ubuf__ P* alphaGradIn);
    __aicore__ inline void VFDoV3ProcessAlphaGradSingleCore(__ubuf__ P* alphaGradOut, __ubuf__ P* alphaGradIn);

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
    GlobalTensor<P> workSpaceGm_;
    WorkspaceBuffer<P> workspaceBuf_; // Workspace 管理

    GlobalTensor<T> xGradGm_;     // 输出 grad_x
    GlobalTensor<P> gradPhiGm_;   // 输出 grad_phi
    GlobalTensor<P> alphaGradGm_; // 输出 grad_alpha
    GlobalTensor<P> gradBiasGm_;  // 输出 grad_bias

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
    uint16_t eleNumPerVf_;
    float scaleMean_;
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
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.bias));
    xGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.grad_x));
    xGradGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_phi));
    alphaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_alpha));
    gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.grad_bias));
    workSpaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.workspace));
    tiling_ = initParams.tilingData;
    smallM_ = tiling_->smallM != 0U;
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
    eleNumPerVf_ = GetVRegSize() / sizeof(P);
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
        pipe_->InitBuffer(
            paramsBuf_,
            MhcPreSinkhornBackwardUtils::CeilAlign<uint32_t>(static_cast<uint32_t>(fusionSize_ * sizeof(P)), 32U) * 2);
        pipe_->InitBuffer(invScratchBuf_,
                          MhcPreSinkhornBackwardUtils::CeilAlign<uint32_t>(vecDealChunk_ * sizeof(P), 32U));
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
template <bool accumulate>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoPreProcessV0(__ubuf__ P* hPreGradAddr,
                                                                            __ubuf__ T* xB16InAddr,
                                                                            __ubuf__ T* gradHInB16InAddr,
                                                                            uint16_t curLenN, uint32_t lenD, uint64_t i,
                                                                            uint32_t j, uint64_t runBSStart)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        uint32_t offset = static_cast<uint32_t>((i - runBSStart) * N_ + j);
        for (uint16_t offsetN = 0; offsetN < curLenN; offsetN++) { // x [curLenN, lenD],  hinGrad [1, lenD]
            Reg::Duplicate(sumReg, 0.0f);
            uint32_t curLenD = lenD;
            Reg::RegTensor<T> xInB16Reg, gradHInB16InReg;
            Reg::RegTensor<P> xFp32Reg, gradHInFp32Reg, mulReg, tmpSumPerVfReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < dLoopCnt; vfBlockIdx++) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(curLenD);

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(
                    xInB16Reg, xB16InAddr + offsetN * lenD + vfBlockIdx * eleNumPerVf);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradHInB16InReg,
                                                                  gradHInB16InAddr + vfBlockIdx * eleNumPerVf);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);

                Reg::Mul(mulReg, xFp32Reg, gradHInFp32Reg, mask);
                Reg::Reduce<Reg::ReduceType::SUM>(tmpSumPerVfReg, mulReg, mask);
                Reg::Add(sumReg, sumReg, tmpSumPerVfReg, mask); // 沿 D 维累加
            }
            uint32_t bufIdx = offset + offsetN;
            if constexpr (accumulate) {
                uint32_t one = 1;
                Reg::MaskReg scalarMask = Reg::UpdateMask<P>(one);
                Reg::RegTensor<P> previous;
                Reg::Load<P>(previous, hPreGradAddr + bufIdx);
                Reg::Add(sumReg, sumReg, previous, scalarMask);
            }
            Reg::Store(hPreGradAddr + bufIdx, sumReg, 1);
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
            VFDoPreProcessV0<false>(hPreGradAddr, xB16InAddr, gradHInB16InAddr, curLenN, copySizeD, i, j, runBSStart);

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
                    VFDoPreProcessV0<false>((__ubuf__ P*)hPreGradBuf.GetPhyAddr(), (__ubuf__ T*)xIn.GetPhyAddr(),
                                            (__ubuf__ T*)gradIn.GetPhyAddr(), currentN, currentD, row, offsetN,
                                            runBSStart);
                } else {
                    VFDoPreProcessV0<true>((__ubuf__ P*)hPreGradBuf.GetPhyAddr(), (__ubuf__ T*)xIn.GetPhyAddr(),
                                           (__ubuf__ T*)gradIn.GetPhyAddr(), currentN, currentD, row, offsetN,
                                           runBSStart);
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
    globalUbOffset_ +=
        MhcPreSinkhornBackwardUtils::CeilAlign<uint32_t>(fusionSize_, CEIL_ALIGN_DEFAULT / sizeof(P)) * sizeof(P);
    // bias 和部分和均按 32 字节对齐，兼容奇数 N。
    LocalTensor<P> biasGradSumBuf =
        paramsBuf_
            .Get<P>()[MhcPreSinkhornBackwardUtils::CeilAlign<uint32_t>(fusionSize_, CEIL_ALIGN_DEFAULT / sizeof(P))];
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
    uint32_t stepOffset1 =
        MhcPreSinkhornBackwardUtils::CeilAlign(uint32_t(buffers.stepLength * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    uint32_t hpreGradTotlenLen =
        MhcPreSinkhornBackwardUtils::CeilAlign(buffers.stepLength, uint32_t(CEIL_ALIGN_DEFAULT));
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
    VFDoV0PreGrad((__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr(), (__ubuf__ P*)hMixBufLocal.GetPhyAddr(),
                  (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), (__ubuf__ P*)buffers.hPreGradBuf.GetPhyAddr(),
                  (__ubuf__ P*)paramBias_.GetPhyAddr(), alphaPre, currentDealBsNum, fusionSize_, N_);
    VFDoV0PostGrad((__ubuf__ P*)buffers.hPostBufS1.GetPhyAddr(), (__ubuf__ P*)hMixBufLocal.GetPhyAddr(),
                   (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), (__ubuf__ P*)gradHPost.GetPhyAddr(),
                   (__ubuf__ P*)paramBias_.GetPhyAddr() + N_, alphaPost, currentDealBsNum, fusionSize_, N_);
    vecInQueue0_.FreeTensor(gradHPost);

    ProcessSinkhorn(runBSStart, currentDealBsNum, buffers.hResBufS1);

    // Sinkhorn 反向完成后计算 exp/max 的梯度。
    VFDoV0ResGrad((__ubuf__ P*)buffers.hResBufS1.GetPhyAddr(), (__ubuf__ P*)hMixBufLocal.GetPhyAddr(),
                  (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), (__ubuf__ P*)buffers.hResBufS1.GetPhyAddr(),
                  (__ubuf__ P*)paramBias_.GetPhyAddr() + 2 * N_, alphaComb, currentDealBsNum, fusionSize_, N_);
    hMixInput = hMixBufLocal;

    AIV02Process(buffers, hMixGradBuf, currentDealBsNum, alphaPre, alphaPost, alphaComb);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessSinkhorn(uint64_t runBSStart,
                                                                           uint32_t currentDealBsNum,
                                                                           LocalTensor<P>& hResBuf)
{
    sinkhorn_.Process(runBSStart, currentDealBsNum, hResBuf, vecInQueue0_);
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV0PreGrad(__ubuf__ P* gradPreOut, __ubuf__ P* hMixAddr,
                                                                         __ubuf__ P* invRmsAddr,
                                                                         __ubuf__ P* gradHPreAddr,
                                                                         __ubuf__ P* biasPreAddr, float alphaPre,
                                                                         uint32_t rows, uint32_t fusionSize, uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> alphaReg;
        Reg::RegTensor<P> oneReg;
        Reg::RegTensor<P> invReg;
        Reg::RegTensor<P> hMixReg;
        Reg::RegTensor<P> zReg;
        Reg::RegTensor<P> negReg;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> sigReg;
        Reg::RegTensor<P> gateReg;
        Reg::RegTensor<P> biasReg;
        Reg::RegTensor<P> upReg;
        Reg::RegTensor<P> outReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(alphaReg, alphaPre, maskAll);
        Reg::Duplicate(oneReg, 1.0f, maskAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            uint32_t len = n;
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + r);
            Reg::Load<P>(hMixReg, hMixAddr + r * fusionSize);
            Reg::Mul(zReg, hMixReg, invReg, mask);
            Reg::Mul(zReg, zReg, alphaReg, mask);
            Reg::Load<P>(biasReg, biasPreAddr);
            Reg::Add(zReg, zReg, biasReg, mask);
            Reg::Muls(negReg, zReg, -1.0f, mask);
            Reg::Exp(expReg, negReg, mask);
            Reg::Adds(expReg, expReg, 1.0f, mask);
            Reg::Div(sigReg, oneReg, expReg, mask);
            Reg::Exp(gateReg, zReg, mask);
            Reg::Adds(gateReg, gateReg, 1.0f, mask);
            Reg::Div(gateReg, oneReg, gateReg, mask);
            Reg::Mul(sigReg, sigReg, gateReg, mask);
            Reg::Load<P>(upReg, gradHPreAddr + r * n);
            Reg::Mul(outReg, sigReg, upReg, mask);
            Reg::Store<P>(gradPreOut + r * n, outReg, n);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV0PostGrad(
    __ubuf__ P* gradPostOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr, __ubuf__ P* gradHPostAddr,
    __ubuf__ P* biasPostAddr, float alphaPost, uint32_t rows, uint32_t fusionSize, uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> alphaReg;
        Reg::RegTensor<P> oneReg;
        Reg::RegTensor<P> invReg;
        Reg::RegTensor<P> hMixReg;
        Reg::RegTensor<P> zReg;
        Reg::RegTensor<P> negReg;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> sigReg;
        Reg::RegTensor<P> gateReg;
        Reg::RegTensor<P> biasReg;
        Reg::RegTensor<P> upReg;
        Reg::RegTensor<P> outReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(alphaReg, alphaPost, maskAll);
        Reg::Duplicate(oneReg, 1.0f, maskAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            uint32_t len = n;
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + r);
            Reg::Load<P>(hMixReg, hMixAddr + r * fusionSize + n);
            Reg::Mul(zReg, hMixReg, invReg, mask);
            Reg::Mul(zReg, zReg, alphaReg, mask);
            Reg::Load<P>(biasReg, biasPostAddr);
            Reg::Add(zReg, zReg, biasReg, mask);
            Reg::Muls(negReg, zReg, -1.0f, mask);
            Reg::Exp(expReg, negReg, mask);
            Reg::Adds(expReg, expReg, 1.0f, mask);
            Reg::Div(sigReg, oneReg, expReg, mask);
            Reg::Exp(gateReg, zReg, mask);
            Reg::Adds(gateReg, gateReg, 1.0f, mask);
            Reg::Div(gateReg, oneReg, gateReg, mask);
            Reg::Mul(sigReg, sigReg, gateReg, mask);
            Reg::Muls(sigReg, sigReg, 2.0f, mask);
            Reg::Load<P>(upReg, gradHPostAddr + r * n);
            Reg::Mul(outReg, sigReg, upReg, mask);
            Reg::Store<P>(gradPostOut + r * n, outReg, n);
        }
    }
}

// 每寄存器按 8×8 通道处理 8 行，独立归约最大值、首个最大值下标及加权和。
template <class T, class P>
template <uint32_t N>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV0PackedRes(__ubuf__ P* output, __ubuf__ P* hc,
                                                                           __ubuf__ P* inv, __ubuf__ P* upstream,
                                                                           __ubuf__ P* bias, float alpha, uint32_t rows)
{
    constexpr uint32_t H = N * N + 2U * N;
    const uint32_t rowCount = rows * N;
    __VEC_SCOPE__
    {
        Reg::MaskReg full = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> order, group, column, absoluteRow, batch, matrixRow, stride;
        Reg::RegTensor<uint32_t> gradientIndex, hcIndex, biasIndex, candidate, first, firstBroadcast, sentinel;
        Reg::Arange((Reg::RegTensor<int32_t>&)order, 0);
        Reg::ShiftRights(group, order, static_cast<int16_t>(3), full);
        Reg::Duplicate(stride, 7U);
        Reg::And(column, order, stride, full);
        Reg::Duplicate(sentinel, N);
        Reg::RegTensor<P> a, invReg, mix, b, z, maximum, maxBroadcast, expReg, weighted, total, totalBroadcast, zero;
        Reg::Duplicate(a, alpha);
        Reg::Duplicate(zero, 0.0f);
        for (uint16_t block = 0; block < static_cast<uint16_t>((rowCount + 7U) / 8U); ++block) {
            Reg::Adds(absoluteRow, group, static_cast<uint32_t>(block) * 8U, full);
            Reg::MaskReg validRows, validCols, active, equalMax, firstMask;
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(validRows, absoluteRow, rowCount, full);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(validCols, column, N, full);
            Reg::And(active, validRows, validCols, full);
            if constexpr (N == 8U) {
                Reg::ShiftRights(batch, absoluteRow, static_cast<int16_t>(3), full);
            } else {
                Reg::Duplicate(stride, N);
                Reg::Div(batch, absoluteRow, stride, full);
            }
            Reg::Duplicate(stride, N);
            Reg::Mul(matrixRow, batch, stride, full);
            Reg::Sub(matrixRow, absoluteRow, matrixRow, full);
            Reg::Mul(gradientIndex, absoluteRow, stride, full);
            Reg::Add(gradientIndex, gradientIndex, column, full);
            Reg::Mul(biasIndex, matrixRow, stride, full);
            Reg::Add(biasIndex, biasIndex, column, full);
            Reg::Duplicate(stride, H);
            Reg::Mul(hcIndex, batch, stride, full);
            Reg::Add(hcIndex, hcIndex, biasIndex, full);
            Reg::Adds(hcIndex, hcIndex, 2U * N, full);
            Reg::DataCopyGather(invReg, inv, batch, active);
            Reg::DataCopyGather(mix, hc, hcIndex, active);
            Reg::DataCopyGather(b, bias, biasIndex, active);
            Reg::Mul(z, mix, invReg, active);
            Reg::Mul(z, z, a, active);
            Reg::Add(z, z, b, active);
            Reg::ReduceMaxWithDataBlock(maximum, z, active);
            Reg::Gather(maxBroadcast, maximum, group);
            Reg::Compare<P, CMPMODE::EQ>(equalMax, z, maxBroadcast, active);
            Reg::Select(candidate, column, sentinel, equalMax);
            Reg::ReduceMinWithDataBlock(first, candidate, active);
            Reg::Gather(firstBroadcast, first, group);
            Reg::Compare<uint32_t, CMPMODE::EQ>(firstMask, column, firstBroadcast, active);
            Reg::Sub(expReg, z, maxBroadcast, active);
            Reg::Exp(expReg, expReg, active);
            Reg::DataCopyGather(weighted, upstream, gradientIndex, active);
            Reg::Mul(weighted, expReg, weighted, active);
            Reg::ReduceSumWithDataBlock(total, weighted, active);
            Reg::Gather(totalBroadcast, total, group);
            Reg::Select(totalBroadcast, totalBroadcast, zero, firstMask);
            Reg::Sub(weighted, weighted, totalBroadcast, active);
            Reg::DataCopyScatter(output, weighted, gradientIndex, active);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV0ResGrad(__ubuf__ P* gradResOut, __ubuf__ P* hMixAddr,
                                                                         __ubuf__ P* invRmsAddr, __ubuf__ P* skAddr,
                                                                         __ubuf__ P* biasResAddr, float alphaRes,
                                                                         uint32_t rows, uint32_t fusionSize, uint32_t n)
{
    if (n == 6U) {
        VFDoV0PackedRes<6U>(gradResOut, hMixAddr, invRmsAddr, skAddr, biasResAddr, alphaRes, rows);
        return;
    }
    if (n == 8U) {
        VFDoV0PackedRes<8U>(gradResOut, hMixAddr, invRmsAddr, skAddr, biasResAddr, alphaRes, rows);
        return;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invReg;
        Reg::RegTensor<P> hMixReg;
        Reg::RegTensor<P> uReg;
        Reg::RegTensor<P> zReg;
        Reg::RegTensor<P> biasReg;
        Reg::RegTensor<P> alphaReg;
        Reg::RegTensor<P> maxReg;
        Reg::RegTensor<P> maxBroadcast;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> upstreamReg;
        Reg::RegTensor<P> weighted;
        Reg::RegTensor<P> weightedSum;
        Reg::RegTensor<P> weightedBroadcast;
        Reg::RegTensor<P> zeroReg;
        Reg::RegTensor<P> correction;
        Reg::RegTensor<P> gzReg;
        Reg::RegTensor<int32_t> order;
        Reg::RegTensor<int32_t> sentinel;
        Reg::RegTensor<int32_t> candidate;
        Reg::RegTensor<int32_t> firstIndex;
        Reg::RegTensor<int32_t> firstBroadcast;
        uint32_t sregN = n;
        Reg::MaskReg maskN = Reg::UpdateMask<P>(sregN);
        Reg::MaskReg equalMax;
        Reg::MaskReg firstMask;
        Reg::Duplicate(alphaReg, alphaRes, maskN);
        Reg::Duplicate(zeroReg, 0.0f, maskN);
        Reg::Arange(order, static_cast<int32_t>(0));
        Reg::Duplicate(sentinel, static_cast<int32_t>(n), maskN);
        uint32_t nn = n * n;
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + r);
            for (uint16_t row = 0; row < static_cast<uint16_t>(n); ++row) {
                Reg::Load<P>(hMixReg, hMixAddr + r * fusionSize + 2 * n + row * n);
                Reg::Load<P>(biasReg, biasResAddr + row * n);
                Reg::Mul(uReg, hMixReg, invReg, maskN);
                Reg::Mul(zReg, uReg, alphaReg, maskN);
                Reg::Add(zReg, zReg, biasReg, maskN);
                Reg::Reduce<Reg::ReduceType::MAX>(maxReg, zReg, maskN);
                Reg::Duplicate(maxBroadcast, maxReg, maskN);
                Reg::Compare<P, CMPMODE::EQ>(equalMax, zReg, maxBroadcast, maskN);
                Reg::Select(candidate, order, sentinel, equalMax);
                Reg::Reduce<Reg::ReduceType::MIN>(firstIndex, candidate, maskN);
                Reg::Duplicate(firstBroadcast, firstIndex, maskN);
                Reg::Compare<int32_t, CMPMODE::EQ>(firstMask, order, firstBroadcast, maskN);
                Reg::Sub(expReg, zReg, maxBroadcast, maskN);
                Reg::Exp(expReg, expReg, maskN);
                Reg::Load<P>(upstreamReg, skAddr + r * nn + row * n);
                Reg::Mul(weighted, expReg, upstreamReg, maskN);
                Reg::Reduce<Reg::ReduceType::SUM>(weightedSum, weighted, maskN);
                Reg::Duplicate(weightedBroadcast, weightedSum, maskN);
                Reg::Select(correction, weightedBroadcast, zeroReg, firstMask);
                Reg::Sub(gzReg, weighted, correction, maskN);
                Reg::Store<P>(gradResOut + r * nn + row * n, gzReg, n);
            }
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV0ProcessGradHPost(__ubuf__ P* hPostIn,
                                                                                  __ubuf__ P* gradHPostIn,
                                                                                  __ubuf__ P* gradHPostOut,
                                                                                  uint32_t stepLen)
{
    uint16_t loopCnt = CeilDiv(stepLen, uint32_t(eleNumPerVf_));
    uint32_t curLen = stepLen;
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<P> gradHPostReg, tmpReg;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(gradHPostReg, hPostIn + vfBlockIdx * eleNumPerVf_);
            Reg::Muls(tmpReg, gradHPostReg, NEG_HALF, mask);
            Reg::Adds(tmpReg, tmpReg, ONE, mask);
            Reg::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            Reg::LoadAlign(tmpReg, gradHPostIn + vfBlockIdx * eleNumPerVf_);
            Reg::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            Reg::StoreAlign(gradHPostOut + vfBlockIdx * eleNumPerVf_, gradHPostReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::AIV02Process(V0V1Buffers<P>& buffers,
                                                                        LocalTensor<P>& hMixGradBuf,
                                                                        uint64_t currentDealBsNum, float alphaPre,
                                                                        float alphaPost, float alphaComb)
{
    __ubuf__ P* hPreBufAddr = (__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr();
    __ubuf__ P* hPostBufAddr = (__ubuf__ P*)buffers.hPostBufS1.GetPhyAddr();
    __ubuf__ P* hResBufAddr = (__ubuf__ P*)buffers.hResBufS1.GetPhyAddr();
    __ubuf__ P* gatherFusionOutBufAddr = (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr();
    __ubuf__ P* hMixGradBufAddr = (__ubuf__ P*)hMixGradBuf.GetPhyAddr();

    uint32_t blockLenPre = N_;
    uint32_t blockLenPost = N_;
    uint32_t blockLenComb = N_ * N_;
    uint32_t totalBlockLen = blockLenPre + blockLenPost + blockLenComb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> gatherReg;
        Reg::RegTensor<P> hMixGradReg;
        Reg::RegTensor<P> alphaPreReg;
        Reg::RegTensor<P> alphaPostReg;
        Reg::RegTensor<P> alphaCombReg;
        Reg::MaskReg maskPre = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskPost = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskComb = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate<P>(alphaPreReg, alphaPre, maskPre);
        Reg::Duplicate<P>(alphaPostReg, alphaPost, maskPost);
        Reg::Duplicate<P>(alphaCombReg, alphaComb, maskComb);

        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(currentDealBsNum); bsIdx++) {
            uint32_t curLenPre = blockLenPre;
            uint32_t curLenPost = blockLenPost;
            uint32_t curLenComb = blockLenComb;
            uint32_t bsFusionOffset = bsIdx * totalBlockLen;

            uint32_t srcOffset = bsIdx * blockLenPre;
            uint32_t dstOffset = bsFusionOffset;
            maskPre = Reg::UpdateMask<P>(curLenPre);
            Reg::Load<P>(gatherReg, hPreBufAddr + srcOffset);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaPreReg, maskPre);
            Reg::Store<P>(hMixGradBufAddr + dstOffset, hMixGradReg);

            uint32_t srcOffset1 = bsIdx * blockLenPre;
            uint32_t dstOffset1 = bsFusionOffset;
            maskPost = Reg::UpdateMask<P>(curLenPost);
            Reg::Load<P>(gatherReg, hPostBufAddr + srcOffset1);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset1 + blockLenPre, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaPostReg, maskPost);
            Reg::Store<P>(hMixGradBufAddr + dstOffset1 + blockLenPre, hMixGradReg);

            uint32_t srcOffset2 = bsIdx * blockLenComb;
            uint32_t dstOffset2 = bsFusionOffset + blockLenPre + blockLenPost;
            maskComb = Reg::UpdateMask<P>(curLenComb);
            Reg::Load<P>(gatherReg, hResBufAddr + srcOffset2);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset2, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaCombReg, maskComb);
            Reg::Store<P>(hMixGradBufAddr + dstOffset2, hMixGradReg);
        }
    }
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
        VFDoV1ProcessBiasGradForN8((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                   (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
    } else if ((fusionSize_ % (CEIL_ALIGN_DEFAULT / sizeof(P))) == 0) {
        VFDoV1ProcessBiasGradForN4N6<true>((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                           (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
    } else {
        VFDoV1ProcessBiasGradForN4N6<false>((__ubuf__ P*)fp32OutBuf.GetPhyAddr(),
                                            (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
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

    VFDoV1Process((__ubuf__ P*)fp32OutBuf.GetPhyAddr(), (__ubuf__ P*)hMixGradBuf.GetPhyAddr(),
                  (__ubuf__ P*)buffers.hFusionBuf.GetPhyAddr(), (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr(),
                  (__ubuf__ P*)hMixInput.GetPhyAddr(), (__ubuf__ P*)buffers.invRmsBuf.GetPhyAddr(), curBSSize);

    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> hMixGradOutBuf = vecOutQueue1_.DeQue<P>();
    dataCopyParams.blockLen = curBSSize * fusionSize_ * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(runBSStart)], hMixGradOutBuf, dataCopyParams);
    PipeBarrier<PIPE_V>();

    uint32_t shape[] = {curBSSize, fusionSize_};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(invRmsGradUb, hMixGradBuf, buffers.brcbTmpBuf, shape, true);
    vecOutQueueSmall_.EnQue(invRmsGradUb);
    LocalTensor<P> invRmsGradUbtBuf = vecOutQueueSmall_.DeQue<P>();

    VFDoV1InvRmsCoeff((__ubuf__ P*)invRmsGradUbtBuf.GetPhyAddr(), (__ubuf__ P*)invRmsBufLocal.GetPhyAddr(), curBSSize,
                      scaleMean_);
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
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV1Process(__ubuf__ P* gradHMixOut,
                                                                         __ubuf__ P* hMixGradBuf,
                                                                         __ubuf__ P* gradAlphaOut, __ubuf__ P* gatherIn,
                                                                         __ubuf__ P* hMixIn, __ubuf__ P* invRmsIn,
                                                                         uint32_t curBSSize)
{
    uint32_t totalElem = curBSSize * fusionSize_;
    uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
    uint32_t curElemCnt = totalElem;

    __VEC_SCOPE__
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);

            Reg::RegTensor<P> h1GradReg, invRmsReg, hMixReg, gatherReg;
            Reg::RegTensor<P> hMixGradReg, invRmsGradReg;
            Reg::RegTensor<P> hMul1Reg, hMul2Reg, hMul3Reg;

            Reg::LoadAlign(h1GradReg, hMixGradBuf + elemOffset);
            Reg::LoadAlign(invRmsReg, invRmsIn + elemOffset);

            Reg::Mul(hMixGradReg, h1GradReg, invRmsReg, mask);
            Reg::StoreAlign(gradHMixOut + elemOffset, hMixGradReg, mask);

            Reg::LoadAlign(hMixReg, hMixIn + elemOffset);
            Reg::Mul(hMul1Reg, h1GradReg, hMixReg, mask);
            Reg::StoreAlign(hMixGradBuf + elemOffset, hMul1Reg, mask);

            Reg::LoadAlign(gatherReg, gatherIn + elemOffset);
            Reg::Mul(hMul2Reg, invRmsReg, hMixReg, mask);
            Reg::Mul(hMul3Reg, hMul2Reg, gatherReg, mask);
            Reg::StoreAlign(gradAlphaOut + elemOffset, hMul3Reg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV1ProcessBiasGradForN8(__ubuf__ P* outBufDst,
                                                                                      __ubuf__ P* gatherFusion,
                                                                                      uint32_t curBSSize)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg1, sumReg2;
        Reg::Duplicate(sumReg1, 0);
        Reg::Duplicate(sumReg2, 0);
        uint32_t dealMask1 = eleNumPerVf_;
        uint32_t dealMask2 = fusionSize_ - eleNumPerVf_;
        Reg::MaskReg mask1 = Reg::UpdateMask<P>(dealMask1);
        Reg::MaskReg mask2 = Reg::UpdateMask<P>(dealMask2);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = bsIdx * fusionSize_;
            uint32_t elemOffset2 = bsIdx * fusionSize_ + eleNumPerVf_;
            Reg::RegTensor<P> gatherReg1, gatherReg2;

            Reg::LoadAlign(gatherReg1, gatherFusion + elemOffset1);
            Reg::LoadAlign(gatherReg2, gatherFusion + elemOffset2);

            Reg::Add(sumReg1, sumReg1, gatherReg1, mask1);
            Reg::Add(sumReg2, sumReg2, gatherReg2, mask2);
        }
        Reg::StoreAlign(outBufDst, sumReg1, mask1);
        Reg::StoreAlign(outBufDst + eleNumPerVf_, sumReg2, mask2);
    }
}

template <class T, class P>
template <bool aligned>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV1ProcessBiasGradForN4N6(__ubuf__ P* outBufDst,
                                                                                        __ubuf__ P* gatherFusion,
                                                                                        uint32_t curBSSize)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        Reg::Duplicate(sumReg, 0);
        uint32_t dealMask = fusionSize_;
        Reg::MaskReg mask = Reg::UpdateMask<P>(dealMask);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = bsIdx * fusionSize_;
            Reg::RegTensor<P> gatherReg;

            if constexpr (aligned) {
                Reg::LoadAlign(gatherReg, gatherFusion + elemOffset1);
            } else {
                // fusionSize 为奇数时，相邻行的 UB 对齐位置不同。
                Reg::Load<P>(gatherReg, gatherFusion + elemOffset1);
            }

            Reg::Add(sumReg, sumReg, gatherReg, mask);
        }
        Reg::StoreAlign(outBufDst, sumReg, mask);
    }
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
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV2CastXToFp32(__ubuf__ P* xRsFp32Addr,
                                                                             __ubuf__ P* xRsOutAddr,
                                                                             __ubuf__ T* bf16InputAddr,
                                                                             uint32_t currentChunkSize,
                                                                             uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xInB16Reg;
        Reg::RegTensor<P> xFp32Reg;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            // copySizeND <= 64，每次处理一行。
            uint32_t lenND = copySizeND;
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;

            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg, bf16InputAddr + offset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

            Reg::StoreAlign(xRsFp32Addr + offset, xFp32Reg, mask); // 后续复用 xFp32

            Reg::StoreAlign(xRsOutAddr + offset, xFp32Reg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV2AccumulateXGrad(
    __ubuf__ P* xGradVec3Addr, __ubuf__ P* xRsGradMmAddr, __ubuf__ P* xRsFp32Addr, __ubuf__ P* invRmsAddr,
    uint32_t currentChunkSize, uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xRsFp32Reg, xRsGradMmReg, xRsGradReg, xGradReg;
        Reg::RegTensor<P> invRmsReg;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            // 每行广播加载一个 FP32 invRms 标量
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invRmsReg, invRmsAddr + tIdx);
            // copySizeND <= 64，每次处理一行。
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;
            Reg::LoadAlign(xRsGradMmReg, xRsGradMmAddr + offset);
            Reg::LoadAlign(xGradReg, xGradVec3Addr + offset);
            Reg::LoadAlign(xRsFp32Reg, xRsFp32Addr + offset);

            Reg::Mul(xRsGradReg, xRsFp32Reg, invRmsReg, mask);
            Reg::Add(xGradReg, xGradReg, xRsGradReg, mask);

            Reg::Add(xGradReg, xGradReg, xRsGradMmReg, mask);
            Reg::StoreAlign(xGradVec3Addr + offset, xGradReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV2HInMulHPre(
    __ubuf__ P* xGradVec3BufAddr, __ubuf__ T* hInGradInBufAddr, __ubuf__ P* hPreUbAddr, uint32_t currentChunkSize,
    uint32_t copySizeND, uint32_t currentN, uint32_t bsGlobalOffset)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> hInReg;
        Reg::RegTensor<P> hInFP32Reg, hPreReg, hPreFullReg;
        Reg::MaskReg mask;
        uint32_t hPreOffset = bsGlobalOffset * N_ + currentN;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            uint32_t hInOffset = tIdx * copySizeND;

            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(hPreFullReg, hPreUbAddr + hPreOffset);
            // copySizeND <= 64，每次处理一行。
            mask = Reg::UpdateMask<P>(lenND);
            // hIn 转为 FP32
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(hInReg, hInGradInBufAddr + hInOffset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(hInFP32Reg, hInReg, mask);
            Reg::Mul(hInFP32Reg, hInFP32Reg, hPreFullReg, mask);
            Reg::StoreAlign(xGradVec3BufAddr + hInOffset, hInFP32Reg, mask);
            hPreOffset += N_;
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
    ubOffset = MhcPreSinkhornBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)),
                                                      uint32_t(CEIL_ALIGN_DEFAULT));

    xGradVec3Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset = MhcPreSinkhornBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)),
                                                      uint32_t(CEIL_ALIGN_DEFAULT));

    invRmsUb_ = fp32TBuf_.GetWithOffset<P>(currentBsSize, ubOffset);
    ubOffset = MhcPreSinkhornBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * sizeof(P)),
                                                      uint32_t(CEIL_ALIGN_DEFAULT));

    uint32_t hPreUbLen =
        MhcPreSinkhornBackwardUtils::CeilAlign(uint32_t(currentBsSize * N_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT)) /
        sizeof(P);
    hPreUb_ = fp32TBuf_.GetWithOffset<P>(hPreUbLen, ubOffset);
    ubOffset = MhcPreSinkhornBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * N_ * sizeof(P)),
                                                      uint32_t(CEIL_ALIGN_DEFAULT));
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    bsCopyParams.blockLen = currentBsSize * N_ * sizeof(P);
    DataCopyPad(hPreUb_, hPreGm_[bsStart * N_], bsCopyParams, dataCopyPadParams);

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
    VFDoV2HInMulHPre(xGradVec3BufAddr, gradHInInBufAddr, hPreUbAddr, currentChunkSize, copySizeND, currentN,
                     (bsChunkStart - bsStart));
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
    VFDoV2CastXToFp32(xRsFp32Addr, xRsOutAddr, bf16InputAddr, currentChunkSize, copySizeND);
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
    VFDoV2AccumulateXGrad(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, currentChunkSize, copySizeND);
    if (!directC0_) {
        vecInQueue0_.FreeTensor(xRsGradMmInBuf_);
    }

    PipeBarrier<PIPE_V>();
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV2CastXGrad(__ubuf__ T* outputAddr,
                                                                           __ubuf__ P* xGradAddr, uint32_t totalElem)
{
    uint16_t vfLoopCnt = CeilDiv(totalElem, static_cast<uint32_t>(eleNumPerVf_));
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xGradReg;
        Reg::RegTensor<T> outputReg;
        static constexpr Reg::CastTrait castFp32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        uint32_t len = totalElem;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            uint32_t offset = vfBlockIdx * eleNumPerVf_;
            Reg::LoadAlign(xGradReg, xGradAddr + offset);
            Reg::Cast<T, P, castFp32ToB16>(outputReg, xGradReg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + offset, outputReg, mask);
        }
    }
}

// 转换并写回 grad_x
template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::ProcessV2ChunkOutputXGrad(
    uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize)
{
    bf16OutputBuf_ = vecOutQueue1_.AllocTensor<T>();
    __ubuf__ T* outputAddr = (__ubuf__ T*)bf16OutputBuf_.GetPhyAddr();
    __ubuf__ P* xGradAddr = (__ubuf__ P*)xGradVec3Buf_.GetPhyAddr();
    VFDoV2CastXGrad(outputAddr, xGradAddr, chunkNDSize);
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
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV1InvRmsCoeff(__ubuf__ P* invRmsGradAddr,
                                                                             __ubuf__ P* invRmsAddr, uint32_t count,
                                                                             float scaleMean)
{
    uint16_t loopCnt = CeilDiv(count, uint32_t(eleNumPerVf_));
    uint32_t curLen = count;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invReg, shMixReg, cubeReg, gradReg, outReg, scaleReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(scaleReg, -scaleMean, maskAll);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            Reg::MaskReg mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(invReg, invRmsAddr + elemOffset);
            Reg::Mul(shMixReg, invReg, invReg, mask);
            Reg::Mul(cubeReg, shMixReg, invReg, mask);
            Reg::LoadAlign(gradReg, invRmsGradAddr + elemOffset);
            Reg::Mul(outReg, cubeReg, gradReg, mask);
            Reg::Mul(outReg, outReg, scaleReg, mask);
            Reg::StoreAlign(invRmsGradAddr + elemOffset, outReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::AIV21Process(LocalTensor<P>& invRmsInBuf,
                                                                        LocalTensor<P>& invRmsUb,
                                                                        LocalTensor<P>& invRmsGradUb,
                                                                        uint32_t currentChunkSize)
{
    __ubuf__ P* invRmsInBufAddr = (__ubuf__ P*)invRmsInBuf.GetPhyAddr();
    __ubuf__ P* invRmsUbAddr = (__ubuf__ P*)invRmsUb.GetPhyAddr();
    __ubuf__ P* invRmsGradUbAddr = (__ubuf__ P*)invRmsGradUb.GetPhyAddr();

    uint32_t vfloopCnt = Ceil(currentChunkSize, eleNumPerVf_);
    uint32_t curLen = currentChunkSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invRmsInReg;
        Reg::RegTensor<P> invRmsTempReg;
        Reg::RegTensor<P> invRmsUbReg;
        Reg::RegTensor<P> invRmsGradUb;
        Reg::RegTensor<P> scaleMeanReg;
        Reg::MaskReg maskAll, mask;
        maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate<P>(scaleMeanReg, (-scaleMean_), maskAll);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < static_cast<uint16_t>(vfloopCnt); vfBlockIdx++) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(invRmsInReg, invRmsInBufAddr + elemOffset);
            Reg::Mul(invRmsTempReg, invRmsInReg, invRmsInReg, mask);
            Reg::Mul(invRmsUbReg, invRmsTempReg, invRmsInReg, mask);
            Reg::LoadAlign(invRmsGradUb, invRmsGradUbAddr + elemOffset);
            Reg::Mul(invRmsUbReg, invRmsUbReg, invRmsGradUb, mask);
            Reg::Mul(invRmsUbReg, invRmsUbReg, scaleMeanReg, mask);
            Reg::StoreAlign(invRmsUbAddr + elemOffset, invRmsUbReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV3ProcessAlphaGradSingleCore(__ubuf__ P* alphaGradOut,
                                                                                            __ubuf__ P* alphaGradIn)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::RegTensor<P> zero, value, sum;
        Reg::Duplicate(zero, 0.0f, mask);
        // 保持 V3 中 0+partial 的累加方式，仅加载三个有效标量。
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(value, alphaGradIn);
        Reg::Add(sum, zero, value, mask);
        Reg::Store<P>(alphaGradOut, sum, 1);
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(value, alphaGradIn + ALPHA_GRAD_SHAPE_2_OFFSET);
        Reg::Add(sum, zero, value, mask);
        Reg::Store<P>(alphaGradOut + 1, sum, 1);
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(value, alphaGradIn + ALPHA_GRAD_SHAPE_3_OFFSET);
        Reg::Add(sum, zero, value, mask);
        Reg::Store<P>(alphaGradOut + 2, sum, 1);
    }
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
    VFDoV3ProcessAlphaGradSingleCore((__ubuf__ P*)alphaGradOutLocal.GetPhyAddr(),
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
        paramsBuf_
            .Get<P>()[MhcPreSinkhornBackwardUtils::CeilAlign<uint32_t>(fusionSize_, CEIL_ALIGN_DEFAULT / sizeof(P))];
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
    VFDoV3ProcessAlphaGrad((__ubuf__ P*)alphaGradOutLocal.GetPhyAddr(), (__ubuf__ P*)alphaGradInLocal.GetPhyAddr());
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

template <class T, class P>
__aicore__ inline void MhcPreSinkhornBackwardKernel<T, P>::VFDoV3ProcessAlphaGrad(__ubuf__ P* alphaGradOut,
                                                                                  __ubuf__ P* alphaGradIn)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        Reg::Duplicate(sumReg, 0);
        uint32_t dealMask = ALPHA_GRAD_PADDING;
        Reg::MaskReg mask = Reg::UpdateMask<P>(dealMask);
        for (uint16_t vcIdx = 0; vcIdx < static_cast<uint16_t>(usedVecCoreNum_); ++vcIdx) {
            uint32_t elemOffset = vcIdx * ALPHA_GRAD_PADDING;
            Reg::RegTensor<P> alphaGradInReg;
            Reg::LoadAlign(alphaGradInReg, alphaGradIn + elemOffset);
            Reg::Add(sumReg, sumReg, alphaGradInReg, mask);
        }
        uint32_t outputCount = ALPHA_GRAD_LAST_DIM_SIZE;
        Reg::MaskReg outputMask = Reg::UpdateMask<P>(outputCount);
        Reg::RegTensor<uint32_t> index, zeroIndex;
        Reg::Arange((Reg::RegTensor<int32_t>&)index, static_cast<int32_t>(0));
        Reg::Muls(index, index, static_cast<uint32_t>(ALPHA_GRAD_SHAPE_2_OFFSET), mask);
        Reg::Duplicate(zeroIndex, 0U);
        Reg::Select(index, index, zeroIndex, outputMask);
        Reg::RegTensor<P> packed;
        Reg::Gather(packed, sumReg, index);
        Reg::StoreAlign(alphaGradOut, packed, outputMask);
    }
}

} // namespace MhcPreSinkhornBackwardNs

#endif // __MHC_PRE_SINKHORN_BACKWARD_KERNEL_H_

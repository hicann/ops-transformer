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
 * \file mhc_pre_backward_kernel.h
 * \brief
 */

#ifndef __mhc_pre_backward_KERNEL_H_
#define __mhc_pre_backward_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_backward_utils.h"
#include "mhc_pre_backward_cube_compute.h"

namespace MhcPreBackward {

using namespace AscendC;
using namespace MhcPreBackwardUtils;

constexpr uint32_t MHC_PRE_BACKWARD_IMPL_MODE_HF32 = 1U;

template <class T, class P>
class MhcPreBackwardKernel {
public:
    __aicore__ inline MhcPreBackwardKernel() = default;
    // 入口函数
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();

    // 初始化函数
    __aicore__ inline void InitGlobalBuffersAndTiling(InitParams initParams);
    __aicore__ inline void InitUBAndAIVBuffers(InitParams initParams);
    __aicore__ inline void InitStage2AndDataCopy();

    // V0 流程 (向量计算)
    __aicore__ inline void ProcessV0Main();
    __aicore__ inline void ProcessV0MainLoop(uint64_t vecRunTimes, float alphaPre, float alphaPost, float alphaComb,
                                             LocalTensor<P> &sumBuf);
    __aicore__ inline void ProcessV0MainAlphaReduce(LocalTensor<P> &sumBuf, uint32_t coreId);
    __aicore__ inline void PreProcessV0(LocalTensor<P> &hPreGradBuf, uint64_t runBSStart, uint64_t runBSEnd);
    __aicore__ inline void AllocV0V1Buffers(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P> &buffers);
    __aicore__ inline void ProcessV0(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P> &buffers, float alphaPre,
                                     float alphaPost, float alphaComb);
    __aicore__ inline void ProcessV1(uint64_t runBSStart, uint64_t runBSEnd, V0V1Buffers<P> &buffers,
                                     uint64_t vecRuntimesId, LocalTensor<P> &sumBuf);
    __aicore__ inline void AIV02Process(V0V1Buffers<P> &buffers, LocalTensor<P> &hMixGradBuf, uint64_t currentDealBsNum,
                                        float alphaPre, float alphaPost, float alphaComb);
    __aicore__ inline void AIV21Process(LocalTensor<P> &invRmsInBuf, LocalTensor<P> &invRmsUb,
                                        LocalTensor<P> &invRmsGradUb, uint32_t currentChunkSize);
    // Cube侧C0/C1流水
    __aicore__ inline void ProcessC0C1Pipeline();
    __aicore__ inline void ProcessC0(uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);
    __aicore__ inline void ProcessC1(uint32_t offsetND, uint32_t currentNDBlock, uint64_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);

    // Vector侧V2流水
    __aicore__ inline void ProcessV2Pipeline();
    __aicore__ inline void ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart, uint64_t bsEnd,
                                     LocalTensor<P> &sumBuf, uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                              uint64_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize,
                                              uint32_t currentN, uint32_t buffId, LocalTensor<P> &sumBuf);
    __aicore__ inline void ProcessV2ChunkLoadInvRms(uint64_t bsChunkStart, uint32_t currentChunkSize);
    __aicore__ inline void ProcessV2ChunkLoadHInGrad(uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t currentN, uint64_t bsStart);
    __aicore__ inline void ProcessV2ChunkLoadXAndGamma(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                                       uint64_t bsChunkStart, uint32_t currentChunkSize,
                                                       uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkProcessXRsGradMm(uint64_t bsStart, uint64_t bsChunkStart, uint32_t copySizeND,
                                                          uint32_t currentChunkSize, uint32_t buffId,
                                                          LocalTensor<P> &sumBuf);
    __aicore__ inline void ProcessV2ChunkOutputXGrad(uint32_t offsetND, uint32_t copySizeND, uint64_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t chunkNDSize);

    // V3 流程 (向量计算)
    __aicore__ inline void ProcessV3();
    __aicore__ inline void ProcessV3AlphaGrad();
    __aicore__ inline void ProcessV3BiasGrad();

    // 向量化计算函数 (VFDo*)
    __aicore__ inline void VFDoPreProcessV0(__ubuf__ P *hPreGradAddr, __ubuf__ T *xB16InAddr,
                                            __ubuf__ T *gradHInB16InAddr, uint16_t curLenN, uint32_t lenD, uint64_t i,
                                            uint32_t j, uint64_t runBSStart);
    __aicore__ inline void VFDoV0HPreGrad(__ubuf__ P *hPreBufS1Addr, __ubuf__ P *hPreAddr, __ubuf__ P *gradHInAddr,
                                          uint32_t totalElem);
    __aicore__ inline void VFDoV0ProcessGradHPost(__ubuf__ P *hPostIn, __ubuf__ P *gradHPostIn,
                                                  __ubuf__ P *gradHPostOut, uint32_t stepLen);
    __aicore__ inline void VFDoV1Process(__ubuf__ P *gradHMix, __ubuf__ P *gradInvRmsOut, __ubuf__ P *gradAlphaOut,
                                         __ubuf__ P *gatherIn, __ubuf__ P *hMixIn, __ubuf__ P *invRmsIn,
                                         uint32_t dealBSSize);
    __aicore__ inline void VFDoV1ProcessBiasGradForN4N6(__ubuf__ P *outBufDst, __ubuf__ P *gatherFusion,
                                                        uint32_t curBSSize);
    __aicore__ inline void VFDoV1ProcessBiasGradForN8(__ubuf__ P *outBufDst, __ubuf__ P *gatherFusion,
                                                      uint32_t curBSSize);
    __aicore__ inline void VFDoV2HInMulHPre(__ubuf__ P *xGradVec3BufAddr, __ubuf__ T *hInGradInBufAddr,
                                            __ubuf__ P *hPreUbAddr, uint32_t currentChunkSize, uint32_t copySizeND,
                                            uint32_t currentN, uint32_t bsGlobalOffset);

    template <bool hasGamma>
    __aicore__ inline void VFDoV2XCastAndMulGamma(__ubuf__ P *xRsFp32Addr, __ubuf__ P *gammaOutAddr,
                                                  __ubuf__ T *bf16InputAddr, __ubuf__ P *gammaInAddr,
                                                  uint32_t currentChunkSize, uint32_t copySizeND);

    template <bool hasGamma>
    __aicore__ inline void VFDoV2GammaMulXRsGradMm(__ubuf__ P *xGradVec3Addr, __ubuf__ P *xRsGradMmAddr,
                                                   __ubuf__ P *xRsFp32Addr, __ubuf__ P *invRmsAddr,
                                                   __ubuf__ P *gammaInAddr, uint32_t currentChunkSize,
                                                   uint32_t copySizeND);

    template <bool hasGradXPost>
    __aicore__ inline void VFDoV2AddGradXPostAndCast(__ubuf__ T *outputAddr, __ubuf__ P *xGradAddr,
                                                     __ubuf__ T *gradXPostAddr, uint32_t totalElem);

    __aicore__ inline void VFDoV3ProcessAlphaGrad(__ubuf__ P *alphaGradOut, __ubuf__ P *alphaGradIn);

private:
    MhcPreBackwardCubeCompute cubeCompute_;
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

    GlobalTensor<T> xGm_;                 // 输入 x
    GlobalTensor<P> phiGm_;               // 输入 phi
    GlobalTensor<P> alphaGm_;             // 输入 alpha
    GlobalTensor<P> gammaGm_;             // 输入 gamma
    GlobalTensor<T> gradHInGm_;           // 输入 grad_h_in
    GlobalTensor<P> gradHPostGm_;         // 输入 grad_h_post
    GlobalTensor<P> gradHResGm_;          // 输入 grad_h_res
    GlobalTensor<T> gradXPostOptionalGm_; // Optional input grad_x_post_optional from GM
    GlobalTensor<P> invRmsGm_;            // 前向预计算 inv_rms
    GlobalTensor<P> hMixGm_;              // 前向预计算 h_mix
    GlobalTensor<P> hPreGm_;              // 前向预计算 h_pre
    GlobalTensor<P> hPostGm_;             // 前向输出 h_post
    GlobalTensor<P> workSpaceGm_;
    WorkspaceBuffer<P> workspaceBuf_; // Workspace buffer管理接口

    GlobalTensor<T> xGradGm_;     // 输出 grad_x
    GlobalTensor<P> gradPhiGm_;   // 输出 grad_phi
    GlobalTensor<P> alphaGradGm_; // 输出 grad_alpha
    GlobalTensor<P> gradBiasGm_;  // 输出 grad_bias
    GlobalTensor<P> gammaGradGm_; // 输出 grad_gamma

    LocalTensor<P> gammaUb;      // gamma
    LocalTensor<T> gradHInUb;    // grad_h_in
    LocalTensor<P> hPreUb;       // h_pre
    LocalTensor<P> invRmsGradUb; // inv_rms_grad
    LocalTensor<P> invRmsUb;     // inv_rms
    LocalTensor<P> xRsGradMmUb;  // x_rs_grad_mm
    LocalTensor<T> xGradUb;      // grad_x bf16

    // ProcessV2 chunk循环使用的buffer
    LocalTensor<P> invRmsInBuf_;
    LocalTensor<P> invRmsGradBuf_;
    LocalTensor<T> gradHInInBuf_;
    LocalTensor<T> bf16InputBuf_;
    LocalTensor<P> gammaOutUb_;
    LocalTensor<P> xRsGradMmInBuf_;
    LocalTensor<T> bf16OutputBuf_;
    LocalTensor<T> gradXPostBuf_;

    // ProcessV2 chunk循环的buffer (由ProcessV2分配)
    LocalTensor<P> gammaIn_;
    LocalTensor<P> xRsFp32Buf_;
    LocalTensor<P> xGradVec3Buf_;
    LocalTensor<P> invRmsUb_;
    LocalTensor<P> hPreUb_;

    GlobalTensor<P> invRmsGradGm_; // V1输出 inv_rms_grad
    GlobalTensor<P> xRsGradMmGm_;  // C0输出 x_rs_grad_mm
    GlobalTensor<P> xRsFp32Gm_;    // V2输出 & C1输入 x_rs

    LocalTensor<uint32_t> hPreGatherOffsetBuf_;
    TPipe *pipe_;
    const MhcPreBackwardTilingData *tiling_;
    TQue<QuePosition::VECIN, 0> vecInQueue0_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue1_;
    TQue<QuePosition::VECIN, 1> vecInQueue1_;
    TQue<QuePosition::VECOUT, 1> vecOutQueueSmall_;

    TBuf<TPosition::VECCALC> fp32TBuf_;

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
    bool withGamma_;
    bool withGradXPost_;
    float scaleMean_;
};

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::Init(InitParams initParams)
{
    InitGlobalBuffersAndTiling(initParams);
    InitUBAndAIVBuffers(initParams);
    InitStage2AndDataCopy();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::InitGlobalBuffersAndTiling(InitParams initParams)
{
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
    xGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
    phiGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_PERSISTENT);
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
    gradHInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.grad_h_in));
    gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_h_post));
    gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_h_res));
    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
    hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));
    hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
    xGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.grad_x));
    xGradGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_phi));
    alphaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_alpha));
    gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_bias));
    workSpaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));
    withGamma_ = (initParams.gamma != nullptr);
    withGradXPost_ = (initParams.grad_x_post_optional != nullptr);
    if (withGradXPost_) {
        gradXPostOptionalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.grad_x_post_optional));
        gradXPostOptionalGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (withGamma_) {
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
        gammaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_gamma));
    }
    tiling_ = initParams.tilingData;
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
    scaleMean_ = (nD_ > 0) ? (1.0f / nD_) : 0.0f;
    eleNumPerVf_ = GetVRegSize() / sizeof(P);
    workspaceBuf_.Init(totalLength_, fusionSize_, vecCoreNum_, coreNum_);
    blockIdx_ = GetBlockIdx();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::InitUBAndAIVBuffers(InitParams initParams)
{
    vecDealChunk_ = N_ > LARGE_N_THRESHOLD ? VEC_DEAL_CHUNK_LARGE_N : VEC_DEAL_CHUNK_SMALL_N;
    hPreMaxBufLen_ = vecDealChunk_ * N_;
    hPostMaxBufLen_ = hPreMaxBufLen_;
    hMixOffset_ = (hPreMaxBufLen_ + hPostMaxBufLen_) * sizeof(P);
    gradHPostOffset_ = hPreMaxBufLen_ * sizeof(P);
    gradHResBufLen_ = vecDealChunk_ * N_ * N_;
    hFusionBufLen_ = hPreMaxBufLen_ + hPostMaxBufLen_ + gradHResBufLen_;
    pipe_ = initParams.tPipeIn;
    if ASCEND_IS_AIC {
        cubeCompute_.Init(pipe_, tiling_);
    }
    if ASCEND_IS_AIV {
        pipe_->InitBuffer(vecInQueue0_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecInQueue1_, DOUBLE_BUFFER, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
        globalUbOffset_ = 0;
        vecDealBSPeCore_ = totalLength_ / vecCoreNum_;
        if (vecDealBSPeCore_ == 0) {
            vecDealBSPeCore_ = totalLength_;
            usedVecCoreNum_ = 1;
        } else {
            vecDealBSPeCore_ = MhcPreBackwardUtils::CeilAlign(vecDealBSPeCore_, static_cast<uint64_t>(CEIL_ALIGN_16));
            uint64_t usedVecCoreNum = CeilDiv(totalLength_, vecDealBSPeCore_);
            usedVecCoreNum_ = usedVecCoreNum > vecCoreNum_ ? vecCoreNum_ : static_cast<uint32_t>(usedVecCoreNum);
        }
        dealStartBS_ = vecDealBSPeCore_ * blockIdx_;
        dealEndBS_ = dealStartBS_ + vecDealBSPeCore_;
        if (dealEndBS_ > totalLength_ || blockIdx_ == usedVecCoreNum_ - 1) {
            dealEndBS_ = totalLength_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::InitStage2AndDataCopy()
{
    v1UsedCubeCoreNum_ = 0;
    cubeDealnDPeCore_ = nD_ / GetBlockNum();
    if (cubeDealnDPeCore_ == 0) {
        cubeDealnDPeCore_ = nD_;
        v1UsedCubeCoreNum_ = 1;
    } else {
        cubeDealnDPeCore_ = MhcPreBackwardUtils::CeilAlign(cubeDealnDPeCore_, CEIL_ALIGN_128);
        v1UsedCubeCoreNum_ = CeilDiv(nD_, cubeDealnDPeCore_);
    }
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::Process()
{
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
        if (GetBlockIdx() == (GetBlockNum() - 1)) {
            ProcessV3();
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoPreProcessV0(__ubuf__ P *hPreGradAddr, __ubuf__ T *xB16InAddr,
                                                                    __ubuf__ T *gradHInB16InAddr, uint16_t curLenN,
                                                                    uint32_t lenD, uint64_t i, uint32_t j,
                                                                    uint64_t runBSStart)
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
            // x[1, lenD]  hinGrad [1, lenD]   res[1, lenD]
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

                Reg::Mul(mulReg, xFp32Reg, gradHInFp32Reg, mask); // mulReg[1,lenD] 0-63
                Reg::Reduce<Reg::ReduceType::SUM>(tmpSumPerVfReg, mulReg,
                                                  mask); // 每个vfBlockIdx循环的临时求和，每64个元素的和
                Reg::Add(sumReg, sumReg, tmpSumPerVfReg, mask); // D方向上的reduceSum
            }
            uint32_t bufIdx = offset + offsetN;
            Reg::Store(hPreGradAddr + bufIdx, sumReg, 1);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::PreProcessV0(LocalTensor<P> &hPreGradBuf, uint64_t runBSStart,
                                                                uint64_t runBSEnd)
{
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

            __ubuf__ T *xB16InAddr = (__ubuf__ T *)bf16InputBuf.GetPhyAddr();
            __ubuf__ T *gradHInB16InAddr = (__ubuf__ T *)gradHInInputBuf.GetPhyAddr();
            __ubuf__ P *hPreGradAddr = (__ubuf__ P *)hPreGradBuf.GetPhyAddr();
            VFDoPreProcessV0(hPreGradAddr, xB16InAddr, gradHInB16InAddr, curLenN, copySizeD, i, j, runBSStart);

            inputOffset += copySizeD * curLenN;
            vecInQueue0_.FreeTensor(bf16InputBuf);
        }
        vecInQueue1_.FreeTensor(gradHInInputBuf);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0Main()
{
    if (blockIdx_ >= usedVecCoreNum_) {
        return;
    }
    uint64_t dealBsNum = dealEndBS_ - dealStartBS_;
    uint64_t vecRunTimes = CeilDiv(dealBsNum, static_cast<uint64_t>(vecDealChunk_));
    float alphaPre = alphaGm_.GetValue(0);
    float alphaPost = alphaGm_.GetValue(1);
    float alphaComb = alphaGm_.GetValue(2);
    LocalTensor<P> sumBuf = fp32TBuf_.GetWithOffset<P>(fusionSize_, globalUbOffset_);
    globalUbOffset_ += MhcPreBackwardUtils::CeilAlign(uint32_t(fusionSize_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    ProcessV0MainLoop(vecRunTimes, alphaPre, alphaPost, alphaComb, sumBuf);
    ProcessV0MainAlphaReduce(sumBuf, GetBlockIdx());
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0MainLoop(uint64_t vecRunTimes, float alphaPre,
                                                                     float alphaPost, float alphaComb,
                                                                     LocalTensor<P> &sumBuf)
{
    for (uint64_t cIdx = 0; cIdx < vecRunTimes; cIdx++) {
        uint64_t runBSStart = cIdx * vecDealChunk_ + dealStartBS_;
        uint64_t runBSEnd = runBSStart + vecDealChunk_;
        if (runBSEnd > dealEndBS_) {
            runBSEnd = dealEndBS_;
        }
        uint32_t currentDealBsNum = static_cast<uint32_t>(runBSEnd - runBSStart);
        V0V1Buffers<P> buffers;
        AllocV0V1Buffers(runBSStart, runBSEnd, buffers);
        PreProcessV0(buffers.hPreGradBuf, runBSStart, runBSEnd);
        ProcessV0(runBSStart, runBSEnd, buffers, alphaPre, alphaPost, alphaComb);
        ProcessV1(runBSStart, runBSEnd, buffers, cIdx, sumBuf);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0MainAlphaReduce(LocalTensor<P> &sumBuf, uint32_t coreId)
{
    AscendC::LocalTensor<P> alphaOutBuf = vecOutQueue1_.AllocTensor<P>();
    uint32_t srcShape1[] = {1, N_};
    uint32_t srcShape2[] = {1, 2 * N_};
    uint32_t srcShape3[] = {1, fusionSize_};
    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf, sumBuf, srcShape1, true);
    PipeBarrier<PIPE_V>();
    Muls(sumBuf, sumBuf, ZERO, N_);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf[8], sumBuf, srcShape2, true);
    PipeBarrier<PIPE_V>();
    Muls(sumBuf, sumBuf, ZERO, N_ * 2);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::AR, true>(alphaOutBuf[16], sumBuf, srcShape3, true);
    PipeBarrier<PIPE_V>();
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::AllocV0V1Buffers(uint64_t runBSStart, uint64_t runBSEnd,
                                                                    V0V1Buffers<P> &buffers)
{
    buffers.stepLength = static_cast<uint32_t>((runBSEnd - runBSStart) * N_);
    uint32_t hMixLength = static_cast<uint32_t>((runBSEnd - runBSStart) * N_ * N_);
    buffers.gatherLength = buffers.stepLength * 2 + hMixLength;
    uint32_t stepOffset1 =
        MhcPreBackwardUtils::CeilAlign(uint32_t(buffers.stepLength * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    // Allocate hPreGradBuf first
    uint32_t hpreGradTotlenLen = MhcPreBackwardUtils::CeilAlign(buffers.stepLength, uint32_t(CEIL_ALIGN_DEFAULT));
    buffers.hPreGradBuf = fp32TBuf_.GetWithOffset<P>(hpreGradTotlenLen, globalUbOffset_);

    uint32_t ubOffset = globalUbOffset_ + hpreGradTotlenLen * sizeof(P);
    uint32_t fusionOffset = ubOffset;

    // V0 buffers
    buffers.hPreBufS1 = fp32TBuf_.GetWithOffset<P>(hPreMaxBufLen_, ubOffset);
    buffers.hPostBufS1 = fp32TBuf_.GetWithOffset<P>(hPostMaxBufLen_, ubOffset + gradHPostOffset_);
    buffers.hResBufS1 = fp32TBuf_.GetWithOffset<P>(gradHResBufLen_, ubOffset + hMixOffset_);
    buffers.hFusionBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    ubOffset += hFusionBufLen_ * sizeof(P);
    fusionOffset = ubOffset;

    buffers.hPreBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    ubOffset += stepOffset1;
    buffers.hPostBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    buffers.gatherFusionOutBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    // V1 buffers
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.invRmsBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.calcTmpBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.brcbTmpBuf = fp32TBuf_.GetWithOffset<uint8_t>(FP32_BUF_SIZE - ubOffset, ubOffset);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV0HPreGrad(__ubuf__ P *hPreBufS1Addr, __ubuf__ P *hPreAddr,
                                                                  __ubuf__ P *gradHInAddr, uint32_t totalElem)
{
    uint32_t eleNumPerVf = 64;
    uint16_t loopCnt = Ceil(totalElem, eleNumPerVf);
    uint32_t curElemCnt = totalElem;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> hPreReg, hInGradReg;
        Reg::RegTensor<P> s1Reg, s2Reg, mulReg, resReg;
        Reg::RegTensor<P> oneReg;
        Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(oneReg, 1.0f, mask);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = Reg::UpdateMask<P>(curElemCnt);
            Reg::LoadAlign(hPreReg, hPreAddr + vfBlockIdx * eleNumPerVf);
            Reg::LoadAlign(hInGradReg, gradHInAddr + vfBlockIdx * eleNumPerVf);
            Reg::Adds(s1Reg, hPreReg, -hcEps_, mask);
            Reg::Sub(s2Reg, oneReg, s1Reg, mask);
            Reg::Mul(mulReg, s1Reg, s2Reg, mask);
            Reg::Mul(resReg, mulReg, hInGradReg, mask);

            Reg::StoreAlign(hPreBufS1Addr + vfBlockIdx * eleNumPerVf, resReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0(uint64_t runBSStart, uint64_t runBSEnd,
                                                             V0V1Buffers<P> &buffers, float alphaPre, float alphaPost,
                                                             float alphaComb)
{
    AscendC::LocalTensor<P> fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    auto hMixGradBuf = buffers.calcTmpBuf;
    dataCopyParams.blockLen = buffers.stepLength * sizeof(P);

    PipeBarrier<PIPE_MTE2>();
    DataCopyPad(fp32InputBuf, hPreGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    LocalTensor<P> hPre = vecInQueue1_.DeQue<P>();

    __ubuf__ P *hPreAddr = (__ubuf__ P *)hPre.GetPhyAddr();
    __ubuf__ P *gradHInAddr = (__ubuf__ P *)buffers.hPreGradBuf.GetPhyAddr();
    __ubuf__ P *hPreBufS1Addr = (__ubuf__ P *)buffers.hPreBufS1.GetPhyAddr();
    VFDoV0HPreGrad(hPreBufS1Addr, hPreAddr, gradHInAddr, buffers.stepLength);
    vecInQueue1_.FreeTensor(hPre);

    fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(fp32InputBuf, hPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    AscendC::LocalTensor<P> gradHPost;
    vecInQueue0_.AllocTensor<P>(gradHPost);
    DataCopyPad(gradHPost, gradHPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHPost);

    LocalTensor<P> hPost = vecInQueue1_.DeQue<P>();
    vecInQueue0_.DeQue<P>(gradHPost);

    AscendC::LocalTensor<P> gradHResBuf;
    vecInQueue0_.AllocTensor<P>(gradHResBuf);
    dataCopyParams.blockLen = buffers.stepLength * N_ * sizeof(P);
    DataCopyPad(gradHResBuf, gradHResGm_[runBSStart * N_ * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHResBuf);

    VFDoV0ProcessGradHPost((__ubuf__ P *)hPost.GetPhyAddr(), (__ubuf__ P *)gradHPost.GetPhyAddr(),
                           (__ubuf__ P *)buffers.hPostBufS1.GetPhyAddr(), buffers.stepLength);
    vecInQueue1_.FreeTensor(hPost);
    vecInQueue0_.FreeTensor(gradHPost);

    vecInQueue0_.DeQue<P>(gradHResBuf);

    Muls(buffers.hResBufS1, gradHResBuf, ONE, buffers.stepLength * N_);
    PipeBarrier<PIPE_V>();
    vecInQueue0_.FreeTensor(gradHResBuf);

    uint32_t currentDealBsNum = static_cast<uint32_t>(runBSEnd - runBSStart);
    AIV02Process(buffers, hMixGradBuf, currentDealBsNum, alphaPre, alphaPost, alphaComb);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV0ProcessGradHPost(__ubuf__ P *hPostIn, __ubuf__ P *gradHPostIn,
                                                                          __ubuf__ P *gradHPostOut, uint32_t stepLen)
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::AIV02Process(V0V1Buffers<P> &buffers, LocalTensor<P> &hMixGradBuf,
                                                                uint64_t currentDealBsNum, float alphaPre,
                                                                float alphaPost, float alphaComb)
{
    __ubuf__ P *hPreBufAddr = (__ubuf__ P *)buffers.hPreBufS1.GetPhyAddr();
    __ubuf__ P *hPostBufAddr = (__ubuf__ P *)buffers.hPostBufS1.GetPhyAddr();
    __ubuf__ P *hResBufAddr = (__ubuf__ P *)buffers.hResBufS1.GetPhyAddr();
    __ubuf__ P *gatherFusionOutBufAddr = (__ubuf__ P *)buffers.gatherFusionOutBuf.GetPhyAddr();
    __ubuf__ P *hMixGradBufAddr = (__ubuf__ P *)hMixGradBuf.GetPhyAddr();

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
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV1(uint64_t runBSStart, uint64_t runBSEnd,
                                                             V0V1Buffers<P> &buffers, uint64_t vecRuntimesId,
                                                             LocalTensor<P> &sumBuf)
{
    uint32_t curBSSize = static_cast<uint32_t>(runBSEnd - runBSStart);
    constexpr bool isReuse = false;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    AscendC::LocalTensor<P> fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    AscendC::LocalTensor<P> invRmsGradUb = vecOutQueueSmall_.AllocTensor<P>();
    if (N_ == 8) {
        VFDoV1ProcessBiasGradForN8((__ubuf__ P *)fp32OutBuf.GetPhyAddr(),
                                   (__ubuf__ P *)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
    } else {
        VFDoV1ProcessBiasGradForN4N6((__ubuf__ P *)fp32OutBuf.GetPhyAddr(),
                                     (__ubuf__ P *)buffers.gatherFusionOutBuf.GetPhyAddr(), curBSSize);
    }
    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> BiasOutBuf = vecOutQueue1_.DeQue<P>();

    uint32_t coreId = GetBlockIdx();
    dataCopyParams.blockLen = fusionSize_ * sizeof(P);
    if (vecRuntimesId != 0) {
        SetAtomicAdd<float>();
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(coreId)], BiasOutBuf, dataCopyParams);
        PipeBarrier<PIPE_MTE3>();
        SetAtomicNone();
    } else {
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(coreId)], BiasOutBuf, dataCopyParams);
        PipeBarrier<PIPE_MTE3>();
    }
    vecOutQueue1_.FreeTensor(BiasOutBuf);

    auto hMixGradBuf = buffers.calcTmpBuf;
    fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    dataCopyParams.blockLen = curBSSize * sizeof(P);
    AscendC::LocalTensor<P> fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(fp32InputBuf, invRmsGm_[runBSStart], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    LocalTensor<P> invRmsBufLocal = vecInQueue1_.DeQue<P>();

    const uint32_t xRowSumBroadCastDst[2] = {curBSSize, fusionSize_};
    const uint32_t xRowSumBroadCastSrc[2] = {curBSSize, 1};
    BroadCast<float, 2, 1>(buffers.invRmsBuf, invRmsBufLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc,
                           buffers.brcbTmpBuf);
    PipeBarrier<PIPE_V>();
    vecInQueue1_.FreeTensor(invRmsBufLocal);

    PipeBarrier<PIPE_V>();
    LocalTensor<P> hMixBuf;
    vecInQueue0_.AllocTensor<P>(hMixBuf);
    dataCopyParams.blockLen = curBSSize * fusionSize_ * sizeof(P);
    DataCopyPad(hMixBuf, hMixGm_[runBSStart * fusionSize_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(hMixBuf);
    vecInQueue0_.DeQue<P>(hMixBuf);

    PipeBarrier<PIPE_V>();
    VFDoV1Process((__ubuf__ P *)fp32OutBuf.GetPhyAddr(), (__ubuf__ P *)hMixGradBuf.GetPhyAddr(),
                  (__ubuf__ P *)buffers.hFusionBuf.GetPhyAddr(), (__ubuf__ P *)buffers.gatherFusionOutBuf.GetPhyAddr(),
                  (__ubuf__ P *)hMixBuf.GetPhyAddr(), (__ubuf__ P *)buffers.invRmsBuf.GetPhyAddr(), curBSSize);

    vecOutQueue1_.EnQue(fp32OutBuf);
    LocalTensor<P> hMixGradOutBuf = vecOutQueue1_.DeQue<P>();
    dataCopyParams.blockLen = curBSSize * fusionSize_ * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(runBSStart)], hMixGradOutBuf, dataCopyParams);
    PipeBarrier<PIPE_V>();

    uint32_t shape[] = {curBSSize, fusionSize_};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(invRmsGradUb, hMixGradBuf, buffers.brcbTmpBuf, shape, true);
    vecOutQueueSmall_.EnQue(invRmsGradUb);
    LocalTensor<P> invRmsGradUbtBuf = vecOutQueueSmall_.DeQue<P>();

    dataCopyParams.blockLen = curBSSize * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(runBSStart)], invRmsGradUbtBuf, dataCopyParams);
    ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(hMixGradBuf, buffers.hFusionBuf, buffers.brcbTmpBuf, shape,
                                                            true);
    PipeBarrier<PIPE_V>();
    vecOutQueue1_.FreeTensor(hMixGradOutBuf);
    vecOutQueueSmall_.FreeTensor(invRmsGradUbtBuf);
    vecInQueue0_.FreeTensor(hMixBuf);

    if (vecRuntimesId == 0) {
        Adds(sumBuf, hMixGradBuf, ZERO, fusionSize_);
    } else {
        Add(sumBuf, hMixGradBuf, sumBuf, fusionSize_);
    }
    PipeBarrier<PIPE_V>();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV1Process(__ubuf__ P *gradHMixOut, __ubuf__ P *hMixGradBuf,
                                                                 __ubuf__ P *gradAlphaOut, __ubuf__ P *gatherIn,
                                                                 __ubuf__ P *hMixIn, __ubuf__ P *invRmsIn,
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV1ProcessBiasGradForN8(__ubuf__ P *outBufDst,
                                                                              __ubuf__ P *gatherFusion,
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV1ProcessBiasGradForN4N6(__ubuf__ P *outBufDst,
                                                                                __ubuf__ P *gatherFusion,
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

            Reg::LoadAlign(gatherReg, gatherFusion + elemOffset1);

            Reg::Add(sumReg, sumReg, gatherReg, mask);
        }
        Reg::StoreAlign(outBufDst, sumReg, mask);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC0C1Pipeline()
{
    if (GetBlockIdx() >= v1UsedCubeCoreNum_) {
        // Init会在所有AIC上设置缓冲区令牌，未参与计算的AIC也必须回收，避免残留事件影响后续下发。
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

    uint32_t currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - dealStartND_);
    uint32_t currentBSBlock = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_));
    uint32_t buffId = 0;
    // 预执行首块C0，为C0/V2/C1双缓冲流水建立初始槽。
    ProcessC0(dealStartND_, currentNDBlock, 0, currentBSBlock, buffId);

    for (uint32_t offsetND = dealStartND_; offsetND < dealEndND_; offsetND += ND_BLOCK_SIZE) {
        currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - offsetND);

        // 先发射下一块C0，再等待当前块V2完成并执行C1。
        for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
            currentBSBlock = static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));

            uint64_t nextOffsetBS = offsetBS + BS_BLOCK_SIZE;
            if (nextOffsetBS < totalLength_) {
                uint32_t nextBSBlock =
                    static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - nextOffsetBS));
                ProcessC0(offsetND, currentNDBlock, nextOffsetBS, nextBSBlock, buffId + 1);
            } else {
                uint32_t nextOffsetND = offsetND + ND_BLOCK_SIZE;
                if (nextOffsetND < dealEndND_) {
                    uint32_t nextNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - nextOffsetND);
                    nextOffsetBS = 0;
                    uint32_t nextBSBlock =
                        static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_));
                    ProcessC0(nextOffsetND, nextNDBlock, nextOffsetBS, nextBSBlock, buffId + 1);
                }
            }

            ProcessC1(offsetND, currentNDBlock, offsetBS, currentBSBlock, buffId);
            buffId++;
        }
    }
    cubeCompute_.End();

    if ASCEND_IS_AIC {
        if (enableHf32) {
            AscendC::SetHF32TransMode(oriHf32TransMode);
            AscendC::SetHF32Mode(oriHf32Mode);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC0(uint32_t offsetND, uint32_t currentNDBlock,
                                                             uint64_t offsetBS, uint32_t currentBSBlock,
                                                             uint32_t buffIdx)
{
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    uint64_t writeOffset = workspaceBuf_.GetXRsGradOffset(GetBlockIdx(), buffIdx);
    cubeCompute_.ProcessC0(mmConfigC0_, workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], phiGm_[offsetND],
                           workSpaceGm_[writeOffset], currentBSBlock, currentNDBlock, fusionSize_, nD_);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC1(uint32_t offsetND, uint32_t currentNDBlock,
                                                             uint64_t offsetBS, uint32_t currentBSBlock,
                                                             uint32_t buffIdx)
{
    AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C1);
    // 两个Vector核写入同一xRs槽的前后半个ND块，C1按currentNDBlock整体读取。
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    uint64_t offsetXs = workspaceBuf_.GetXRsOffset(GetBlockIdx(), buffIdx);
    bool enAtomic = (offsetBS != 0);
    cubeCompute_.ProcessC1(mmConfigC1_, workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], workSpaceGm_[offsetXs],
                           gradPhiGm_[offsetND], fusionSize_, currentNDBlock, currentBSBlock, fusionSize_,
                           ND_BLOCK_SIZE, nD_, enAtomic);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2Pipeline()
{
    if (blockIdx_ >= 2 * v1UsedCubeCoreNum_) {
        return;
    }
    hPreGatherOffsetBuf_ = fp32TBuf_.GetWithOffset<uint32_t>(PROCESS_V2_CHUNK_SIZE, 0);
    globalUbOffset_ = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t); // V2部分开始分配fp32TBuf_
    for (uint32_t i = 0; i < PROCESS_V2_CHUNK_SIZE; i++) {
        hPreGatherOffsetBuf_.SetValue(i, i * N_ * sizeof(P));
    }

    // 确定处理范围 （1C2V 模式）
    uint32_t vecIdx = blockIdx_ % VEC_CORE_VECIDX_MOD;
    uint32_t buffId = 0;
    // ND方向切分：每个ND block大小为nDBlockSize_，在1C2V模式下，每个V核处理一半
    for (uint32_t offsetNDBase = dealStartND_; offsetNDBase < dealEndND_; offsetNDBase += ND_BLOCK_SIZE) {
        // 根据vecIdx选择处理ND block的前半部分(0)或后半部分(1)
        uint32_t offsetND = offsetNDBase + (ND_BLOCK_SIZE / 2) * vecIdx;
        if (offsetND >= dealEndND_) {
            continue; // 超出范围，跳过
        }
        uint32_t copySizeND = Min(ND_BLOCK_SIZE / 2, dealEndND_ - offsetND);

        // BS方向切分：与C0保持一致，按BS_BLOCK_SIZE切分
        LocalTensor<P> sumBuf = vecOutQueueSmall_.AllocTensor<P>();
        for (uint64_t offsetBS = 0; offsetBS < totalLength_; offsetBS += BS_BLOCK_SIZE) {
            uint32_t currentBSBlock =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(BS_BLOCK_SIZE), totalLength_ - offsetBS));
            // 执行当前的V2
            ProcessV2(offsetND, copySizeND, offsetBS, offsetBS + currentBSBlock, sumBuf, buffId);
            buffId++;
            // 通知C1计算
            AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
        }
        if (withGamma_) {
            vecOutQueueSmall_.EnQue(sumBuf);
            sumBuf = vecOutQueueSmall_.DeQue<P>();
            DataCopyExtParams fp32ExtCopyParams;
            fp32ExtCopyParams.blockCount = 1;
            fp32ExtCopyParams.blockLen = copySizeND * sizeof(P);
            fp32ExtCopyParams.srcStride = 0;
            fp32ExtCopyParams.dstStride = 0;
            DataCopyPad(gammaGradGm_[offsetND], sumBuf, fp32ExtCopyParams);
        }
        vecOutQueueSmall_.FreeTensor(sumBuf);
    }
}

template <class T, class P>
template <bool hasGamma>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2XCastAndMulGamma(
    __ubuf__ P *xRsFp32Addr, __ubuf__ P *gammaOutAddr, __ubuf__ T *bf16InputAddr, __ubuf__ P *gammaInAddr,
    uint32_t currentChunkSize, uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xInB16Reg;
        Reg::RegTensor<P> xFp32Reg, gammaReg, resultReg;
        if constexpr (hasGamma) {
            // copySizeND最大为ND_BLOCK_SIZE的一半，即64个元素，可完整驻留在一个向量寄存器中。
            Reg::LoadAlign(gammaReg, gammaInAddr);
        }
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            // copySizeND最大为64，一个向量指令即可处理整行，无需额外的VF分块循环。
            uint32_t lenND = copySizeND;
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;

            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg, bf16InputAddr + offset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

            Reg::StoreAlign(xRsFp32Addr + offset, xFp32Reg, mask); // 后续其他计算需要xFp32

            if constexpr (hasGamma) {
                Reg::Mul(resultReg, gammaReg, xFp32Reg, mask); // 逐元素乘：gamma * x
                Reg::StoreAlign(gammaOutAddr + offset, resultReg, mask);
            } else {
                Reg::StoreAlign(gammaOutAddr + offset, xFp32Reg, mask);
            }
        }
    }
}

template <class T, class P>
template <bool hasGamma>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2GammaMulXRsGradMm(
    __ubuf__ P *xGradVec3Addr, __ubuf__ P *xRsGradMmAddr, __ubuf__ P *xRsFp32Addr, __ubuf__ P *invRmsAddr,
    __ubuf__ P *gammaInAddr, uint32_t currentChunkSize, uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> gammaReg, xRsFp32Reg, xRsGradMmReg, xRsGradReg, gammaGradReg, xGradReg;
        Reg::RegTensor<P> invRmsReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        if constexpr (hasGamma) {
            // copySizeND最大为ND_BLOCK_SIZE的一半，即64个元素，可完整驻留在一个向量寄存器中。
            Reg::LoadAlign(gammaReg, gammaInAddr);
        }
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            // invRms每行只使用一个FP32标量，直接广播加载，减少标量读取和Duplicate指令。
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invRmsReg, invRmsAddr + tIdx);
            // copySizeND最大为64，当前T行只需一次向量处理，删除恒单次的内层循环控制。
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;
            Reg::LoadAlign(xRsGradMmReg, xRsGradMmAddr + offset);
            Reg::LoadAlign(xGradReg, xGradVec3Addr + offset);
            Reg::LoadAlign(xRsFp32Reg, xRsFp32Addr + offset);

            Reg::Mul(xRsGradReg, xRsFp32Reg, invRmsReg, mask);
            Reg::Add(xGradReg, xGradReg, xRsGradReg, mask);

            if constexpr (hasGamma) {
                Reg::Mul(xRsGradReg, gammaReg, xRsGradMmReg, mask);
                Reg::Add(xGradReg, xGradReg, xRsGradReg, mask);
                Reg::Mul(gammaGradReg, xRsFp32Reg, xRsGradMmReg, mask);
                Reg::StoreAlign(xRsGradMmAddr + offset, gammaGradReg, mask);
            } else {
                Reg::Add(xGradReg, xGradReg, xRsGradMmReg, mask);
            }
            Reg::StoreAlign(xGradVec3Addr + offset, xGradReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2HInMulHPre(__ubuf__ P *xGradVec3BufAddr,
                                                                    __ubuf__ T *hInGradInBufAddr,
                                                                    __ubuf__ P *hPreUbAddr, uint32_t currentChunkSize,
                                                                    uint32_t copySizeND, uint32_t currentN,
                                                                    uint32_t bsGlobalOffset)
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
            // copySizeND最大为64，当前T行只需一次向量处理，删除恒单次的内层循环控制。
            mask = Reg::UpdateMask<P>(lenND);
            // hIn B16 -> B32
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(hInReg, hInGradInBufAddr + hInOffset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(hInFP32Reg, hInReg, mask);
            Reg::Mul(hInFP32Reg, hInFP32Reg, hPreFullReg, mask);
            Reg::StoreAlign(xGradVec3BufAddr + hInOffset, hInFP32Reg, mask);
            hPreOffset += N_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint64_t bsStart,
                                                             uint64_t bsEnd, LocalTensor<P> &sumBuf, uint32_t buffId)
{
    // ==========================================================
    // 初始化
    // ==========================================================
    uint32_t currentN = offsetND / D_;
    // 统一使用64行chunk，使N=4时两个FP32主工作区各由32KB降为16KB，
    // 从而压缩V2的TBuf峰值，并为后续输出队列双缓冲腾出UB空间。
    uint32_t chunkSize = PROCESS_V2_CHUNK_SIZE / NUMBER_TWO;
    uint32_t currentBsSize = static_cast<uint32_t>(bsEnd - bsStart);
    buffId = buffId % VEC_CORE_VECIDX_MOD;

    // ==========================================================
    // Buffer分配 - 按最大CHUNK大小分配 (存到全局变量)
    // ==========================================================
    uint32_t ubOffset = globalUbOffset_; // sumBuf offset
    uint32_t maxChunkNDSize = chunkSize * copySizeND;
    if (withGamma_) {
        gammaIn_ = fp32TBuf_.GetWithOffset<P>(copySizeND, ubOffset);
        ubOffset =
            MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + copySizeND * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    }

    xRsFp32Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    xGradVec3Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    invRmsUb_ = fp32TBuf_.GetWithOffset<P>(currentBsSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    uint32_t hPreUbLen =
        MhcPreBackwardUtils::CeilAlign(uint32_t(currentBsSize * N_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT)) /
        sizeof(P);
    hPreUb_ = fp32TBuf_.GetWithOffset<P>(hPreUbLen, ubOffset);
    ubOffset = MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * N_ * sizeof(P)),
                                              uint32_t(CEIL_ALIGN_DEFAULT));

    // ==========================================================
    // 参数初始化和hPre预加载
    // ==========================================================
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    if (withGamma_) {
        bsCopyParams.blockLen = copySizeND * sizeof(P);
        DataCopyPad(gammaIn_, gammaGm_[offsetND], bsCopyParams, dataCopyPadParams);
    }
    bsCopyParams.blockLen = currentBsSize * N_ * sizeof(P);
    DataCopyPad(hPreUb_, hPreGm_[bsStart * N_], bsCopyParams, dataCopyPadParams);

    // 当前BS块的invRms系数一次搬入并计算，后续各chunk按行偏移复用已有UB空间。
    ProcessV2ChunkLoadInvRms(bsStart, currentBsSize);
    // ==========================================================
    // 主循环: 按chunk=64切分BS方向
    // ==========================================================
    for (uint64_t bsChunkStart = bsStart; bsChunkStart < bsEnd; bsChunkStart += chunkSize) {
        uint64_t bsChunkEnd = Min(bsChunkStart + chunkSize, bsEnd);
        uint32_t currentChunkSize = static_cast<uint32_t>(bsChunkEnd - bsChunkStart);
        uint32_t chunkNDSize = currentChunkSize * copySizeND;

        // C0每完成128行发布一次；两个64行chunk共同消费一块C0结果。
        if ((bsChunkStart - bsStart) % MATMUL_WRITE_OFFSET_M == 0) {
            AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C0);
        }
        ProcessV2ChunkLoop(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, chunkNDSize, currentN, buffId,
                           sumBuf);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND,
                                                                      uint64_t bsStart, uint64_t bsChunkStart,
                                                                      uint32_t currentChunkSize, uint32_t chunkNDSize,
                                                                      uint32_t currentN, uint32_t buffId,
                                                                      LocalTensor<P> &sumBuf)
{
    // 2. 加载hInGrad并进行向量计算
    ProcessV2ChunkLoadHInGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, currentN, bsStart);

    // 3. 加载x和gamma，计算xRs
    ProcessV2ChunkLoadXAndGamma(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, buffId);

    // 4. 融合计算xRsGrad并处理xRsGradMm
    ProcessV2ChunkProcessXRsGradMm(bsStart, bsChunkStart, copySizeND, currentChunkSize, buffId, sumBuf);

    // 5. 融合处理gradXPost并输出grad_x
    ProcessV2ChunkOutputXGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, chunkNDSize);
}

// 子函数1: 加载inv_rms和inv_rms_grad
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkLoadInvRms(uint64_t bsChunkStart,
                                                                            uint32_t currentChunkSize)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = currentChunkSize * sizeof(P);
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;

    vecInQueue0_.AllocTensor<P>(invRmsInBuf_);
    DataCopyPad(invRmsInBuf_, invRmsGm_[bsChunkStart], bsCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(invRmsInBuf_);
    vecInQueue0_.DeQue<P>(invRmsInBuf_);

    invRmsGradBuf_ = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(invRmsGradBuf_, workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(bsChunkStart)], bsCopyParams,
                dataCopyPadParams);
    vecInQueue1_.EnQue(invRmsGradBuf_);
    LocalTensor<P> invRmsGradUb = vecInQueue1_.DeQue<P>();

    AIV21Process(invRmsInBuf_, invRmsUb_, invRmsGradUb, currentChunkSize);
    vecInQueue0_.FreeTensor(invRmsInBuf_);
    vecInQueue1_.FreeTensor(invRmsGradUb);
}

// 子函数2: 加载hInGrad并进行向量计算
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkLoadHInGrad(uint32_t offsetND, uint32_t copySizeND,
                                                                             uint64_t bsChunkStart,
                                                                             uint32_t currentChunkSize,
                                                                             uint32_t currentN, uint64_t bsStart)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    DataCopyParams blockCopyParams;
    blockCopyParams.blockCount = currentChunkSize;
    blockCopyParams.blockLen = copySizeND * sizeof(T);
    blockCopyParams.srcStride = (D_ - copySizeND) * sizeof(T);
    blockCopyParams.dstStride = 0;
    vecInQueue0_.AllocTensor<T>(gradHInInBuf_);
    DataCopyPad(gradHInInBuf_, gradHInGm_[bsChunkStart * D_ + (offsetND % D_)], blockCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHInInBuf_);
    vecInQueue0_.DeQue<T>(gradHInInBuf_);

    __ubuf__ T *gradHInInBufAddr = (__ubuf__ T *)gradHInInBuf_.GetPhyAddr();
    __ubuf__ P *hPreUbAddr = (__ubuf__ P *)hPreUb_.GetPhyAddr();
    __ubuf__ P *xGradVec3BufAddr = (__ubuf__ P *)xGradVec3Buf_.GetPhyAddr();
    VFDoV2HInMulHPre(xGradVec3BufAddr, gradHInInBufAddr, hPreUbAddr, currentChunkSize, copySizeND, currentN,
                     (bsChunkStart - bsStart));
    vecInQueue0_.FreeTensor(gradHInInBuf_);
}

// 子函数3: 加载x和gamma，计算xRs
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkLoadXAndGamma(uint32_t offsetND, uint32_t copySizeND,
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
    // 先提升为64位再乘，避免大T乘ND超过2^32个元素时地址回绕。
    DataCopyPad(bf16InputBuf_, xGm_[bsChunkStart * nD_ + offsetND], xCopyParams, xCopyPadParams);
    vecInQueue1_.EnQue(bf16InputBuf_);
    bf16InputBuf_ = vecInQueue1_.DeQue<T>();

    // 计算xRs和gamma
    __ubuf__ T *bf16InputAddr = (__ubuf__ T *)bf16InputBuf_.GetPhyAddr();
    __ubuf__ P *xRsFp32Addr = (__ubuf__ P *)xRsFp32Buf_.GetPhyAddr();
    gammaOutUb_ = vecOutQueue1_.AllocTensor<P>();
    __ubuf__ P *gammaOutAddr = (__ubuf__ P *)gammaOutUb_.GetPhyAddr();
    if (withGamma_) {
        __ubuf__ P *gammaInAddr = (__ubuf__ P *)gammaIn_.GetPhyAddr();
        VFDoV2XCastAndMulGamma<true>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, gammaInAddr, currentChunkSize,
                                     copySizeND);
    } else {
        VFDoV2XCastAndMulGamma<false>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, nullptr, currentChunkSize, copySizeND);
    }
    vecInQueue1_.FreeTensor(bf16InputBuf_);
    vecOutQueue1_.EnQue(gammaOutUb_);
    gammaOutUb_ = vecOutQueue1_.DeQue<P>();

    // 输出gamma到workspace
    DataCopyExtParams fp32ExtCopyParams;
    fp32ExtCopyParams.blockCount = currentChunkSize;
    fp32ExtCopyParams.blockLen = copySizeND * sizeof(P);
    fp32ExtCopyParams.srcStride = 0;
    fp32ExtCopyParams.dstStride = (ND_BLOCK_SIZE - copySizeND) * sizeof(P);

    uint64_t offset = workspaceBuf_.GetXRsOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
    uint32_t vecIdx = blockIdx_ % NUMBER_TWO;
    offset += (bsChunkStart - bsStart) * ND_BLOCK_SIZE;
    offset += vecIdx * (ND_BLOCK_SIZE / 2);

    DataCopyPad(workSpaceGm_[offset], gammaOutUb_, fp32ExtCopyParams);
    vecOutQueue1_.FreeTensor(gammaOutUb_);
}

// 子函数4: 融合计算xRsGrad并处理xRsGradMm
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkProcessXRsGradMm(
    uint64_t bsStart, uint64_t bsChunkStart, uint32_t copySizeND, uint32_t currentChunkSize, uint32_t buffId,
    LocalTensor<P> &sumBuf)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    uint32_t vecIdx = blockIdx_ % NUMBER_TWO;
    uint64_t offsetXRsGradMm = workspaceBuf_.GetXRsGradOffset(GetBlockIdx() / VEC_DEAL_VECIDX_DIV, buffId);
    offsetXRsGradMm += (bsChunkStart - bsStart) * ND_BLOCK_SIZE + vecIdx * (ND_BLOCK_SIZE / 2);

    vecInQueue0_.AllocTensor<P>(xRsGradMmInBuf_);
    DataCopyParams blockCopyParams;
    blockCopyParams.blockCount = currentChunkSize;
    blockCopyParams.blockLen = copySizeND * sizeof(P);
    blockCopyParams.srcStride = (ND_BLOCK_SIZE - copySizeND) * sizeof(P);
    blockCopyParams.dstStride = 0;
    DataCopyPad(xRsGradMmInBuf_, workSpaceGm_[offsetXRsGradMm], blockCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(xRsGradMmInBuf_);
    vecInQueue0_.DeQue<P>(xRsGradMmInBuf_);

    __ubuf__ P *xRsGradMmInBufAddr = (__ubuf__ P *)xRsGradMmInBuf_.GetPhyAddr();
    __ubuf__ P *xGradVec3Addr = (__ubuf__ P *)xGradVec3Buf_.GetPhyAddr();
    __ubuf__ P *xRsFp32Addr = (__ubuf__ P *)xRsFp32Buf_.GetPhyAddr();
    __ubuf__ P *invRmsAddr = (__ubuf__ P *)invRmsUb_.GetPhyAddr() + (bsChunkStart - bsStart);
    if (withGamma_) {
        __ubuf__ P *gammaInAddr = (__ubuf__ P *)gammaIn_.GetPhyAddr();
        VFDoV2GammaMulXRsGradMm<true>(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, gammaInAddr,
                                      currentChunkSize, copySizeND);
    } else {
        VFDoV2GammaMulXRsGradMm<false>(xGradVec3Addr, xRsGradMmInBufAddr, xRsFp32Addr, invRmsAddr, nullptr,
                                       currentChunkSize, copySizeND);
    }

    uint32_t srcReduceShape[] = {currentChunkSize, copySizeND};
    if (withGamma_) {
        LocalTensor<P> reduceSumBuf = xRsFp32Buf_;
        PipeBarrier<PIPE_V>();
        if (bsChunkStart == 0) {
            ReduceSum<P, Pattern::Reduce::RA, true>(sumBuf, xRsGradMmInBuf_, srcReduceShape, true);
        } else {
            ReduceSum<P, Pattern::Reduce::RA, true>(reduceSumBuf, xRsGradMmInBuf_, srcReduceShape, true);
            PipeBarrier<PIPE_V>();
            Add(sumBuf, sumBuf, reduceSumBuf, copySizeND);
        }
        PipeBarrier<PIPE_V>();
    }
    vecInQueue0_.FreeTensor(xRsGradMmInBuf_);

    PipeBarrier<PIPE_V>();
}

template <class T, class P>
template <bool hasGradXPost>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2AddGradXPostAndCast(__ubuf__ T *outputAddr,
                                                                             __ubuf__ P *xGradAddr,
                                                                             __ubuf__ T *gradXPostAddr,
                                                                             uint32_t totalElem)
{
    uint16_t vfLoopCnt = CeilDiv(totalElem, static_cast<uint32_t>(eleNumPerVf_));
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xGradReg, gradXPostFp32Reg;
        Reg::RegTensor<T> gradXPostReg, outputReg;
        static constexpr Reg::CastTrait castFp32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        uint32_t len = totalElem;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            uint32_t offset = vfBlockIdx * eleNumPerVf_;
            Reg::LoadAlign(xGradReg, xGradAddr + offset);
            if constexpr (hasGradXPost) {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradXPostReg, gradXPostAddr + offset);
                Reg::Cast<P, T, ctHalf2Fp32Zero>(gradXPostFp32Reg, gradXPostReg, mask);
                Reg::Add(xGradReg, xGradReg, gradXPostFp32Reg, mask);
            }
            Reg::Cast<T, P, castFp32ToB16>(outputReg, xGradReg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + offset, outputReg, mask);
        }
    }
}

// 子函数6: 融合处理gradXPost并输出grad_x
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkOutputXGrad(uint32_t offsetND, uint32_t copySizeND,
                                                                             uint64_t bsChunkStart,
                                                                             uint32_t currentChunkSize,
                                                                             uint32_t chunkNDSize)
{
    bf16OutputBuf_ = vecOutQueue1_.AllocTensor<T>();
    __ubuf__ T *outputAddr = (__ubuf__ T *)bf16OutputBuf_.GetPhyAddr();
    __ubuf__ P *xGradAddr = (__ubuf__ P *)xGradVec3Buf_.GetPhyAddr();
    if (withGradXPost_) {
        DataCopyExtParams gradXPostCopyParams;
        DataCopyPadExtParams<T> gradXPostPadParams;
        gradXPostCopyParams.blockCount = currentChunkSize;
        gradXPostCopyParams.blockLen = copySizeND * sizeof(T);
        gradXPostCopyParams.srcStride = (nD_ - copySizeND) * sizeof(T);
        gradXPostCopyParams.dstStride = 0;
        gradXPostBuf_ = vecInQueue1_.AllocTensor<T>();
        DataCopyPad(gradXPostBuf_, gradXPostOptionalGm_[bsChunkStart * nD_ + offsetND], gradXPostCopyParams,
                    gradXPostPadParams);
        vecInQueue1_.EnQue(gradXPostBuf_);
        gradXPostBuf_ = vecInQueue1_.DeQue<T>();
        __ubuf__ T *gradXPostAddr = (__ubuf__ T *)gradXPostBuf_.GetPhyAddr();
        VFDoV2AddGradXPostAndCast<true>(outputAddr, xGradAddr, gradXPostAddr, chunkNDSize);
    } else {
        VFDoV2AddGradXPostAndCast<false>(outputAddr, xGradAddr, nullptr, chunkNDSize);
    }
    PipeBarrier<PIPE_V>();
    if (withGradXPost_) {
        vecInQueue1_.FreeTensor(gradXPostBuf_);
    }
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::AIV21Process(LocalTensor<P> &invRmsInBuf, LocalTensor<P> &invRmsUb,
                                                                LocalTensor<P> &invRmsGradUb, uint32_t currentChunkSize)
{
    __ubuf__ P *invRmsInBufAddr = (__ubuf__ P *)invRmsInBuf.GetPhyAddr();
    __ubuf__ P *invRmsUbAddr = (__ubuf__ P *)invRmsUb.GetPhyAddr();
    __ubuf__ P *invRmsGradUbAddr = (__ubuf__ P *)invRmsGradUb.GetPhyAddr();

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
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV3()
{
    ProcessV3AlphaGrad();
    ProcessV3BiasGrad();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV3AlphaGrad()
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
    VFDoV3ProcessAlphaGrad((__ubuf__ P *)alphaGradOutLocal.GetPhyAddr(), (__ubuf__ P *)alphaGradInLocal.GetPhyAddr());
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    alphaGradOutLocal.SetValue(1, alphaGradOutLocal.GetValue(ALPHA_GRAD_SHAPE_2_OFFSET));
    alphaGradOutLocal.SetValue(2, alphaGradOutLocal.GetValue(ALPHA_GRAD_SHAPE_3_OFFSET));
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV3BiasGrad()
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV3ProcessAlphaGrad(__ubuf__ P *alphaGradOut,
                                                                          __ubuf__ P *alphaGradIn)
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
        Reg::StoreAlign(alphaGradOut, sumReg, mask);
    }
}

} // namespace MhcPreBackward

#endif // __mhc_pre_backward_KERNEL_H_

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
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_backward_utils.h"

namespace MhcPreBackward {

using namespace matmul;
using namespace AscendC;
using namespace MhcPreBackwardUtils;

template <class T, class P>
class MhcPreBackwardKernel {
public:
    __aicore__ inline MhcPreBackwardKernel(MT_C0 &matmul0, MT_C1 &matmul1)
        : mm0(matmul0),
          mm1(matmul1)
    {}
    // 入口函数
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();

    // 初始化函数
    __aicore__ inline void InitGlobalBuffersAndTiling(InitParams initParams);
    __aicore__ inline void InitUBAndAIVBuffers(InitParams initParams);
    __aicore__ inline void InitStage2AndDataCopy();

    // V0 流程 (向量计算)
    __aicore__ inline void ProcessV0Main();
    __aicore__ inline void ProcessV0MainLoop(uint32_t vecRunTimes, float alphaPre, float alphaPost, float alphaComb,
                                             LocalTensor<P> &sumBuf);
    __aicore__ inline void ProcessV0MainAlphaReduce(LocalTensor<P> &sumBuf, uint32_t coreId);
    __aicore__ inline void PreProcessV0(LocalTensor<P> &hPreGradBuf, uint32_t runBSStart, uint32_t runBSEnd);
    __aicore__ inline void AllocV0V1Buffers(uint32_t runBSStart, uint32_t runBSEnd, V0V1Buffers<P> &buffers);
    __aicore__ inline void ProcessV0(uint32_t runBSStart, uint32_t runBSEnd, V0V1Buffers<P> &buffers, float alphaPre,
                                     float alphaPost, float alphaComb);
    __aicore__ inline void ProcessV1(uint32_t runBSStart, uint32_t runBSEnd, V0V1Buffers<P> &buffers,
                                     uint32_t vecRuntimesId, LocalTensor<P> &sumBuf);
    __aicore__ inline void AIV02Process(V0V1Buffers<P> &buffers, LocalTensor<P> &hMixGradBuf, uint64_t currentDealBsNum,
                                        float alphaPre, float alphaPost, float alphaComb);
    __aicore__ inline void AIV21Process(LocalTensor<P> &invRmsInBuf, LocalTensor<P> &invRmsUb,
                                        LocalTensor<P> &invRmsGradUb, uint32_t currentChunkSize);
    __aicore__ inline void AIV22Process(LocalTensor<P> &invRmsUb, LocalTensor<P> &xRsFp32Buf,
                                        LocalTensor<P> &xRsGradInvUb, uint32_t currentChunkSize, uint32_t copySizeND);

    // C0/C1 流程 (矩阵计算)
    __aicore__ inline void ProcessC0C1Pipeline();
    __aicore__ inline void ProcessC0(uint32_t offsetND, uint32_t currentNDBlock, uint32_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);
    __aicore__ inline void ProcessC1(uint32_t offsetND, uint32_t currentNDBlock, uint32_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);

    // V2 流程 (向量计算)
    __aicore__ inline void ProcessV2Pipeline();
    __aicore__ inline void ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint32_t bsStart, uint32_t bsEnd,
                                     LocalTensor<P> &sumBuf, uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND, uint32_t bsStart,
                                              uint32_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize,
                                              uint32_t currentN, uint32_t buffId, LocalTensor<P> &sumBuf);
    __aicore__ inline void ProcessV2ChunkLoadInvRms(uint32_t bsChunkStart, uint32_t currentChunkSize);
    __aicore__ inline void ProcessV2ChunkLoadHInGrad(uint32_t offsetND, uint32_t copySizeND, uint32_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t currentN, uint32_t bsStart);
    __aicore__ inline void ProcessV2ChunkLoadXAndGamma(uint32_t offsetND, uint32_t copySizeND, uint32_t bsStart,
                                                       uint32_t bsChunkStart, uint32_t currentChunkSize,
                                                       uint32_t buffId);
    __aicore__ inline void ProcessV2ChunkComputeXRsGrad(uint32_t copySizeND, uint32_t currentChunkSize,
                                                        uint32_t chunkNDSize);
    __aicore__ inline void ProcessV2ChunkProcessXRsGradMm(uint32_t bsStart, uint32_t bsChunkStart, uint32_t copySizeND,
                                                          uint32_t currentChunkSize, uint32_t chunkNDSize,
                                                          uint32_t buffId, LocalTensor<P> &sumBuf);
    __aicore__ inline void ProcessV2ChunkProcessGradXPost(uint32_t offsetND, uint32_t bsChunkStart,
                                                          uint32_t currentChunkSize, uint32_t chunkNDSize,
                                                          uint32_t copySizeND);
    __aicore__ inline void ProcessV2ChunkOutputXGrad(uint32_t offsetND, uint32_t copySizeND, uint32_t bsChunkStart,
                                                     uint32_t currentChunkSize, uint32_t chunkNDSize);

    // V3 流程 (向量计算)
    __aicore__ inline void ProcessV3();
    __aicore__ inline void ProcessV3AlphaGrad();
    __aicore__ inline void ProcessV3BiasGrad();

    // 向量化计算函数 (VFDo*)
    __aicore__ inline void VFDoPreProcessV0(__ubuf__ P *hPreGradAddr, __ubuf__ T *xB16InAddr,
                                            __ubuf__ T *gradHInB16InAddr, uint16_t curLenN, uint32_t lenD, uint32_t i,
                                            uint32_t j, uint32_t runBSStart);
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

    template <bool hasGamma, bool isFirstChunk>
    __aicore__ inline void VFDoV2XCastAndMulGamma(__ubuf__ P *xRsFp32Addr, __ubuf__ P *gammaOutAddr,
                                                  __ubuf__ T *bf16InputAddr, __ubuf__ P *gammaSrcAddr,
                                                  __ubuf__ P *gammaInAddr, uint32_t currentChunkSize,
                                                  uint32_t copySizeND);

    template <bool hasGamma>
    __aicore__ inline void VFDoV2GammaMulXRsGradMm(__ubuf__ P *xRsGradUbAddr, __ubuf__ P *xRsGradMmAddr,
                                                   __ubuf__ P *gammaInAddr, uint32_t currentChunkSize,
                                                   uint32_t copySizeND);

    __aicore__ inline void VFDoV3ProcessAlphaGrad(__ubuf__ P *alphaGradOut, __ubuf__ P *alphaGradIn);

private:
    MT_C0 &mm0;
    MT_C1 &mm1;
    uint32_t coreNum_;
    uint32_t totalLength_;
    uint32_t vecDealBSPeCore_;
    uint32_t vecCoreNum_;
    uint32_t dealStartBS_;
    uint32_t dealEndBS_;
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
    LocalTensor<P> gammaUb_;
    LocalTensor<P> gammaOutUb_;
    LocalTensor<P> xRsGradMmInBuf_;
    LocalTensor<T> bf16OutputBuf_;
    LocalTensor<T> gradXPostBuf_;

    // ProcessV2 chunk循环的buffer (由ProcessV2分配)
    LocalTensor<P> gammaIn_;
    LocalTensor<P> xRsFp32Buf_;
    LocalTensor<P> xGradVec3Buf_;
    LocalTensor<P> xRsGradMmUb_;
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
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
    gradHInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.grad_h_in));
    gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_h_post));
    gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.grad_h_res));
    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
    hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));
    hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
    xGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.grad_x));
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
    nD_ = static_cast<uint32_t>(tiling_->nD);
    D_ = static_cast<uint32_t>(tiling_->D);
    N_ = static_cast<uint32_t>(tiling_->N);
    fusionSize_ = static_cast<uint32_t>(tiling_->fusionSize);
    coreNum_ = tiling_->coreNum;
    totalLength_ = static_cast<uint32_t>(tiling_->totalLength);
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
    if ASCEND_IS_AIV {
        pipe_ = initParams.tPipeIn;
        pipe_->InitBuffer(vecInQueue0_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueue1_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecInQueue1_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(vecOutQueueSmall_, 1, SMALL_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);
        globalUbOffset_ = 0; // V0V1部分开始分配fp32TBuf_
        globalUbOffset_ += hFusionBufLen_ * sizeof(P);
        vecDealBSPeCore_ = totalLength_ / vecCoreNum_;
        if (vecDealBSPeCore_ == 0) {
            vecDealBSPeCore_ = totalLength_;
            usedVecCoreNum_ = 1;
        } else {
            vecDealBSPeCore_ = MhcPreBackwardUtils::CeilAlign(vecDealBSPeCore_, CEIL_ALIGN_16);
            usedVecCoreNum_ = CeilDiv(totalLength_, vecDealBSPeCore_) > vecCoreNum_ ?
                                  vecCoreNum_ :
                                  CeilDiv(totalLength_, vecDealBSPeCore_);
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
        // 执行矩阵乘法 C0 -> C1
        ProcessC0C1Pipeline();
    }

    if ASCEND_IS_AIV {
        // 执行向量计算V2
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
                                                                    uint32_t lenD, uint32_t i, uint32_t j,
                                                                    uint32_t runBSStart)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> sumReg;
        uint32_t offset = static_cast<uint32_t>((i - runBSStart) * N_ + j);
        for (uint16_t offsetN = 0; offsetN < curLenN; offsetN++) { // x [curLenN, lenD],  hinGrad [1, lenD]
            MicroAPI::Duplicate(sumReg, 0.0f);
            uint32_t curLenD = lenD;
            // x[1, lenD]  hinGrad [1, lenD]   res[1, lenD]
            MicroAPI::RegTensor<T> xInB16Reg, gradHInB16InReg;
            MicroAPI::RegTensor<P> xFp32Reg, gradHInFp32Reg, mulReg, tmpSumPerVfReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < dLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);

                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xInB16Reg, xB16InAddr + offsetN * lenD + vfBlockIdx * eleNumPerVf);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    gradHInB16InReg, gradHInB16InAddr + vfBlockIdx * eleNumPerVf);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);

                MicroAPI::Mul(mulReg, xFp32Reg, gradHInFp32Reg, mask); // mulReg[1,lenD] 0-63
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(tmpSumPerVfReg, mulReg,
                                                            mask); // 每个vfBlockIdx循环的临时求和，每64个元素的和
                MicroAPI::Add(sumReg, sumReg, tmpSumPerVfReg, mask); // D方向上的reduceSum
            }
            uint32_t bufIdx = offset + offsetN;
            MicroAPI::Store(hPreGradAddr + bufIdx, sumReg, 1);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::PreProcessV0(LocalTensor<P> &hPreGradBuf, uint32_t runBSStart,
                                                                uint32_t runBSEnd)
{
    uint32_t maxDLen = MAX_D_LEN;
    uint32_t queueCopySize = INOUT_QUEUE_SIZE / sizeof(T);
    uint32_t copySizeD = D_ <= maxDLen ? D_ : maxDLen;
    uint32_t lenN = static_cast<uint32_t>(queueCopySize / copySizeD > N_ ? N_ : queueCopySize / copySizeD);
    AscendC::LocalTensor<T> bf16InputBuf;
    DataCopyParams dataCopyParams;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    for (uint32_t i = runBSStart; i < runBSEnd; i++) {
        uint64_t inputOffset = static_cast<uint64_t>(i) * N_ * copySizeD;
        for (uint32_t j = 0; j < N_; j += lenN) {
            uint32_t curLenN = static_cast<uint32_t>(j + lenN > N_ ? N_ - j : lenN);
            dataCopyParams.blockLen = copySizeD * sizeof(T);
            dataCopyParams.blockCount = curLenN;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            vecInQueue0_.AllocTensor<T>(bf16InputBuf);
            DataCopyPad(bf16InputBuf, xGm_[inputOffset], dataCopyParams, dataCopyPadParams);
            vecInQueue0_.EnQue(bf16InputBuf);
            vecInQueue0_.DeQue<T>(bf16InputBuf);

            dataCopyParams.blockCount = 1;
            AscendC::LocalTensor<T> gradHInInput = vecInQueue1_.AllocTensor<T>();
            DataCopyPad(gradHInInput, gradHInGm_[i * copySizeD], dataCopyParams, dataCopyPadParams);
            vecInQueue1_.EnQue(gradHInInput);

            LocalTensor<T> gradHInInputBuf = vecInQueue1_.DeQue<T>();

            __ubuf__ T *xB16InAddr = (__ubuf__ T *)bf16InputBuf.GetPhyAddr();
            __ubuf__ T *gradHInB16InAddr = (__ubuf__ T *)gradHInInputBuf.GetPhyAddr();
            __ubuf__ P *hPreGradAddr = (__ubuf__ P *)hPreGradBuf.GetPhyAddr();
            VFDoPreProcessV0(hPreGradAddr, xB16InAddr, gradHInB16InAddr, curLenN, copySizeD, i, j, runBSStart);

            inputOffset += copySizeD * curLenN;
            vecInQueue0_.FreeTensor(bf16InputBuf);
            vecInQueue1_.FreeTensor(gradHInInputBuf);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0Main()
{
    if (blockIdx_ >= usedVecCoreNum_) {
        return;
    }
    uint32_t dealBsNum = dealEndBS_ - dealStartBS_;
    uint32_t vecRunTimes = CeilDiv(dealBsNum, vecDealChunk_);
    float alphaPre = alphaGm_.GetValue(0);
    float alphaPost = alphaGm_.GetValue(1);
    float alphaComb = alphaGm_.GetValue(2);
    LocalTensor<P> sumBuf = fp32TBuf_.GetWithOffset<P>(fusionSize_, globalUbOffset_);
    globalUbOffset_ += MhcPreBackwardUtils::CeilAlign(uint32_t(fusionSize_ * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));
    ProcessV0MainLoop(vecRunTimes, alphaPre, alphaPost, alphaComb, sumBuf);
    ProcessV0MainAlphaReduce(sumBuf, GetBlockIdx());
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0MainLoop(uint32_t vecRunTimes, float alphaPre,
                                                                     float alphaPost, float alphaComb,
                                                                     LocalTensor<P> &sumBuf)
{
    for (uint32_t cIdx = 0; cIdx < vecRunTimes; cIdx++) {
        uint32_t runBSStart = cIdx * vecDealChunk_ + dealStartBS_;
        uint32_t runBSEnd = runBSStart + vecDealChunk_;
        if (runBSEnd > dealEndBS_) {
            runBSEnd = dealEndBS_;
        }
        uint32_t currentDealBsNum = runBSEnd - runBSStart;
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
    uint32_t srcShape1[] = {1, static_cast<uint32_t>(N_)};
    uint32_t srcShape2[] = {1, static_cast<uint32_t>(2 * N_)};
    uint32_t srcShape3[] = {1, static_cast<uint32_t>(fusionSize_)};
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::AllocV0V1Buffers(uint32_t runBSStart, uint32_t runBSEnd,
                                                                    V0V1Buffers<P> &buffers)
{
    buffers.stepLength = (runBSEnd - runBSStart) * N_;
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
        MicroAPI::RegTensor<P> hPreReg, hInGradReg;
        MicroAPI::RegTensor<P> s1Reg, s2Reg, mulReg, resReg;
        MicroAPI::RegTensor<P> oneReg;
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(oneReg, 1.0f, mask);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = MicroAPI::UpdateMask<P>(curElemCnt);
            MicroAPI::LoadAlign(hPreReg, hPreAddr + vfBlockIdx * eleNumPerVf);
            MicroAPI::LoadAlign(hInGradReg, gradHInAddr + vfBlockIdx * eleNumPerVf);
            MicroAPI::Adds(s1Reg, hPreReg, -hcEps_, mask);
            MicroAPI::Sub(s2Reg, oneReg, s1Reg, mask);
            MicroAPI::Mul(mulReg, s1Reg, s2Reg, mask);
            MicroAPI::Mul(resReg, mulReg, hInGradReg, mask);

            MicroAPI::StoreAlign(hPreBufS1Addr + vfBlockIdx * eleNumPerVf, resReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0(uint32_t runBSStart, uint32_t runBSEnd,
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

    fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(fp32InputBuf, hPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    AscendC::LocalTensor<P> gradHPost;
    vecInQueue0_.AllocTensor<P>(gradHPost);
    DataCopyPad(gradHPost, gradHPostGm_[runBSStart * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue0_.EnQue(gradHPost);

    LocalTensor<P> hPost = vecInQueue1_.DeQue<P>();
    vecInQueue0_.DeQue<P>(gradHPost);
    VFDoV0ProcessGradHPost((__ubuf__ P *)hPost.GetPhyAddr(), (__ubuf__ P *)gradHPost.GetPhyAddr(),
                           (__ubuf__ P *)buffers.hPostBufS1.GetPhyAddr(), buffers.stepLength);
    vecInQueue1_.FreeTensor(hPost);
    vecInQueue0_.FreeTensor(gradHPost);

    fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    dataCopyParams.blockLen = buffers.stepLength * N_ * sizeof(P);
    DataCopyPad(fp32InputBuf, gradHResGm_[runBSStart * N_ * N_], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    LocalTensor<P> gradHResBuf = vecInQueue1_.DeQue<P>();

    Muls(buffers.hResBufS1, gradHResBuf, ONE, buffers.stepLength * N_);
    PipeBarrier<PIPE_V>();
    vecInQueue1_.FreeTensor(gradHResBuf);

    uint32_t currentDealBsNum = runBSEnd - runBSStart;
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
        MicroAPI::MaskReg mask;
        MicroAPI::RegTensor<P> gradHPostReg, tmpReg;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = MicroAPI::UpdateMask<P>(curLen);
            MicroAPI::LoadAlign(gradHPostReg, hPostIn + vfBlockIdx * eleNumPerVf_);
            MicroAPI::Muls(tmpReg, gradHPostReg, NEG_HALF, mask);
            MicroAPI::Adds(tmpReg, tmpReg, ONE, mask);
            MicroAPI::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            MicroAPI::LoadAlign(tmpReg, gradHPostIn + vfBlockIdx * eleNumPerVf_);
            MicroAPI::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            MicroAPI::StoreAlign(gradHPostOut + vfBlockIdx * eleNumPerVf_, gradHPostReg, mask);
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

    uint32_t blockLenPre = static_cast<uint32_t>(N_);
    uint32_t blockLenPost = static_cast<uint32_t>(N_);
    uint32_t blockLenComb = static_cast<uint32_t>(N_ * N_);
    uint32_t totalBlockLen = blockLenPre + blockLenPost + blockLenComb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> gatherReg;
        MicroAPI::RegTensor<P> hMixGradReg;
        MicroAPI::RegTensor<P> alphaPreReg;
        MicroAPI::RegTensor<P> alphaPostReg;
        MicroAPI::RegTensor<P> alphaCombReg;
        MicroAPI::MaskReg maskPre = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskPost = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskComb = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<P>(alphaPreReg, alphaPre, maskPre);
        MicroAPI::Duplicate<P>(alphaPostReg, alphaPost, maskPost);
        MicroAPI::Duplicate<P>(alphaCombReg, alphaComb, maskComb);

        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(currentDealBsNum); bsIdx++) {
            uint32_t curLenPre = blockLenPre;
            uint32_t curLenPost = blockLenPost;
            uint32_t curLenComb = blockLenComb;
            uint32_t bsFusionOffset = bsIdx * totalBlockLen;

            uint32_t srcOffset = bsIdx * blockLenPre;
            uint32_t dstOffset = bsFusionOffset;
            maskPre = MicroAPI::UpdateMask<P>(curLenPre);
            MicroAPI::Load<P>(gatherReg, hPreBufAddr + srcOffset);
            MicroAPI::Store<P>(gatherFusionOutBufAddr + dstOffset, gatherReg);
            MicroAPI::Mul(hMixGradReg, gatherReg, alphaPreReg, maskPre);
            MicroAPI::Store<P>(hMixGradBufAddr + dstOffset, hMixGradReg);

            uint32_t srcOffset1 = bsIdx * blockLenPre;
            uint32_t dstOffset1 = bsFusionOffset;
            maskPost = MicroAPI::UpdateMask<P>(curLenPost);
            MicroAPI::Load<P>(gatherReg, hPostBufAddr + srcOffset1);
            MicroAPI::Store<P>(gatherFusionOutBufAddr + dstOffset1 + blockLenPre, gatherReg);
            MicroAPI::Mul(hMixGradReg, gatherReg, alphaPostReg, maskPost);
            MicroAPI::Store<P>(hMixGradBufAddr + dstOffset1 + blockLenPre, hMixGradReg);

            uint32_t srcOffset2 = bsIdx * blockLenComb;
            uint32_t dstOffset2 = bsFusionOffset + blockLenPre + blockLenPost;
            maskComb = MicroAPI::UpdateMask<P>(curLenComb);
            MicroAPI::Load<P>(gatherReg, hResBufAddr + srcOffset2);
            MicroAPI::Store<P>(gatherFusionOutBufAddr + dstOffset2, gatherReg);
            MicroAPI::Mul(hMixGradReg, gatherReg, alphaCombReg, maskComb);
            MicroAPI::Store<P>(hMixGradBufAddr + dstOffset2, hMixGradReg);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV1(uint32_t runBSStart, uint32_t runBSEnd,
                                                             V0V1Buffers<P> &buffers, uint32_t vecRuntimesId,
                                                             LocalTensor<P> &sumBuf)
{
    uint32_t curBSSize = runBSEnd - runBSStart;
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
    PipeBarrier<PIPE_V>();
    vecOutQueue1_.FreeTensor(BiasOutBuf);

    auto hMixGradBuf = buffers.calcTmpBuf;
    fp32OutBuf = vecOutQueue1_.AllocTensor<P>();
    dataCopyParams.blockLen = curBSSize * sizeof(P);
    AscendC::LocalTensor<P> fp32InputBuf = vecInQueue1_.AllocTensor<P>();
    DataCopyPad(fp32InputBuf, invRmsGm_[runBSStart], dataCopyParams, dataCopyPadParams);
    vecInQueue1_.EnQue(fp32InputBuf);
    LocalTensor<P> invRmsBufLocal = vecInQueue1_.DeQue<P>();

    const uint32_t xRowSumBroadCastDst[2] = {curBSSize, static_cast<uint32_t>(fusionSize_)};
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

    uint32_t shape[] = {curBSSize, static_cast<uint32_t>(fusionSize_)};
    // inv rms grad
    ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(invRmsGradUb, hMixGradBuf, buffers.brcbTmpBuf, shape, true);
    vecOutQueueSmall_.EnQue(invRmsGradUb);
    LocalTensor<P> invRmsGradUbtBuf = vecOutQueueSmall_.DeQue<P>();

    dataCopyParams.blockLen = curBSSize * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(runBSStart)], invRmsGradUbtBuf, dataCopyParams);
    // alpha grad (inv rms 梯度计算完成，复用hMixGradBuf)
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
    uint32_t totalElem = static_cast<uint32_t>(curBSSize * fusionSize_);
    uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
    uint32_t curElemCnt = totalElem;

    __VEC_SCOPE__
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);

            MicroAPI::RegTensor<P> h1GradReg, invRmsReg, hMixReg, gatherReg;
            MicroAPI::RegTensor<P> hMixGradReg, invRmsGradReg;
            MicroAPI::RegTensor<P> hMul1Reg, hMul2Reg, hMul3Reg;

            MicroAPI::LoadAlign(h1GradReg, hMixGradBuf + elemOffset);
            MicroAPI::LoadAlign(invRmsReg, invRmsIn + elemOffset);

            MicroAPI::Mul(hMixGradReg, h1GradReg, invRmsReg, mask);
            MicroAPI::StoreAlign(gradHMixOut + elemOffset, hMixGradReg, mask);

            MicroAPI::LoadAlign(hMixReg, hMixIn + elemOffset);
            MicroAPI::Mul(hMul1Reg, h1GradReg, hMixReg, mask);
            MicroAPI::StoreAlign(hMixGradBuf + elemOffset, hMul1Reg, mask);

            MicroAPI::LoadAlign(gatherReg, gatherIn + elemOffset);
            MicroAPI::Mul(hMul2Reg, invRmsReg, hMixReg, mask);
            MicroAPI::Mul(hMul3Reg, hMul2Reg, gatherReg, mask);
            MicroAPI::StoreAlign(gradAlphaOut + elemOffset, hMul3Reg, mask);
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
        MicroAPI::RegTensor<P> sumReg1, sumReg2;
        MicroAPI::Duplicate(sumReg1, 0);
        MicroAPI::Duplicate(sumReg2, 0);
        uint32_t dealMask1 = eleNumPerVf_;
        uint32_t dealMask2 = static_cast<uint32_t>(fusionSize_ - eleNumPerVf_);
        MicroAPI::MaskReg mask1 = MicroAPI::UpdateMask<P>(dealMask1);
        MicroAPI::MaskReg mask2 = MicroAPI::UpdateMask<P>(dealMask2);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = static_cast<uint32_t>(bsIdx * fusionSize_);
            uint32_t elemOffset2 = static_cast<uint32_t>(bsIdx * fusionSize_ + eleNumPerVf_);
            MicroAPI::RegTensor<P> gatherReg1, gatherReg2;

            MicroAPI::LoadAlign(gatherReg1, gatherFusion + elemOffset1);
            MicroAPI::LoadAlign(gatherReg2, gatherFusion + elemOffset2);

            MicroAPI::Add(sumReg1, sumReg1, gatherReg1, mask1);
            MicroAPI::Add(sumReg2, sumReg2, gatherReg2, mask2);
        }
        MicroAPI::StoreAlign(outBufDst, sumReg1, mask1);
        MicroAPI::StoreAlign(outBufDst + eleNumPerVf_, sumReg2, mask2);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV1ProcessBiasGradForN4N6(__ubuf__ P *outBufDst,
                                                                                __ubuf__ P *gatherFusion,
                                                                                uint32_t curBSSize)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> sumReg;
        MicroAPI::Duplicate(sumReg, 0);
        uint32_t dealMask = static_cast<uint32_t>(fusionSize_);
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(dealMask);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = static_cast<uint32_t>(bsIdx * fusionSize_);
            MicroAPI::RegTensor<P> gatherReg;

            MicroAPI::LoadAlign(gatherReg, gatherFusion + elemOffset1);

            MicroAPI::Add(sumReg, sumReg, gatherReg, mask);
        }
        MicroAPI::StoreAlign(outBufDst, sumReg, mask);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC0C1Pipeline()
{
    if (GetBlockIdx() >= v1UsedCubeCoreNum_) {
        return;
    }

    uint32_t currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - dealStartND_);
    uint32_t currentBSBlock = Min(SINGLE_M, uint32_t(totalLength_));
    uint32_t buffId = 0;
    // 预执行第一个C0
    ProcessC0(dealStartND_, currentNDBlock, 0, currentBSBlock, buffId);

    for (uint32_t offsetND = dealStartND_; offsetND < dealEndND_; offsetND += ND_BLOCK_SIZE) {
        currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - offsetND);

        for (uint32_t offsetBS = 0; offsetBS < totalLength_; offsetBS += SINGLE_M) {
            currentBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - offsetBS));

            uint32_t nextOffsetBS = offsetBS + SINGLE_M;
            if (nextOffsetBS < totalLength_) { // BS还有未计算
                uint32_t nextBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - nextOffsetBS));
                ProcessC0(offsetND, currentNDBlock, nextOffsetBS, nextBSBlock, (buffId + 1));
            } else { // 预计算下一个nD块的C0
                uint32_t nextOffsetND = offsetND + ND_BLOCK_SIZE;
                if (nextOffsetND < dealEndND_) { // ND 还有未计算
                    uint32_t nextNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - nextOffsetND);
                    nextOffsetBS = 0;
                    uint32_t nextBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - nextOffsetBS));
                    ProcessC0(nextOffsetND, nextNDBlock, nextOffsetBS, nextBSBlock, (buffId + 1));
                }
            }

            ProcessC1(offsetND, currentNDBlock, offsetBS, currentBSBlock, buffId);
            buffId++;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC0(uint32_t offsetND, uint32_t currentNDBlock,
                                                             uint32_t offsetBS, uint32_t currentBSBlock,
                                                             uint32_t buffIdx)
{
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    // 设置矩阵形状
    mm0.SetOrgShape(totalLength_, nD_, fusionSize_);
    mm0.SetSingleShape(currentBSBlock, currentNDBlock, fusionSize_);
    // h_mix_grad @ phi
    mm0.SetTensorA(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)]); // [BS, N^2+2N]
    mm0.SetTensorB(phiGm_[offsetND], false);                                 // [N^2+2N, ND]
    // 计算
    uint64_t writeOffset = workspaceBuf_.GetXRsGradOffset(GetBlockIdx(), buffIdx);
    while (mm0.Iterate()) {
        mm0.GetTensorC(workSpaceGm_[writeOffset], 0, true); // 连续写
        writeOffset += MATMUL_WRITE_OFFSET_M * MATMUL_WRITE_OFFSET_N;
    }
    mm0.End();

    // 通知V2计算
    AscendC::CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_FIX>(CROSS_CORE_WAIT_FLAG_C0);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC1(uint32_t offsetND, uint32_t currentNDBlock,
                                                             uint32_t offsetBS, uint32_t currentBSBlock,
                                                             uint32_t buffIdx)
{
    AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C1);
    // 1C2V 对应操作两块 workspaceGm_
    buffIdx = buffIdx % VEC_CORE_VECIDX_MOD;
    // 设置矩阵形状
    uint64_t offsetXs = workspaceBuf_.GetXRsOffset(GetBlockIdx(), buffIdx);
    mm1.SetOrgShape(fusionSize_, currentNDBlock, totalLength_, currentBSBlock, nD_); // [M,N,Ka,Kb,orgKc]
    mm1.SetSingleShape(fusionSize_, currentNDBlock, currentBSBlock);                 // [sM, sN, sK]
    // h_mix_grad^T @ x_rs
    mm1.SetTensorA(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], true); // [BS, N^2+2N]
    mm1.SetTensorB(workSpaceGm_[offsetXs], false);                                 // [BS, NDBlock]
    // 计算
    bool enAtomic = (offsetBS != 0);
    mm1.IterateAll(gradPhiGm_[offsetND], enAtomic); // 非连续写
    mm1.End();
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

        // BS方向切分：与C0保持一致，按SINGLE_M切分
        LocalTensor<P> sumBuf = vecOutQueueSmall_.AllocTensor<P>();
        for (uint32_t offsetBS = 0; offsetBS < totalLength_; offsetBS += SINGLE_M) {
            uint32_t currentBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - offsetBS));
            // 等待C0完成计算
            AscendC::CrossCoreWaitFlag(CROSS_CORE_WAIT_FLAG_C0);

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
template <bool hasGamma, bool isFirstChunk>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2XCastAndMulGamma(
    __ubuf__ P *xRsFp32Addr, __ubuf__ P *gammaOutAddr, __ubuf__ T *bf16InputAddr, __ubuf__ P *gammaSrcAddr,
    __ubuf__ P *gammaInAddr, uint32_t currentChunkSize, uint32_t copySizeND)
{
    uint32_t eleNumPerVf = 64;
    uint16_t vfLoopCnt = CeilDiv(copySizeND, eleNumPerVf);
    __VEC_SCOPE__
    {
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            MicroAPI::RegTensor<T> xInB16Reg;
            MicroAPI::RegTensor<P> xFp32Reg, gammaReg, resultReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(lenND);
                uint32_t offset = tIdx * copySizeND + vfBlockIdx * eleNumPerVf;

                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInB16Reg, bf16InputAddr + offset);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                MicroAPI::StoreAlign(xRsFp32Addr + offset, xFp32Reg, mask); // 后续其他计算需要xFp32

                if constexpr (hasGamma) {
                    MicroAPI::LoadAlign(gammaReg, gammaSrcAddr + vfBlockIdx * eleNumPerVf);
                    if constexpr (isFirstChunk) {
                        MicroAPI::StoreAlign(gammaInAddr + offset, gammaReg, mask);
                    }

                    MicroAPI::Mul(resultReg, gammaReg, xFp32Reg, mask); // 逐元素乘：gamma * x
                    MicroAPI::StoreAlign(gammaOutAddr + offset, resultReg, mask);
                } else { // 没有gamma，原始逻辑是 xFp32Reg * 1.0，等价于不处理，这里直接store
                    MicroAPI::StoreAlign(gammaOutAddr + offset, xFp32Reg, mask);
                }
            }
        }
    }
}

template <class T, class P>
template <bool hasGamma>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2GammaMulXRsGradMm(__ubuf__ P *xRsGradUbAddr,
                                                                           __ubuf__ P *xRsGradMmAddr,
                                                                           __ubuf__ P *gammaInAddr,
                                                                           uint32_t currentChunkSize,
                                                                           uint32_t copySizeND)
{
    uint32_t eleNumPerVf = 64;
    uint16_t vfLoopCnt = CeilDiv(copySizeND, eleNumPerVf);
    __VEC_SCOPE__
    {
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            MicroAPI::RegTensor<P> gammaReg, xRsGradMmReg, resultReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(lenND);
                uint32_t offset = tIdx * copySizeND + vfBlockIdx * eleNumPerVf;
                MicroAPI::LoadAlign(xRsGradMmReg, xRsGradMmAddr + offset);

                if constexpr (hasGamma) {
                    MicroAPI::LoadAlign(gammaReg, gammaInAddr + vfBlockIdx * eleNumPerVf);

                    MicroAPI::Mul(resultReg, gammaReg, xRsGradMmReg, mask); // 逐元素乘：gamma * xRsGradMm
                    MicroAPI::StoreAlign(xRsGradUbAddr + offset, resultReg, mask);
                } else { // 没有gamma，原始逻辑是 xFp32Reg * 1.0，等价于不处理，这里直接store
                    MicroAPI::StoreAlign(xRsGradUbAddr + offset, xRsGradMmReg, mask);
                }
            }
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
    uint16_t vfLoopCnt = CeilDiv(copySizeND, static_cast<uint32_t>(eleNumPerVf_));
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> hInReg;
        MicroAPI::RegTensor<P> hInFP32Reg, hPreReg, hPreFullReg;
        MicroAPI::RegTensor<uint32_t> hPreIndexReg;
        MicroAPI::MaskReg mask, maskAll;
        maskAll = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
        uint32_t hPreOffset = static_cast<uint32_t>(bsGlobalOffset * N_ + currentN);
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            uint32_t hInOffset = tIdx * copySizeND;

            // Load hPre [1] -> [copySizeND]
            MicroAPI::Duplicate(hPreIndexReg, hPreOffset, maskAll);
            MicroAPI::Gather(hPreFullReg, hPreUbAddr, hPreIndexReg, maskAll);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
                mask = MicroAPI::UpdateMask<P>(lenND);
                // hIn B16 -> B32
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(hInReg, hInGradInBufAddr + hInOffset);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(hInFP32Reg, hInReg, mask);
                MicroAPI::Mul(hInFP32Reg, hInFP32Reg, hPreFullReg, mask);
                MicroAPI::StoreAlign(xGradVec3BufAddr + hInOffset, hInFP32Reg, mask);
                hInOffset += eleNumPerVf_;
            }
            hPreOffset += N_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint32_t bsStart,
                                                             uint32_t bsEnd, LocalTensor<P> &sumBuf, uint32_t buffId)
{
    // ==========================================================
    // 初始化
    // ==========================================================
    uint32_t currentN = static_cast<uint32_t>(offsetND / D_);
    uint32_t chunkSize = N_ > 4 ? (PROCESS_V2_CHUNK_SIZE / 2) : PROCESS_V2_CHUNK_SIZE;
    uint32_t currentBsSize = bsEnd - bsStart;
    buffId = buffId % VEC_CORE_VECIDX_MOD;

    // ==========================================================
    // Buffer分配 - 按最大CHUNK大小分配 (存到全局变量)
    // ==========================================================
    uint32_t ubOffset = globalUbOffset_; // sumBuf offset
    uint32_t maxChunkNDSize = chunkSize * copySizeND;
    if (withGamma_) {
        gammaIn_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
        ubOffset = MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)),
                                                  uint32_t(CEIL_ALIGN_DEFAULT));
    }

    xRsFp32Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    xGradVec3Buf_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    xRsGradMmUb_ = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    invRmsUb_ = fp32TBuf_.GetWithOffset<P>(currentBsSize, ubOffset);
    ubOffset =
        MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * sizeof(P)), uint32_t(CEIL_ALIGN_DEFAULT));

    hPreUb_ = fp32TBuf_.GetWithOffset<P>(currentBsSize * N_, ubOffset);
    ubOffset = MhcPreBackwardUtils::CeilAlign(uint32_t(ubOffset + currentBsSize * N_ * sizeof(P)),
                                              uint32_t(CEIL_ALIGN_DEFAULT));

    // ==========================================================
    // 参数初始化和hPre预加载
    // ==========================================================
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;

    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = currentBsSize * N_ * sizeof(P);
    bsCopyParams.srcStride = 0;
    bsCopyParams.dstStride = 0;
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    DataCopyPad(hPreUb_, hPreGm_[bsStart * N_], bsCopyParams, dataCopyPadParams);

    // ==========================================================
    // 主循环: 按chunk=64切分BS方向
    // ==========================================================
    for (uint32_t bsChunkStart = bsStart; bsChunkStart < bsEnd; bsChunkStart += chunkSize) {
        uint32_t bsChunkEnd = Min(bsChunkStart + chunkSize, bsEnd);
        uint32_t currentChunkSize = bsChunkEnd - bsChunkStart;
        uint32_t chunkNDSize = currentChunkSize * copySizeND;

        ProcessV2ChunkLoop(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, chunkNDSize, currentN, buffId,
                           sumBuf);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkLoop(uint32_t offsetND, uint32_t copySizeND,
                                                                      uint32_t bsStart, uint32_t bsChunkStart,
                                                                      uint32_t currentChunkSize, uint32_t chunkNDSize,
                                                                      uint32_t currentN, uint32_t buffId,
                                                                      LocalTensor<P> &sumBuf)
{
    // 1. 加载inv_rms和inv_rms_grad
    ProcessV2ChunkLoadInvRms(bsChunkStart, currentChunkSize);

    // 2. 加载hInGrad并进行向量计算
    ProcessV2ChunkLoadHInGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, currentN, bsStart);

    // 3. 加载x和gamma，计算xRs
    ProcessV2ChunkLoadXAndGamma(offsetND, copySizeND, bsStart, bsChunkStart, currentChunkSize, buffId);

    // 4. 计算xRsGrad
    ProcessV2ChunkComputeXRsGrad(copySizeND, currentChunkSize, chunkNDSize);

    // 5. 处理xRsGradMm
    ProcessV2ChunkProcessXRsGradMm(bsStart, bsChunkStart, copySizeND, currentChunkSize, chunkNDSize, buffId, sumBuf);

    // 6. 处理gradXPost
    ProcessV2ChunkProcessGradXPost(offsetND, bsChunkStart, currentChunkSize, chunkNDSize, copySizeND);

    // 7. 输出grad_x
    ProcessV2ChunkOutputXGrad(offsetND, copySizeND, bsChunkStart, currentChunkSize, chunkNDSize);
}

// 子函数1: 加载inv_rms和inv_rms_grad
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkLoadInvRms(uint32_t bsChunkStart,
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

    PipeBarrier<PIPE_MTE2>();
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
                                                                             uint32_t bsChunkStart,
                                                                             uint32_t currentChunkSize,
                                                                             uint32_t currentN, uint32_t bsStart)
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
                                                                               uint32_t bsStart, uint32_t bsChunkStart,
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
    DataCopyPad(bf16InputBuf_, xGm_[bsChunkStart * nD_ + offsetND], xCopyParams, xCopyPadParams);
    vecInQueue1_.EnQue(bf16InputBuf_);
    bf16InputBuf_ = vecInQueue1_.DeQue<T>();

    // 首次加载gamma
    if (withGamma_ && bsChunkStart == bsStart) {
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = false;
        vecInQueue0_.AllocTensor<P>(gammaUb_);
        DataCopyParams fp32BlockCopyParams;
        fp32BlockCopyParams.blockCount = 1;
        fp32BlockCopyParams.blockLen = copySizeND * sizeof(P);
        fp32BlockCopyParams.srcStride = 0;
        fp32BlockCopyParams.dstStride = 0;
        DataCopyPad(gammaUb_, gammaGm_[offsetND], fp32BlockCopyParams, dataCopyPadParams);
        vecInQueue0_.EnQue(gammaUb_);
        vecInQueue0_.DeQue<P>(gammaUb_);
    }

    // 计算xRs和gamma
    __ubuf__ T *bf16InputAddr = (__ubuf__ T *)bf16InputBuf_.GetPhyAddr();
    __ubuf__ P *xRsFp32Addr = (__ubuf__ P *)xRsFp32Buf_.GetPhyAddr();
    gammaOutUb_ = vecOutQueue1_.AllocTensor<P>();
    __ubuf__ P *gammaOutAddr = (__ubuf__ P *)gammaOutUb_.GetPhyAddr();
    if (withGamma_) {
        __ubuf__ P *gammaInAddr = (__ubuf__ P *)gammaIn_.GetPhyAddr();
        if (bsChunkStart == bsStart) {
            __ubuf__ P *gammaSrcAddr = (__ubuf__ P *)gammaUb_.GetPhyAddr();
            VFDoV2XCastAndMulGamma<true, true>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, gammaSrcAddr, gammaInAddr,
                                               currentChunkSize, copySizeND);
            vecInQueue0_.FreeTensor(gammaUb_);
        } else {
            VFDoV2XCastAndMulGamma<true, false>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, gammaInAddr, gammaInAddr,
                                                currentChunkSize, copySizeND);
        }
    } else {
        VFDoV2XCastAndMulGamma<false, false>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, nullptr, nullptr,
                                             currentChunkSize, copySizeND);
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
    uint32_t vecIdx = blockIdx_ % 2;
    offset += (bsChunkStart - bsStart) * ND_BLOCK_SIZE;
    offset += vecIdx * (ND_BLOCK_SIZE / 2);

    DataCopyPad(workSpaceGm_[offset], gammaOutUb_, fp32ExtCopyParams);
    vecOutQueue1_.FreeTensor(gammaOutUb_);
}

// 子函数4: 计算xRsGrad
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkComputeXRsGrad(uint32_t copySizeND,
                                                                                uint32_t currentChunkSize,
                                                                                uint32_t chunkNDSize)
{
    auto xRsGradInvUb = xRsGradMmUb_;
    AIV22Process(invRmsUb_, xRsFp32Buf_, xRsGradInvUb, currentChunkSize, copySizeND);
    Add(xGradVec3Buf_, xGradVec3Buf_, xRsGradInvUb, chunkNDSize);
    PipeBarrier<PIPE_V>();
}

// 子函数5: 处理xRsGradMm
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkProcessXRsGradMm(
    uint32_t bsStart, uint32_t bsChunkStart, uint32_t copySizeND, uint32_t currentChunkSize, uint32_t chunkNDSize,
    uint32_t buffId, LocalTensor<P> &sumBuf)
{
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = false;
    uint32_t vecIdx = blockIdx_ % 2;
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
    __ubuf__ P *xRsGradMmUbAddr = (__ubuf__ P *)xRsGradMmUb_.GetPhyAddr();
    if (withGamma_) {
        __ubuf__ P *gammaInAddr = (__ubuf__ P *)gammaIn_.GetPhyAddr();
        VFDoV2GammaMulXRsGradMm<true>(xRsGradMmUbAddr, xRsGradMmInBufAddr, gammaInAddr, currentChunkSize, copySizeND);
    } else {
        VFDoV2GammaMulXRsGradMm<false>(xRsGradMmUbAddr, xRsGradMmInBufAddr, nullptr, currentChunkSize, copySizeND);
    }

    uint32_t srcReduceShape[] = {currentChunkSize, copySizeND};
    if (withGamma_) {
        Mul(xRsGradMmInBuf_, xRsFp32Buf_, xRsGradMmInBuf_, copySizeND * currentChunkSize);
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

    Add(xGradVec3Buf_, xGradVec3Buf_, xRsGradMmUb_, chunkNDSize);
    PipeBarrier<PIPE_V>();
}

// 子函数6: 处理gradXPost
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkProcessGradXPost(
    uint32_t offsetND, uint32_t bsChunkStart, uint32_t currentChunkSize, uint32_t chunkNDSize, uint32_t copySizeND)
{
    LocalTensor<P> gradXPostFp32 = xRsGradMmUb_;
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
        Cast(gradXPostFp32, gradXPostBuf_, RoundMode::CAST_NONE, chunkNDSize);
        PipeBarrier<PIPE_V>();
        vecInQueue1_.FreeTensor(gradXPostBuf_);

        Add(xGradVec3Buf_, xGradVec3Buf_, gradXPostFp32, chunkNDSize);
        PipeBarrier<PIPE_V>();
    }
}

// 子函数7: 输出grad_x
template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2ChunkOutputXGrad(uint32_t offsetND, uint32_t copySizeND,
                                                                             uint32_t bsChunkStart,
                                                                             uint32_t currentChunkSize,
                                                                             uint32_t chunkNDSize)
{
    bf16OutputBuf_ = vecOutQueue1_.AllocTensor<T>();
    Cast(bf16OutputBuf_, xGradVec3Buf_, RoundMode::CAST_RINT, chunkNDSize);
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
__aicore__ inline void MhcPreBackwardKernel<T, P>::AIV22Process(LocalTensor<P> &invRmsUb, LocalTensor<P> &xRsFp32Buf,
                                                                LocalTensor<P> &xRsGradInvUb, uint32_t currentChunkSize,
                                                                uint32_t copySizeND)
{
    __ubuf__ P *invRmsUbAddr = (__ubuf__ P *)invRmsUb.GetPhyAddr();
    __ubuf__ P *xRsGradInvUbAddr = (__ubuf__ P *)xRsGradInvUb.GetPhyAddr();
    __ubuf__ P *xRsFp32BufAddr = (__ubuf__ P *)xRsFp32Buf.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> invRmsReg;
        MicroAPI::RegTensor<P> NDReg;
        MicroAPI::RegTensor<uint32_t> invRmsIdexReg;
        MicroAPI::MaskReg maskAll, mask;
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(currentChunkSize); bsIdx++) {
            uint32_t vfloopCnt1 = Ceil(copySizeND, eleNumPerVf_);
            uint32_t curLen1 = copySizeND;
            maskAll = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
            P invRmsSclar = invRmsUbAddr[bsIdx];
            MicroAPI::Duplicate<P>(invRmsReg, invRmsSclar, maskAll);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < static_cast<uint16_t>(vfloopCnt1); vfBlockIdx++) {
                uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
                uint32_t srcOffset = bsIdx * copySizeND;
                mask = MicroAPI::UpdateMask<P>(curLen1);
                MicroAPI::LoadAlign(NDReg, xRsFp32BufAddr + srcOffset + elemOffset);
                MicroAPI::Mul(NDReg, NDReg, invRmsReg, mask);
                MicroAPI::StoreAlign(xRsGradInvUbAddr + srcOffset + elemOffset, NDReg, mask);
            }
        }
    }
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
        MicroAPI::RegTensor<P> invRmsInReg;
        MicroAPI::RegTensor<P> invRmsTempReg;
        MicroAPI::RegTensor<P> invRmsUbReg;
        MicroAPI::RegTensor<P> invRmsGradUb;
        MicroAPI::RegTensor<P> scaleMeanReg;
        MicroAPI::MaskReg maskAll, mask;
        maskAll = MicroAPI::CreateMask<P, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<P>(scaleMeanReg, (-scaleMean_), maskAll);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < static_cast<uint16_t>(vfloopCnt); vfBlockIdx++) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf_;
            mask = MicroAPI::UpdateMask<P>(curLen);
            MicroAPI::LoadAlign(invRmsInReg, invRmsInBufAddr + elemOffset);
            MicroAPI::Mul(invRmsTempReg, invRmsInReg, invRmsInReg, mask);
            MicroAPI::Mul(invRmsUbReg, invRmsTempReg, invRmsInReg, mask);
            MicroAPI::LoadAlign(invRmsGradUb, invRmsGradUbAddr + elemOffset);
            MicroAPI::Mul(invRmsUbReg, invRmsUbReg, invRmsGradUb, mask);
            MicroAPI::Mul(invRmsUbReg, invRmsUbReg, scaleMeanReg, mask);
            MicroAPI::StoreAlign(invRmsUbAddr + elemOffset, invRmsUbReg, mask);
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
    uint32_t biasGradShapeSrc[] = {usedVecCoreNum_, static_cast<uint32_t>(fusionSize_)};
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
        MicroAPI::RegTensor<P> sumReg;
        MicroAPI::Duplicate(sumReg, 0);
        uint32_t dealMask = ALPHA_GRAD_PADDING;
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(dealMask);
        for (uint16_t vcIdx = 0; vcIdx < static_cast<uint16_t>(usedVecCoreNum_); ++vcIdx) {
            uint32_t elemOffset = vcIdx * ALPHA_GRAD_PADDING;
            MicroAPI::RegTensor<P> alphaGradInReg;
            MicroAPI::LoadAlign(alphaGradInReg, alphaGradIn + elemOffset);
            MicroAPI::Add(sumReg, sumReg, alphaGradInReg, mask);
        }
        MicroAPI::StoreAlign(alphaGradOut, sumReg, mask);
    }
}

} // namespace MhcPreBackward

#endif // __mhc_pre_backward_KERNEL_H_

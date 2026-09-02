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
 * \file mhc_pre_sinkhorn_backward_deterministic.h
 * \brief MhcPreSinkhornBackward 确定性路径 Kernel (arch35)
 *
 * 计算流程 (2 轮 BS 循环):
 *
 *   BS 循环 #1: PhaseA(C) + PhaseB(K)
 *     PhaseACLoop:
 *       for idxC in ⌈c / cLoopDataLen⌉:
 *         CopyIn x[bs,n,cSlice], gradHin[bs,cSlice]           GM→UB
 *         gradHPre += (gradHin * x).reduce_sum(C)              UB (累加)
 *         gradXFromHin = gradHin * hPre                        UB→WS
 *         xFp32 = x cast to fp32                               UB→WS
 *
 *     PhaseBKDim:
 *       CopyIn invRms, hcBeforeNorm                             GM→UB
 *       [pre/post/res 三段]:
 *         ComputeGradNormOut (z = hcBeforeNorm * invRms + bias) UB
 *         SigmoidGrad → gradZ                                   UB
 *         ComputeGradAlphaAndBias → gradAlpha/gradBias          UB→WS
 *         ComputeGradHcBeforeNormAndInvRms → WS                 UB→WS
 *       SinkhornGradSimdVf + ExpGradSimdVf (res path)           UB
 *
 *   SyncAll → InitPhase2Buffers → CrossCoreSetFlag
 *
 *   BS 循环 #2: PhaseE(C)
 *     for idxC: ComputeGradXFromRms → gradXFromRmsWs            WS→UB→WS
 *
 *   AddAlphaAndBias (blockIdx==0)                                WS→GM
 *   CrossCoreWaitFlag + SyncAll
 *   PhaseFFinal: gradX = rms + hin + matmul                    WS→GM
 *
 *   Phase D [AIC]:
 *     PhaseDMm1: gradHcBeforeNorm @ phi → gradXFromMatmulWs    WS (AIC 写)
 *     PhaseDMm2: gradHcBeforeNorm^T @ xFp32 → gradPhi          GM (AIC 写)
 */
#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H

#include <algorithm>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "mhc_pre_sinkhorn_backward_deterministic_base.h"

using namespace AscendC;

template <typename T>
__aicore__ inline T MinValue(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline T MaxValue(T a, T b)
{
    return a > b ? a : b;
}

constexpr uint8_t SIGMOID_GRAD_PRE = 1;
constexpr uint8_t SIGMOID_GRAD_POST = 2;
constexpr int64_t WS_BUFFER_INTERVAL = 1024;
constexpr float FLT_MAX = 3.402823466e+38F;
constexpr int64_t CUBE_MAX_N_SIZE = 16;
constexpr uint16_t PIPE_MTE3_FLAG = 0x5;
constexpr uint16_t PIPE_FIX_FLAG = 0x6;
constexpr uint8_t AIV_AIC_MODE = 0x2;

template <typename X_T, typename GRADHIN_T, typename U>
class MhcPreSinkhornBackwardDeterministic {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;

    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, C2Type, C2Type, MHC_PRE_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreSinkhornBackwardDeterministic() = default;
    __aicore__ inline void Process(void);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
                                GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha,
                                GM_ADDR gradBias, GM_ADDR workspace,
                                const MhcPreSinkhornBackwardArch35DeterminiticTilingData *tilingData, TPipe *pipe);

private:
    // Phase A: C 维循环
    __aicore__ inline void PhaseACLoop(uint64_t bsIdx, uint64_t bsLen);
    __aicore__ inline void CopyInHpreAndGradHPost(const GlobalTensor<U> &gmTensor, uint64_t bsLen);
    __aicore__ inline void CopyInXAndGradHin(uint64_t xGmOffset, uint64_t gradHinGmOffset, uint64_t bsLen,
                                             uint64_t cLen);
    __aicore__ inline void ComputeGradHPreAndGradXFromHin(LocalTensor<U> &hPreLocal, LocalTensor<U> &gradHPreLocal,
                                                          const BsCLoopInfo &info);

    // Phase B: K 维计算
    __aicore__ inline void PhaseBKDim(uint64_t bsIdx, uint64_t bsLen);
    __aicore__ inline void ComputeSigmoidGrad(uint64_t bsLen, uint64_t colLen, uint8_t num);
    __aicore__ inline void ComputeGradAlphaAndBias(uint64_t bsLen, uint64_t colLen);
    __aicore__ inline void ComputeGradNormOut(const LocalTensor<U> &invRmsLocal, uint64_t bsLen, uint64_t colLen,
                                              uint64_t oft, const U &alpha, uint64_t biasOft);
    __aicore__ inline void CopyInHcBeforeNorm(uint64_t bsLen, uint64_t colLen, uint64_t gmOft);
    __aicore__ inline void CopyInInvRms(uint64_t bsLen, uint64_t gmOft);
    __aicore__ inline void CopyOutAlphaAndBias(uint64_t gmAlphaOft, uint64_t gmBiasOft, uint64_t colLen);
    __aicore__ inline void ComputeGradHcBeforeNormAndInvRms(uint64_t bsOffset, uint64_t bsLen, U alphaScalar,
                                                            uint64_t count, uint64_t offset);
    __aicore__ inline void SinkhornGradSimdVf(const LocalTensor<float> &skGradLocal,
                                              const LocalTensor<float> &sumOutRowLocal,
                                              const LocalTensor<float> &sumOutColLocal,
                                              const LocalTensor<float> &normOutLocal, uint64_t bsIdx, uint64_t bsLen);
    __aicore__ inline void SinkhornGradAndExpGrad(uint64_t bsOffset, uint64_t bsIdx, uint64_t bsLen);
    __aicore__ inline void ExpGradSimdVf(LocalTensor<float> skGrad, LocalTensor<float> expGrad,
                                         LocalTensor<float> zResLocal, uint64_t bsIdx, uint64_t bsLen);
    // SinkhornGradSimdVf sub-functions (for-skIter loop in caller)
    __aicore__ inline void SinkhornGradComputeTmpMul(__local_mem__ U *rowNormPtr, __local_mem__ U *skGradPtr,
                                                     __local_mem__ U *tmpPtr, uint16_t bsLenLoop);
    __aicore__ inline void SinkhornGradComputeTmpDiv(__local_mem__ U *colSumPtr, __local_mem__ U *skGradPtr,
                                                     __local_mem__ U *tmpPtr, __local_mem__ U *tmpDivLocalPtr,
                                                     __local_mem__ U *tmpDiv2LocalPtr, uint16_t bsLenLoop);
    __aicore__ inline void SinkhornGradReduceSumCol(__local_mem__ U *tmpDiv2LocalPtr, __local_mem__ U *tmpPtr,
                                                    uint16_t bsLenLoop);
    __aicore__ inline void SinkhornGradComputeRowNorm(__local_mem__ U *tmpDivLocalPtr, __local_mem__ U *rowNormPtr,
                                                      __local_mem__ U *tmpPtr, __local_mem__ U *rowSumPtr,
                                                      __local_mem__ U *tmpDiv3LocalPtr,
                                                      __local_mem__ U *tmpDiv4LocalPtr, uint16_t bsLenLoop);
    __aicore__ inline void SinkhornGradUpdateSkGrad(__local_mem__ U *tmpDiv3LocalPtr, __local_mem__ U *tmpDiv4LocalPtr,
                                                    __local_mem__ U *skGradPtr, uint16_t bsLenLoop);
    // ExpGradSimdVf sub-functions
    __aicore__ inline void ExpGradComputeMaxAndIsMax(__local_mem__ U *zResPtr, __local_mem__ U *zResMaxPtr,
                                                     __local_mem__ U *zResMaxIdxPtr, __local_mem__ U *isMaxIdxPtr,
                                                     uint16_t bsLenLoop);
    __aicore__ inline void ExpGradComputeExpMul(__local_mem__ U *zResPtr, __local_mem__ U *zResMaxPtr,
                                                __local_mem__ U *skGradPtr, __local_mem__ U *tmpMulPtr,
                                                uint16_t bsLenLoop);
    __aicore__ inline void ExpGradComputeFinal(__local_mem__ U *tmpMulPtr, __local_mem__ U *isMaxIdxPtr,
                                               __local_mem__ U *expGradPtr, uint16_t bsLenLoop);

    // Phase D: AIC Matmul
    __aicore__ inline void PhaseDMm1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void PhaseDMm2(const int32_t taskOffset, const int32_t mm2N);

    // Phase E/F: 第二轮 BS 循环
    __aicore__ inline void PhaseECLoop(uint64_t bsOffset, uint64_t bsLen);
    __aicore__ inline void PhaseFFinal(void);

    // Phase 2 初始化
    __aicore__ inline void InitPhase2Buffers(void);

    // Alpha/Bias 辅助
    __aicore__ inline void GetAlphaAndBias(void);
    __aicore__ inline void AddAlphaAndBias(void);
    __aicore__ inline void CopyInAlphaAndBias(int64_t wsBiasOft, int64_t wsAlphaOft, int64_t biasLen);
    __aicore__ inline void CopyOutAlphaAndBiasSum(int64_t gmBiasOft, int64_t gmAlphaOft, int64_t biasLen);

private:
    GlobalTensor<X_T> xGm_; // 输入
    GlobalTensor<GRADHIN_T> gradHinGm_;
    GlobalTensor<U> gradHPostGm_;
    GlobalTensor<U> gradHResGm_;
    GlobalTensor<U> phiGm_;
    GlobalTensor<U> alphaGm_;
    GlobalTensor<U> biasGm_;
    GlobalTensor<U> hPreGm_;
    GlobalTensor<U> hcBeforeNormGm_;
    GlobalTensor<U> invRmsGm_;
    GlobalTensor<U> sumOutGm_;
    GlobalTensor<U> normOutGm_;

    GlobalTensor<U> xFp32Ws_;
    GlobalTensor<U> gradAlphaWs_;
    GlobalTensor<U> gradBiasWs_;
    GlobalTensor<U> gradXFromHinWs_;
    GlobalTensor<U> gradHcBeforeNormWs_;
    GlobalTensor<U> gradNormOutWs_;
    GlobalTensor<U> gradXFromMatmulWs_;
    GlobalTensor<U> gradXFromRmsWs_;

    GlobalTensor<X_T> gradXGm_; // 输出
    GlobalTensor<U> gradPhiGm_;
    GlobalTensor<U> gradAlphaGm_;
    GlobalTensor<U> gradBiasGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> hPreAndGradHPostQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradHinQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> hcBeforeNormQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> invRmsQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradHResQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradAlphaQue_;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> xFp32Que_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradXFromHinQue_;
    TQue<QuePosition::VECOUT, 1> gradAlphaAndBiasQue_;
    TQue<QuePosition::VECOUT, 1> gradAlphaSumQue_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gradBiasQue_;

    TBuf<TPosition::VECCALC> gradHPreBuf_;
    TBuf<TPosition::VECCALC> gradZBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> normOutForwardBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> gradCalcBuf1_;
    TBuf<TPosition::VECCALC> gradCalcBuf2_;
    TBuf<TPosition::VECCALC> tmpDivBuf_;
    TBuf<TPosition::VECCALC> tmpDiv2Buf_;
    TBuf<TPosition::VECCALC> tmpDiv3Buf_;
    TBuf<TPosition::VECCALC> tmpDiv4Buf_;
    TBuf<TPosition::VECCALC> tmpMulBuf_;
    TBuf<TPosition::VECCALC> zResMaxBuf_;
    TBuf<TPosition::VECCALC> zResMaxIdxBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> isMaxBuf_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumOutRowQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumOutColQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> normOutQue_;

    int64_t usedAivNum_ = 0;
    int64_t bsCount_ = 0;
    uint16_t n_ = 0;
    uint16_t nn_ = 0;
    int64_t c_ = 0;
    int64_t curCoreProcessbsTask_ = 0;
    int64_t bsLoopDataLen_ = 0;
    int64_t cLoopDataLen_ = 0;
    int64_t skIterCount_ = 0;
    float eps_ = 0;
    int64_t k_ = 0;
    int64_t blockIdx_ = 0;
    int64_t mm1K_ = 0;
    int64_t mm1M_ = 0;
    int64_t mm1N_ = 0;
    int64_t mm2K_ = 0;
    int64_t mm2M_ = 0;
    int64_t mm2N_ = 0;
    int64_t blockSize_ = 0;
    int64_t nAlignFp32_ = 0;
    int64_t nnAlignFp32_ = 0;
    int64_t cLenAlignFp32_ = 0;
    int64_t cLenAlignFp16_ = 0;
    int64_t bsNAlignFp32_ = 0;
    int64_t bsAlignFp32_ = 0;
    int64_t bsnnAlignFp32_ = 0;
    int64_t kAlignFp32_ = 0;
    int64_t gradBiasOft_ = 0;
    int64_t uTypeSize_ = sizeof(U);
    int64_t xTypeSize_ = sizeof(X_T);
    int64_t gradHinTypeSize_ = sizeof(GRADHIN_T);
    U alphaPre_, alphaPost_, alphaRes_;
    TPipe *pipe_ = nullptr;
    const MhcPreSinkhornBackwardArch35DeterminiticTilingData *tilingData_ = nullptr;
    constexpr static uint32_t V_REG_SIZE = Ops::Base::GetVRegSize();
    int64_t aicNum_ = 0;
    int64_t bsTaskCount_ = 0;
    int64_t totalAicTasks_ = 0;
    int64_t calcBSPerTask_ = 0;
    int64_t totalAicTasksAligned_ = 0;

    // vector中计算RMSNormGrad的循环
    int64_t cLoopsPerBS_ = 0;
    int64_t cCountPerBS_ = 0;
    int64_t cTailPerBS_ = 0;

    // vector中最后计算gradX的循环
    int64_t cLoopsPerBsForFinal_ = 0;
    int64_t cTailLoopsPerBsForFinal_ = 0;
    int64_t cBlockSizeForFinal_ = 0;
    int64_t cBlockTailSizeForFinal_ = 0;
    int64_t blockTotalNum_ = 0;
    int64_t finalUsedAivNum_ = 0;
};

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::Init(
    GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost, GM_ADDR gradHRes, GM_ADDR alpha,
    GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
    GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace,
    const MhcPreSinkhornBackwardArch35DeterminiticTilingData *tilingData, TPipe *pipe)
{
    tilingData_ = tilingData;
    pipe_ = pipe;
    bsCount_ = tilingData->bs;
    n_ = static_cast<uint16_t>(tilingData->n);
    nn_ = n_ * n_;
    c_ = tilingData->c;
    cLoopDataLen_ = tilingData->cLoopDataLen;
    bsLoopDataLen_ = tilingData->bsLoopDataLen;
    skIterCount_ = tilingData->skIterCount;
    eps_ = tilingData->eps;
    k_ = n_ * n_ + 2 * n_;
    usedAivNum_ = tilingData->usedAivNum;
    blockIdx_ = GetBlockIdx();
    blockSize_ = Ops::Base::GetUbBlockSize();
    curCoreProcessbsTask_ = (blockIdx_ == usedAivNum_ - 1) ? tilingData->tailBsTaskCount : tilingData->bsTaskCount;
    nAlignFp32_ = Ops::Base::CeilAlign(int64_t(n_ * uTypeSize_), blockSize_) / uTypeSize_;
    cLenAlignFp32_ = Ops::Base::CeilAlign(int64_t(cLoopDataLen_ * uTypeSize_), blockSize_) / uTypeSize_;
    cLenAlignFp16_ = Ops::Base::CeilAlign(int64_t(cLoopDataLen_ * xTypeSize_), blockSize_) / xTypeSize_;
    nnAlignFp32_ = Ops::Base::CeilAlign(int64_t(n_ * n_ * uTypeSize_), blockSize_) / uTypeSize_;
    bsNAlignFp32_ = Ops::Base::CeilAlign(int64_t(n_ * bsLoopDataLen_ * uTypeSize_), blockSize_) / uTypeSize_;
    bsAlignFp32_ = Ops::Base::CeilAlign(int64_t(bsLoopDataLen_ * uTypeSize_), blockSize_) / uTypeSize_;
    bsnnAlignFp32_ = Ops::Base::CeilAlign(int64_t(bsLoopDataLen_ * n_ * n_ * uTypeSize_), blockSize_) / uTypeSize_;
    kAlignFp32_ = Ops::Base::CeilAlign(int64_t(k_ * uTypeSize_), blockSize_) / uTypeSize_;
    gradBiasOft_ = blockSize_ / uTypeSize_;

    // 每个BS中nc循环
    cCountPerBS_ = cLenAlignFp32_;
    cLoopsPerBS_ = Ops::Base::CeilDiv(int64_t(n_ * c_), cCountPerBS_);
    cTailPerBS_ = n_ * c_ - (cLoopsPerBS_ - 1) * cCountPerBS_;
    // BSNC每核循环
    cLoopsPerBsForFinal_ =
        (blockIdx_ == tilingData->finalUsedAivNum - 1) ? tilingData->cBlockTailLoops : tilingData->cBlockLoops;
    cBlockSizeForFinal_ = tilingData->cBlockCount;
    cBlockTailSizeForFinal_ =
        (blockIdx_ == tilingData->finalUsedAivNum - 1) ? tilingData->cBlockTailTailCount : tilingData->cBlockTailCount;
    blockTotalNum_ = (cLoopsPerBsForFinal_ - 1) * cBlockSizeForFinal_ + tilingData->cBlockTailCount;
    finalUsedAivNum_ = tilingData->finalUsedAivNum;
    aicNum_ = tilingData->aicNum;

    gradXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(gradX));
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(phi));
    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradPhi));
    constexpr int64_t FP32_BYTE_SIZE = sizeof(U);
    int64_t gradAlphaOffset = bsCount_ * c_ * n_ + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;
    int64_t gradBiasOffset = gradAlphaOffset + 3 * usedAivNum_ + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;
    int64_t gradXFromHinOffset =
        gradBiasOffset + (2 * usedAivNum_ * n_ + usedAivNum_ * n_ * n_) + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;
    int64_t gradXFromRmsOffset = gradXFromHinOffset + bsCount_ * n_ * c_ + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;
    int64_t gradXFromMatmulOffset = gradXFromRmsOffset + bsCount_ * n_ * c_ + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;
    int64_t gradHcBeforeNormOffset = gradXFromMatmulOffset + bsCount_ * n_ * c_ + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;
    int64_t gradNormOutOffset =
        gradHcBeforeNormOffset + bsCount_ * (2 * n_ + n_ * n_) + WS_BUFFER_INTERVAL / FP32_BYTE_SIZE;

    xFp32Ws_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace));
    gradAlphaWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradAlphaOffset);
    gradBiasWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradBiasOffset);
    gradXFromHinWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradXFromHinOffset);
    gradHcBeforeNormWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradHcBeforeNormOffset);
    gradNormOutWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradNormOutOffset);
    gradXFromMatmulWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradXFromMatmulOffset);
    gradXFromRmsWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(workspace) + gradXFromRmsOffset);

    if ASCEND_IS_AIV {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        gradHinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GRADHIN_T *>(gradHin));
        gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradHPost));
        gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradHRes));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(bias));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(hPre));
        hcBeforeNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(hcBeforeNorm));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(invRms));
        sumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(sumOut));
        normOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(normOut));
        gradAlphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradAlpha));
        gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradBias));

        // Init UB
        // VECIN
        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, cLenAlignFp16_ * n_ * bsLoopDataLen_ * xTypeSize_);
        pipe_->InitBuffer(hPreAndGradHPostQue_, DOUBLE_BUFFER, bsNAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradHinQue_, DOUBLE_BUFFER, cLenAlignFp16_ * bsLoopDataLen_ * gradHinTypeSize_);
        pipe_->InitBuffer(hcBeforeNormQue_, DOUBLE_BUFFER, bsLoopDataLen_ * nnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(invRmsQue_, DOUBLE_BUFFER, bsAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradHResQue_, DOUBLE_BUFFER, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);

        // VECOUT
        pipe_->InitBuffer(xFp32Que_, DOUBLE_BUFFER, cLenAlignFp32_ * n_ * bsLoopDataLen_ * uTypeSize_);
        pipe_->InitBuffer(gradXFromHinQue_, DOUBLE_BUFFER, cLenAlignFp32_ * n_ * bsLoopDataLen_ * uTypeSize_);
        pipe_->InitBuffer(gradAlphaAndBiasQue_, 1, nnAlignFp32_ * uTypeSize_ + blockSize_);

        // BUF
        pipe_->InitBuffer(gradHPreBuf_, bsNAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradZBuf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(biasBuf_, kAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(normOutForwardBuf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(zBuf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradCalcBuf1_, kAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradCalcBuf2_, kAlignFp32_ * uTypeSize_);

        pipe_->InitBuffer(tmpDivBuf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(tmpDiv2Buf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(tmpDiv3Buf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(tmpDiv4Buf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(tmpMulBuf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(sumOutRowQue_, DOUBLE_BUFFER, skIterCount_ * bsLoopDataLen_ * nAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(sumOutColQue_, DOUBLE_BUFFER, skIterCount_ * bsLoopDataLen_ * nAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(normOutQue_, DOUBLE_BUFFER, skIterCount_ * bsLoopDataLen_ * nnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(zResMaxBuf_, bsLoopDataLen_ * nnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(zResMaxIdxBuf_, bsLoopDataLen_ * nnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(tmpBuf_, bsLoopDataLen_ * nnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(isMaxBuf_, bsLoopDataLen_ * nnAlignFp32_ * uTypeSize_);

        if (blockIdx_ == 0) {
            InitOutput<U>(gradAlphaGm_, 3, U(0));
            InitOutput<U>(gradAlphaWs_, 3 * usedAivNum_, U(0));
            InitOutput<U>(gradBiasGm_, n_ * n_ + 2 * n_, U(0));
            InitOutput<U>(gradBiasWs_, 2 * usedAivNum_ * n_ + usedAivNum_ * n_ * n_, U(0));
        }
        SyncAll();
    }

    if ASCEND_IS_AIC {
        mm1K_ = n_ * n_ + 2 * n_;
        mm1M_ = bsCount_;
        mm1N_ = n_ * c_;

        mm2K_ = bsCount_;
        mm2M_ = n_ * n_ + 2 * n_;
        mm2N_ = n_ * c_;
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInXAndGradHin(
    uint64_t xGmOffset, uint64_t gradHinGmOffset, uint64_t bsLen, uint64_t cLen)
{
    LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>(); // in
    LocalTensor<GRADHIN_T> gradHinLocal = gradHinQue_.AllocTensor<GRADHIN_T>();

    CopyIn(xLocal, xGm_[xGmOffset], bsLen * n_, cLen, 0, (c_ - cLen) * xTypeSize_);
    CopyIn(gradHinLocal, gradHinGm_[gradHinGmOffset], bsLen, cLen, 0, (c_ - cLen) * gradHinTypeSize_);

    xQue_.EnQue(xLocal);
    gradHinQue_.EnQue(gradHinLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInHpreAndGradHPost(
    const GlobalTensor<U> &gMTensor, uint64_t bsLen)
{
    LocalTensor<U> hPreLocal = hPreAndGradHPostQue_.AllocTensor<U>();
    CopyIn(hPreLocal, gMTensor, 1, bsLen * n_, 0, 0);
    hPreAndGradHPostQue_.EnQue(hPreLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInHcBeforeNorm(uint64_t bsLen,
                                                                                                  uint64_t colLen,
                                                                                                  uint64_t gmOft)
{
    LocalTensor<U> hcBeforeNormLocal = hcBeforeNormQue_.AllocTensor<U>();
    CopyIn(hcBeforeNormLocal, hcBeforeNormGm_[gmOft], bsLen, colLen, 0, (k_ - colLen) * uTypeSize_);
    hcBeforeNormQue_.EnQue(hcBeforeNormLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInInvRms(uint64_t bsLen,
                                                                                            uint64_t gmOft)
{
    LocalTensor<U> invRmsLocal = invRmsQue_.AllocTensor<U>();
    CopyIn(invRmsLocal, invRmsGm_[gmOft], 1, bsLen, 0, 0);
    invRmsQue_.EnQue(invRmsLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyOutAlphaAndBias(uint64_t gmAlphaOft,
                                                                                                   uint64_t gmBiasOft,
                                                                                                   uint64_t colLen)
{
    LocalTensor<U> gradAlphaLocal = gradAlphaAndBiasQue_.DeQue<U>();
    LocalTensor<U> gradBiasLocal = gradAlphaLocal[gradBiasOft_];
    SetAtomicAdd<U>();
    CopyOut(gradAlphaWs_[gmAlphaOft], gradAlphaLocal, 1, 1, 0, 0);
    CopyOut(gradBiasWs_[gmBiasOft], gradBiasLocal, 1, colLen, 0, 0);
    SetAtomicNone();
    gradAlphaAndBiasQue_.FreeTensor(gradAlphaLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInAlphaAndBias(int64_t wsBiasOft,
                                                                                                  int64_t wsAlphaOft,
                                                                                                  int64_t biasLen)
{
    LocalTensor<U> gradBiasLocal = gradBiasQue_.AllocTensor<U>();
    LocalTensor<U> gradAlphaLocal = gradAlphaQue_.AllocTensor<U>();
    CopyIn(gradBiasLocal, gradBiasWs_[wsBiasOft], usedAivNum_, biasLen, 0, 0);
    CopyIn(gradAlphaLocal, gradAlphaWs_[wsAlphaOft], 1, usedAivNum_, 0, 0);
    gradBiasQue_.EnQue(gradBiasLocal);
    gradAlphaQue_.EnQue(gradAlphaLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyOutAlphaAndBiasSum(
    int64_t gmBiasOft, int64_t gmAlphaOft, int64_t biasLen)
{
    auto biasAlignFp32 = Ops::Base::CeilAlign(int64_t(biasLen * uTypeSize_), blockSize_) / uTypeSize_;
    LocalTensor<U> gradBiasSumLocal = gradBiasQue_.DeQue<U>();
    LocalTensor<U> gradAlphaSumLocal = gradAlphaSumQue_.DeQue<U>();
    CopyOut(gradAlphaGm_[gmAlphaOft], gradAlphaSumLocal, 1, 1, 0, 0);
    for (int i = 1; i < usedAivNum_; i++) {
        Add(gradBiasSumLocal, gradBiasSumLocal, gradBiasSumLocal[i * biasAlignFp32], biasLen);
    }
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    CopyOut(gradBiasGm_[gmBiasOft], gradBiasSumLocal, 1, biasLen, 0, 0);
    gradBiasQue_.FreeTensor(gradBiasSumLocal);
    gradAlphaSumQue_.FreeTensor(gradAlphaSumLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::AddAlphaAndBias(void)
{
    for (uint16_t i = 0; i < 3; i++) {
        auto biasLen = (i == 2) ? n_ * n_ : n_;
        CopyInAlphaAndBias(i * usedAivNum_ * n_, i * usedAivNum_, biasLen);
        LocalTensor<U> gradAlphaSumLocal = gradAlphaSumQue_.AllocTensor<U>();
        LocalTensor<U> gradAlphaLocal = gradAlphaQue_.DeQue<U>();

        __local_mem__ U *gradAlphaAddr = (__ubuf__ U *)gradAlphaLocal.GetPhyAddr();
        __local_mem__ U *gradAlphaSumAddr = (__ubuf__ U *)gradAlphaSumLocal.GetPhyAddr();
        uint32_t vfLen = V_REG_SIZE / sizeof(U);
        uint16_t loopCnt = (usedAivNum_ + vfLen - 1) / vfLen;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<U> gradAlphaReg;
            AscendC::MicroAPI::RegTensor<U> gradAlphaSumTempReg;
            AscendC::MicroAPI::RegTensor<U> gradAlphaSumReg;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg allMaskReg =
                AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
            uint32_t maskLen = static_cast<uint32_t>(usedAivNum_);
            MicroAPI::Duplicate(gradAlphaSumReg, U(0), allMaskReg);
            for (uint16_t i = 0; i < loopCnt; i++) {
                maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg gradAlphaOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::DataCopy(gradAlphaReg, gradAlphaAddr, gradAlphaOfst);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradAlphaSumTempReg, gradAlphaReg,
                                                                           maskReg); // reduce_sum
                AscendC::MicroAPI::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaSumTempReg, maskReg);
            }
            AscendC::MicroAPI::Store(gradAlphaSumAddr, gradAlphaSumReg, 1);
        }
        gradAlphaSumQue_.EnQue(gradAlphaSumLocal);
        gradAlphaQue_.FreeTensor(gradAlphaLocal);
        CopyOutAlphaAndBiasSum(i * n_, i, biasLen);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradHcBeforeNormAndInvRms(
    uint64_t bsOffset, uint64_t bsLen, U alphaScalar, uint64_t count, uint64_t offset)
{
    LocalTensor<U> srcLocal = gradZBuf_.Get<U>();
    for (uint64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
        int64_t curBsIdx = bsOffset + bsLoop;
        LocalTensor<U> dstLocal = xFp32Que_.AllocTensor<U>();
        ComputeMulScalar(dstLocal, srcLocal, alphaScalar, count, bsLoop * count);
        xFp32Que_.EnQue<U>(dstLocal);
        dstLocal = xFp32Que_.DeQue<U>();
        CopyOut(gradNormOutWs_[curBsIdx * k_ + offset], dstLocal, 1, count, 0, 0);
        xFp32Que_.FreeTensor(dstLocal);
        dstLocal = xFp32Que_.AllocTensor<U>();
        U invRms = invRmsGm_(curBsIdx);
        ComputeMulScalar(dstLocal, srcLocal, invRms * alphaScalar, count, bsLoop * count);
        xFp32Que_.EnQue<U>(dstLocal);
        dstLocal = xFp32Que_.DeQue<U>();
        CopyOut(gradHcBeforeNormWs_[curBsIdx * k_ + offset], dstLocal, 1, count, 0, 0);
        xFp32Que_.FreeTensor(dstLocal);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseACLoop(uint64_t bsIdx,
                                                                                           uint64_t bsLen)
{
    LocalTensor<U> gradHPreLocal = gradHPreBuf_.Get<U>(); // buf 中间结果
    Duplicate(gradHPreLocal, float(0), bsLen * n_);

    uint64_t cLoopDataLen = cLoopDataLen_;
    uint64_t cLoopNum = Ops::Base::CeilDiv(c_, cLoopDataLen_);
    uint64_t cTailDataLen = c_ - cLoopDataLen * (cLoopNum - 1);
    uint64_t bsOffset = blockIdx_ * tilingData_->bsTaskCount + bsIdx * bsLoopDataLen_;
    uint64_t hPreOffset = bsOffset * n_;
    uint64_t xOffset = bsOffset * n_ * c_;
    uint64_t gradHinOffset = bsOffset * c_;
    uint64_t gradXFromHinOffset = bsOffset * n_ * c_;
    CopyInHpreAndGradHPost(hPreGm_[hPreOffset], bsLen);
    LocalTensor<U> hPreLocal = hPreAndGradHPostQue_.DeQue<U>();

    BsCLoopInfo info;
    info.bsLen = bsLen;
    for (uint64_t cIdx = 0; cIdx < cLoopNum; cIdx++) {
        auto cLen = cIdx == cLoopNum - 1 ? cTailDataLen : cLoopDataLen;
        int64_t cLenXTAlign = Ops::Base::CeilAlign(int64_t(cLen * xTypeSize_), blockSize_) / xTypeSize_;
        int64_t cLenUAlign = Ops::Base::CeilAlign(int64_t(cLen * uTypeSize_), blockSize_) / uTypeSize_;
        int64_t cLenGRADHINTAlign =
            Ops::Base::CeilAlign(int64_t(cLen * gradHinTypeSize_), blockSize_) / gradHinTypeSize_;
        info.cLen = cLen;
        info.cLenXTAlign = cLenXTAlign;
        info.cLenUAlign = cLenUAlign;
        info.cLenGRADHINTAlign = cLenGRADHINTAlign;
        uint64_t xStartOft = xOffset + cIdx * cLoopDataLen;
        uint64_t gradHinStartOft = gradHinOffset + cIdx * cLoopDataLen;
        uint64_t gradXFromHinStartOft = gradXFromHinOffset + cIdx * cLoopDataLen;
        CopyInXAndGradHin(xStartOft, gradHinStartOft, bsLen, cLen);
        ComputeGradHPreAndGradXFromHin(hPreLocal, gradHPreLocal, info);
        LocalTensor<U> xFp32Local = xFp32Que_.DeQue<U>();                                        // out
        LocalTensor<U> gradXFromHinLocal = gradXFromHinQue_.DeQue<U>();                          // out
        CopyOut(xFp32Ws_[xStartOft], xFp32Local, bsLen * n_, cLen, 0, (c_ - cLen) * uTypeSize_); // 搬运x_f32到workspace
        CopyOut(gradXFromHinWs_[gradXFromHinStartOft], gradXFromHinLocal, bsLen * n_, cLen, 0,
                (c_ - cLen) * uTypeSize_); // 搬运grad_x_from_hin到workspace
        xFp32Que_.FreeTensor(xFp32Local);
        gradXFromHinQue_.FreeTensor(gradXFromHinLocal);
    }
    hPreAndGradHPostQue_.FreeTensor(hPreLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseBKDim(uint64_t bsIdx,
                                                                                          uint64_t bsLen)
{
    uint64_t bsOffset = blockIdx_ * tilingData_->bsTaskCount + bsIdx * bsLoopDataLen_;
    uint64_t hcBeforeNormOffset = bsOffset * k_;
    uint64_t invRmsOffset = bsOffset;
    uint64_t gradHPostOffset = bsOffset * n_;

    CopyInInvRms(bsLen, invRmsOffset);
    LocalTensor<U> invRmsLocal = invRmsQue_.DeQue<U>();
    ComputeGradNormOut(invRmsLocal, bsLen, n_, hcBeforeNormOffset, alphaPre_, 0);
    ComputeSigmoidGrad(bsLen, n_, SIGMOID_GRAD_PRE);
    ComputeGradAlphaAndBias(bsLen, n_);
    CopyOutAlphaAndBias(blockIdx_, blockIdx_ * n_, n_);
    ComputeGradHcBeforeNormAndInvRms(bsOffset, bsLen, alphaPre_, n_, 0);

    ComputeGradNormOut(invRmsLocal, bsLen, n_, hcBeforeNormOffset + n_, alphaPost_, n_);
    CopyInHpreAndGradHPost(gradHPostGm_[gradHPostOffset], bsLen);
    ComputeSigmoidGrad(bsLen, n_, SIGMOID_GRAD_POST);
    ComputeGradAlphaAndBias(bsLen, n_);
    CopyOutAlphaAndBias(blockIdx_ + usedAivNum_, blockIdx_ * n_ + usedAivNum_ * n_, n_);
    ComputeGradHcBeforeNormAndInvRms(bsOffset, bsLen, alphaPost_, n_, n_);

    ComputeGradNormOut(invRmsLocal, bsLen, n_ * n_, hcBeforeNormOffset + 2 * n_, alphaRes_, 2 * n_);
    invRmsQue_.FreeTensor(invRmsLocal);

    SinkhornGradAndExpGrad(bsOffset, bsIdx, bsLen);
    ComputeGradAlphaAndBias(bsLen, n_ * n_);
    CopyOutAlphaAndBias(blockIdx_ + 2 * usedAivNum_, blockIdx_ * n_ * n_ + usedAivNum_ * n_ * 2, n_ * n_);

    ComputeGradHcBeforeNormAndInvRms(bsOffset, bsLen, alphaRes_, n_ * n_, 2 * n_);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradNormOut(
    const LocalTensor<U> &invRmsLocal, uint64_t bsLen, uint64_t colLen, uint64_t oft, const U &alpha, uint64_t biasOft)
{
    CopyInHcBeforeNorm(bsLen, colLen, oft);
    LocalTensor<U> hcBeforeNormLocal = hcBeforeNormQue_.DeQue<U>();

    LocalTensor<U> zLocal = zBuf_.Get<U>();
    LocalTensor<U> normOutForwardLocal = normOutForwardBuf_.Get<U>();
    LocalTensor<U> biasLocal = biasBuf_.Get<U>();

    __local_mem__ U *zAddr = (__ubuf__ U *)zLocal.GetPhyAddr();
    __local_mem__ U *hcBeformNormAddr = (__ubuf__ U *)hcBeforeNormLocal.GetPhyAddr();
    __local_mem__ U *invRmsAddr = (__ubuf__ U *)invRmsLocal.GetPhyAddr();
    __local_mem__ U *normOutForwardAddr = (__ubuf__ U *)normOutForwardLocal.GetPhyAddr();
    __local_mem__ U *biasAddr = (__ubuf__ U *)biasLocal[biasOft].GetPhyAddr();

    auto cLenAlign = Ops::Base::CeilAlign(int64_t(colLen * uTypeSize_), blockSize_) / uTypeSize_;
    ComputeForwardPart<U>(hcBeformNormAddr, invRmsAddr, normOutForwardAddr, bsLen, colLen, cLenAlign, V_REG_SIZE,
                          alpha);
    PipeBarrier<PIPE_V>();
    ComputeZAndForwardPart<U>(zAddr, normOutForwardAddr, biasAddr, bsLen, colLen, cLenAlign, V_REG_SIZE, alpha);
    hcBeforeNormQue_.FreeTensor(hcBeforeNormLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeSigmoidGrad(uint64_t bsLen,
                                                                                                  uint64_t colLen,
                                                                                                  uint8_t num)
{
    LocalTensor<U> gradHLocal;
    LocalTensor<U> zPreLocal = zBuf_.Get<U>();
    LocalTensor<U> gradZLocal = gradZBuf_.Get<U>();

    __local_mem__ U *gradHAddr;
    __local_mem__ U *zPreAddr = (__ubuf__ U *)zPreLocal.GetPhyAddr();
    __local_mem__ U *gradZAddr = (__ubuf__ U *)gradZLocal.GetPhyAddr();
    if (num == SIGMOID_GRAD_PRE) {
        gradHLocal = gradHPreBuf_.Get<U>();
        gradHAddr = (__ubuf__ U *)gradHLocal.GetPhyAddr();
        SigmoidGrad<U, true>(gradHAddr, zPreAddr, gradZAddr, bsLen, colLen, V_REG_SIZE, eps_);
    } else {
        gradHLocal = hPreAndGradHPostQue_.DeQue<U>();
        gradHAddr = (__ubuf__ U *)gradHLocal.GetPhyAddr();
        SigmoidGrad<U, false>(gradHAddr, zPreAddr, gradZAddr, bsLen, colLen, V_REG_SIZE, eps_);
        hPreAndGradHPostQue_.FreeTensor(gradHLocal);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradAlphaAndBias(uint64_t bsLen,
                                                                                                       uint64_t colLen)
{
    LocalTensor<U> normOutForwardLocal = normOutForwardBuf_.Get<U>();
    LocalTensor<U> gradZLocal = gradZBuf_.Get<U>();
    LocalTensor<U> gradAlphaLocal = gradAlphaAndBiasQue_.AllocTensor<U>();
    LocalTensor<U> gradBiasLocal = gradAlphaLocal[gradBiasOft_];

    __local_mem__ U *normOutForwardAddr = (__ubuf__ U *)normOutForwardLocal.GetPhyAddr();
    __local_mem__ U *gradZAddr = (__ubuf__ U *)gradZLocal.GetPhyAddr();
    __local_mem__ U *gradAlphaAddr = (__ubuf__ U *)gradAlphaLocal.GetPhyAddr();
    __local_mem__ U *gradBiasAddr = (__ubuf__ U *)gradBiasLocal.GetPhyAddr();

    ComputeGradAlpha<U>(normOutForwardAddr, gradZAddr, gradAlphaAddr, bsLen, colLen, V_REG_SIZE);
    ComputeGradBias<U>(gradBiasAddr, gradZAddr, bsLen, colLen, V_REG_SIZE);
    gradAlphaAndBiasQue_.EnQue<U>(gradAlphaLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradHPreAndGradXFromHin(
    LocalTensor<U> &hPreLocal, LocalTensor<U> &gradHPreLocal, const BsCLoopInfo &info)
{
    LocalTensor<U> xFp32Local = xFp32Que_.AllocTensor<U>();
    LocalTensor<U> gradXFromHinLocal = gradXFromHinQue_.AllocTensor<U>();
    LocalTensor<X_T> xLocal = xQue_.DeQue<X_T>();
    LocalTensor<GRADHIN_T> gradHInLocal = gradHinQue_.DeQue<GRADHIN_T>();

    __local_mem__ X_T *xAddr = (__ubuf__ X_T *)xLocal.GetPhyAddr();
    __local_mem__ U *xCastAddr = (__ubuf__ U *)xFp32Local.GetPhyAddr();
    __local_mem__ GRADHIN_T *gradHInAddr = (__ubuf__ GRADHIN_T *)gradHInLocal.GetPhyAddr();
    __local_mem__ U *gradHPreAddr = (__ubuf__ U *)gradHPreLocal.GetPhyAddr();
    __local_mem__ U *hPreAddr = (__ubuf__ U *)hPreLocal.GetPhyAddr();
    __local_mem__ U *gradXFromHinAddr = (__ubuf__ U *)gradXFromHinLocal.GetPhyAddr();

    uint32_t vfLen = V_REG_SIZE / uTypeSize_;
    uint16_t loopCnt = (info.cLen + vfLen - 1) / vfLen;
    uint16_t bsLen = static_cast<uint16_t>(info.bsLen);
    uint32_t cLen = static_cast<uint32_t>(info.cLen);
    int64_t cLenGRADHINTAlign = info.cLenGRADHINTAlign;
    int64_t cLenUAlign = info.cLenUAlign;
    int64_t cLenXTAlign = info.cLenXTAlign;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<X_T> xReg;
        AscendC::MicroAPI::RegTensor<U> xCastReg;
        AscendC::MicroAPI::RegTensor<GRADHIN_T> gradHInReg;
        AscendC::MicroAPI::RegTensor<U> gradHInCastReg;
        AscendC::MicroAPI::RegTensor<U> gradHPreReg;
        AscendC::MicroAPI::RegTensor<U> gradXFromHinReg;
        AscendC::MicroAPI::RegTensor<U> gradHPreRegSumReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();

        for (uint16_t i = 0; i < bsLen; i++) {
            auto gradHPreStart = gradHPreAddr + i * n_;
            uint32_t maskLen = cLen;
            for (uint16_t j = 0; j < loopCnt; j++) {
                auto gradHInStart = gradHInAddr + i * cLenGRADHINTAlign + j * vfLen;
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::DataCopy<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(gradHInReg,
                                                                                                     gradHInStart);
                AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHInCastReg, gradHInReg, valueMaskReg);
                for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                                   AscendC::MicroAPI::MemType::SCALAR_LOAD>();
                    U hPre = *(hPreAddr + i * n_ + k);
                    U gradHPre = *(gradHPreStart + k);
                    auto xCastStart = xCastAddr + (i * n_ + k) * cLenUAlign + j * vfLen;
                    auto xAddrStart = xAddr + (i * n_ + k) * cLenXTAlign + j * vfLen;
                    auto gradXFromHinStart = gradXFromHinAddr + (i * n_ + k) * cLenUAlign + j * vfLen;

                    AscendC::MicroAPI::DataCopy<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddrStart);
                    AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xCastReg, xReg, valueMaskReg);
                    AscendC::MicroAPI::Muls(gradXFromHinReg, gradHInCastReg, hPre,
                                            valueMaskReg); // grad_h_in_f32 * h_pre
                    AscendC::MicroAPI::Mul(gradHPreReg, xCastReg, gradHInCastReg, valueMaskReg); // x_f32 *
                                                                                                 // grad_h_in_f32
                    AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradHPreReg, gradHPreReg,
                                                                               valueMaskReg);      // reduce_sum
                    AscendC::MicroAPI::Adds(gradHPreRegSumReg, gradHPreReg, gradHPre, OneMaskReg); // 将原值进行累加

                    AscendC::MicroAPI::DataCopy(gradXFromHinStart, gradXFromHinReg, valueMaskReg);
                    AscendC::MicroAPI::DataCopy(xCastStart, xCastReg, valueMaskReg);
                    AscendC::MicroAPI::Store(gradHPreStart + k, gradHPreRegSumReg, 1);
                }
            }
        }
    }
    xFp32Que_.EnQue(xFp32Local);
    gradXFromHinQue_.EnQue(gradXFromHinLocal);
    xQue_.FreeTensor(xLocal);
    gradHinQue_.FreeTensor(gradHInLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::GetAlphaAndBias(void)
{
    LocalTensor<U> biasLocal_ = biasBuf_.Get<U>();
    alphaPre_ = alphaGm_.GetValue(0);
    alphaPost_ = alphaGm_.GetValue(1);
    alphaRes_ = alphaGm_.GetValue(2);
    DataCopyPad(biasLocal_, biasGm_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(U)), 0, 0, 0},
                {false, 0, 0, 0});
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradAndExpGrad(uint64_t bsOffset,
                                                                                                      uint64_t bsIdx,
                                                                                                      uint64_t bsLen)
{
    int64_t gmOffset = static_cast<int64_t>(bsOffset);
    int64_t sumOutRowOffset = gmOffset * n_;
    int64_t sumOutColOffset = (gmOffset + bsCount_) * n_;
    int64_t normOutOffset = gmOffset * n_ * n_;

    LocalTensor<float> sumOutRowLocal = sumOutRowQue_.AllocTensor<float>();
    LocalTensor<float> sumOutColLocal = sumOutColQue_.AllocTensor<float>();
    LocalTensor<float> normOutLocal = normOutQue_.AllocTensor<float>();
    LocalTensor<float> gradHResLocal = gradHResQue_.AllocTensor<U>();

    event_t eventIdVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVMte2);

    CopyIn(sumOutRowLocal, sumOutGm_[sumOutRowOffset], skIterCount_, bsLen * n_, 0,
           (bsCount_ - bsLen + bsCount_) * n_ * uTypeSize_);
    CopyIn(sumOutColLocal, sumOutGm_[sumOutColOffset], skIterCount_, bsLen * n_, 0,
           (bsCount_ - bsLen + bsCount_) * n_ * uTypeSize_);
    CopyIn(normOutLocal, normOutGm_[normOutOffset], skIterCount_, bsLen * n_ * n_, 0,
           (bsCount_ - bsLen + bsCount_) * n_ * n_ * uTypeSize_);
    CopyIn(gradHResLocal, gradHResGm_[gmOffset * n_ * n_], 1, bsLen * n_ * n_, 0, 0);

    sumOutRowQue_.EnQue<float>(sumOutRowLocal);
    sumOutColQue_.EnQue<float>(sumOutColLocal);
    normOutQue_.EnQue<float>(normOutLocal);
    gradHResQue_.EnQue(gradHResLocal);

    sumOutRowLocal = sumOutRowQue_.DeQue<float>();
    sumOutColLocal = sumOutColQue_.DeQue<float>();
    normOutLocal = normOutQue_.DeQue<float>();
    LocalTensor<float> skGradLocal = gradHResQue_.DeQue<float>();

    LocalTensor<float> expGradLocal = gradZBuf_.Get<float>();
    LocalTensor<float> zReslocal = zBuf_.Get<float>();

    SinkhornGradSimdVf(skGradLocal, sumOutRowLocal, sumOutColLocal, normOutLocal, bsIdx, bsLen);
    ExpGradSimdVf(skGradLocal, expGradLocal, zReslocal, bsIdx, bsLen);

    gradHResQue_.FreeTensor(skGradLocal);
    sumOutRowQue_.FreeTensor(sumOutRowLocal);
    sumOutColQue_.FreeTensor(sumOutColLocal);
    normOutQue_.FreeTensor(normOutLocal);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradSimdVf(
    const LocalTensor<float> &skGradLocal, const LocalTensor<float> &sumOutRowLocal,
    const LocalTensor<float> &sumOutColLocal, const LocalTensor<float> &normOutLocal, uint64_t bsIdx, uint64_t bsLen)
{
    auto bsLennAlignFp32_ = Ops::Base::CeilAlign(int64_t(bsLen * n_ * uTypeSize_), blockSize_) / uTypeSize_;
    auto bsLennnAlignFp32_ = Ops::Base::CeilAlign(int64_t(bsLen * n_ * n_ * uTypeSize_), blockSize_) / uTypeSize_;

    __local_mem__ U *skGradPtr = (__local_mem__ U *)skGradLocal.GetPhyAddr();
    __local_mem__ U *colSumBase = (__local_mem__ U *)sumOutColLocal.GetPhyAddr();
    __local_mem__ U *rowSumBase = (__local_mem__ U *)sumOutRowLocal.GetPhyAddr();
    __local_mem__ U *normOutBase = (__local_mem__ U *)normOutLocal.GetPhyAddr();
    __local_mem__ U *tmpDivLocalPtr = (__local_mem__ U *)tmpDivBuf_.Get<float>().GetPhyAddr();
    __local_mem__ U *tmpDiv2LocalPtr = (__local_mem__ U *)tmpDiv2Buf_.Get<float>().GetPhyAddr();
    __local_mem__ U *tmpDiv3LocalPtr = (__local_mem__ U *)tmpDiv3Buf_.Get<float>().GetPhyAddr();
    __local_mem__ U *tmpDiv4LocalPtr = (__local_mem__ U *)tmpDiv4Buf_.Get<float>().GetPhyAddr();
    __local_mem__ U *tmpPtr = (__local_mem__ U *)tmpBuf_.Get<float>().GetPhyAddr();

    uint16_t bsLenLoop = static_cast<uint16_t>(bsLen);

    for (int64_t i = skIterCount_ - 1; i >= 0; --i) {
        auto rowNormPtr = normOutBase + i * bsLennnAlignFp32_;
        auto colSumPtr = colSumBase + i * bsLennAlignFp32_;
        auto rowSumPtr = rowSumBase + i * bsLennAlignFp32_;

        SinkhornGradComputeTmpMul(rowNormPtr, skGradPtr, tmpPtr, bsLenLoop);
        SinkhornGradComputeTmpDiv(colSumPtr, skGradPtr, tmpPtr, tmpDivLocalPtr, tmpDiv2LocalPtr, bsLenLoop);
        SinkhornGradReduceSumCol(tmpDiv2LocalPtr, tmpPtr, bsLenLoop);
        SinkhornGradComputeRowNorm(tmpDivLocalPtr, rowNormPtr, tmpPtr, rowSumPtr, tmpDiv3LocalPtr, tmpDiv4LocalPtr,
                                   bsLenLoop);
        SinkhornGradUpdateSkGrad(tmpDiv3LocalPtr, tmpDiv4LocalPtr, skGradPtr, bsLenLoop);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradSimdVf(
    LocalTensor<float> skGrad, LocalTensor<float> expGrad, LocalTensor<float> zResLocal, uint64_t bsIdx, uint64_t bsLen)
{
    LocalTensor<float> zResMax = zResMaxBuf_.Get<float>();
    LocalTensor<float> zResMaxIdx = zResMaxIdxBuf_.Get<float>();
    LocalTensor<float> tmpMul = tmpMulBuf_.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
    LocalTensor<float> isMaxIdx = isMaxBuf_.Get<float>();

    __local_mem__ U *expGradPtr = (__local_mem__ U *)expGrad.GetPhyAddr();
    __local_mem__ U *zResPtr = (__local_mem__ U *)zResLocal.GetPhyAddr();
    __local_mem__ U *skGradPtr = (__local_mem__ U *)skGrad.GetPhyAddr();
    __local_mem__ U *tmpMulPtr = (__local_mem__ U *)tmpMul.GetPhyAddr();
    __local_mem__ U *tmpPtr = (__local_mem__ U *)tmpLocal.GetPhyAddr();
    __local_mem__ U *isMaxIdxPtr = (__local_mem__ U *)isMaxIdx.GetPhyAddr();

    uint32_t maskLenN = static_cast<uint32_t>(n_);
    uint32_t maskLenNN = static_cast<uint32_t>(nn_);

    uint16_t bsLenLoop = static_cast<uint16_t>(bsLen);

    AscendC::DataCopy(zResMaxIdx, zResLocal, maskLenNN);
    AscendC::DataCopy(zResMax, zResLocal, maskLenNN);

    __local_mem__ U *zResMaxPtr = (__local_mem__ U *)zResMax.GetPhyAddr();
    __local_mem__ U *zResMaxIdxPtr = (__local_mem__ U *)zResMaxIdx.GetPhyAddr();

    ExpGradComputeMaxAndIsMax(zResPtr, zResMaxPtr, zResMaxIdxPtr, isMaxIdxPtr, bsLenLoop);
    ExpGradComputeExpMul(zResPtr, zResMaxPtr, skGradPtr, tmpMulPtr, bsLenLoop);
    ExpGradComputeFinal(tmpMulPtr, isMaxIdxPtr, expGradPtr, bsLenLoop);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradComputeTmpMul(
    __local_mem__ U *rowNormPtr, __local_mem__ U *skGradPtr, __local_mem__ U *tmpPtr, uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> rowNormReg;
        Reg::RegTensor<float> skGradReg;
        Reg::RegTensor<float> tmpMulReg;
        Reg::UnalignRegForLoad ureg0, ureg1;
        Reg::MaskReg maskNN;

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);
            maskNN = Reg::UpdateMask<float>(maskLenNN);

            int64_t bsOffset = j * nn_;
            auto tmpPtr1 = tmpPtr + bsOffset;

            Reg::LoadUnAlignPre(ureg0, rowNormPtr + bsOffset);
            Reg::LoadUnAlign(rowNormReg, ureg0, rowNormPtr + bsOffset);
            Reg::LoadUnAlignPre(ureg1, skGradPtr + bsOffset);
            Reg::LoadUnAlign(skGradReg, ureg1, skGradPtr + bsOffset);

            Reg::Mul(tmpMulReg, rowNormReg, skGradReg, maskNN);

            Reg::Store(tmpPtr1, tmpMulReg, nn_);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradComputeTmpDiv(
    __local_mem__ U *colSumPtr, __local_mem__ U *skGradPtr, __local_mem__ U *tmpPtr, __local_mem__ U *tmpDivLocalPtr,
    __local_mem__ U *tmpDiv2LocalPtr, uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        Reg::RegTensor<float> colSumReg;
        Reg::RegTensor<float> colSumSquareReg;
        Reg::RegTensor<float> tmpDivReg;
        Reg::RegTensor<float> tmpDiv2Reg;
        Reg::UnalignRegForLoad ureg2, ureg3, ureg4;
        Reg::UnalignRegForStore uregStore0, uregStore1;
        Reg::MaskReg maskN;
        Reg::MaskReg maskNN;

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);
            maskNN = Reg::UpdateMask<float>(maskLenNN);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = j * nn_;

            auto tmpDivLocalPtr1 = tmpDivLocalPtr + bsOffset;
            auto tmpDiv2LocalPtr1 = tmpDiv2LocalPtr + bsOffset;
            auto tmpPtr1 = tmpPtr + bsOffset;
            auto skGradPtr1 = skGradPtr + bsOffset;

            Reg::LoadUnAlignPre(ureg2, colSumPtr + j * n_);
            Reg::LoadUnAlign(colSumReg, ureg2, colSumPtr + j * n_);

            Reg::Mul(colSumSquareReg, colSumReg, colSumReg, maskN);

            for (uint16_t k = 0; k < n_; k++) {
                Reg::RegTensor<float> tmpReg, tmpReg1;

                Reg::LoadUnAlignPre(ureg3, tmpPtr1 + k * n_);
                Reg::LoadUnAlign(tmpReg, ureg3, tmpPtr1 + k * n_);

                Reg::LoadUnAlignPre(ureg4, skGradPtr1 + k * n_);
                Reg::LoadUnAlign(tmpReg1, ureg4, skGradPtr1 + k * n_);

                Reg::Div(tmpDivReg, tmpReg1, colSumReg, maskNN);
                Reg::Div(tmpDiv2Reg, tmpReg, colSumSquareReg, maskNN);

                Reg::StoreUnAlign(tmpDivLocalPtr1, tmpDivReg, uregStore0, n_);
                Reg::StoreUnAlign(tmpDiv2LocalPtr1, tmpDiv2Reg, uregStore1, n_);
            }
            Reg::StoreUnAlignPost(tmpDivLocalPtr1, uregStore0, 0);
            Reg::StoreUnAlignPost(tmpDiv2LocalPtr1, uregStore1, 0);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradReduceSumCol(
    __local_mem__ U *tmpDiv2LocalPtr, __local_mem__ U *tmpPtr, uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        Reg::RegTensor<float> reduceSumReg;
        Reg::UnalignRegForLoad ureg5;
        Reg::MaskReg maskN;

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = j * nn_;

            auto tmpDiv2LocalPtr1 = tmpDiv2LocalPtr + bsOffset;
            auto tmpPtr1 = tmpPtr + j * nAlignFp32_;

            Reg::Duplicate(reduceSumReg, (float)0.0, maskN);
            for (uint16_t k = 0; k < n_; k++) {
                Reg::RegTensor<float> tmpReg;
                Reg::LoadUnAlignPre(ureg5, tmpDiv2LocalPtr1 + k * n_);
                Reg::LoadUnAlign(tmpReg, ureg5, tmpDiv2LocalPtr1 + k * n_);
                Reg::Add(reduceSumReg, reduceSumReg, tmpReg, maskN);
            }

            Reg::Store(tmpPtr1, reduceSumReg, n_);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradComputeRowNorm(
    __local_mem__ U *tmpDivLocalPtr, __local_mem__ U *rowNormPtr, __local_mem__ U *tmpPtr, __local_mem__ U *rowSumPtr,
    __local_mem__ U *tmpDiv3LocalPtr, __local_mem__ U *tmpDiv4LocalPtr, uint16_t bsLenLoop)
{
    auto gradXRowPtr = tmpPtr;
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        Reg::RegTensor<float> rowSumBroadcastReg;
        Reg::RegTensor<float> tmpDivReg;
        Reg::RegTensor<float> tmpDiv3Reg;
        Reg::RegTensor<float> tmpDiv4Reg;
        Reg::RegTensor<float> tmpMul2Reg;
        Reg::RegTensor<float> gradXRowNormedReg;
        Reg::RegTensor<float> rowNormReg;
        Reg::RegTensor<float> tmpSubReg;
        Reg::UnalignRegForLoad ureg6, ureg7;
        Reg::UnalignRegForStore uregStore3, uregStore4;
        Reg::MaskReg maskN;
        Reg::MaskReg maskNN;

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);
            maskNN = Reg::UpdateMask<float>(maskLenNN);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = j * nn_;
            int64_t nAlignOffset = j * nAlignFp32_;

            auto tmpDiv3LocalPtr1 = tmpDiv3LocalPtr + bsOffset;
            auto tmpDiv4LocalPtr1 = tmpDiv4LocalPtr + bsOffset;

            Reg::LoadAlign(gradXRowNormedReg, gradXRowPtr + nAlignOffset);
            for (uint16_t k = 0; k < n_; k++) {
                auto rowSumValue = *(rowSumPtr + j * n_ + k);

                Reg::LoadUnAlignPre(ureg6, tmpDivLocalPtr + bsOffset + k * n_);
                Reg::LoadUnAlign(tmpDivReg, ureg6, tmpDivLocalPtr + bsOffset + k * n_);
                Reg::LoadUnAlignPre(ureg7, rowNormPtr + bsOffset + k * n_);
                Reg::LoadUnAlign(rowNormReg, ureg7, rowNormPtr + bsOffset + k * n_);

                Reg::Sub(tmpSubReg, tmpDivReg, gradXRowNormedReg, maskN);
                Reg::Mul(tmpMul2Reg, tmpSubReg, rowNormReg, maskN);
                Reg::Duplicate(rowSumBroadcastReg, rowSumValue, maskN);
                Reg::Div(tmpDiv3Reg, tmpSubReg, rowSumBroadcastReg, maskN);
                Reg::Div(tmpDiv4Reg, tmpMul2Reg, rowSumBroadcastReg, maskN);
                Reg::StoreUnAlign(tmpDiv3LocalPtr1, tmpDiv3Reg, uregStore3, n_);
                Reg::StoreUnAlign(tmpDiv4LocalPtr1, tmpDiv4Reg, uregStore4, n_);
            }
            Reg::StoreUnAlignPost(tmpDiv3LocalPtr1, uregStore3, 0);
            Reg::StoreUnAlignPost(tmpDiv4LocalPtr1, uregStore4, 0);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradUpdateSkGrad(
    __local_mem__ U *tmpDiv3LocalPtr, __local_mem__ U *tmpDiv4LocalPtr, __local_mem__ U *skGradPtr, uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        Reg::RegTensor<float> tmpDiv3Reg;
        Reg::RegTensor<float> tmpDiv4Reg;
        Reg::RegTensor<float> skGradReg;
        Reg::UnalignRegForLoad ureg10, ureg11;
        Reg::UnalignRegForStore uregStore3;
        Reg::MaskReg maskN;

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = j * nn_;

            auto tmpDiv3LocalPtr1 = tmpDiv3LocalPtr + bsOffset;
            auto tmpDiv4LocalPtr1 = tmpDiv4LocalPtr + bsOffset;
            auto skGradPtr1 = skGradPtr + bsOffset;

            for (uint16_t k = 0; k < n_; k++) {
                Reg::RegTensor<float> reduceSumReg, tmpReg, tmpReg1;

                Reg::LoadUnAlignPre(ureg10, tmpDiv3LocalPtr1 + k * n_);
                Reg::LoadUnAlign(tmpDiv3Reg, ureg10, tmpDiv3LocalPtr1 + k * n_);

                Reg::LoadUnAlignPre(ureg11, tmpDiv4LocalPtr1 + k * n_);
                Reg::LoadUnAlign(tmpDiv4Reg, ureg11, tmpDiv4LocalPtr1 + k * n_);

                Reg::Reduce<Reg::ReduceType::SUM>(reduceSumReg, tmpDiv4Reg, maskN);
                Reg::Muls(tmpReg, reduceSumReg, (float)-1.0, maskN);
                Reg::Duplicate(tmpReg1, tmpReg, maskN);
                Reg::Add(skGradReg, tmpDiv3Reg, tmpReg1, maskN);

                Reg::StoreUnAlign(skGradPtr1, skGradReg, uregStore3, n_);
            }
            Reg::StoreUnAlignPost(skGradPtr1, uregStore3, 0);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradComputeMaxAndIsMax(
    __local_mem__ U *zResPtr, __local_mem__ U *zResMaxPtr, __local_mem__ U *zResMaxIdxPtr, __local_mem__ U *isMaxIdxPtr,
    uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> zResReg;
        Reg::RegTensor<float> zResMaxReg;
        Reg::RegTensor<float> zResMaxIdxReg;
        Reg::RegTensor<float> zResMaxBcastReg;
        Reg::RegTensor<float> zResMaxIdxBcastReg;
        Reg::RegTensor<float> isMaxReg;
        Reg::RegTensor<float> zeroReg;
        Reg::RegTensor<float> oneReg;
        Reg::RegTensor<float> posInfReg;
        Reg::UnalignRegForLoad ureg;
        Reg::UnalignRegForStore uregStore0, uregStore1, uregStore2;
        Reg::MaskReg maxMask;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskN;
        Reg::MaskReg maskNN;

        Reg::Duplicate(zeroReg, (float)0.0, mask);
        Reg::Duplicate(oneReg, (float)1.0, mask);
        Reg::Duplicate(posInfReg, (float)FLT_MAX, mask);

        for (uint16_t i = 0; i < bsLenLoop; i++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);

            maskNN = Reg::UpdateMask<float>(maskLenNN);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = i * nn_;

            auto zResMaxPtr1 = zResMaxPtr + bsOffset;
            auto zResMaxIdxPtr1 = zResMaxIdxPtr + bsOffset;
            auto isMaxIdxPtr1 = isMaxIdxPtr + bsOffset;

            Reg::RegTensor<float> idxReg, idxSelectReg;
            Reg::Arange(idxReg, 0);

            for (uint16_t j = 0; j < n_; j++) {
                int64_t nOffset = j * n_;
                auto zResPtr1 = zResPtr + bsOffset + nOffset;
                auto zResMaxPtr2 = zResMaxPtr1 + nOffset;
                auto zResMaxIdxPtr2 = zResMaxIdxPtr1 + nOffset;

                Reg::LoadUnAlignPre(ureg, zResPtr1);
                Reg::LoadUnAlign(zResReg, ureg, zResPtr1);

                Reg::Reduce<Reg::ReduceType::MAX>(zResMaxReg, zResReg, maskN);
                Reg::Duplicate(zResMaxBcastReg, zResMaxReg, maskN);
                Reg::Compare<float>(maxMask, zResMaxBcastReg, zResReg, maskN);
                Reg::Select(idxSelectReg, idxReg, posInfReg, maxMask);
                Reg::Reduce<Reg::ReduceType::MIN>(zResMaxIdxReg, idxSelectReg, maskN);
                Reg::Select(isMaxReg, oneReg, zeroReg, maxMask);
                Reg::Duplicate(zResMaxIdxBcastReg, zResMaxIdxReg, maskN);

                for (uint16_t k = 0; k < n_; k++) {
                    Reg::StoreUnAlign<float>(zResMaxPtr2, zResMaxBcastReg, uregStore0, 1);
                    Reg::StoreUnAlign<float>(zResMaxIdxPtr2, zResMaxIdxBcastReg, uregStore1, 1);
                }
                Reg::StoreUnAlignPost(zResMaxPtr2, uregStore0, 0);
                Reg::StoreUnAlignPost(zResMaxIdxPtr2, uregStore1, 0);

                Reg::StoreUnAlign<float>(isMaxIdxPtr1, isMaxReg, uregStore2, n_);
            }
            Reg::StoreUnAlignPost(isMaxIdxPtr1, uregStore2, 0);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradComputeExpMul(
    __local_mem__ U *zResPtr, __local_mem__ U *zResMaxPtr, __local_mem__ U *skGradPtr, __local_mem__ U *tmpMulPtr,
    uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        Reg::RegTensor<float> zResReg;
        Reg::RegTensor<float> zResMaxReg;
        Reg::RegTensor<float> skGradReg;
        Reg::RegTensor<float> tmpSubReg;
        Reg::RegTensor<float> tmpExpReg;
        Reg::RegTensor<float> tmpMulReg;
        Reg::UnalignRegForLoad ureg0, ureg1, ureg2;
        Reg::MaskReg maskN;
        Reg::MaskReg maskNN;

        for (uint16_t i = 0; i < bsLenLoop; i++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);

            maskNN = Reg::UpdateMask<float>(maskLenNN);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = i * nn_;

            Reg::LoadUnAlignPre(ureg0, zResPtr + bsOffset);
            Reg::LoadUnAlign(zResReg, ureg0, zResPtr + bsOffset);
            Reg::LoadUnAlignPre(ureg1, zResMaxPtr + bsOffset);
            Reg::LoadUnAlign(zResMaxReg, ureg1, zResMaxPtr + bsOffset);
            Reg::LoadUnAlignPre(ureg2, skGradPtr + bsOffset);
            Reg::LoadUnAlign(skGradReg, ureg2, skGradPtr + bsOffset);

            Reg::Sub<float>(tmpSubReg, zResReg, zResMaxReg, maskNN);
            Reg::Exp<float>(tmpExpReg, tmpSubReg, maskNN);
            Reg::Mul<float>(tmpMulReg, tmpExpReg, skGradReg, maskNN);

            Reg::Store<float>(tmpMulPtr + bsOffset, tmpMulReg, nn_);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradComputeFinal(
    __local_mem__ U *tmpMulPtr, __local_mem__ U *isMaxIdxPtr, __local_mem__ U *expGradPtr, uint16_t bsLenLoop)
{
    __VEC_SCOPE__
    {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        Reg::RegTensor<float> tmpMulReg;
        Reg::RegTensor<float> isMaxReg;
        Reg::RegTensor<float> sumAllReg;
        Reg::RegTensor<float> sumAllBcastReg;
        Reg::RegTensor<float> tmpMul2Reg;
        Reg::RegTensor<float> expGradReg;
        Reg::UnalignRegForLoad ureg;
        Reg::UnalignRegForStore uregStore3;
        Reg::MaskReg maskN;

        for (uint16_t i = 0; i < bsLenLoop; i++) {
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            maskN = Reg::UpdateMask<float>(maskLenN);

            int64_t bsOffset = i * nn_;

            auto expGradPtr1 = expGradPtr + bsOffset;

            for (uint16_t j = 0; j < n_; j++) {
                Reg::LoadUnAlignPre(ureg, tmpMulPtr + bsOffset + j * n_);
                Reg::LoadUnAlign(tmpMulReg, ureg, tmpMulPtr + bsOffset + j * n_);

                Reg::LoadUnAlignPre(ureg, isMaxIdxPtr + bsOffset + j * n_);
                Reg::LoadUnAlign(isMaxReg, ureg, isMaxIdxPtr + bsOffset + j * n_);

                Reg::Reduce<Reg::ReduceType::SUM>(sumAllReg, tmpMulReg, maskN);
                Reg::Duplicate(sumAllBcastReg, sumAllReg, maskN);
                Reg::Mul(tmpMul2Reg, isMaxReg, sumAllBcastReg, maskN);
                Reg::Sub(expGradReg, tmpMulReg, tmpMul2Reg, maskN);

                Reg::StoreUnAlign<float>(expGradPtr1, expGradReg, uregStore3, n_);
            }
            Reg::StoreUnAlignPost(expGradPtr1, uregStore3, 0);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseECLoop(uint64_t bsOffset,
                                                                                           uint64_t bsLen)
{
    for (int64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
        auto curBsIdx = bsOffset + bsLoop;
        U curGradInvRms = 0;
        U curInvRms = invRmsGm_(curBsIdx);
        auto curBSNCOffset = curBsIdx * n_ * c_;
        auto curBSKOffset = curBsIdx * k_;
        curInvRms = curInvRms * curInvRms * curInvRms;
        LocalTensor<U> gradNormOutLocal = gradCalcBuf1_.Get<U>();
        LocalTensor<U> hcBeforeNormLocal = gradCalcBuf2_.Get<U>();
        CopyIn(gradNormOutLocal, gradNormOutWs_[curBSKOffset], 1, k_, 0, 0);
        CopyIn(hcBeforeNormLocal, hcBeforeNormGm_[curBSKOffset], 1, k_, 0, 0);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        ComputeGradInvRms(gradNormOutLocal, hcBeforeNormLocal, k_, curGradInvRms);

        for (int64_t cLoop = 0; cLoop < cLoopsPerBS_; cLoop++) { // 切k
            auto cCount = cLoop == cLoopsPerBS_ - 1 ? cTailPerBS_ : cCountPerBS_;
            auto cOffset = curBSNCOffset + cCountPerBS_ * cLoop;
            LocalTensor<U> xTmpLocal = xQue_.AllocTensor<U>();
            CopyIn(xTmpLocal, xFp32Ws_[cOffset], 1, cCount, 0, 0);
            xQue_.EnQue(xTmpLocal);
            xTmpLocal = xQue_.DeQue<U>();
            LocalTensor<U> xFp32LocalOut = xFp32Que_.AllocTensor<U>();
            ComputeGradXFromRmsVF(xTmpLocal, xFp32LocalOut, curInvRms, curGradInvRms, cCount, n_ * c_);
            xFp32Que_.EnQue(xFp32LocalOut);
            xFp32LocalOut = xFp32Que_.DeQue<U>();
            CopyOut(gradXFromRmsWs_[cOffset], xFp32LocalOut, 1, cCount, 0, 0);
            xQue_.FreeTensor(xTmpLocal);
            xFp32Que_.FreeTensor(xFp32LocalOut);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseFFinal()
{
    for (int64_t dataLoop = 0; dataLoop < cLoopsPerBsForFinal_; dataLoop++) { // 切c
        int64_t calcCount = dataLoop == cLoopsPerBsForFinal_ - 1 ? cBlockTailSizeForFinal_ : cBlockSizeForFinal_;
        int64_t calcOffset = blockIdx_ * blockTotalNum_ + dataLoop * cBlockSizeForFinal_;
        LocalTensor<U> xFp32Local = xFp32Que_.AllocTensor<U>();
        LocalTensor<U> xLocal = xQue_.AllocTensor<U>();
        LocalTensor<U> gradXFromHinLocal = gradHinQue_.AllocTensor<U>();
        LocalTensor<U> gradXFromMatMulLocal = hPreAndGradHPostQue_.AllocTensor<U>();
        CopyIn(xLocal, gradXFromRmsWs_[calcOffset], 1, calcCount, 0, 0);
        xQue_.EnQue(xLocal);
        xLocal = xQue_.DeQue<U>();
        CopyIn(gradXFromHinLocal, gradXFromHinWs_[calcOffset], 1, calcCount, 0, 0);
        gradHinQue_.EnQue(gradXFromHinLocal);
        gradXFromHinLocal = gradHinQue_.DeQue<U>();
        CopyIn(gradXFromMatMulLocal, gradXFromMatmulWs_[calcOffset], 1, calcCount, 0, 0);
        hPreAndGradHPostQue_.EnQue(gradXFromMatMulLocal);
        gradXFromMatMulLocal = hPreAndGradHPostQue_.DeQue<U>();
        Add(xFp32Local, gradXFromHinLocal, xLocal, calcCount);
        Add(xFp32Local, xFp32Local, gradXFromMatMulLocal, calcCount);
        if constexpr (!std::is_same<X_T, U>::value) {
            Cast(xFp32Local.template ReinterpretCast<X_T>(), xFp32Local, RoundMode::CAST_RINT, calcCount);
        }
        xFp32Que_.EnQue(xFp32Local);
        LocalTensor<X_T> xOutLocal = xFp32Que_.DeQue<X_T>();
        CopyOut(gradXGm_[calcOffset], xOutLocal, 1, calcCount, 0, 0);
        xFp32Que_.FreeTensor(xOutLocal);
        xQue_.FreeTensor(xLocal);
        gradHinQue_.FreeTensor(gradXFromHinLocal);
        hPreAndGradHPostQue_.FreeTensor(gradXFromMatMulLocal);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::InitPhase2Buffers(void)
{
    pipe_->Reset();
    auto aivNumAlign = Ops::Base::CeilAlign(int64_t(usedAivNum_ * uTypeSize_), blockSize_) / uTypeSize_;
    pipe_->InitBuffer(gradAlphaQue_, DOUBLE_BUFFER, aivNumAlign * uTypeSize_);
    pipe_->InitBuffer(gradAlphaSumQue_, 1, blockSize_);
    pipe_->InitBuffer(gradBiasQue_, 1, usedAivNum_ * nnAlignFp32_ * uTypeSize_);
    pipe_->InitBuffer(gradCalcBuf1_, kAlignFp32_ * uTypeSize_);
    pipe_->InitBuffer(gradCalcBuf2_, kAlignFp32_ * uTypeSize_);
    pipe_->InitBuffer(hPreAndGradHPostQue_, DOUBLE_BUFFER, cBlockSizeForFinal_ * sizeof(U));
    pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, MaxValue(cLenAlignFp32_, cBlockSizeForFinal_) * sizeof(U));
    pipe_->InitBuffer(gradHinQue_, DOUBLE_BUFFER, cBlockSizeForFinal_ * sizeof(U));
    pipe_->InitBuffer(xFp32Que_, DOUBLE_BUFFER, MaxValue(cLenAlignFp32_, cBlockSizeForFinal_) * sizeof(U));
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::Process(void)
{
    if ASCEND_IS_AIV {
        // === BS 循环 #1: PhaseA(C) + PhaseB(K) ===
        if (blockIdx_ < usedAivNum_) {
            GetAlphaAndBias();
            uint64_t bsLoopNum = Ops::Base::CeilDiv(curCoreProcessbsTask_, bsLoopDataLen_);
            uint64_t bsTailDataLen = curCoreProcessbsTask_ - bsLoopDataLen_ * (bsLoopNum - 1);
            for (uint64_t bsIdx = 0; bsIdx < bsLoopNum; bsIdx++) {
                auto bsLen = bsIdx == bsLoopNum - 1 ? bsTailDataLen : bsLoopDataLen_;
                PhaseACLoop(bsIdx, bsLen);
                PhaseBKDim(bsIdx, bsLen);
            }
        }

        // === Phase 1→2 同步 ===
        SyncAll();
        InitPhase2Buffers();
        CrossCoreSetFlag<AIV_AIC_MODE, PIPE_MTE3>(PIPE_MTE3_FLAG);

        // === BS 循环 #2: PhaseE(C) ===
        if (blockIdx_ < usedAivNum_) {
            uint64_t bsLoopNum = Ops::Base::CeilDiv(curCoreProcessbsTask_, bsLoopDataLen_);
            uint64_t bsTailDataLen = curCoreProcessbsTask_ - bsLoopDataLen_ * (bsLoopNum - 1);
            for (uint64_t bsIdx = 0; bsIdx < bsLoopNum; bsIdx++) {
                auto bsLen = bsIdx == bsLoopNum - 1 ? bsTailDataLen : bsLoopDataLen_;
                int64_t gmOffset = blockIdx_ * tilingData_->bsTaskCount + bsIdx * bsLoopDataLen_;
                PhaseECLoop(gmOffset, bsLen);
            }
        }

        // === 归约 + 最终输出 ===
        if (blockIdx_ == 0) {
            AddAlphaAndBias();
        }

        CrossCoreWaitFlag<AIV_AIC_MODE>(PIPE_FIX_FLAG);
        SyncAll();
        if (blockIdx_ < finalUsedAivNum_) {
            PhaseFFinal();
        }
    }

    // === Phase D: AIC Matmul ===
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<AIV_AIC_MODE>(PIPE_MTE3_FLAG);
        int64_t totalSize = bsCount_;
        int64_t taskNumPerCore = Ops::Base::CeilDiv(totalSize, aicNum_);
        int64_t taskOffset = blockIdx_ * taskNumPerCore;
        taskNumPerCore = MinValue(taskNumPerCore, totalSize - blockIdx_ * taskNumPerCore);
        PhaseDMm1(static_cast<int32_t>(taskOffset), static_cast<int32_t>(taskNumPerCore));

        CrossCoreSetFlag<AIV_AIC_MODE, PIPE_FIX>(PIPE_FIX_FLAG);

        int64_t totalN = n_ * c_;
        int64_t taskNCount = Ops::Base::CeilDiv(totalN, aicNum_);
        taskNCount = MinValue(taskNCount, CUBE_MAX_N_SIZE);
        int64_t taskLoops = Ops::Base::CeilDiv(totalN, taskNCount);
        int64_t taskNTailCount = totalN - (taskLoops - 1) * taskNCount;
        for (int32_t taskLoop = blockIdx_; taskLoop < taskLoops; taskLoop += aicNum_) {
            int32_t taskOft = taskLoop * taskNCount;
            int32_t taskCount = taskLoop == (taskLoops - 1) ? taskNTailCount : taskNCount;
            PhaseDMm2(taskOft, taskCount);
        }
    }
}

// mm1K_ = n_ * n_ + 2 * n_;
// mm1M_ = b * s;
// mm1N_ = n_ * c_;

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseDMm1(const int32_t taskOffset,
                                                                                         const int32_t mm1M)
{
    if (mm1M <= 0)
        return;
    mm1_.SetTensorA(gradHcBeforeNormWs_[taskOffset * mm1K_]);
    mm1_.SetTensorB(phiGm_);
    mm1_.SetHF32(false, 1);
    mm1_.SetOrgShape(mm1M_, mm1N_, mm1K_);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false>(gradXFromMatmulWs_[taskOffset * (n_ * c_)], 0);
    mm1_.End();
}

// mm2K_ = b * s;
// mm2M_ = n_ * n_ + 2 * n_;
// mm2N_ = n_ * c_;

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseDMm2(const int32_t taskOffset,
                                                                                         const int32_t mm2N)
{
    if (mm2N <= 0)
        return;
    mm2_.SetTensorA(gradHcBeforeNormWs_, true);
    mm2_.SetTensorB(xFp32Ws_[taskOffset]);
    mm2_.SetHF32(false, 1);
    mm2_.SetOrgShape(mm2M_, mm2N_, mm2K_);
    mm2_.SetSingleShape(mm2M_, mm2N, mm2K_);
    mm2_.template IterateAll<false>(gradPhiGm_[taskOffset], 0);
    mm2_.End();
}

#endif //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H

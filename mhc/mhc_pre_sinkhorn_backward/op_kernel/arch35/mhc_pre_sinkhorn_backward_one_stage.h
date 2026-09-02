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
 * \file mhc_pre_sinkhorn_backward_one_stage.h
 * \brief mhc_pre_sinkhorn_backward
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ONE_STAGE_H
#define MHC_PRE_SINKHORN_BACKWARD_ONE_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_backward_simd_vf.h"
#include "mhc_pre_sinkhorn_backward_simt_vf.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t FP32_PER_REPEAT = 64;

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
} // namespace

template <typename T, typename U>
class MhcPreSinkhornBackwardOneStage {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, C2Type, C2Type, MHC_PRE_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreSinkhornBackwardOneStage() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
                                GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha,
                                GM_ADDR gradBias, GM_ADDR workspace,
                                const MhcPreSinkhornBackwardArch35TilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, phi, hPre, gradHin, gradHPost, gradHRes, alpha, bias, hcBeforeNorm, invRms, sumOut, normOut, gradX,
               gradPhi, gradAlpha, gradBias, workspace);
        InitBuffer();
    }
    __aicore__ inline void Process();

protected:
    int8_t ping1_ = 0;
    int64_t blkIdx_, aivNum_, aicNum_;
    TPipe *pipe_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_;
    int64_t c_, n_, c0_, c1_;
    int64_t tile_, tileAligned_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    T alphaPre_, alphaPost_, alphaRes_;

    TBuf<TPosition::VECCALC> gradHinBuf_, hPreBuf_, hResBuf_, xBuf_;
    TBuf<TPosition::VECCALC> xFp32Buf_, gradXBuf_;
    TBuf<TPosition::VECCALC> fusedGradHPreAndGradHPostBuf_, gradHResBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_, biasBuf_, invRmsBuf_, gradInvRmsBuf_;
    TBuf<TPosition::VECCALC> fusedHPre2AndHPost2Buf_;
    TBuf<TPosition::VECCALC> sigmoidGradBuf1_, sigmoidGradBuf2_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> gradRMSNormBuf_;
    TBuf<TPosition::VECOUT> gradH2Buf_;
    TBuf<TPosition::VECOUT> gradAlphaBuf_, gradBiasBuf_;

    LocalTensor<U> xLocal_;
    LocalTensor<T> xFp32Local_;
    LocalTensor<U> gradHinLocal_;
    LocalTensor<U> gradXLocal_;
    LocalTensor<T> hPreLocal_;
    LocalTensor<T> fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> biasLocal_, alphaLocal_;
    LocalTensor<T> gradBiasLocal_, gradAlphaLocal_;

    LocalTensor<T> invRmsLocal_, gradInvRmsLocal_;
    LocalTensor<T> gradH2Local_;
    LocalTensor<T> fusedHPre2AndHPost2Local_, hRes2Local_;
    LocalTensor<T> tmpLocal_;
    LocalTensor<T> gradSigmoidLocal1_;
    LocalTensor<T> gradSigmoidLocal2_;
    LocalTensor<T> gradRMSNormLocal_;

    GlobalTensor<U> xGlobal_;
    GlobalTensor<T> xFp32Global_;
    GlobalTensor<U> gradHinGlobal_;
    GlobalTensor<T> hPreGlobal_;
    GlobalTensor<U> gradXGlobal_;
    GlobalTensor<T> gradHPostGlobal_;
    GlobalTensor<T> alphaGlobal_, biasGlobal_;
    GlobalTensor<T> invRmsGlobal_;
    GlobalTensor<T> gradAlphaGlobal_, gradBiasGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> skNormGlobal_, skSumGlobal_;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> gradH2Global_;
    GlobalTensor<T> gradPhiGlobal_;
    GlobalTensor<T> phiGlobal_;

private:
    __aicore__ inline void InitTiling(const MhcPreSinkhornBackwardArch35TilingData *tilingData)
    {
        batchSize_ = tilingData->batchSize;
        seqLength_ = tilingData->seqLength;
        aivNum_ = tilingData->aivNum;
        aicNum_ = tilingData->aivNum / 2;
        c_ = tilingData->c;
        n_ = tilingData->n;
        c0_ = tilingData->c0;
        c1_ = tilingData->c1;
        skIterCount_ = tilingData->skIterCount;
        tile_ = tilingData->tileSize;
        tileAligned_ = AlignUp(tile_, BYTE_SIZE_PER_BLOCK / sizeof(T));
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tile_);

        if ASCEND_IS_AIC {
            mm1K_ = n_ * n_ + 2 * n_;
            mm1M_ = tile_ * 2;
            mm1N_ = n_ * c_;

            mm2K_ = batchSize_ * seqLength_;
            mm2M_ = n_ * n_ + 2 * n_;
            mm2N_ = n_ * c_;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
                                  GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                  GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha,
                                  GM_ADDR gradBias, GM_ADDR workspace)
    {
        // cube 和 vector 共用
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(x));             // fp32
        xFp32Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace)); // fp32
        gradPhiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gradPhi)); // fp32
        phiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(phi));         // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradX));     // fp32
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + batchSize_ * seqLength_ * n_ * c_);

        if ASCEND_IS_AIV {
            hPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hPre));       // fp32
            gradHinGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradHin)); // fp32

            gradHPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gradHPost)); // fp32

            alphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(alpha)); // fp32
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));   // fp32

            invRmsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(invRms)); // fp32
            gradAlphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gradAlpha));
            gradBiasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gradBias));

            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hcBeforeNorm)); // fp32

            skNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(normOut)); // fp32
            skSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(sumOut));   // fp32

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gradHRes)); // fp32
        }
    }

    __aicore__ inline void InitBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(xBuf_, n_ * c_ * sizeof(U));
            pipe_->InitBuffer(xFp32Buf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(gradHinBuf_, DOUBLE_BUF * c_ * sizeof(U));
            pipe_->InitBuffer(gradXBuf_, DOUBLE_BUF * c_ * sizeof(U));
            pipe_->InitBuffer(hPreBuf_, DOUBLE_BUF * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(hResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(invRmsBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradH2Buf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));

            int32_t sigmoidGradBufSize = AlignUp(tile_ * 2 * n_ * sizeof(T), VEC_LENGTH);

            pipe_->InitBuffer(sigmoidGradBuf1_, sigmoidGradBufSize);
            pipe_->InitBuffer(sigmoidGradBuf2_, sigmoidGradBufSize);

            pipe_->InitBuffer(biasBuf_, (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(alphaBuf_, (n_ * n_ + 2 * n_) * sizeof(T));

            pipe_->InitBuffer(tmpBuf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(gradInvRmsBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradHResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(fusedGradHPreAndGradHPostBuf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(fusedHPre2AndHPost2Buf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradAlphaBuf_, 8 * sizeof(T));
            pipe_->InitBuffer(gradBiasBuf_, 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradRMSNormBuf_, tile_ * sizeof(T));

            xLocal_ = xBuf_.Get<U>();
            xFp32Local_ = xFp32Buf_.Get<T>();
            gradHinLocal_ = gradHinBuf_.Get<U>();

            fusedGradHPreAndGradHPostLocal_ = fusedGradHPreAndGradHPostBuf_.Get<T>();
            fusedHPre2AndHPost2Local_ = fusedHPre2AndHPost2Buf_.Get<T>();

            biasLocal_ = biasBuf_.Get<T>();
            alphaLocal_ = alphaBuf_.Get<T>();

            gradH2Local_ = gradH2Buf_.Get<T>();
            tmpLocal_ = tmpBuf_.Get<T>();

            gradHResLocal_ = gradHResBuf_.Get<T>();

            invRmsLocal_ = invRmsBuf_.Get<T>();
            gradInvRmsLocal_ = gradInvRmsBuf_.Get<T>();

            gradSigmoidLocal1_ = sigmoidGradBuf1_.Get<T>();
            gradSigmoidLocal2_ = sigmoidGradBuf2_.Get<T>();

            gradBiasLocal_ = gradBiasBuf_.Get<T>();
            gradAlphaLocal_ = gradAlphaBuf_.Get<T>();

            gradRMSNormLocal_ = gradRMSNormBuf_.Get<T>();
            gradXLocal_ = gradXBuf_.Get<U>();

            hPreLocal_ = hPreBuf_.Get<T>();

            hRes2Local_ = hResBuf_.Get<T>();
        }
    }

    __aicore__ inline void GetAlphaAndBias();
    __aicore__ inline void ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradSigmoid(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline T ComputeGradInvRms(const int32_t i);
    __aicore__ inline void ComputeGradRMSNorm(const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradXVector(const T gradRsqrtVal, const T gradRMSNormVal, const int32_t taskOffset,
                                              bool copyNext);
    __aicore__ inline void ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K);
};

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::GetAlphaAndBias()
{
    // HcBase 常驻 UB
    alphaPre_ = alphaGlobal_.GetValue(0);
    alphaPost_ = alphaGlobal_.GetValue(1);
    alphaRes_ = alphaGlobal_.GetValue(2);
    DataCopyPad(biasLocal_, biasGlobal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});

    for (int32_t i = 0; i < n_; i++) {
        alphaLocal_.SetValue(i, alphaPre_);
        alphaLocal_.SetValue(i + n_, alphaPost_);
    }
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradResHHat2(const int32_t taskOffset,
                                                                                 const int32_t tileTaskCount)
{
    // Compute sigmoidGrad
    DataCopyPad(hRes2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_) + 2 * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * n_ * sizeof(T)),
                 static_cast<int64_t>(2 * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * n_ * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(invRmsLocal_, invRmsGlobal_[taskOffset],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * n_));
    uint32_t xThreadCount = n_ * n_;
    uint32_t yThreadCount = thread_num / xThreadCount;

    Duplicate(gradInvRmsLocal_, 0.f, tileTaskCount);
    Simt::VF_CALL<SinkhornKnoppSimt<T>>(
        Simt::Dim3{xThreadCount, yThreadCount}, (__ubuf__ T *)gradH2Local_.GetPhyAddr(),
        (__ubuf__ T *)gradInvRmsLocal_.GetPhyAddr(), (__gm__ T *)gradAlphaGlobal_[2].GetPhyAddr(),
        (__gm__ T *)gradBiasGlobal_[n_ * 2].GetPhyAddr(), (__gm__ T *)skSumGlobal_.GetPhyAddr(),
        (__gm__ T *)skNormGlobal_.GetPhyAddr(), (__ubuf__ T *)gradHResLocal_.GetPhyAddr(),
        (__ubuf__ T *)hRes2Local_.GetPhyAddr(), (__ubuf__ T *)invRmsLocal_.GetPhyAddr(),
        (__ubuf__ T *)biasLocal_[n_ * 2].GetPhyAddr(), (__ubuf__ T *)tmpLocal_.GetPhyAddr(), alphaRes_, totalTasks_,
        taskOffset, tileTaskCount, skIterCount_, n_);
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradSigmoid(const int32_t taskOffset,
                                                                                const int32_t tileTaskCount)
{
    DataCopyPad(fusedHPre2AndHPost2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_)],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(2 * n_ * sizeof(T)),
                 static_cast<int64_t>(n_ * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint16_t repeatTimes = Ceil(tileTaskCount * n_ * 2, FP32_PER_VL);
    ComputeGradSigmoidVf<T>(gradSigmoidLocal1_, gradSigmoidLocal2_, fusedHPre2AndHPost2Local_, invRmsLocal_,
                            alphaLocal_, biasLocal_, repeatTimes, tileTaskCount * n_ * 2);
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradRMSNorm(const int32_t tileTaskCount)
{
    Mul(gradRMSNormLocal_, invRmsLocal_, invRmsLocal_, tileTaskCount);
    Mul(gradRMSNormLocal_, gradRMSNormLocal_, invRmsLocal_, tileTaskCount);
    Muls(gradRMSNormLocal_, gradRMSNormLocal_, -1.f / static_cast<T>(n_ * c_), tileTaskCount);
}

template <typename T, typename U>
__aicore__ inline T MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradInvRms(const int32_t i)
{
    Mul(gradSigmoidLocal1_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_], gradSigmoidLocal1_[i * 2 * n_],
        2 * n_);
    Mul(gradSigmoidLocal2_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_], gradSigmoidLocal2_[i * 2 * n_],
        2 * n_);
    Mul(tmpLocal_[i * 2 * n_], gradSigmoidLocal1_[i * 2 * n_], alphaLocal_, 2 * n_);
    Muls(gradH2Local_[i * (2 * n_ + n_ * n_)], tmpLocal_[i * 2 * n_], invRmsLocal_.GetValue(i), 2 * n_);
    Mul(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], fusedHPre2AndHPost2Local_[i * 2 * n_], 2 * n_);
    ReduceSum<T>(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], 2 * n_);
    T gradRsqrtVal = tmpLocal_.GetValue(i * 2 * n_) + gradInvRmsLocal_.GetValue(i);
    return gradRsqrtVal;
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradXVector(const T gradRsqrtVal,
                                                                                const T gradRMSNormVal,
                                                                                const int32_t taskOffset, bool copyNext)
{
    int8_t ping = 0;

    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    for (int32_t i = 0; i < n_; i++) {
        WaitFlag<HardEvent::MTE3_V>(ping);

        ComputeGradXVf<T, U>(gradXLocal_[ping * c_], xLocal_[i * c_], gradHinLocal_[ping1_ * c_],
                             hPreLocal_[ping1_ * 8 + i], gradRsqrtVal, gradRMSNormVal, c_, c1_);
        Cast(xFp32Local_[ping * c_], xLocal_[i * c_], RoundMode::CAST_NONE, c_);

        SetFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + i * c_], gradXLocal_[ping * c_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0});
        DataCopyPad(xFp32Global_[taskOffset * n_ * c_ + i * c_], xFp32Local_[ping * c_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0});

        SetFlag<HardEvent::MTE3_V>(ping);
        WaitFlag<HardEvent::V_MTE2>(0);

        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);

        if (copyNext) {
            DataCopyPad(xLocal_[i * c_], xGlobal_[(taskOffset + 1) * n_ * c_ + i * c_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
        }

        ping = 1 - ping;
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/
template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::Process()
{
    if ASCEND_IS_AIV {
        if (blkIdx_ == 0) {
            InitOutput<T>(gradPhiGlobal_, (n_ * n_ + 2 * n_) * n_ * c_, 0);
            InitOutput<T>(gradAlphaGlobal_, 3, 0);
            InitOutput<T>(gradBiasGlobal_, n_ * n_ + 2 * n_, 0);
        }
    }
    SyncAll<false>();

    if ASCEND_IS_AIV {
        int8_t ping = 0;
        GetAlphaAndBias();

        for (int32_t taskOffset = blkIdx_ * tile_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tile_) {
            int32_t tileTaskCount = min(tile_, totalTasks_ - taskOffset);

            if (tileTaskCount > 0) {
                ComputeGradResHHat2(taskOffset, tileTaskCount);
                ComputeGradSigmoid(taskOffset, tileTaskCount);
                ComputeGradRMSNorm(tileTaskCount);

                DataCopyPad(fusedGradHPreAndGradHPostLocal_, gradHPostGlobal_[taskOffset * n_],
                            {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                            {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
                DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_],
                            {static_cast<uint16_t>(n_), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0},
                            {false, 0, 0, 0});

                SetFlag<HardEvent::V_MTE2>(0);
                SetFlag<HardEvent::V_MTE2>(1);
                for (int32_t i = 0; i < tileTaskCount; i++) {
                    WaitFlag<HardEvent::V_MTE2>(ping1_);
                    DataCopyPad(hPreLocal_[ping1_ * 8], hPreGlobal_[(taskOffset + i) * n_],
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(gradHinLocal_[ping1_ * c_], gradHinGlobal_[(taskOffset + i) * c_],
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0},
                                {false, 0, 0, 0});

                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);

                    ComputeGradPreVf<T>(fusedGradHPreAndGradHPostLocal_[i * 2 * n_], xLocal_,
                                        gradHinLocal_[ping1_ * c_], n_, c_ / FP32_PER_VL);
                    T gradRsqrtVal = ComputeGradInvRms(i);
                    ComputeGradXVector(gradRsqrtVal, gradRMSNormLocal_.GetValue(i), taskOffset + i,
                                       i != tileTaskCount - 1);
                    SetFlag<HardEvent::V_MTE2>(ping1_);
                    ping1_ = 1 - ping1_;
                }
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], gradH2Local_,
                            {static_cast<uint16_t>(tileTaskCount),
                             static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});

                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(1);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                int32_t dupSize = AlignUp(tile_ * n_ * 2, FP32_PER_REPEAT) - tileTaskCount * 2 * n_;
                Duplicate(gradSigmoidLocal1_[tileTaskCount * 2 * n_], 0.f, dupSize);
                Duplicate(gradSigmoidLocal2_[tileTaskCount * 2 * n_], 0.f, dupSize);
                // compute gradHcScale
                int32_t repeatTimes = Ceil(tileTaskCount * n_ * 2, FP32_PER_REPEAT);
                uint64_t preMask[] = {0X0F0F0F0F0F0F0F0F};
                uint64_t postMask[] = {0XF0F0F0F0F0F0F0F0};
                ReduceSum<T>(gradAlphaLocal_, gradSigmoidLocal2_, tmpLocal_, preMask, repeatTimes, 8);
                ReduceSum<T>(gradAlphaLocal_[1], gradSigmoidLocal2_, tmpLocal_, postMask, repeatTimes, 8);
                uint32_t shape[] = {static_cast<uint32_t>(tileTaskCount), static_cast<uint32_t>(2 * n_)};
                ReduceSum<T, Pattern::Reduce::RA, true>(gradBiasLocal_, gradSigmoidLocal1_,
                                                        tmpLocal_.template ReinterpretCast<uint8_t>(), shape, true);

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                SetAtomicAdd<T>();
                DataCopyPad(gradAlphaGlobal_, gradAlphaLocal_,
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * sizeof(T)), 0, 0, 0});
                DataCopyPad(gradBiasGlobal_, gradBiasLocal_,
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * n_ * sizeof(T)), 0, 0, 0});
                SetAtomicNone();
            }
        }
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int32_t taskOffset = blkIdx_ * 2 * tile_; taskOffset < totalTasksAligned_;
             taskOffset += aicNum_ * 2 * tile_) {
            int32_t tileTaskCount = min(2 * tile_, totalTasks_ - taskOffset);
            CrossCoreWaitFlag<0x2>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
                ProcessMatmul2(taskOffset, tileTaskCount);
            }
        }
    }
}

// mm1K_ = n_ * n_ + 2 * n_;
// mm1M_ = tile_ * 2;
// mm1N_ = n_ * c_;

// mm2K_ = tile_ * 2;
// mm2M_ = n_ * n_ + 2 * n_;
// mm2N_ = n_ * c_;
template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ProcessMatmul1(const int32_t taskOffset,
                                                                            const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(phiGlobal_);
    mm1_.SetHF32(true, 1);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false>(gradXGlobal_[taskOffset * (n_ * c_)], 1);
    mm1_.End();
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ProcessMatmul2(const int32_t taskOffset,
                                                                            const int32_t mm2K)
{
    if (mm2K <= 0)
        return;

    mm2_.SetTensorA(gradH2Global_[taskOffset * mm2M_], true);
    mm2_.SetTensorB(xFp32Global_[taskOffset * mm2N_]);
    mm2_.SetHF32(true, 1);
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K);
    mm2_.template IterateAll<false>(gradPhiGlobal_, 1);
    mm2_.End();
}

#endif // MHC_PRE_SINKHORN_BACKWARD_ONE_STAGE_H

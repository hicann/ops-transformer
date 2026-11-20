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
 * \file mhc_pre_vector_compute.h
 * \brief Shared AIV buffers, data movement, and vector processing for MHC Pre
 */

#ifndef MHC_PRE_VECTOR_COMPUTE_H_
#define MHC_PRE_VECTOR_COMPUTE_H_

#include "mhc_pre_common.h"
#include "mhc_pre_tiling_key.h"

namespace MhcPre {

using namespace AscendC;

template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitBS;
template <class T, class P, int8_t RESI_MODE>
#if defined(MHC_PRE_ENABLE_ASCEND950)
class MhcPreSplitBSWave;
template <class T, class P, int8_t RESI_MODE, bool WEIGHT_REUSE>
#endif
class MhcPreMKPart1;
#if defined(MHC_PRE_ENABLE_ASCEND950)
template <class T, class P, int8_t RESI_MODE, bool WEIGHT_REUSE>
#else
template <class T, class P, int8_t RESI_MODE>
#endif
class MhcPreMKPart2;
#if !defined(MHC_PRE_ENABLE_ASCEND950)
template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitND;
#endif

// Vector cast and division traits shared by all templates.
constexpr Reg::CastTrait MHC_PRE_CAST_FP32_TO_B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr Reg::DivSpecificMode MHC_PRE_DIV_ZEROING_MODE = {Reg::MaskMergeMode::ZEROING, true};

template <class P>
#if defined(MHC_PRE_ENABLE_ASCEND950)
__simd_callee__ inline void MhcPreLoadHInValue(Reg::RegTensor<P>& dst, __ubuf__ P* src, uint32_t offset)
#else
__aicore__ inline void MhcPreLoadHInValue(Reg::RegTensor<P>& dst, __ubuf__ P* src, uint32_t offset)
#endif
{
    Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(dst, src + offset);
}

template <class T, class P>
#if defined(MHC_PRE_ENABLE_ASCEND950)
__simd_callee__ inline void MhcPreAccumulateHIn(Reg::RegTensor<P>& acc, Reg::RegTensor<P>& xFp32,
                                                Reg::RegTensor<T>& xIn, Reg::RegTensor<P>& hPre, __ubuf__ T* x,
                                                uint32_t offset, Reg::MaskReg mask)
#else
__aicore__ inline void MhcPreAccumulateHIn(Reg::RegTensor<P>& acc, Reg::RegTensor<P>& xFp32, Reg::RegTensor<T>& xIn,
                                           Reg::RegTensor<P>& hPre, __ubuf__ T* x, uint32_t offset, Reg::MaskReg mask)
#endif
{
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xIn, x + offset);
    Reg::Cast<P, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32, xIn, mask);
#if defined(MHC_PRE_ENABLE_ASCEND950)
    Reg::MulAddDst(acc, xFp32, hPre, mask);
}

template <class P, uint32_t HEAD_NUM, bool HAS_RESI_VALUE>
__simd_vf__ inline void MhcPreVFPostSegmentsImpl(uint16_t rowCount, uint32_t invRmsOffset, uint32_t mmRowStride,
                                                 __ubuf__ P* hPre, __ubuf__ P* hPost, __ubuf__ P* mm,
                                                 __ubuf__ P* invRms, __ubuf__ P* alpha, __ubuf__ P* bias,
                                                 __ubuf__ P* hRes)
{
    Reg::RegTensor<P> dataReg;
    Reg::RegTensor<P> invRmsReg;
    Reg::RegTensor<P> alphaPreReg;
    Reg::RegTensor<P> alphaPostReg;
    Reg::RegTensor<P> alphaResReg;
    Reg::RegTensor<P> biasPreReg;
    Reg::RegTensor<P> biasPostReg;
    Reg::RegTensor<P> biasResReg;
    uint32_t nMaskSize = HEAD_NUM;
    uint32_t resMaskSize = HEAD_NUM * HEAD_NUM;
    Reg::MaskReg nMask = Reg::UpdateMask<P>(nMaskSize);
    Reg::MaskReg resMask = Reg::UpdateMask<P>(resMaskSize);

    Reg::RegTensor<uint32_t> indices;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
    Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(alphaPreReg, alpha);
    Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(alphaPostReg, alpha + HEAD_NUM);
    Reg::Gather(biasPreReg, bias, indices, nMask);
    Reg::Gather(biasPostReg, bias + HEAD_NUM, indices, nMask);
    if constexpr (HAS_RESI_VALUE) {
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(alphaResReg, alpha + 2U * HEAD_NUM);
        Reg::Gather(biasResReg, bias + 2U * HEAD_NUM, indices, resMask);
    }

    // Consume pre/post/res by fixed UB offsets in one VF loop. This replaces Gather and stages
    // every output at a 32-byte-aligned local-buffer base before DMA.
    for (uint16_t row = 0; row < rowCount; ++row) {
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invRmsReg, invRms + invRmsOffset + row);
        uint32_t mmOffset = static_cast<uint32_t>(row) * mmRowStride;
        Reg::Gather(dataReg, mm + mmOffset, indices, nMask);
        Reg::Mul(dataReg, dataReg, invRmsReg, nMask);
        {
            Reg::RegTensor<P> affineReg;
            affineReg = biasPreReg;
            Reg::MulAddDst(affineReg, dataReg, alphaPreReg, nMask);
            dataReg = affineReg;
        }
        Reg::Store<P>(hPre + row * HEAD_NUM, dataReg, HEAD_NUM);

        Reg::Gather(dataReg, mm + mmOffset + HEAD_NUM, indices, nMask);
        Reg::Mul(dataReg, dataReg, invRmsReg, nMask);
        {
            Reg::RegTensor<P> affineReg;
            affineReg = biasPostReg;
            Reg::MulAddDst(affineReg, dataReg, alphaPostReg, nMask);
            dataReg = affineReg;
        }
        Reg::Store<P>(hPost + row * HEAD_NUM, dataReg, HEAD_NUM);

        if constexpr (HAS_RESI_VALUE) {
            Reg::Gather(dataReg, mm + mmOffset + 2U * HEAD_NUM, indices, resMask);
            Reg::Mul(dataReg, dataReg, invRmsReg, resMask);
            {
                Reg::RegTensor<P> affineReg;
                affineReg = biasResReg;
                Reg::MulAddDst(affineReg, dataReg, alphaResReg, resMask);
                dataReg = affineReg;
            }
            Reg::Store<P>(hRes + row * HEAD_NUM * HEAD_NUM, dataReg, HEAD_NUM * HEAD_NUM);
        }
    }
#else
    Reg::Mul(xFp32, xFp32, hPre, mask);
    Reg::Add(acc, acc, xFp32, mask);
#endif
}

template <class P, uint32_t HEAD_NUM, bool HAS_RESI_VALUE>
__aicore__ inline void MhcPreVFPostSegments(const LocalTensor<P>& hPreLocal, const LocalTensor<P>& hPostLocal,
                                            const LocalTensor<P>& hResLocal, const LocalTensor<P>& mmLocal,
                                            const LocalTensor<P>& invRmsLocal, const LocalTensor<P>& alphaLocal,
                                            const LocalTensor<P>& biasLocal, uint16_t rowCount, uint32_t invRmsOffset,
                                            uint32_t mmRowStride)
{
    __ubuf__ P* hPre = (__ubuf__ P*)hPreLocal.GetPhyAddr();
    __ubuf__ P* hPost = (__ubuf__ P*)hPostLocal.GetPhyAddr();
    __ubuf__ P* mm = (__ubuf__ P*)mmLocal.GetPhyAddr();
    __ubuf__ P* invRms = (__ubuf__ P*)invRmsLocal.GetPhyAddr();
    __ubuf__ P* alpha = (__ubuf__ P*)alphaLocal.GetPhyAddr();
    __ubuf__ P* bias = (__ubuf__ P*)biasLocal.GetPhyAddr();
    __ubuf__ P* hRes = nullptr;
    if constexpr (HAS_RESI_VALUE) {
        hRes = (__ubuf__ P*)hResLocal.GetPhyAddr();
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    asc_vf_call<MhcPreVFPostSegmentsImpl<P, HEAD_NUM, HAS_RESI_VALUE>>(rowCount, invRmsOffset, mmRowStride, hPre, hPost,
                                                                       mm, invRms, alpha, bias, hRes);
#else
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> dataReg;
        Reg::RegTensor<P> invRmsReg;
        Reg::RegTensor<P> invRmsBroadReg;
        Reg::RegTensor<P> alphaPreReg;
        Reg::RegTensor<P> alphaPostReg;
        Reg::RegTensor<P> alphaResReg;
        Reg::RegTensor<P> biasPreReg;
        Reg::RegTensor<P> biasPostReg;
        Reg::RegTensor<P> biasResReg;
        uint32_t nMaskSize = HEAD_NUM;
        uint32_t resMaskSize = HEAD_NUM * HEAD_NUM;
        Reg::MaskReg nMask = Reg::UpdateMask<P>(nMaskSize);
        Reg::MaskReg resMask = Reg::UpdateMask<P>(resMaskSize);

        Reg::Load<P>(alphaPreReg, alpha);
        Reg::Load<P>(alphaPostReg, alpha + HEAD_NUM);
        Reg::Load<P>(biasPreReg, bias);
        Reg::Load<P>(biasPostReg, bias + HEAD_NUM);
        if constexpr (HAS_RESI_VALUE) {
            Reg::Load<P>(alphaResReg, alpha + 2U * HEAD_NUM);
            Reg::Load<P>(biasResReg, bias + 2U * HEAD_NUM);
        }

        // Consume pre/post/res by fixed UB offsets in one VF loop. This replaces Gather and stages
        // every output at a 32-byte-aligned local-buffer base before DMA.
        for (uint16_t row = 0; row < rowCount; ++row) {
            Reg::Load<P>(invRmsReg, invRms + invRmsOffset + row);
            Reg::Duplicate(invRmsBroadReg, invRmsReg, nMask);
            uint32_t mmOffset = static_cast<uint32_t>(row) * mmRowStride;
            Reg::Load<P>(dataReg, mm + mmOffset);
            Reg::Mul(dataReg, dataReg, invRmsBroadReg, nMask);
            Reg::Mul(dataReg, dataReg, alphaPreReg, nMask);
            Reg::Add(dataReg, dataReg, biasPreReg, nMask);
            Reg::Store<P>(hPre + row * HEAD_NUM, dataReg, HEAD_NUM);

            Reg::Load<P>(dataReg, mm + mmOffset + HEAD_NUM);
            Reg::Mul(dataReg, dataReg, invRmsBroadReg, nMask);
            Reg::Mul(dataReg, dataReg, alphaPostReg, nMask);
            Reg::Add(dataReg, dataReg, biasPostReg, nMask);
            Reg::Store<P>(hPost + row * HEAD_NUM, dataReg, HEAD_NUM);

            if constexpr (HAS_RESI_VALUE) {
                Reg::Duplicate(invRmsBroadReg, invRmsReg, resMask);
                Reg::Load<P>(dataReg, mm + mmOffset + 2U * HEAD_NUM);
                Reg::Mul(dataReg, dataReg, invRmsBroadReg, resMask);
                Reg::Mul(dataReg, dataReg, alphaResReg, resMask);
                Reg::Add(dataReg, dataReg, biasResReg, resMask);
                Reg::Store<P>(hRes + row * HEAD_NUM * HEAD_NUM, dataReg, HEAD_NUM * HEAD_NUM);
            }
        }
    }
#endif
}
template <class P, uint32_t HEAD_NUM, bool HAS_RESI_VALUE>
#if defined(MHC_PRE_ENABLE_ASCEND950)
__simd_vf__ inline void MhcPreVFExpandAlphaImpl(__ubuf__ P* alpha)
#else
__aicore__ inline void MhcPreVFExpandAlpha(const LocalTensor<P>& alphaLocal)
{
    __ubuf__ P* alpha = (__ubuf__ P*)alphaLocal.GetPhyAddr();
    __VEC_SCOPE__
#endif
{
    Reg::RegTensor<P> alphaPreReg;
    Reg::RegTensor<P> alphaPostReg;
    Reg::RegTensor<P> alphaResReg;
    Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(alphaPreReg, alpha);
    Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(alphaPostReg, alpha + 1U);
    if constexpr (HAS_RESI_VALUE) {
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(alphaResReg, alpha + 2U);
    }
    Reg::Store<P>(alpha, alphaPreReg, HEAD_NUM);
    Reg::Store<P>(alpha + HEAD_NUM, alphaPostReg, HEAD_NUM);
    if constexpr (HAS_RESI_VALUE) {
        Reg::Store<P>(alpha + 2U * HEAD_NUM, alphaResReg, HEAD_NUM * HEAD_NUM);
    }
}

#if defined(MHC_PRE_ENABLE_ASCEND950)
template <class P, uint32_t HEAD_NUM, bool HAS_RESI_VALUE>
__aicore__ inline void MhcPreVFExpandAlpha(const LocalTensor<P>& alphaLocal)
{
    __ubuf__ P* alpha = (__ubuf__ P*)alphaLocal.GetPhyAddr();
    asc_vf_call<MhcPreVFExpandAlphaImpl<P, HEAD_NUM, HAS_RESI_VALUE>>(alpha);
#endif
}
template <class T, class P, int8_t RESI_MODE>
class MhcPreVectorCompute {
public:
    static_assert(RESI_MODE == MHC_PRE_HAS_RESI || RESI_MODE == MHC_PRE_NO_RESI, "Unsupported MhcPre residual mode");
    // Buffering and supported head-count constants.
    static constexpr uint32_t kDoubleBufferCount = 2;
    static constexpr uint32_t kSingleBufferCount = 1;
    static constexpr uint32_t kSupportedN4 = 4;
    static constexpr uint32_t kSupportedN6 = 6;
    static constexpr uint32_t kSupportedN8 = 8;

    // Common vector arithmetic constants.
    static constexpr uint32_t kBlockLenSingle = 1;
    static constexpr uint32_t kAlignmentBytes = 32;
    static constexpr uint32_t kHalfSplitDivisor = 2;
    static constexpr float kOneValue = 1.0f;
    static constexpr float kTwoValue = 2.0f;

    // Shared BS vector tile sizes.
#if defined(MHC_PRE_ENABLE_ASCEND950)
    static constexpr uint32_t ND_LENGTH = MHC_PRE_RMS_K_BLOCK;
#else
    static constexpr uint32_t PARALLEL_NUM = 2;
    static constexpr uint32_t ND_LENGTH = 256;
#endif
    static constexpr uint32_t V0_BASE_T = 16;
    static constexpr uint64_t V1_BASE_T = 16;
    static constexpr uint64_t V1_BASE_D = 32;

    // Shared AIV buffer sizes.
#if defined(MHC_PRE_ENABLE_ASCEND950)
    static constexpr uint32_t kXInQueueBufferBytes = MHC_PRE_MK_PART2_INPUT_BYTES;
    static constexpr uint32_t kOutQueueBufferBytes = MHC_PRE_MK_PART2_OUTPUT_BYTES;
    static constexpr uint32_t kTmpBufferBytes = MHC_PRE_MK_PART2_TEMP_BYTES;
#else
    static constexpr uint32_t kXInQueueBufferBytes = 80U * 1024U;
    static constexpr uint32_t kOutQueueBufferBytes = 20U * 1024U;
    static constexpr uint32_t kTmpBufferBytes = 40U * 1024U;
#endif
    static constexpr int8_t kResiMode = RESI_MODE;

    __aicore__ inline MhcPreVectorCompute() = default;

    __aicore__ inline void InitPipeAndCoreIdx(TPipe* pipe)
    {
        pipe_ = pipe;
        coreIdx_ = GetBlockIdx();
        subBlockIdx_ = GetSubBlockIdx();
    }

    __aicore__ inline void BindGlobalTensors(InitParams initParams)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.bias));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.gamma));
        hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.hin));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_post));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_res));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.inv_rms));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_pre));
    }
#else
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.bias));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.gamma));
        hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initParams.hin));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_post));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_res));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.inv_rms));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_pre));
        xFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.workspace));
    }
#endif

    __aicore__ inline void InitFromTilingData(const MhcPreTilingData* tilingData)
    {
        tiling_ = tilingData;
        matrixInfo_.totalLength = tiling_->totalLength;
        matrixInfo_.nD = tiling_->nD;
        N_ = tiling_->N;
        D_ = tiling_->D;
        matrixInfo_.fusionSize = tiling_->fusionSize;
        matrixInfo_.normEps = tiling_->normEps;
        matrixInfo_.hcEps = tiling_->hcEps;
        coreNum_ = tiling_->coreNum;
        totalLength_ = tiling_->totalLength;
        outFlag_ = (tiling_->outFlag != 0);
        scaleMean_ = tiling_->scaleMean;
        chunkTSize_ = tiling_->chunkTSize;
        v1ChunkDSize_ = tiling_->v1ChunkDSize;
        hasGamma_ = (tiling_->hasGamma != 0);
        hasResi_ = (tiling_->hasResi != 0);
        implMode_ = tiling_->implMode;
        eleNumPerVf_ = MhcPreGetVRegSize() / sizeof(P);
    }

    __aicore__ inline void InitMNConfig()
    {
        mnConfig_.m = matrixInfo_.totalLength;
        mnConfig_.n = matrixInfo_.fusionSize;
        mnConfig_.k = matrixInfo_.nD;
        mnConfig_.singleCoreM = chunkTSize_;
        mnConfig_.singleCoreN = mnConfig_.n;
        mnConfig_.singleCoreK = ND_LENGTH;
        mnConfig_.curSingleCoreM = mnConfig_.singleCoreM;
        mnConfig_.curSingleCoreN = mnConfig_.singleCoreN;
        mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
        curSingleT_ = chunkTSize_;
    }

    __aicore__ inline void InitHMixBuffer(InitParams initParams)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        if (outFlag_) {
            hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_mix));
        }
    }

    __aicore__ inline void InitPart2UbBuffers()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        // Double-buffer X and general output queues to overlap MTE2, Vector, and MTE3.
        pipe_->InitBuffer(xInQueue_, kDoubleBufferCount, kXInQueueBufferBytes);
        pipe_->InitBuffer(outQueue_, kDoubleBufferCount, kOutQueueBufferBytes);
        pipe_->InitBuffer(invRmsOutQueue_, kSingleBufferCount, MHC_PRE_MK_PART2_ROWS * sizeof(P));

        if (hasGamma_) {
            pipe_->InitBuffer(gammaInQueue_, kSingleBufferCount, ND_LENGTH * sizeof(P));
        }

        pipe_->InitBuffer(tmpBuff_, kTmpBufferBytes);
        uint32_t parameterBufferBytes = MhcPreAlign(mnConfig_.n * sizeof(P), kAlignmentBytes);
        pipe_->InitBuffer(biasInQue_, kSingleBufferCount, parameterBufferBytes);
        pipe_->InitBuffer(alphaBuf_, parameterBufferBytes);
        alphaInUb_ = alphaBuf_.template Get<P>();

        uint64_t buffOffset = 0;
        hPreBuff_ = tmpBuff_.template GetWithOffset<P>(MHC_PRE_MK_PART2_ROWS * N_, buffOffset);
        buffOffset += MHC_PRE_MK_PART2_ROWS * N_ * sizeof(P);
        hPostBuff_ = tmpBuff_.template GetWithOffset<P>(MHC_PRE_MK_PART2_ROWS * N_, buffOffset);
    }
#else
    {
        if (outFlag_) {
            hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_mix));
        } else {
            hMixGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P*>(initParams.workspace + mnConfig_.singleCoreM * mnConfig_.singleCoreK *
                                                                       PARALLEL_NUM * sizeof(P) * coreNum_));
        }
    }

    __aicore__ inline void InitUbBuffers(bool splitAcrossSubBlocks = true, uint32_t v1BufferRows = V1_BASE_T)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        // Double-buffer X and general output queues to overlap MTE2, Vector, and MTE3.
        pipe_->InitBuffer(xInQueue_, kDoubleBufferCount, kXInQueueBufferBytes);
        pipe_->InitBuffer(outQueue_, kDoubleBufferCount, kOutQueueBufferBytes);
        uint32_t invRmsRows =
            splitAcrossSubBlocks ? (curSingleT_ + kHalfSplitDivisor - 1U) / kHalfSplitDivisor : curSingleT_;
        uint32_t invRmsBytes = MhcPreAlign(invRmsRows, kAlignmentBytes / sizeof(P)) * sizeof(P);
        pipe_->InitBuffer(invRmsOutQueue_, kSingleBufferCount, invRmsBytes);

        if (hasGamma_) {
            pipe_->InitBuffer(gammaInQueue_, kSingleBufferCount, ND_LENGTH * sizeof(P));
        }

        pipe_->InitBuffer(tmpBuff_, kTmpBufferBytes);
        uint32_t parameterBufferBytes = MhcPreAlign(mnConfig_.n * sizeof(P), kAlignmentBytes);
        pipe_->InitBuffer(biasInQue_, kSingleBufferCount, parameterBufferBytes);
        pipe_->InitBuffer(alphaBuf_, parameterBufferBytes);
        alphaInUb_ = alphaBuf_.template Get<P>();

        uint64_t buffOffset = 0;
        hPreBuff_ = tmpBuff_.template GetWithOffset<P>(v1BufferRows * N_, buffOffset);
        buffOffset += v1BufferRows * N_ * sizeof(P);
        hPostBuff_ = tmpBuff_.template GetWithOffset<P>(v1BufferRows * N_, buffOffset);
    }
#endif

    __aicore__ inline void InitBlockParams(uint64_t curblock, uint32_t tBlockNum)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        globalOffsetM_ = curblock * chunkTSize_;
        curSingleT_ = chunkTSize_;
        if (curblock == tBlockNum - 1U) {
            mnConfig_.curSingleCoreM = totalLength_ - globalOffsetM_;
            curSingleT_ = matrixInfo_.totalLength - curblock * chunkTSize_;
        }
        if ASCEND_IS_AIV {
            VectorComputeOffset();
        }
    }
#else
    {
        globalOffsetM_ = curblock * chunkTSize_;
        curSingleT_ = chunkTSize_;
        if (curblock == tBlockNum - 1U) {
            mnConfig_.curSingleCoreM = totalLength_ - globalOffsetM_;
            curSingleT_ = matrixInfo_.totalLength - curblock * chunkTSize_;
        }
        if ASCEND_IS_AIV {
            VectorComputeOffset();
        }
    }

    __aicore__ inline void V0Prologue(uint32_t curNdLen, uint32_t offsetNd)
    {
        if (hasGamma_) {
            gammaUb_ = gammaInQueue_.template AllocTensor<P>();
            DataCopyGamma(curNdLen, offsetNd);
            gammaUb_ = gammaInQueue_.template DeQue<P>();
        }

        for (uint32_t offsetM = vectorOffset_.offsetMStart; offsetM < vectorOffset_.offsetMEnd; offsetM += V0_BASE_T) {
            uint32_t curMLen = V0_BASE_T;
            if (offsetM + V0_BASE_T >= vectorOffset_.offsetMEnd) {
                curMLen = vectorOffset_.offsetMEnd - offsetM;
            }
            uint64_t invRmsOffset = offsetM - vectorOffset_.offsetMStart;
            LocalTensor<P> invRmsUb = invRmsUb_[invRmsOffset];
            xLocal_ = xInQueue_.template AllocTensor<T>();
            DataCopyX(curMLen, curNdLen, offsetM, offsetNd);
            xLocal_ = xInQueue_.template DeQue<T>();

            LocalTensor<P> aL1Ub = outQueue_.template AllocTensor<P>();
            if (hasGamma_) {
                if (offsetNd == 0U) {
                    VFDoV0ProcessXIn<true, true>((__ubuf__ P*)aL1Ub.GetPhyAddr(), (__ubuf__ P*)invRmsUb.GetPhyAddr(),
                                                 (__ubuf__ T*)xLocal_.GetPhyAddr(), (__ubuf__ P*)gammaUb_.GetPhyAddr(),
                                                 curMLen, curNdLen);
                } else {
                    VFDoV0ProcessXIn<true, false>((__ubuf__ P*)aL1Ub.GetPhyAddr(), (__ubuf__ P*)invRmsUb.GetPhyAddr(),
                                                  (__ubuf__ T*)xLocal_.GetPhyAddr(), (__ubuf__ P*)gammaUb_.GetPhyAddr(),
                                                  curMLen, curNdLen);
                }
            } else if (offsetNd == 0U) {
                VFDoV0ProcessXIn<false, true>((__ubuf__ P*)aL1Ub.GetPhyAddr(), (__ubuf__ P*)invRmsUb.GetPhyAddr(),
                                              (__ubuf__ T*)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            } else {
                VFDoV0ProcessXIn<false, false>((__ubuf__ P*)aL1Ub.GetPhyAddr(), (__ubuf__ P*)invRmsUb.GetPhyAddr(),
                                               (__ubuf__ T*)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            }
            outQueue_.template EnQue<P>(aL1Ub);
            aL1Ub = outQueue_.template DeQue<P>();
            DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd);
            xInQueue_.FreeTensor(xLocal_);
            outQueue_.FreeTensor(aL1Ub);
        }

        if (hasGamma_) {
            gammaInQueue_.FreeTensor(gammaUb_);
        }
    }
#endif

    __aicore__ inline void V0PostProcess(uint32_t curblock, uint32_t tBlockNum)
    {
        if (vectorOffset_.singleCoreM == 0U) {
            return;
        }
        VFDoV0ProcessInvRms((__ubuf__ P*)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_,
                            matrixInfo_.normEps);
        DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
    }

    __aicore__ inline void VectorComputeOffset()
    {
        uint64_t alignSingleM = Ceil(curSingleT_ / 2, 8) * 8;
        vectorOffset_.singleCoreM = alignSingleM < curSingleT_ ? alignSingleM : curSingleT_;
        if (subBlockIdx_ == 0) {
            vectorOffset_.offsetMStart = 0;
            vectorOffset_.offsetMEnd = vectorOffset_.singleCoreM;
        } else {
            vectorOffset_.offsetMStart = vectorOffset_.singleCoreM;
            vectorOffset_.singleCoreM = curSingleT_ - vectorOffset_.singleCoreM;
            vectorOffset_.offsetMEnd = curSingleT_;
        }
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    template <bool hasGamma, bool isFirstND, bool OUTPUT_NZ, bool COMPUTE_RMS, bool COMPENSATE_RMS>
    static __simd_vf__ inline void VFDoV0ProcessXInImpl(__ubuf__ P* xDst, __ubuf__ P* invRmsDst, __ubuf__ T* xIn,
                                                        __ubuf__ P* gamma, uint16_t mSize, uint16_t nSize,
                                                        uint16_t nzRows, uint32_t nSrcUbAligned, uint32_t nDstUbAligned,
                                                        uint16_t nLoopCnt, uint32_t elemsPerVf, __ubuf__ P* correction)
    {
        Reg::MaskReg full = Reg::CreateMask<P>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            __ubuf__ P* nzDst = xDst + mIdx * MHC_PRE_BASIC_API_C0_SIZE;
            uint32_t elementNum = nSize;
            Reg::RegTensor<P> squareLanes;
            if constexpr (COMPUTE_RMS) {
                Reg::Duplicate(squareLanes, 0.0f);
            }
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                Reg::RegTensor<T> xInReg;
                Reg::RegTensor<P> gammaReg;
                Reg::RegTensor<P> xFp32Reg;
                Reg::RegTensor<P> xMulReg;

                uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * elemsPerVf;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                Reg::MaskReg maskN4B32 = Reg::UpdateMask<P>(elementNum);
                Reg::Cast<float, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32Reg, xInReg, maskN4B32);
                if constexpr (hasGamma) {
                    Reg::LoadAlign(gammaReg, gamma + vfBlockIdx * elemsPerVf);
                    Reg::Mul(xMulReg, gammaReg, xFp32Reg, maskN4B32);
                } else {
                    xMulReg = xFp32Reg;
                }
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * elemsPerVf;
                if constexpr (OUTPUT_NZ) {
                    Reg::StoreAlign<P, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        nzDst, xMulReg, nzRows, 8U * nzRows, maskN4B32);
                } else {
                    Reg::StoreAlign(xDst + dstUbOffset, xMulReg, maskN4B32);
                }

                if constexpr (COMPUTE_RMS) {
                    MhcPreAccumulateRmsLanes(squareLanes, xFp32Reg, maskN4B32, full);
                }
            }
            if constexpr (COMPUTE_RMS) {
                Reg::RegTensor<P> sum;
                Reg::Reduce<Reg::ReduceType::SUM>(sum, squareLanes, full);
                if constexpr (COMPENSATE_RMS) {
                    Reg::RegTensor<P> error;
                    if constexpr (isFirstND) {
                        Reg::Duplicate(error, 0.0f);
                    } else {
                        Reg::RegTensor<P> previous;
                        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(previous, invRmsDst + mIdx);
                        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(error, correction + mIdx);
                        MhcPreCompensatedAdd(previous, error, sum, full);
                        sum = previous;
                    }
                    Reg::Store(correction + mIdx, error, 1U);
                } else if constexpr (!isFirstND) {
                    Reg::RegTensor<P> previous;
                    Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(previous, invRmsDst + mIdx);
                    Reg::Add(sum, previous, sum, full);
                }
                Reg::Store(invRmsDst + mIdx, sum, 1U);
            }
        }
    }

    template <bool hasGamma, bool isFirstND, bool OUTPUT_NZ = false, bool COMPUTE_RMS = true,
              bool COMPENSATE_RMS = false>
    __aicore__ inline void VFDoV0ProcessXIn(__ubuf__ P* xDst, __ubuf__ P* invRmsDst, __ubuf__ T* xIn, __ubuf__ P* gamma,
                                            uint16_t mSize, uint16_t nSize, uint16_t nzPadding = 0U,
                                            __ubuf__ P* correction = nullptr)
    {
        uint32_t nSrcUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(T)));
        uint32_t nDstUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreCeilDiv(nSize, eleNumPerVf_);
        uint16_t nzRows = BasicApiAlign(mSize, MHC_PRE_BASIC_API_C0_SIZE) + nzPadding;
        asc_vf_call<VFDoV0ProcessXInImpl<hasGamma, isFirstND, OUTPUT_NZ, COMPUTE_RMS, COMPENSATE_RMS>>(
            xDst, invRmsDst, xIn, gamma, mSize, nSize, nzRows, nSrcUbAligned, nDstUbAligned, nLoopCnt, eleNumPerVf_,
            correction);
    }

    static __simd_vf__ inline void VFDoV0ProcessInvRmsImpl(__ubuf__ P* invRms, uint16_t nSize, float scaleMean,
                                                           float normEps, uint16_t nLoopCnt, uint32_t elemsPerVf)
    {
        Reg::RegTensor<uint32_t> indices;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
        uint32_t elementNum = nSize;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            uint32_t count = AscendC::Std::min(elementNum, elemsPerVf);
            Reg::MaskReg maskN4B32 = Reg::UpdateMask<P>(elementNum);
            Reg::RegTensor<P> invRmsReg;
            auto src = invRms + vfBlockIdx * elemsPerVf;
            if (count == elemsPerVf) {
                Reg::Load(invRmsReg, src);
            } else {
                Reg::Gather(invRmsReg, src, indices, maskN4B32);
            }
            MhcPreNormalizeRms(invRmsReg, maskN4B32, scaleMean, normEps);
            Reg::Store<P>(invRms + vfBlockIdx * elemsPerVf, invRmsReg, count);
        }
    }
#else
    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXIn(__ubuf__ P* xDst, __ubuf__ P* invRmsDst, __ubuf__ T* xIn, __ubuf__ P* gamma,
                                            uint16_t mSize, uint16_t nSize)
    {
        uint32_t nSrcUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(T)));
        uint32_t nDstUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreCeilDiv(nSize, eleNumPerVf_);
        __VEC_SCOPE__
        {
            Reg::MaskReg mask = Reg::CreateMask<P>();
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
                uint32_t elementNum = nSize;
                Reg::RegTensor<P> sumReg;
                if constexpr (isFirstND) {
                    Reg::Duplicate(sumReg, 0);
                } else {
                    Reg::Load(sumReg, invRmsDst + mIdx);
                }
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                    Reg::RegTensor<T> xInReg;
                    Reg::RegTensor<P> gammaReg;
                    Reg::RegTensor<P> xFp32Reg, xMulReg, xSquaReg;
                    Reg::RegTensor<P> tmpSumReg;

                    uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf_;
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                    Reg::MaskReg maskN4B32 = Reg::UpdateMask<P>(elementNum);
                    Reg::Cast<float, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32Reg, xInReg, maskN4B32);
                    if constexpr (hasGamma) {
                        Reg::LoadAlign(gammaReg, gamma + vfBlockIdx * eleNumPerVf_);
                        Reg::Mul(xMulReg, gammaReg, xFp32Reg, maskN4B32);
                    } else {
                        xMulReg = xFp32Reg;
                    }
                    uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf_;
                    Reg::StoreAlign(xDst + dstUbOffset, xMulReg, maskN4B32);

                    Reg::Mul(xSquaReg, xFp32Reg, xFp32Reg, maskN4B32);
                    Reg::Reduce<Reg::ReduceType::SUM>(tmpSumReg, xSquaReg, maskN4B32);
                    Reg::Add(sumReg, sumReg, tmpSumReg, maskN4B32);
                }

                Reg::Store(invRmsDst + mIdx, sumReg, 1);
            }
        }
    }
#endif

    __aicore__ inline void VFDoV0ProcessInvRms(__ubuf__ P* invRms, uint16_t nSize, float scaleMean, float normEps)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        uint16_t nLoopCnt = MhcPreCeilDiv(nSize, eleNumPerVf_);
        asc_vf_call<VFDoV0ProcessInvRmsImpl>(invRms, nSize, scaleMean, normEps, nLoopCnt, eleNumPerVf_);
    }

    // The MK post owner scans each complete input row. Cube can cut K anywhere,
    // including inside a square-sum block on the short-K or weight-reuse path.
    __aicore__ inline void ComputeFullRowRms(const LocalTensor<P>& rms, const LocalTensor<P>& scratch,
                                             uint32_t scratchBytes, uint64_t rowStart, uint32_t rows)
    {
        constexpr uint32_t kCopyColumns = 1024U;
        auto x = scratch.template ReinterpretCast<T>();
        // Retain each row's low rounding error across 1024-column DMA chunks.
        uint32_t correctionBytes = BasicApiRoundUp<P>(rows) * sizeof(P);
        auto correction = (__ubuf__ P*)scratch[(scratchBytes - correctionBytes) / sizeof(P)].GetPhyAddr();
        uint32_t kStride = MhcPreAlign(MhcPreMin(kCopyColumns, matrixInfo_.nD), MHC_PRE_BASIC_API_VL_FP32);
        uint32_t rowsPerCopy =
            MhcPreMin(rows, (scratchBytes - correctionBytes) / (kStride * static_cast<uint32_t>(sizeof(T))));
        auto ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        auto readDone = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        for (uint32_t row = 0; row < rows; row += rowsPerCopy) {
            uint32_t count = MhcPreMin(rowsPerCopy, rows - row);
            for (uint32_t k = 0; k < matrixInfo_.nD; k += kCopyColumns) {
                uint32_t columns = MhcPreMin(kCopyColumns, matrixInfo_.nD - k);
                uint32_t xStride = BasicApiRoundUp<T>(columns);
                DataCopyExtParams copy{static_cast<uint16_t>(count), columns * static_cast<uint32_t>(sizeof(T)),
                                       (matrixInfo_.nD - columns) * static_cast<uint32_t>(sizeof(T)), 0U, 0U};
                DataCopyPadExtParams<T> pad{true, 0U, static_cast<uint8_t>(xStride - columns), 0U};
                DataCopyPad(x, xGm_[(rowStart + row) * matrixInfo_.nD + k], copy, pad);
                SetFlag<HardEvent::MTE2_V>(ready);
                WaitFlag<HardEvent::MTE2_V>(ready);
                auto rmsAddr = (__ubuf__ float*)rms[row].GetPhyAddr();
                auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
                if (k == 0U) {
                    asc_vf_call<MhcPreVFAccumulateRms<T, true>>(rmsAddr, correction + row, xAddr, count, columns,
                                                                xStride);
                } else {
                    asc_vf_call<MhcPreVFAccumulateRms<T, false>>(rmsAddr, correction + row, xAddr, count, columns,
                                                                 xStride);
                }
                SetFlag<HardEvent::V_MTE2>(readDone);
                WaitFlag<HardEvent::V_MTE2>(readDone);
            }
        }
        PipeBarrier<PIPE_V>();
        VFDoV0ProcessInvRms((__ubuf__ P*)rms.GetPhyAddr(), rows, scaleMean_, matrixInfo_.normEps);
    }
#else
    {
        uint32_t nUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreCeilDiv(nSize, eleNumPerVf_);
        __VEC_SCOPE__
        {
            uint32_t elementNum = nSize;
            Reg::MaskReg mask = Reg::CreateMask<P>();
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                Reg::MaskReg maskN4B32 = Reg::UpdateMask<P>(elementNum);
                Reg::RegTensor<P> invrmsReg, onesReg;

                Reg::LoadAlign(invrmsReg, invRms + vfBlockIdx * eleNumPerVf_);
                Reg::Muls(invrmsReg, invrmsReg, scaleMean, maskN4B32);
                Reg::Adds(invrmsReg, invrmsReg, normEps, maskN4B32);
                Reg::Sqrt(invrmsReg, invrmsReg, maskN4B32);
                Reg::Duplicate(onesReg, 1);
                Reg::Div(invrmsReg, onesReg, invrmsReg, maskN4B32);
                Reg::StoreAlign(invRms + vfBlockIdx * eleNumPerVf_, invrmsReg, maskN4B32);
            }
        }
    }
#endif

    __aicore__ inline void DataCopyX(uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curMLen);
        copyParams.blockLen = uint32_t(curNdLen * sizeof(T));
        copyParams.srcStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(T));
        copyParams.dstStride = uint32_t(0);

        uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

        uint64_t offset = globalOffsetM_ * matrixInfo_.nD + offsetM * matrixInfo_.nD + offsetNd;
        DataCopyPad(xLocal_, xGm_[offset], copyParams, padParams);
        xInQueue_.EnQue<T>(xLocal_);
    }

    __aicore__ inline void DataCopyGamma(uint32_t curNdLen, uint32_t offsetNd)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);

        uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
        DataCopyPadExtParams<P> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

        uint64_t offset = offsetNd;

        DataCopyPad(gammaUb_, gammaGm_[offset], copyParams, padParams);
        gammaInQueue_.EnQue<P>(gammaUb_);
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline LocalTensor<P> LoadGammaTile(uint32_t offsetK)
    {
        gammaUb_ = gammaInQueue_.template AllocTensor<P>();
        DataCopyGamma(MHC_PRE_RMS_K_BLOCK, offsetK);
        return gammaInQueue_.template DeQue<P>();
    }
#else
    __aicore__ inline void HMixCopyIn(uint64_t offset, uint64_t lenT)
    {
        LocalTensor<P> hMixLocal = xInQueue_.AllocTensor<P>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(lenT * mnConfig_.n * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

        DataCopyPad(hMixLocal, hMixGm_[offset], copyParams, copyPadParams);

        xInQueue_.EnQue(hMixLocal);
    }
#endif

    __aicore__ inline void DataCopyOutInvRmsUb(uint32_t curMLen, uint32_t offsetM)
    {
        invRmsOutQueue_.EnQue<P>(invRmsUb_);
        invRmsUb_ = invRmsOutQueue_.DeQue<P>();

        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(curMLen * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);

        uint64_t offset = globalOffsetM_ + offsetM;

        DataCopyPad(invRmsGm_[offset], invRmsUb_, copyParams);
    }

#if !defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P>& x, uint32_t curMLen, uint32_t curNdLen,
                                                  uint32_t offsetM, uint32_t offsetNd)
    {
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curMLen);
        copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);

        uint64_t offset =
            chunkTSize_ * ND_LENGTH * (coreIdx_ + (vectorCount_ % PARALLEL_NUM) * coreNum_) + offsetM * curNdLen;
        DataCopyPad(xFloatGm_[offset], x, copyParams);
    }
#endif

    __aicore__ inline void DataCopyOutHPre(uint64_t offset, uint32_t totalElem)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(totalElem * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3)));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3)));
        DataCopyPad(hPreGm_[offset * N_], hPreBuff_, copyParams);
    }
#else
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(totalElem * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(hPreGm_[offset * N_], hPreBuff_, copyParams);
    }
#endif

    __aicore__ inline void AIVPreLoad()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        invRmsUb_ = invRmsOutQueue_.AllocTensor<P>();

        DataCopyExtParams alphaCopyParams;
        alphaCopyParams.blockCount = 1U;
        uint32_t alphaInputCount = hasResi_ ? 3U : 2U;
        alphaCopyParams.blockLen = alphaInputCount * sizeof(P);
        alphaCopyParams.srcStride = 0U;
        alphaCopyParams.dstStride = 0U;
        uint32_t alphaRightPadding = kAlignmentBytes / sizeof(P) - alphaInputCount;
        DataCopyPadExtParams<P> alphaPadParams{true, 0, static_cast<uint8_t>(alphaRightPadding), 0};
        DataCopyPad(alphaInUb_, alphaGm_, alphaCopyParams, alphaPadParams);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V)));
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V)));

        if (N_ == kSupportedN4) {
            if (hasResi_) {
                MhcPreVFExpandAlpha<P, kSupportedN4, true>(alphaInUb_);
            } else {
                MhcPreVFExpandAlpha<P, kSupportedN4, false>(alphaInUb_);
            }
        } else if (N_ == kSupportedN6) {
            if (hasResi_) {
                MhcPreVFExpandAlpha<P, kSupportedN6, true>(alphaInUb_);
            } else {
                MhcPreVFExpandAlpha<P, kSupportedN6, false>(alphaInUb_);
            }
        } else if (hasResi_) {
            MhcPreVFExpandAlpha<P, kSupportedN8, true>(alphaInUb_);
        } else {
            MhcPreVFExpandAlpha<P, kSupportedN8, false>(alphaInUb_);
        }
        BiasCopyIn();
        biasInUb_ = biasInQue_.DeQue<P>();
    }
#else
    {
        invRmsUb_ = invRmsOutQueue_.AllocTensor<P>();

        DataCopyExtParams alphaCopyParams;
        alphaCopyParams.blockCount = 1U;
        uint32_t alphaInputCount = hasResi_ ? 3U : 2U;
        alphaCopyParams.blockLen = alphaInputCount * sizeof(P);
        alphaCopyParams.srcStride = 0U;
        alphaCopyParams.dstStride = 0U;
        uint32_t alphaRightPadding = kAlignmentBytes / sizeof(P) - alphaInputCount;
        DataCopyPadExtParams<P> alphaPadParams{true, 0, static_cast<uint8_t>(alphaRightPadding), 0};
        DataCopyPad(alphaInUb_, alphaGm_, alphaCopyParams, alphaPadParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        if (N_ == kSupportedN4) {
            if (hasResi_) {
                MhcPreVFExpandAlpha<P, kSupportedN4, true>(alphaInUb_);
            } else {
                MhcPreVFExpandAlpha<P, kSupportedN4, false>(alphaInUb_);
            }
        } else if (N_ == kSupportedN6) {
            if (hasResi_) {
                MhcPreVFExpandAlpha<P, kSupportedN6, true>(alphaInUb_);
            } else {
                MhcPreVFExpandAlpha<P, kSupportedN6, false>(alphaInUb_);
            }
        } else if (hasResi_) {
            MhcPreVFExpandAlpha<P, kSupportedN8, true>(alphaInUb_);
        } else {
            MhcPreVFExpandAlpha<P, kSupportedN8, false>(alphaInUb_);
        }
        BiasCopyIn();
        biasInUb_ = biasInQue_.DeQue<P>();
    }
#endif

    __aicore__ inline void BiasCopyIn()
    {
        LocalTensor<P> biasLocal = biasInQue_.AllocTensor<P>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(matrixInfo_.fusionSize * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

        DataCopyPad(biasLocal, biasGm_, copyParams, copyPadParams);
        biasInQue_.EnQue(biasLocal);
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    static __simd_vf__ inline void AIV1ProcessHPreImpl(__ubuf__ P* hPreBuffAddr, uint16_t nLoopCnt, uint32_t curElemCnt,
                                                       uint32_t elemsPerVf, float hcEps)
    {
        Reg::RegTensor<P> hPreReg;
        Reg::RegTensor<P> negReg;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> addOneReg;
        Reg::RegTensor<P> sigmoidReg;
        Reg::RegTensor<P> resultReg;
        Reg::RegTensor<P> oneReg;

        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);
            Reg::LoadAlign(hPreReg, hPreBuffAddr + vfBlockIdx * elemsPerVf);
            Reg::Neg(negReg, hPreReg, mask);
            Reg::Exp(expReg, negReg, mask);
            Reg::Adds(addOneReg, expReg, static_cast<P>(kOneValue), mask);
            Reg::Duplicate(oneReg, static_cast<P>(kOneValue), mask);
            Reg::Div<P, &MHC_PRE_DIV_ZEROING_MODE>(sigmoidReg, oneReg, addOneReg, mask);

            Reg::Adds(resultReg, sigmoidReg, hcEps, mask);
            Reg::StoreAlign(hPreBuffAddr + vfBlockIdx * elemsPerVf, resultReg, mask);
        }
    }
#endif

    __aicore__ inline void AIV1ProcessHPre(uint64_t offsetT, uint64_t lenT)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        __ubuf__ P* hPreBuffAddr = (__ubuf__ P*)hPreBuff_.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
        uint32_t curElemCnt = totalElem;
        asc_vf_call<AIV1ProcessHPreImpl>(hPreBuffAddr, nLoopCnt, curElemCnt, eleNumPerVf_, matrixInfo_.hcEps);

        if (outFlag_) {
            uint64_t offset = globalOffsetM_ + offsetT;
            DataCopyOutHPre(offset, totalElem);
        }
    }

    template <bool REFINE_SIGMOID>
    static __simd_vf__ inline void AIV1ProcessHPostImpl(__ubuf__ P* hPostBuffAddr, __ubuf__ P* hPostOutAddr,
                                                        uint32_t regCapacityFp32, uint16_t nLoopCnt, float scalarValue,
                                                        uint32_t curElemCnt, float hcEps)
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * regCapacityFp32;
            Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);
            Reg::RegTensor<P> hPostReg;
            Reg::RegTensor<P> negReg;
            Reg::RegTensor<P> expReg;
            Reg::RegTensor<P> addOneReg;
            Reg::RegTensor<P> sigmoidReg;
            Reg::RegTensor<P> resultReg;
            Reg::RegTensor<P> oneReg;

            Reg::LoadAlign(hPostReg, hPostBuffAddr + elemOffset);
            Reg::Neg(negReg, hPostReg, mask);
            Reg::Exp(expReg, negReg, mask);
            Reg::Adds(addOneReg, expReg, kOneValue, mask);
            Reg::Duplicate(oneReg, kOneValue, mask);
            Reg::Div<P, &MHC_PRE_DIV_ZEROING_MODE>(sigmoidReg, oneReg, addOneReg, mask);

            if constexpr (REFINE_SIGMOID) {
                // Retain the low part of 1 + exp(-x), then correct the reciprocal.
                Reg::MaskReg finite;
                Reg::CompareScalar<P, CMPMODE::LE>(finite, addOneReg, 3.402823466e38f, mask);
                Reg::RegTensor<P> roundedExp, residual, error, refined;
                Reg::Sub(roundedExp, addOneReg, oneReg, finite);
                Reg::Sub(residual, addOneReg, roundedExp, finite);
                Reg::Sub(residual, oneReg, residual, finite);
                Reg::Sub(expReg, expReg, roundedExp, finite);
                Reg::Add(residual, residual, expReg, finite);
                Reg::Neg(addOneReg, addOneReg, finite);
                Reg::Neg(residual, residual, finite);
                error = oneReg;
                Reg::MulAddDst(error, addOneReg, sigmoidReg, finite);
                Reg::MulAddDst(error, residual, sigmoidReg, finite);
                refined = sigmoidReg;
                Reg::MulAddDst(refined, sigmoidReg, error, finite);
                Reg::Select(sigmoidReg, refined, sigmoidReg, finite);
            }

            if constexpr (RESI_MODE == MHC_PRE_NO_RESI) {
                Reg::RegTensor<P> scaleReg;
                Reg::Duplicate(scaleReg, scalarValue, mask);
                Reg::Duplicate(resultReg, static_cast<P>(hcEps), mask);
                Reg::MulAddDst(resultReg, sigmoidReg, scaleReg, mask);
            } else {
                Reg::Muls(resultReg, sigmoidReg, scalarValue, mask);
            }
            Reg::StoreAlign(hPostOutAddr + elemOffset, resultReg, mask);
        }
    }
#else
    {
        __ubuf__ P* hPreBuffAddr = (__ubuf__ P*)hPreBuff_.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
        uint32_t curElemCnt = totalElem;
        __VEC_SCOPE__
        {
            Reg::RegTensor<P> hPreReg;
            Reg::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);
                Reg::LoadAlign(hPreReg, hPreBuffAddr + vfBlockIdx * eleNumPerVf_);
                Reg::Neg(negReg, hPreReg, mask);
                Reg::Exp(expReg, negReg, mask);
                Reg::Adds(addOneReg, expReg, static_cast<P>(kOneValue), mask);
                Reg::Duplicate(oneReg, static_cast<P>(kOneValue), mask);
                Reg::Div<P, &MHC_PRE_DIV_ZEROING_MODE>(sigmoidReg, oneReg, addOneReg, mask);

                Reg::Adds(resultReg, sigmoidReg, matrixInfo_.hcEps, mask);
                Reg::StoreAlign(hPreBuffAddr + vfBlockIdx * eleNumPerVf_, resultReg, mask);
            }
        }

        if (outFlag_) {
            uint64_t offset = globalOffsetM_ + offsetT;
            DataCopyOutHPre(offset, totalElem);
        }
    }
#endif

    __aicore__ inline void AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        uint64_t offset = globalOffsetM_ + offsetT;
        LocalTensor<P> hPostOutLocal = outQueue_.AllocTensor<P>();
        __ubuf__ P* hPostBuffAddr = (__ubuf__ P*)hPostBuff_.GetPhyAddr();
        __ubuf__ P* hPostOutAddr = (__ubuf__ P*)hPostOutLocal.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint32_t regCapacityFp32 = 64;
        uint16_t nLoopCnt = Ceil(totalElem, regCapacityFp32);
        float scalarValue = kTwoValue;
        uint32_t curElemCnt = totalElem;

        if (matrixInfo_.nD < 512U) {
            asc_vf_call<AIV1ProcessHPostImpl<true>>(hPostBuffAddr, hPostOutAddr, regCapacityFp32, nLoopCnt, scalarValue,
                                                    curElemCnt, matrixInfo_.hcEps);
        } else {
            asc_vf_call<AIV1ProcessHPostImpl<false>>(hPostBuffAddr, hPostOutAddr, regCapacityFp32, nLoopCnt,
                                                     scalarValue, curElemCnt, matrixInfo_.hcEps);
        }
        outQueue_.EnQue(hPostOutLocal);
        hPostOutLocal = outQueue_.DeQue<P>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(lenT * N_ * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPad(hPostGm_[offset * N_], hPostOutLocal, copyParams);
        outQueue_.FreeTensor(hPostOutLocal);
    }

    template <uint32_t SPLIT_HEADS>
    static __simd_vf__ inline void VFDoTailHInSplitImpl(__ubuf__ T* firstAddr, __ubuf__ T* restAddr,
                                                        __ubuf__ T* outAddr, __ubuf__ P* preAddr, uint32_t offsetD,
                                                        uint32_t lenD, uint16_t loops, uint32_t elemsPerVf,
                                                        uint64_t hiddenSize)
    {
        Reg::RegTensor<P> xFp32;
        Reg::RegTensor<P> acc;
        Reg::RegTensor<P> w0;
        Reg::RegTensor<P> w1;
        Reg::RegTensor<P> w2;
        Reg::RegTensor<P> w3;
        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> outReg;
        MhcPreLoadHInValue<P>(w0, preAddr, 0U);
        MhcPreLoadHInValue<P>(w1, preAddr, 1U);
        MhcPreLoadHInValue<P>(w2, preAddr, 2U);
        MhcPreLoadHInValue<P>(w3, preAddr, 3U);
        uint32_t remaining = lenD;
        for (uint16_t i = 0U; i < loops; ++i) {
            Reg::MaskReg mask = Reg::UpdateMask<P>(remaining);
            uint32_t d = offsetD + i * elemsPerVf;
            Reg::Duplicate<P>(acc, static_cast<P>(0.0f), mask);
            MhcPreAccumulateHIn<T, P>(acc, xFp32, xReg, w0, firstAddr, d, mask);
            MhcPreAccumulateHIn<T, P>(acc, xFp32, xReg, w1, firstAddr, static_cast<uint32_t>(hiddenSize) + d, mask);
            if constexpr (SPLIT_HEADS == 3U) {
                MhcPreAccumulateHIn<T, P>(acc, xFp32, xReg, w2, firstAddr, (2U * static_cast<uint32_t>(hiddenSize)) + d,
                                          mask);
            } else {
                MhcPreAccumulateHIn<T, P>(acc, xFp32, xReg, w2, restAddr, d, mask);
            }
            MhcPreAccumulateHIn<T, P>(acc, xFp32, xReg, w3, restAddr,
                                      (3U - SPLIT_HEADS) * static_cast<uint32_t>(hiddenSize) + d, mask);
            Reg::Cast<T, P, MHC_PRE_CAST_FP32_TO_B16>(outReg, acc, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outAddr + d - offsetD, outReg, mask);
        }
    }

    template <uint32_t SPLIT_HEADS>
    __aicore__ inline void VFDoTailHInSplit(__ubuf__ T* firstAddr, __ubuf__ T* restAddr, __ubuf__ T* outAddr,
                                            __ubuf__ P* preAddr, uint32_t offsetD, uint32_t lenD)
    {
        uint16_t loops = (lenD + eleNumPerVf_ - 1U) / eleNumPerVf_;
        asc_vf_call<VFDoTailHInSplitImpl<SPLIT_HEADS>>(firstAddr, restAddr, outAddr, preAddr, offsetD, lenD, loops,
                                                       eleNumPerVf_, D_);
    }
#else
    {
        uint64_t offset = globalOffsetM_ + offsetT;
        LocalTensor<P> hPostOutLocal = outQueue_.AllocTensor<P>();
        __ubuf__ P* hPostBuffAddr = (__ubuf__ P*)hPostBuff_.GetPhyAddr();
        __ubuf__ P* hPostOutAddr = (__ubuf__ P*)hPostOutLocal.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint32_t regCapacityFp32 = 64;
        uint16_t nLoopCnt = Ceil(totalElem, regCapacityFp32);
        float scalarValue = kTwoValue;
        uint32_t curElemCnt = totalElem;

        __VEC_SCOPE__
        {
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
                uint32_t elemOffset = vfBlockIdx * regCapacityFp32;
                Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);
                Reg::RegTensor<P> hPostReg;
                Reg::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

                Reg::LoadAlign(hPostReg, hPostBuffAddr + elemOffset);
                Reg::Neg(negReg, hPostReg, mask);
                Reg::Exp(expReg, negReg, mask);
                Reg::Adds(addOneReg, expReg, kOneValue, mask);
                Reg::Duplicate(oneReg, kOneValue, mask);
                Reg::Div<P, &MHC_PRE_DIV_ZEROING_MODE>(sigmoidReg, oneReg, addOneReg, mask);

                Reg::Muls(resultReg, sigmoidReg, scalarValue, mask);
                if constexpr (RESI_MODE == MHC_PRE_NO_RESI) {
                    Reg::Adds(resultReg, resultReg, matrixInfo_.hcEps, mask);
                }
                Reg::StoreAlign(hPostOutAddr + elemOffset, resultReg, mask);
            }
        }
        outQueue_.EnQue(hPostOutLocal);
        hPostOutLocal = outQueue_.DeQue<P>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(lenT * N_ * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPad(hPostGm_[offset * N_], hPostOutLocal, copyParams);
        outQueue_.FreeTensor(hPostOutLocal);
    }
#endif

    template <uint32_t HEAD_NUM>
#if defined(MHC_PRE_ENABLE_ASCEND950)
    static __simd_vf__ inline void VFDoV1ProcessHInRowsImpl(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr,
                                                            __ubuf__ P* hPreAddr, uint16_t rowCount, uint32_t lenD,
                                                            uint32_t localStride, uint16_t dLoopCnt,
                                                            uint32_t elemsPerVf)
    {
        Reg::RegTensor<P> xFp32Reg;
        Reg::RegTensor<T> xInReg;
        Reg::RegTensor<P> accFp32Reg;
        Reg::RegTensor<T> outB16Reg;
        Reg::RegTensor<P> hPreReg0;
        Reg::RegTensor<P> hPreReg1;
        Reg::RegTensor<P> hPreReg2;
        Reg::RegTensor<P> hPreReg3;
        Reg::RegTensor<P> hPreReg4;
        Reg::RegTensor<P> hPreReg5;
        Reg::RegTensor<P> hPreReg6;
        Reg::RegTensor<P> hPreReg7;

        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t hPreOffset = static_cast<uint32_t>(row) * HEAD_NUM;
            MhcPreLoadHInValue<P>(hPreReg0, hPreAddr, hPreOffset);
            MhcPreLoadHInValue<P>(hPreReg1, hPreAddr, hPreOffset + 1U);
            MhcPreLoadHInValue<P>(hPreReg2, hPreAddr, hPreOffset + 2U);
            MhcPreLoadHInValue<P>(hPreReg3, hPreAddr, hPreOffset + 3U);
            if constexpr (HEAD_NUM >= kSupportedN6) {
                MhcPreLoadHInValue<P>(hPreReg4, hPreAddr, hPreOffset + 4U);
                MhcPreLoadHInValue<P>(hPreReg5, hPreAddr, hPreOffset + 5U);
            }
            if constexpr (HEAD_NUM == kSupportedN8) {
                MhcPreLoadHInValue<P>(hPreReg6, hPreAddr, hPreOffset + 6U);
                MhcPreLoadHInValue<P>(hPreReg7, hPreAddr, hPreOffset + 7U);
            }

            uint32_t remaining = lenD;
            uint32_t xRowOffset = static_cast<uint32_t>(row) * HEAD_NUM * localStride;
            uint32_t outRowOffset = static_cast<uint32_t>(row) * localStride;
            for (uint16_t dIdx = 0; dIdx < dLoopCnt; ++dIdx) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(remaining);
                uint32_t dOffset = static_cast<uint32_t>(dIdx) * elemsPerVf;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + xRowOffset + dOffset);
                Reg::Cast<P, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32Reg, xInReg, mask);
                Reg::Mul(accFp32Reg, xFp32Reg, hPreReg0, mask);
                MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg1, xInAddr,
                                          xRowOffset + localStride + dOffset, mask);
                MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg2, xInAddr,
                                          xRowOffset + 2U * localStride + dOffset, mask);
                MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg3, xInAddr,
                                          xRowOffset + 3U * localStride + dOffset, mask);
                if constexpr (HEAD_NUM >= kSupportedN6) {
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg4, xInAddr,
                                              xRowOffset + 4U * localStride + dOffset, mask);
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg5, xInAddr,
                                              xRowOffset + 5U * localStride + dOffset, mask);
                }
                if constexpr (HEAD_NUM == kSupportedN8) {
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg6, xInAddr,
                                              xRowOffset + 6U * localStride + dOffset, mask);
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg7, xInAddr,
                                              xRowOffset + 7U * localStride + dOffset, mask);
                }
                Reg::Cast<T, P, MHC_PRE_CAST_FP32_TO_B16>(outB16Reg, accFp32Reg, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hinOutAddr + outRowOffset + dOffset, outB16Reg, mask);
            }
        }
    }

    template <uint32_t HEAD_NUM>
    __aicore__ inline void VFDoV1ProcessHInRows(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, __ubuf__ P* hPreAddr,
                                                uint16_t rowCount, uint32_t lenD, uint32_t localStride)
    {
        uint16_t dLoopCnt = (lenD + eleNumPerVf_ - 1) / eleNumPerVf_;
        asc_vf_call<VFDoV1ProcessHInRowsImpl<HEAD_NUM>>(xInAddr, hinOutAddr, hPreAddr, rowCount, lenD, localStride,
                                                        dLoopCnt, eleNumPerVf_);
    }
#else
    __aicore__ inline void VFDoV1ProcessHInRows(__ubuf__ T* xInAddr, __ubuf__ T* hinOutAddr, __ubuf__ P* hPreAddr,
                                                uint16_t rowCount, uint32_t lenD, uint32_t localStride)
    {
        uint16_t dLoopCnt = (lenD + eleNumPerVf_ - 1) / eleNumPerVf_;
        __VEC_SCOPE__
        {
            Reg::RegTensor<P> xFp32Reg;
            Reg::RegTensor<T> xInReg;
            Reg::RegTensor<P> accFp32Reg;
            Reg::RegTensor<T> outB16Reg;
            Reg::RegTensor<P> hPreReg0;
            Reg::RegTensor<P> hPreReg1;
            Reg::RegTensor<P> hPreReg2;
            Reg::RegTensor<P> hPreReg3;
            Reg::RegTensor<P> hPreReg4;
            Reg::RegTensor<P> hPreReg5;
            Reg::RegTensor<P> hPreReg6;
            Reg::RegTensor<P> hPreReg7;

            for (uint16_t row = 0; row < rowCount; ++row) {
                uint32_t hPreOffset = static_cast<uint32_t>(row) * HEAD_NUM;
                MhcPreLoadHInValue<P>(hPreReg0, hPreAddr, hPreOffset);
                MhcPreLoadHInValue<P>(hPreReg1, hPreAddr, hPreOffset + 1U);
                MhcPreLoadHInValue<P>(hPreReg2, hPreAddr, hPreOffset + 2U);
                MhcPreLoadHInValue<P>(hPreReg3, hPreAddr, hPreOffset + 3U);
                if constexpr (HEAD_NUM >= kSupportedN6) {
                    MhcPreLoadHInValue<P>(hPreReg4, hPreAddr, hPreOffset + 4U);
                    MhcPreLoadHInValue<P>(hPreReg5, hPreAddr, hPreOffset + 5U);
                }
                if constexpr (HEAD_NUM == kSupportedN8) {
                    MhcPreLoadHInValue<P>(hPreReg6, hPreAddr, hPreOffset + 6U);
                    MhcPreLoadHInValue<P>(hPreReg7, hPreAddr, hPreOffset + 7U);
                }

                uint32_t remaining = lenD;
                uint32_t xRowOffset = static_cast<uint32_t>(row) * HEAD_NUM * localStride;
                uint32_t outRowOffset = static_cast<uint32_t>(row) * localStride;
                for (uint16_t dIdx = 0; dIdx < dLoopCnt; ++dIdx) {
                    Reg::MaskReg mask = Reg::UpdateMask<P>(remaining);
                    uint32_t dOffset = static_cast<uint32_t>(dIdx) * eleNumPerVf_;
                    Reg::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg0, xInAddr, xRowOffset + dOffset,
                                              mask);
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg1, xInAddr,
                                              xRowOffset + localStride + dOffset, mask);
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg2, xInAddr,
                                              xRowOffset + 2U * localStride + dOffset, mask);
                    MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg3, xInAddr,
                                              xRowOffset + 3U * localStride + dOffset, mask);
                    if constexpr (HEAD_NUM >= kSupportedN6) {
                        MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg4, xInAddr,
                                                  xRowOffset + 4U * localStride + dOffset, mask);
                        MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg5, xInAddr,
                                                  xRowOffset + 5U * localStride + dOffset, mask);
                    }
                    if constexpr (HEAD_NUM == kSupportedN8) {
                        MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg6, xInAddr,
                                                  xRowOffset + 6U * localStride + dOffset, mask);
                        MhcPreAccumulateHIn<T, P>(accFp32Reg, xFp32Reg, xInReg, hPreReg7, xInAddr,
                                                  xRowOffset + 7U * localStride + dOffset, mask);
                    }
                    Reg::Cast<T, P, MHC_PRE_CAST_FP32_TO_B16>(outB16Reg, accFp32Reg, mask);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hinOutAddr + outRowOffset + dOffset, outB16Reg,
                                                                      mask);
                }
            }
        }
    }
#endif

    __aicore__ inline void AIV1ProcessHInBatched(uint64_t offsetT, uint64_t lenT, uint32_t xBufferBytes = 80U * 1024U,
                                                 uint32_t outBufferBytes = 20U * 1024U)
    {
        if (N_ != kSupportedN4 && N_ != kSupportedN6 && N_ != kSupportedN8) {
            return;
        }

        constexpr uint32_t blockElements = kAlignmentBytes / sizeof(T);
        uint32_t fullD = static_cast<uint32_t>(D_);
        bool fullDFits = xBufferBytes / (N_ * fullD * sizeof(T)) > 0U && outBufferBytes / (fullD * sizeof(T)) > 0U;
        uint32_t dChunkSize = fullDFits ? fullD : v1ChunkDSize_;
        for (uint32_t offsetD = 0; offsetD < D_; offsetD += dChunkSize) {
            uint32_t lenD = dChunkSize < D_ - offsetD ? dChunkSize : D_ - offsetD;
            bool contiguous = offsetD == 0U && lenD == D_ && (lenD % blockElements == 0U);
            uint32_t localStride = contiguous ? lenD : Ceil(lenD, blockElements) * blockElements;
            uint32_t maxRowsByX = xBufferBytes / (N_ * localStride * sizeof(T));
            uint32_t maxRowsByOut = outBufferBytes / (localStride * sizeof(T));
            // Cap row batches to control register pressure while amortizing DMA and VF setup.
            uint32_t maxBatchRows = chunkTSize_ <= 32U ? 2U : 4U;
            uint32_t rowsPerBatch =
                MhcPreMin(static_cast<uint64_t>(MhcPreMin(MhcPreMin(maxRowsByX, maxRowsByOut), maxBatchRows)), lenT);
            if (rowsPerBatch == 0U) {
                return;
            }

            for (uint32_t rowOffset = 0; rowOffset < lenT; rowOffset += rowsPerBatch) {
                uint32_t rowCount = MhcPreMin(static_cast<uint64_t>(rowsPerBatch), lenT - rowOffset);
                LocalTensor<T> xIn = xInQueue_.AllocTensor<T>();
                DataCopyExtParams xCopyParams;
                xCopyParams.blockCount = contiguous ? 1U : static_cast<uint16_t>(rowCount * N_);
                xCopyParams.blockLen = contiguous ? rowCount * N_ * lenD * sizeof(T) : lenD * sizeof(T);
                xCopyParams.srcStride = contiguous ? 0U : (D_ - lenD) * sizeof(T);
                xCopyParams.dstStride = 0U;
                DataCopyPadExtParams<T> xPadParams{!contiguous, 0, static_cast<uint8_t>(localStride - lenD), 0};
                uint64_t globalRow = globalOffsetM_ + offsetT + rowOffset;
                DataCopyPad(xIn, xGm_[globalRow * N_ * D_ + offsetD], xCopyParams, xPadParams);
                xInQueue_.EnQue(xIn);
                xIn = xInQueue_.DeQue<T>();

                LocalTensor<T> hinOut = outQueue_.AllocTensor<T>();
                __ubuf__ T* xInAddr = (__ubuf__ T*)xIn.GetPhyAddr();
                __ubuf__ T* hinOutAddr = (__ubuf__ T*)hinOut.GetPhyAddr();
                __ubuf__ P* hPreAddr = (__ubuf__ P*)hPreBuff_[rowOffset * N_].GetPhyAddr();
                if (N_ == kSupportedN4) {
                    VFDoV1ProcessHInRows<kSupportedN4>(xInAddr, hinOutAddr, hPreAddr, rowCount, lenD, localStride);
                } else if (N_ == kSupportedN6) {
                    VFDoV1ProcessHInRows<kSupportedN6>(xInAddr, hinOutAddr, hPreAddr, rowCount, lenD, localStride);
                } else {
                    VFDoV1ProcessHInRows<kSupportedN8>(xInAddr, hinOutAddr, hPreAddr, rowCount, lenD, localStride);
                }
                xInQueue_.FreeTensor(xIn);

                outQueue_.EnQue(hinOut);
                hinOut = outQueue_.DeQue<T>();
                DataCopyExtParams outCopyParams;
                outCopyParams.blockCount = contiguous ? 1U : static_cast<uint16_t>(rowCount);
                outCopyParams.blockLen = contiguous ? rowCount * lenD * sizeof(T) : lenD * sizeof(T);
                outCopyParams.srcStride = contiguous ? 0U : (localStride - lenD) * sizeof(T);
                outCopyParams.dstStride = contiguous ? 0U : (D_ - lenD) * sizeof(T);
                DataCopyPad(hinGm_[globalRow * D_ + offsetD], hinOut, outCopyParams);
                outQueue_.FreeTensor(hinOut);
            }
        }
    }

    template <bool WAIT_HRES_COPY = false>
    __aicore__ inline void AIV1PostProcessTile(const LocalTensor<P>& hMixLocal, uint64_t offsetT, uint64_t lenT,
                                               uint32_t invRmsOffset, uint32_t hMixRowStride,
                                               uint32_t xBufferBytes = kXInQueueBufferBytes,
                                               uint32_t outBufferBytes = kOutQueueBufferBytes)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        uint64_t outputOffset = globalOffsetM_ + offsetT;
        LocalTensor<P> hResOutLocal;
        if constexpr (RESI_MODE == MHC_PRE_HAS_RESI) {
            hResOutLocal = outQueue_.template AllocTensor<P>();
        }

        matmulRes_ = hMixLocal;
        if (N_ == kSupportedN4) {
            MhcPreVFPostSegments<P, kSupportedN4, RESI_MODE == MHC_PRE_HAS_RESI>(
                hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_, alphaInUb_, biasInUb_,
                static_cast<uint16_t>(lenT), invRmsOffset, hMixRowStride);
        } else if (N_ == kSupportedN6) {
            MhcPreVFPostSegments<P, kSupportedN6, RESI_MODE == MHC_PRE_HAS_RESI>(
                hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_, alphaInUb_, biasInUb_,
                static_cast<uint16_t>(lenT), invRmsOffset, hMixRowStride);
        } else {
            MhcPreVFPostSegments<P, kSupportedN8, RESI_MODE == MHC_PRE_HAS_RESI>(
                hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_, alphaInUb_, biasInUb_,
                static_cast<uint16_t>(lenT), invRmsOffset, hMixRowStride);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (RESI_MODE == MHC_PRE_HAS_RESI) {
            outQueue_.EnQue(hResOutLocal);
            hResOutLocal = outQueue_.template DeQue<P>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1U;
            copyParams.blockLen = static_cast<uint32_t>(lenT * N_ * N_ * sizeof(P));
            copyParams.srcStride = 0U;
            copyParams.dstStride = 0U;
            DataCopyPad(hResGm_[outputOffset * N_ * N_], hResOutLocal, copyParams);
            if constexpr (WAIT_HRES_COPY) {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V)));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V)));
            }
            outQueue_.FreeTensor(hResOutLocal);
        }

        AIV1ProcessHPost(offsetT, lenT);
        AIV1ProcessHPre(offsetT, lenT);
        AIV1ProcessHInBatched(offsetT, lenT, xBufferBytes, outBufferBytes);
    }
#else
    {
        uint64_t outputOffset = globalOffsetM_ + offsetT;
        LocalTensor<P> hResOutLocal;
        if constexpr (RESI_MODE == MHC_PRE_HAS_RESI) {
            hResOutLocal = outQueue_.template AllocTensor<P>();
        }

        matmulRes_ = hMixLocal;
        if (N_ == kSupportedN4) {
            MhcPreVFPostSegments<P, kSupportedN4, RESI_MODE == MHC_PRE_HAS_RESI>(
                hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_, alphaInUb_, biasInUb_,
                static_cast<uint16_t>(lenT), invRmsOffset, hMixRowStride);
        } else if (N_ == kSupportedN6) {
            MhcPreVFPostSegments<P, kSupportedN6, RESI_MODE == MHC_PRE_HAS_RESI>(
                hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_, alphaInUb_, biasInUb_,
                static_cast<uint16_t>(lenT), invRmsOffset, hMixRowStride);
        } else {
            MhcPreVFPostSegments<P, kSupportedN8, RESI_MODE == MHC_PRE_HAS_RESI>(
                hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_, alphaInUb_, biasInUb_,
                static_cast<uint16_t>(lenT), invRmsOffset, hMixRowStride);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (RESI_MODE == MHC_PRE_HAS_RESI) {
            outQueue_.EnQue(hResOutLocal);
            hResOutLocal = outQueue_.template DeQue<P>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1U;
            copyParams.blockLen = static_cast<uint32_t>(lenT * N_ * N_ * sizeof(P));
            copyParams.srcStride = 0U;
            copyParams.dstStride = 0U;
            DataCopyPad(hResGm_[outputOffset * N_ * N_], hResOutLocal, copyParams);
            if constexpr (WAIT_HRES_COPY) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            }
            outQueue_.FreeTensor(hResOutLocal);
        }

        AIV1ProcessHPost(offsetT, lenT);
        AIV1ProcessHPre(offsetT, lenT);
        AIV1ProcessHInBatched(offsetT, lenT, xBufferBytes, outBufferBytes);
    }
#endif

private:
    template <class U, class V, int8_t MODE>
    friend class MhcPreSplitBS;
    template <class U, class V, int8_t MODE>
#if defined(MHC_PRE_ENABLE_ASCEND950)
    friend class MhcPreSplitBSWave;
    template <class U, class V, int8_t MODE, bool REUSE>
#endif
    friend class MhcPreMKPart1;
#if defined(MHC_PRE_ENABLE_ASCEND950)
    template <class U, class V, int8_t MODE, bool REUSE>
#else
    template <class U, class V, int8_t MODE>
#endif
    friend class MhcPreMKPart2;
#if !defined(MHC_PRE_ENABLE_ASCEND950)
    template <class U, class V, int8_t MODE>
    friend class MhcPreSplitND;
#endif
    MNConfig mnConfig_;
    uint32_t coreNum_;
    uint64_t totalLength_;
    uint64_t N_;
    uint64_t D_;
    bool outFlag_;

    bool hasGamma_;
    bool hasResi_ = true;
    uint32_t implMode_ = MHC_PRE_IMPL_MODE_FP32;

    GlobalTensor<T> xGm_;
    GlobalTensor<P> phiGm_;
    GlobalTensor<P> alphaGm_;
    GlobalTensor<P> biasGm_;
    GlobalTensor<P> gammaGm_;
    GlobalTensor<T> hinGm_;
    GlobalTensor<P> hPostGm_;
    GlobalTensor<P> hResGm_;
    GlobalTensor<P> invRmsGm_;
    GlobalTensor<P> hMixGm_;
    GlobalTensor<P> hPreGm_;
#if !defined(MHC_PRE_ENABLE_ASCEND950)
    GlobalTensor<P> xFloatGm_;
#endif

    TQue<QuePosition::VECIN, 2> xInQueue_;
    TQue<QuePosition::VECIN, 1> gammaInQueue_;
    TQue<QuePosition::VECOUT, 1> invRmsOutQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TQue<QuePosition::VECIN, 1> biasInQue_;

    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::VECCALC> alphaBuf_;

    LocalTensor<P> hPreBuff_;
    LocalTensor<P> hPostBuff_;

    LocalTensor<P> alphaInUb_;
    LocalTensor<P> biasInUb_;
    LocalTensor<P> matmulRes_;

    LocalTensor<T> xLocal_;
    LocalTensor<P> invRmsUb_;
    LocalTensor<P> gammaUb_;

    TPipe* pipe_;
    MatrixInfo matrixInfo_;
    VectorOffsetParams vectorOffset_;

    const MhcPreTilingData* tiling_;

    uint32_t chunkTSize_ = 64;
    uint32_t v1ChunkDSize_ = 5120;
    uint32_t curSingleT_ = 64;
    uint32_t coreIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    float scaleMean_ = 0.0f;
    uint64_t globalOffsetM_ = 0;
    uint32_t vectorCount_ = 0;
    uint32_t cubeCount_ = 0;
    uint64_t mmCount_ = 0;
    uint64_t vec1Count_ = 0;
    uint16_t eleNumPerVf_ = 0;
};

} // namespace MhcPre

#endif // MHC_PRE_VECTOR_COMPUTE_H_

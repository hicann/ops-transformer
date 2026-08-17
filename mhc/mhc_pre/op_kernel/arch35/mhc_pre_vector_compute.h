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
class MhcPreMKPart1;
template <class T, class P, int8_t RESI_MODE>
class MhcPreMKPart2;
template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitND;

// Vector cast and division traits shared by all templates.
constexpr MicroAPI::CastTrait MHC_PRE_CAST_FP32_TO_B16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MHC_PRE_CAST_B16_TO_FP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::DivSpecificMode MHC_PRE_DIV_ZEROING_MODE = {MicroAPI::MaskMergeMode::ZEROING, true};

template <class P>
__aicore__ inline void MhcPreLoadHInValue(MicroAPI::RegTensor<P> &dst, __ubuf__ P *src, uint32_t offset)
{
    MicroAPI::DataCopy<P, MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + offset);
}

template <class T, class P>
__aicore__ inline void MhcPreAccumulateHIn(MicroAPI::RegTensor<P> &acc, MicroAPI::RegTensor<P> &xFp32,
                                           MicroAPI::RegTensor<T> &xIn, MicroAPI::RegTensor<P> &hPre, __ubuf__ T *x,
                                           uint32_t offset, MicroAPI::MaskReg mask)
{
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xIn, x + offset);
    MicroAPI::Cast<P, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32, xIn, mask);
    MicroAPI::Mul(xFp32, xFp32, hPre, mask);
    MicroAPI::Add(acc, acc, xFp32, mask);
}

template <class P, uint32_t HEAD_NUM, bool HAS_RESI_VALUE>
__aicore__ inline void MhcPreVFPostSegments(const LocalTensor<P> &hPreLocal, const LocalTensor<P> &hPostLocal,
                                            const LocalTensor<P> &hResLocal, const LocalTensor<P> &mmLocal,
                                            const LocalTensor<P> &invRmsLocal, const LocalTensor<P> &alphaLocal,
                                            const LocalTensor<P> &biasLocal, uint16_t rowCount, uint32_t invRmsOffset,
                                            uint32_t mmRowStride)
{
    __ubuf__ P *hPre = (__ubuf__ P *)hPreLocal.GetPhyAddr();
    __ubuf__ P *hPost = (__ubuf__ P *)hPostLocal.GetPhyAddr();
    __ubuf__ P *mm = (__ubuf__ P *)mmLocal.GetPhyAddr();
    __ubuf__ P *invRms = (__ubuf__ P *)invRmsLocal.GetPhyAddr();
    __ubuf__ P *alpha = (__ubuf__ P *)alphaLocal.GetPhyAddr();
    __ubuf__ P *bias = (__ubuf__ P *)biasLocal.GetPhyAddr();
    __ubuf__ P *hRes = nullptr;
    if constexpr (HAS_RESI_VALUE) {
        hRes = (__ubuf__ P *)hResLocal.GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> dataReg;
        MicroAPI::RegTensor<P> invRmsReg;
        MicroAPI::RegTensor<P> invRmsBroadReg;
        MicroAPI::RegTensor<P> alphaPreReg;
        MicroAPI::RegTensor<P> alphaPostReg;
        MicroAPI::RegTensor<P> alphaResReg;
        MicroAPI::RegTensor<P> biasPreReg;
        MicroAPI::RegTensor<P> biasPostReg;
        MicroAPI::RegTensor<P> biasResReg;
        uint32_t nMaskSize = HEAD_NUM;
        uint32_t resMaskSize = HEAD_NUM * HEAD_NUM;
        MicroAPI::MaskReg nMask = MicroAPI::UpdateMask<P>(nMaskSize);
        MicroAPI::MaskReg resMask = MicroAPI::UpdateMask<P>(resMaskSize);

        MicroAPI::Load<P>(alphaPreReg, alpha);
        MicroAPI::Load<P>(alphaPostReg, alpha + HEAD_NUM);
        MicroAPI::Load<P>(biasPreReg, bias);
        MicroAPI::Load<P>(biasPostReg, bias + HEAD_NUM);
        if constexpr (HAS_RESI_VALUE) {
            MicroAPI::Load<P>(alphaResReg, alpha + 2U * HEAD_NUM);
            MicroAPI::Load<P>(biasResReg, bias + 2U * HEAD_NUM);
        }

        // Consume pre/post/res by fixed UB offsets in one VF loop. This replaces Gather and stages
        // every output at a 32-byte-aligned local-buffer base before DMA.
        for (uint16_t row = 0; row < rowCount; ++row) {
            MicroAPI::Load<P>(invRmsReg, invRms + invRmsOffset + row);
            MicroAPI::Duplicate(invRmsBroadReg, invRmsReg, nMask);
            uint32_t mmOffset = static_cast<uint32_t>(row) * mmRowStride;
            MicroAPI::Load<P>(dataReg, mm + mmOffset);
            MicroAPI::Mul(dataReg, dataReg, invRmsBroadReg, nMask);
            MicroAPI::Mul(dataReg, dataReg, alphaPreReg, nMask);
            MicroAPI::Add(dataReg, dataReg, biasPreReg, nMask);
            MicroAPI::Store<P>(hPre + row * HEAD_NUM, dataReg, HEAD_NUM);

            MicroAPI::Load<P>(dataReg, mm + mmOffset + HEAD_NUM);
            MicroAPI::Mul(dataReg, dataReg, invRmsBroadReg, nMask);
            MicroAPI::Mul(dataReg, dataReg, alphaPostReg, nMask);
            MicroAPI::Add(dataReg, dataReg, biasPostReg, nMask);
            MicroAPI::Store<P>(hPost + row * HEAD_NUM, dataReg, HEAD_NUM);

            if constexpr (HAS_RESI_VALUE) {
                MicroAPI::Duplicate(invRmsBroadReg, invRmsReg, resMask);
                MicroAPI::Load<P>(dataReg, mm + mmOffset + 2U * HEAD_NUM);
                MicroAPI::Mul(dataReg, dataReg, invRmsBroadReg, resMask);
                MicroAPI::Mul(dataReg, dataReg, alphaResReg, resMask);
                MicroAPI::Add(dataReg, dataReg, biasResReg, resMask);
                MicroAPI::Store<P>(hRes + row * HEAD_NUM * HEAD_NUM, dataReg, HEAD_NUM * HEAD_NUM);
            }
        }
    }
}
template <class P, uint32_t HEAD_NUM, bool HAS_RESI_VALUE>
__aicore__ inline void MhcPreVFExpandAlpha(const LocalTensor<P> &alphaLocal)
{
    __ubuf__ P *alpha = (__ubuf__ P *)alphaLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> alphaPreReg;
        MicroAPI::RegTensor<P> alphaPostReg;
        MicroAPI::RegTensor<P> alphaResReg;
        MicroAPI::DataCopy<P, MicroAPI::LoadDist::DIST_BRC_B32>(alphaPreReg, alpha);
        MicroAPI::DataCopy<P, MicroAPI::LoadDist::DIST_BRC_B32>(alphaPostReg, alpha + 1U);
        if constexpr (HAS_RESI_VALUE) {
            MicroAPI::DataCopy<P, MicroAPI::LoadDist::DIST_BRC_B32>(alphaResReg, alpha + 2U);
        }
        MicroAPI::Store<P>(alpha, alphaPreReg, HEAD_NUM);
        MicroAPI::Store<P>(alpha + HEAD_NUM, alphaPostReg, HEAD_NUM);
        if constexpr (HAS_RESI_VALUE) {
            MicroAPI::Store<P>(alpha + 2U * HEAD_NUM, alphaResReg, HEAD_NUM * HEAD_NUM);
        }
    }
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
    static constexpr uint32_t PARALLEL_NUM = 2;
    static constexpr uint32_t ND_LENGTH = 256;
    static constexpr uint32_t V0_BASE_T = 16;
    static constexpr uint64_t V1_BASE_T = 16;
    static constexpr uint64_t V1_BASE_D = 32;

    // Shared AIV buffer sizes.
    static constexpr uint32_t kXInQueueBufferBytes = 80U * 1024U;
    static constexpr uint32_t kOutQueueBufferBytes = 20U * 1024U;
    static constexpr uint32_t kTmpBufferBytes = 40U * 1024U;
    static constexpr int8_t kResiMode = RESI_MODE;

    __aicore__ inline MhcPreVectorCompute() = default;

    __aicore__ inline void InitPipeAndCoreIdx(TPipe *pipe)
    {
        pipe_ = pipe;
        coreIdx_ = GetBlockIdx();
        subBlockIdx_ = GetSubBlockIdx();
    }

    __aicore__ inline void BindGlobalTensors(InitParams initParams)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
        hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.hin));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_res));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));
        xFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));
    }

    __aicore__ inline void InitFromTilingData(const MhcPreTilingData *tilingData)
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
    {
        if (outFlag_) {
            hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
        } else {
            hMixGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P *>(initParams.workspace + mnConfig_.singleCoreM * mnConfig_.singleCoreK *
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

    __aicore__ inline void InitBlockParams(uint64_t curblock, uint32_t tBlockNum)
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
                    VFDoV0ProcessXIn<true, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                                 (__ubuf__ T *)xLocal_.GetPhyAddr(),
                                                 (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                } else {
                    VFDoV0ProcessXIn<true, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                                  (__ubuf__ T *)xLocal_.GetPhyAddr(),
                                                  (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                }
            } else if (offsetNd == 0U) {
                VFDoV0ProcessXIn<false, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                              (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            } else {
                VFDoV0ProcessXIn<false, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                               (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
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

    __aicore__ inline void V0PostProcess(uint32_t curblock, uint32_t tBlockNum)
    {
        if (vectorOffset_.singleCoreM == 0U) {
            return;
        }
        VFDoV0ProcessInvRms((__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_,
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

    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXIn(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn, __ubuf__ P *gamma,
                                            uint16_t mSize, uint16_t nSize)
    {
        uint32_t nSrcUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(T)));
        uint32_t nDstUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreCeilDiv(nSize, eleNumPerVf_);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
                uint32_t elementNum = nSize;
                MicroAPI::RegTensor<P> sumReg;
                if constexpr (isFirstND) {
                    MicroAPI::Duplicate(sumReg, 0);
                } else {
                    MicroAPI::Load(sumReg, invRmsDst + mIdx);
                }
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                    MicroAPI::RegTensor<T> xInReg;
                    MicroAPI::RegTensor<P> gammaReg;
                    MicroAPI::RegTensor<P> xFp32Reg, xMulReg, xSquaReg;
                    MicroAPI::RegTensor<P> tmpSumReg;

                    uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf_;
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                    MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
                    MicroAPI::Cast<float, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32Reg, xInReg, maskN4B32);
                    if constexpr (hasGamma) {
                        MicroAPI::LoadAlign(gammaReg, gamma + vfBlockIdx * eleNumPerVf_);
                        MicroAPI::Mul(xMulReg, gammaReg, xFp32Reg, maskN4B32);
                    } else {
                        xMulReg = xFp32Reg;
                    }
                    uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf_;
                    MicroAPI::StoreAlign(xDst + dstUbOffset, xMulReg, maskN4B32);

                    MicroAPI::Mul(xSquaReg, xFp32Reg, xFp32Reg, maskN4B32);
                    MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(tmpSumReg, xSquaReg, maskN4B32);
                    MicroAPI::Add(sumReg, sumReg, tmpSumReg, maskN4B32);
                }

                MicroAPI::Store(invRmsDst + mIdx, sumReg, 1);
            }
        }
    }

    __aicore__ inline void VFDoV0ProcessInvRms(__ubuf__ P *invRms, uint16_t nSize, float scaleMean, float normEps)
    {
        uint32_t nUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreCeilDiv(nSize, eleNumPerVf_);
        __VEC_SCOPE__
        {
            uint32_t elementNum = nSize;
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
                MicroAPI::RegTensor<P> invrmsReg, onesReg;

                MicroAPI::LoadAlign(invrmsReg, invRms + vfBlockIdx * eleNumPerVf_);
                MicroAPI::Muls(invrmsReg, invrmsReg, scaleMean, maskN4B32);
                MicroAPI::Adds(invrmsReg, invrmsReg, normEps, maskN4B32);
                MicroAPI::Sqrt(invrmsReg, invrmsReg, maskN4B32);
                MicroAPI::Duplicate(onesReg, 1);
                MicroAPI::Div(invrmsReg, onesReg, invrmsReg, maskN4B32);
                MicroAPI::StoreAlign(invRms + vfBlockIdx * eleNumPerVf_, invrmsReg, maskN4B32);
            }
        }
    }

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

    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen,
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

    __aicore__ inline void DataCopyOutHPre(uint64_t offset, uint32_t totalElem)
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

    __aicore__ inline void AIVPreLoad()
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

    __aicore__ inline void AIV1ProcessHPre(uint64_t offsetT, uint64_t lenT)
    {
        __ubuf__ P *hPreBuffAddr = (__ubuf__ P *)hPreBuff_.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
        uint32_t curElemCnt = totalElem;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<P> hPreReg;
            MicroAPI::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
                MicroAPI::LoadAlign(hPreReg, hPreBuffAddr + vfBlockIdx * eleNumPerVf_);
                MicroAPI::Neg(negReg, hPreReg, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addOneReg, expReg, static_cast<P>(kOneValue), mask);
                MicroAPI::Duplicate(oneReg, static_cast<P>(kOneValue), mask);
                MicroAPI::Div<P, &MHC_PRE_DIV_ZEROING_MODE>(sigmoidReg, oneReg, addOneReg, mask);

                MicroAPI::Adds(resultReg, sigmoidReg, matrixInfo_.hcEps, mask);
                MicroAPI::StoreAlign(hPreBuffAddr + vfBlockIdx * eleNumPerVf_, resultReg, mask);
            }
        }

        if (outFlag_) {
            uint64_t offset = globalOffsetM_ + offsetT;
            DataCopyOutHPre(offset, totalElem);
        }
    }

    __aicore__ inline void AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT)
    {
        uint64_t offset = globalOffsetM_ + offsetT;
        LocalTensor<P> hPostOutLocal = outQueue_.AllocTensor<P>();
        __ubuf__ P *hPostBuffAddr = (__ubuf__ P *)hPostBuff_.GetPhyAddr();
        __ubuf__ P *hPostOutAddr = (__ubuf__ P *)hPostOutLocal.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint32_t regCapacityFp32 = 64;
        uint16_t nLoopCnt = Ceil(totalElem, regCapacityFp32);
        float scalarValue = kTwoValue;
        uint32_t curElemCnt = totalElem;

        __VEC_SCOPE__
        {
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
                uint32_t elemOffset = vfBlockIdx * regCapacityFp32;
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
                MicroAPI::RegTensor<P> hPostReg;
                MicroAPI::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

                MicroAPI::LoadAlign(hPostReg, hPostBuffAddr + elemOffset);
                MicroAPI::Neg(negReg, hPostReg, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addOneReg, expReg, kOneValue, mask);
                MicroAPI::Duplicate(oneReg, kOneValue, mask);
                MicroAPI::Div<P, &MHC_PRE_DIV_ZEROING_MODE>(sigmoidReg, oneReg, addOneReg, mask);

                MicroAPI::Muls(resultReg, sigmoidReg, scalarValue, mask);
                if constexpr (RESI_MODE == MHC_PRE_NO_RESI) {
                    MicroAPI::Adds(resultReg, resultReg, matrixInfo_.hcEps, mask);
                }
                MicroAPI::StoreAlign(hPostOutAddr + elemOffset, resultReg, mask);
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

    template <uint32_t HEAD_NUM>
    __aicore__ inline void VFDoV1ProcessHInRows(__ubuf__ T *xInAddr, __ubuf__ T *hinOutAddr, __ubuf__ P *hPreAddr,
                                                uint16_t rowCount, uint32_t lenD, uint32_t localStride)
    {
        uint16_t dLoopCnt = (lenD + eleNumPerVf_ - 1) / eleNumPerVf_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<P> xFp32Reg;
            MicroAPI::RegTensor<T> xInReg;
            MicroAPI::RegTensor<P> accFp32Reg;
            MicroAPI::RegTensor<T> outB16Reg;
            MicroAPI::RegTensor<P> hPreReg0;
            MicroAPI::RegTensor<P> hPreReg1;
            MicroAPI::RegTensor<P> hPreReg2;
            MicroAPI::RegTensor<P> hPreReg3;
            MicroAPI::RegTensor<P> hPreReg4;
            MicroAPI::RegTensor<P> hPreReg5;
            MicroAPI::RegTensor<P> hPreReg6;
            MicroAPI::RegTensor<P> hPreReg7;

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
                    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(remaining);
                    uint32_t dOffset = static_cast<uint32_t>(dIdx) * eleNumPerVf_;
                    MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);
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
                    MicroAPI::Cast<T, P, MHC_PRE_CAST_FP32_TO_B16>(outB16Reg, accFp32Reg, mask);
                    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + outRowOffset + dOffset,
                                                                                outB16Reg, mask);
                }
            }
        }
    }

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
                __ubuf__ T *xInAddr = (__ubuf__ T *)xIn.GetPhyAddr();
                __ubuf__ T *hinOutAddr = (__ubuf__ T *)hinOut.GetPhyAddr();
                __ubuf__ P *hPreAddr = (__ubuf__ P *)hPreBuff_[rowOffset * N_].GetPhyAddr();
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
    __aicore__ inline void AIV1PostProcessTile(const LocalTensor<P> &hMixLocal, uint64_t offsetT, uint64_t lenT,
                                               uint32_t invRmsOffset, uint32_t hMixRowStride,
                                               uint32_t xBufferBytes = kXInQueueBufferBytes,
                                               uint32_t outBufferBytes = kOutQueueBufferBytes)
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

private:
    template <class U, class V, int8_t MODE>
    friend class MhcPreSplitBS;
    template <class U, class V, int8_t MODE>
    friend class MhcPreMKPart1;
    template <class U, class V, int8_t MODE>
    friend class MhcPreMKPart2;
    template <class U, class V, int8_t MODE>
    friend class MhcPreSplitND;
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
    GlobalTensor<P> xFloatGm_;

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

    TPipe *pipe_;
    MatrixInfo matrixInfo_;
    VectorOffsetParams vectorOffset_;

    const MhcPreTilingData *tiling_;

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

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
 * \file mhc_pre_split_nd.h
 * \brief ND-split Basic API implementation for MHC Pre
 */

#ifndef MHC_PRE_SPLIT_ND_H_
#define MHC_PRE_SPLIT_ND_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_vector_compute.h"
#include "mhc_pre_cube_compute.h"
#include "mhc_pre_tiling_key.h"

namespace MhcPre {

using namespace matmul;
using namespace AscendC;

// ND pipeline cross-core synchronization flags.
constexpr uint64_t SYNC_V0toV0 = 0x1;
constexpr uint64_t SYNC_V0toC = 0x2;
constexpr uint64_t SYNC_CtoC = 0x3;
constexpr uint64_t SYNC_CtoV1 = 0x4;

template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitND {
public:
    // Compile-time residual mode.
    static constexpr int8_t kResiMode = RESI_MODE;

    __aicore__ inline MhcPreSplitND() = default;
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();
    __aicore__ inline void AICProcess();
    __aicore__ inline void InitLocalBuffers();
    __aicore__ inline void VectorComputeOffset();
    __aicore__ inline void AIVPreLoad();
    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXInSingleReduce(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn,
                                                        __ubuf__ P *gamma, uint16_t mSize, uint16_t nSize);
    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen,
                                                  uint32_t offsetM, uint32_t offsetNd);
    __aicore__ inline void V0Prologue(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void AIV1Process(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void HMixProcess(uint64_t offsetT, uint64_t lenT);

    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    GlobalTensor<P> mmResGm_;
    GlobalTensor<P> tempMMResGm_;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    MhcPreCubeCompute mmService_;

    // ND template tile sizes.
    static constexpr uint32_t ND_LENGTH = 2048;
    static constexpr uint64_t V1_BASE_T = 8;

private:
    // UB allocation sizes.
    static constexpr uint32_t kXInQueueBufferBytes = 80 * 1024;
    static constexpr uint32_t kOutQueueBufferBytes = 16 * 1024;
    static constexpr uint32_t kTmpBufferBytes = 20 * 1024;

    // Shape-dependent scheduling defaults.
    static constexpr uint32_t kMinTForN4 = 2;
    static constexpr uint32_t kMinTForN6 = 4;
    static constexpr uint32_t kDefaultCurSingleM = 2;
    static constexpr uint32_t kDefaultMinT = 1;
    static constexpr uint32_t kDefaultVectorCoreNum = 2;
    static constexpr uint32_t kDefaultV0BaseT = 1;

    uint32_t chunNDSize_ = 320;
    uint32_t curSingleM_ = kDefaultCurSingleM;
    uint32_t minT_ = kDefaultMinT;
    uint32_t vectorCoreNum_ = kDefaultVectorCoreNum;
    uint32_t v0BaseT_ = kDefaultV0BaseT;
};

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::Init(InitParams initParams)
{
    vector_.xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
    vector_.phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
    vector_.alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
    vector_.biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias));
    vector_.gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
    vector_.hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.hin));
    vector_.hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
    vector_.hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_res));

    vector_.invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
    vector_.hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));

    vector_.xFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));

    vector_.InitFromTilingData(initParams.tilingData);

    if (vector_.N_ == vector_.kSupportedN4) {
        minT_ = kMinTForN4;
    }
    if (vector_.N_ == vector_.kSupportedN6) {
        minT_ = kMinTForN6;
    }

    chunNDSize_ = Ceil(vector_.matrixInfo_.nD, vector_.coreNum_);
    vector_.chunkTSize_ = Ceil(vector_.totalLength_, vector_.coreNum_);

    vector_.mnConfig_.m = vector_.matrixInfo_.totalLength;
    vector_.mnConfig_.n = vector_.matrixInfo_.fusionSize;
    vector_.mnConfig_.k = vector_.matrixInfo_.nD;
    vector_.mnConfig_.singleCoreM = vector_.totalLength_;
    vector_.mnConfig_.singleCoreN = vector_.mnConfig_.n;
    vector_.mnConfig_.singleCoreK = chunNDSize_;
    vector_.mnConfig_.curSingleCoreM = vector_.mnConfig_.singleCoreM;
    vector_.mnConfig_.curSingleCoreN = vector_.mnConfig_.singleCoreN;
    vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.singleCoreK;
    curSingleM_ = vector_.chunkTSize_;

    constexpr uint64_t kWorkspaceAlignBytes = 32UL;
    uint64_t xFloatWorkspaceBytes = vector_.totalLength_ * vector_.matrixInfo_.nD * sizeof(P);
    xFloatWorkspaceBytes =
        (xFloatWorkspaceBytes + kWorkspaceAlignBytes - 1UL) / kWorkspaceAlignBytes * kWorkspaceAlignBytes;
    if (vector_.outFlag_) {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    } else {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));
    }
    tempMMResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));

    vector_.pipe_ = initParams.tPipeIn;
    vector_.coreIdx_ = GetBlockIdx();
    vector_.subBlockIdx_ = GetSubBlockIdx();

    if ASCEND_IS_AIV {
        InitLocalBuffers();
    }
    if ASCEND_IS_AIC {
        vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        mmService_.Init(this->vector_.implMode_);
    }

    SyncAll<false>();
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::InitLocalBuffers()
{
    vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, kXInQueueBufferBytes);
    vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, kOutQueueBufferBytes);
    vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, vector_.kSingleBufferCount,
                              Ceil(curSingleM_, vector_.kDoubleBufferCount) * sizeof(P));

    if (vector_.hasGamma_) {
        vector_.pipe_->InitBuffer(vector_.gammaInQueue_, vector_.kSingleBufferCount, ND_LENGTH * sizeof(P));
    }

    vector_.pipe_->InitBuffer(vector_.tmpBuff_, kTmpBufferBytes);

    vector_.pipe_->InitBuffer(vector_.biasInQue_, vector_.kSingleBufferCount, vector_.mnConfig_.n * sizeof(P));
    vector_.pipe_->InitBuffer(vector_.alphaBuf_, vector_.mnConfig_.n * sizeof(P));
    vector_.alphaInUb_ = vector_.alphaBuf_.template Get<P>();

    uint64_t buffOffset = 0;
    vector_.hPreBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * vector_.N_), buffOffset);
    buffOffset += V1_BASE_T * vector_.N_ * sizeof(P);
    buffOffset = Ceil(buffOffset, 32) * 32;

    vector_.hPostBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * vector_.N_), buffOffset);
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::Process()
{
    if ASCEND_IS_AIV {
        vector_.coreIdx_ = GetBlockIdx() / vector_.kDoubleBufferCount;
        this->AIVPreLoad();

        uint32_t tBlockNum = Ceil(vector_.totalLength_, vector_.chunkTSize_);
        if (vector_.coreIdx_ < tBlockNum) {
            vector_.globalOffsetM_ = vector_.coreIdx_ * vector_.chunkTSize_;
            V0Prologue(vector_.coreIdx_, tBlockNum);
        } else {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
            AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        }
        // Every launched AIV drains its paired AIC completion token, including
        // idle cores and the zero-row subblock of an odd M tile.
        AscendC::CrossCoreWaitFlag(SYNC_CtoV1);
        if (vector_.coreIdx_ < tBlockNum) {
            AIV1Process(vector_.coreIdx_, tBlockNum);
        }
    }

    if ASCEND_IS_AIC {
        uint32_t ndBlockNum = Ceil(vector_.matrixInfo_.nD, chunNDSize_);
        if (vector_.coreIdx_ < ndBlockNum) {
            AICProcess();
        }
        // All launched AICs must join this barrier, including cores without a K slice.
        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_CtoC);
        AscendC::CrossCoreWaitFlag(SYNC_CtoC);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_CtoV1);
        mmService_.End(this->vector_.implMode_);
    }

    if ASCEND_IS_AIV {
        vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
        vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
    }
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::AIVPreLoad()
{
    vector_.AIVPreLoad();
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::AICProcess()
{
    AscendC::CrossCoreWaitFlag(SYNC_V0toC);

    uint64_t offset = chunNDSize_ * vector_.coreIdx_;
    uint64_t outOffset = vector_.mnConfig_.singleCoreM * vector_.mnConfig_.singleCoreN * vector_.coreIdx_;
    if (offset + vector_.mnConfig_.singleCoreK > vector_.mnConfig_.k) {
        vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.k - offset;
    }

    uint64_t mAlign = BasicApiAlign(vector_.mnConfig_.singleCoreM, AscendC::BLOCK_CUBE);
    uint64_t nAlign = BasicApiAlign(vector_.mnConfig_.singleCoreN, AscendC::BLOCK_CUBE);
    uint64_t maxKL1 = MHC_PRE_BASIC_API_L1_BUF_OFFSET / mAlign;
    maxKL1 = BasicApiAlign(maxKL1, MHC_PRE_BASIC_API_C0_SIZE) - MHC_PRE_BASIC_API_C0_SIZE;
    maxKL1 = AscendC::Std::max(maxKL1, static_cast<uint64_t>(MHC_PRE_BASIC_API_C0_SIZE));
    // Derive baseK/maxKL1 from the active M/N footprint so the ND path generalizes across D values.
    uint64_t baseK = (256U / AscendC::Std::max(mAlign, nAlign)) * 32U;
    baseK = AscendC::Std::max(baseK, 32UL);
    // Keep each B GM-to-L1 transfer within the low-level cube K block.
    maxKL1 = AscendC::Std::min(maxKL1, baseK);

    uint8_t aL1BufferId = 0;
    uint64_t endOffset = offset + vector_.mnConfig_.curSingleCoreK;
    for (uint64_t kOffset = offset; kOffset < endOffset; kOffset += maxKL1) {
        uint64_t currentK = AscendC::Std::min(maxKL1, endOffset - kOffset);
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();
        LocalTensor<P> currentAL1 = aL1_[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        LocalTensor<P> currentBL1 = bL1_[bL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        mmService_.CopyInA1Nd2Nz(vector_.mnConfig_.singleCoreM, currentK, vector_.mnConfig_.k,
                                 vector_.xFloatGm_[kOffset], currentAL1);
        mmService_.CopyInB1Nd2Nz(vector_.mnConfig_.k, currentK, vector_.mnConfig_.singleCoreN, vector_.phiGm_[kOffset],
                                 currentBL1);
        bool isFirst = kOffset == offset;
        bool isLast = kOffset + currentK >= endOffset;
        mmService_.Process(vector_.mnConfig_.singleCoreM, vector_.mnConfig_.singleCoreN, vector_.mnConfig_.singleCoreM,
                           baseK, isFirst, isLast, currentAL1, currentBL1);
        if (isLast) {
            mmService_.CopyOut(tempMMResGm_[outOffset], vector_.mnConfig_.singleCoreN);
        }
        aL1BufferId ^= 1U;
    }
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::VectorComputeOffset()
{
    uint64_t alignSingleM = Ceil(curSingleM_, 2);
    vector_.vectorOffset_.singleCoreM = alignSingleM < curSingleM_ ? alignSingleM : curSingleM_;
    if (vector_.subBlockIdx_ == 0) {
        vector_.vectorOffset_.offsetMStart = 0;
        vector_.vectorOffset_.offsetMEnd = vector_.vectorOffset_.singleCoreM;
    } else {
        vector_.vectorOffset_.offsetMStart = vector_.vectorOffset_.singleCoreM;
        vector_.vectorOffset_.singleCoreM = curSingleM_ - vector_.vectorOffset_.singleCoreM;
        vector_.vectorOffset_.offsetMEnd = curSingleM_;
    }
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen,
                                                                              uint32_t curNdLen, uint32_t offsetM,
                                                                              uint32_t offsetNd)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t((vector_.matrixInfo_.nD - curNdLen) * sizeof(P));

    uint64_t offset = (vector_.globalOffsetM_ + offsetM) * vector_.matrixInfo_.nD + offsetNd;
    DataCopyPad(vector_.xFloatGm_[offset], x, copyParams);
}

template <class T, class P, int8_t RESI_MODE>
template <bool hasGamma, bool isFirstND>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::VFDoV0ProcessXInSingleReduce(__ubuf__ P *xDst,
                                                                                    __ubuf__ P *invRmsDst,
                                                                                    __ubuf__ T *xIn, __ubuf__ P *gamma,
                                                                                    uint16_t mSize, uint16_t nSize)
{
    uint32_t nSrcUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(T)));
    uint32_t nDstUbAligned = MhcPreAlign(nSize, static_cast<uint16_t>(MHC_PRE_UB_ALIGN_SIZE / sizeof(P)));
    uint16_t nLoopCnt = MhcPreCeilDiv(nSize, vector_.eleNumPerVf_);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<P>();
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elementNum = nSize;
            // Keep the whole D-slice sum in one register and reduce once per row instead of once per vector block.
            MicroAPI::RegTensor<P> squareSumReg;
            MicroAPI::Duplicate(squareSumReg, 0.0f);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
                MicroAPI::RegTensor<T> xInReg;
                MicroAPI::RegTensor<P> gammaReg;
                MicroAPI::RegTensor<P> xFp32Reg;
                MicroAPI::RegTensor<P> xMulReg;
                MicroAPI::RegTensor<P> squareReg;
                uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * vector_.eleNumPerVf_;
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                MicroAPI::MaskReg blockMask = MicroAPI::UpdateMask<P>(elementNum);
                MicroAPI::Cast<P, T, MHC_PRE_CAST_B16_TO_FP32>(xFp32Reg, xInReg, blockMask);
                if constexpr (hasGamma) {
                    MicroAPI::LoadAlign(gammaReg, gamma + vfBlockIdx * vector_.eleNumPerVf_);
                    MicroAPI::Mul(xMulReg, gammaReg, xFp32Reg, blockMask);
                } else {
                    xMulReg = xFp32Reg;
                }
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * vector_.eleNumPerVf_;
                MicroAPI::StoreAlign(xDst + dstUbOffset, xMulReg, blockMask);
                MicroAPI::Mul(squareReg, xFp32Reg, xFp32Reg, blockMask);
                MicroAPI::Add<P, MicroAPI::MaskMergeMode::MERGING>(squareSumReg, squareSumReg, squareReg, blockMask);
            }
            MicroAPI::RegTensor<P> partialReg;
            MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(partialReg, squareSumReg, fullMask);
            if constexpr (isFirstND) {
                MicroAPI::Store(invRmsDst + mIdx, partialReg, 1U);
            } else {
                MicroAPI::RegTensor<P> sumReg;
                MicroAPI::Load(sumReg, invRmsDst + mIdx);
                MicroAPI::Add(sumReg, sumReg, partialReg, fullMask);
                MicroAPI::Store(invRmsDst + mIdx, sumReg, 1U);
            }
        }
    }
}
template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::V0Prologue(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_ = vector_.chunkTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_ = vector_.matrixInfo_.totalLength - curBlock * vector_.chunkTSize_;
    }

    VectorComputeOffset();
    if (vector_.vectorOffset_.singleCoreM == 0) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
        AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        return;
    }

    for (uint32_t offsetNd = 0; offsetNd < vector_.matrixInfo_.nD; offsetNd += ND_LENGTH) {
        uint32_t curNdLen = ND_LENGTH;
        if (offsetNd + ND_LENGTH >= vector_.matrixInfo_.nD) {
            curNdLen = vector_.matrixInfo_.nD - offsetNd;
        }

        if (vector_.hasGamma_) {
            vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
            vector_.DataCopyGamma(curNdLen, offsetNd);
            vector_.gammaUb_ = vector_.gammaInQueue_.template DeQue<P>();
        }

        for (uint32_t offsetM = vector_.vectorOffset_.offsetMStart; offsetM < vector_.vectorOffset_.offsetMEnd;
             offsetM += v0BaseT_) {
            uint32_t curMLen = v0BaseT_;
            if (offsetM + v0BaseT_ >= vector_.vectorOffset_.offsetMEnd) {
                curMLen = vector_.vectorOffset_.offsetMEnd - offsetM;
            }
            uint64_t invRmsOffset = offsetM - vector_.vectorOffset_.offsetMStart;
            LocalTensor<P> invRmsUb = vector_.invRmsUb_[invRmsOffset];
            vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
            vector_.DataCopyX(curMLen, curNdLen, offsetM, offsetNd);
            vector_.xLocal_ = vector_.xInQueue_.template DeQue<T>();

            LocalTensor<P> aL1Ub = vector_.outQueue_.template AllocTensor<P>();

            if (vector_.hasGamma_) {
                if (offsetNd == 0) {
                    this->template VFDoV0ProcessXInSingleReduce<true, true>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)vector_.xLocal_.GetPhyAddr(), (__ubuf__ P *)vector_.gammaUb_.GetPhyAddr(),
                        curMLen, curNdLen);
                } else {
                    this->template VFDoV0ProcessXInSingleReduce<true, false>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)vector_.xLocal_.GetPhyAddr(), (__ubuf__ P *)vector_.gammaUb_.GetPhyAddr(),
                        curMLen, curNdLen);
                }
            } else {
                if (offsetNd == 0) {
                    this->template VFDoV0ProcessXInSingleReduce<false, true>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)vector_.xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                } else {
                    this->template VFDoV0ProcessXInSingleReduce<false, false>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)vector_.xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                }
            }

            vector_.outQueue_.template EnQue<P>(aL1Ub);
            aL1Ub = vector_.outQueue_.template DeQue<P>();
            this->DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd);

            vector_.xInQueue_.FreeTensor(vector_.xLocal_);
            vector_.outQueue_.FreeTensor(aL1Ub);
        }
        if (vector_.hasGamma_) {
            vector_.gammaInQueue_.FreeTensor(vector_.gammaUb_);
        }
    }
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
    AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);

    vector_.VFDoV0ProcessInvRms((__ubuf__ P *)vector_.invRmsUb_.GetPhyAddr(), vector_.vectorOffset_.singleCoreM,
                                vector_.scaleMean_, vector_.matrixInfo_.normEps);
    vector_.DataCopyOutInvRmsUb(vector_.vectorOffset_.singleCoreM, vector_.vectorOffset_.offsetMStart);
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::AIV1Process(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_ = vector_.chunkTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_ = vector_.matrixInfo_.totalLength - curBlock * vector_.chunkTSize_;
    }
    VectorComputeOffset();
    if (vector_.vectorOffset_.singleCoreM == 0) {
        return;
    }
    uint64_t lenT = 0;
    uint64_t singleCoreOffset = 0;
    for (int offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;
         offsetT += V1_BASE_T) {
        lenT = V1_BASE_T < vector_.vectorOffset_.offsetMEnd - offsetT ? V1_BASE_T :
                                                                        vector_.vectorOffset_.offsetMEnd - offsetT;
        HMixProcess(offsetT, lenT);
        LocalTensor<P> hMixLocal = vector_.xInQueue_.template DeQue<P>();
        vector_.AIV1PostProcessTile(hMixLocal, offsetT, lenT, static_cast<uint32_t>(singleCoreOffset),
                                    vector_.mnConfig_.n, kXInQueueBufferBytes, kOutQueueBufferBytes);
        vector_.xInQueue_.FreeTensor(hMixLocal);

        singleCoreOffset += lenT;
    }
}

template <class T, class P, int8_t RESI_MODE>
__aicore__ inline void MhcPreSplitND<T, P, RESI_MODE>::HMixProcess(uint64_t offsetT, uint64_t lenT)
{
    uint32_t mmResGmBlockNum = Ceil(vector_.matrixInfo_.nD, chunNDSize_);
    uint32_t computeLen = lenT * vector_.mnConfig_.n;
    // Keep each partial at a 32-byte-aligned UB stride so vector Add and DMA start aligned.
    uint32_t alignLen = MhcPreAlign(computeLen, 8);
    uint64_t HMixOffset = (vector_.globalOffsetM_ + offsetT) * vector_.mnConfig_.n;

    LocalTensor<P> hMixLocal = vector_.xInQueue_.template AllocTensor<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(mmResGmBlockNum);
    copyParams.blockLen = uint32_t(computeLen * sizeof(P));
    copyParams.srcStride =
        uint32_t((vector_.mnConfig_.curSingleCoreM * vector_.mnConfig_.curSingleCoreN - computeLen) * sizeof(P));
    copyParams.dstStride = uint32_t((alignLen - computeLen) * sizeof(P) / MHC_PRE_UB_ALIGN_SIZE);
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};
    DataCopyPad(hMixLocal, tempMMResGm_[HMixOffset], copyParams, copyPadParams);
    vector_.xInQueue_.EnQue(hMixLocal);
    hMixLocal = vector_.xInQueue_.template DeQue<P>();

    uint64_t addOffset = 0;
    for (uint32_t mmResGmBlockIdx = 1; mmResGmBlockIdx < mmResGmBlockNum; mmResGmBlockIdx++) {
        addOffset += alignLen;
        Add(hMixLocal, hMixLocal, hMixLocal[addOffset], computeLen);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
    if (vector_.outFlag_) {
        DataCopyExtParams MixOutCopyParams;
        MixOutCopyParams.blockCount = static_cast<uint16_t>(1);
        MixOutCopyParams.blockLen = uint32_t(computeLen * sizeof(P));
        MixOutCopyParams.srcStride = uint32_t(0);
        MixOutCopyParams.dstStride = uint32_t(0);
        DataCopyPad(mmResGm_[HMixOffset], hMixLocal, MixOutCopyParams);
    }
    vector_.xInQueue_.EnQue(hMixLocal);
}

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_ND_H_

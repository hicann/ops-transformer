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
 * \file mhc_pre_split_bs.h
 * \brief BS-split Basic API implementation for MHC Pre
 */

#ifndef MHC_PRE_SPLIT_BS_H
#define MHC_PRE_SPLIT_BS_H

#include "mhc_pre_vector_compute.h"
#include "mhc_pre_cube_compute.h"

namespace MhcPre {

constexpr uint32_t MHC_PRE_BS_SEQUENTIAL_PARTIAL_K = 1024U;

template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitBS {
public:
    __aicore__ inline MhcPreSplitBS() = default;

    __aicore__ inline void Init(InitParams initParams)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.InitHMixBuffer(initParams);
        vector_.InitPipeAndCoreIdx(initParams.tPipeIn);

        InitUbBuffersBasicApi();
        SyncAll<false>();

        if ASCEND_IS_AIV {
            vector_.coreIdx_ = GetBlockIdx() / vector_.kDoubleBufferCount;
            vector_.AIVPreLoad();
        }
        if ASCEND_IS_AIC {
            vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
            aL1_ = l1Buffer_.Get<float>();
            bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
            mmService_.Init(vector_.implMode_);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t tBlockNum = Ceil(vector_.totalLength_, vector_.chunkTSize_);

        for (uint64_t curblock = vector_.coreIdx_; curblock < tBlockNum; curblock += vector_.coreNum_) {
            vector_.InitBlockParams(curblock, tBlockNum);
            ConfigureDirectUbRowPartition();

            if ASCEND_IS_AIC {
                vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.singleCoreK;
            }

            uint32_t partialIndex = 0U;
            for (uint32_t offsetNd = 0; offsetNd < vector_.matrixInfo_.nD; offsetNd += vector_.ND_LENGTH) {
                uint32_t curNdLen = MhcPreMin(vector_.ND_LENGTH, vector_.matrixInfo_.nD - offsetNd);
                bool isPartialEnd = IsSequentialPartialEnd(offsetNd, curNdLen);
                // Two X staging slots are free initially. From the third slice on,
                // AIV waits until AIC returns the slot through X_CONSUMED_FLAG.
                if ASCEND_IS_AIV {
                    if (vector_.vectorCount_ >= 2) {
                        AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);
                    }
                    vector_.V0Prologue(curNdLen, offsetNd);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
                    vector_.vectorCount_++;
                }

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_READY_FLAG);
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_READY_FLAG +
                                                                                        MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    AICProcessBasicApi(offsetNd, partialIndex);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_CONSUMED_FLAG);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_CONSUMED_FLAG +
                                                                              MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                }
                if (isPartialEnd) {
                    partialIndex++;
                }
            }
            vector_.mmCount_++;
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG);
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG +
                                                                         MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            }

            if ASCEND_IS_AIV {
                vector_.V0PostProcess(curblock, tBlockNum);
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_MM_READY_FLAG);
                AIV1ProcessDirect();
                vector_.vec1Count_++;
            }
        }

        if ASCEND_IS_AIV {
            vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
            vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
        } else {
            mmService_.End(vector_.implMode_);
        }
    }

private:
    __aicore__ inline void InitUbBuffersBasicApi()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, vector_.kXInQueueBufferBytes);
        vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, vector_.kOutQueueBufferBytes);

        uint32_t invRmsRows = (vector_.curSingleT_ + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
        uint32_t invRmsBytes = MhcPreAlign(invRmsRows, vector_.kAlignmentBytes / sizeof(P)) * sizeof(P);
        vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, vector_.kSingleBufferCount, invRmsBytes);

        if (vector_.hasGamma_) {
            vector_.pipe_->InitBuffer(vector_.gammaInQueue_, vector_.kSingleBufferCount, vector_.ND_LENGTH * sizeof(P));
        }

        uint32_t parameterBufferBytes = MhcPreAlign(vector_.mnConfig_.n * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.biasInQue_, vector_.kSingleBufferCount, parameterBufferBytes);
        vector_.pipe_->InitBuffer(vector_.alphaBuf_, parameterBufferBytes);
        vector_.alphaInUb_ = vector_.alphaBuf_.template Get<P>();

        uint32_t hSegmentBytes = MhcPreAlign(vector_.V1_BASE_T * vector_.N_ * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.tmpBuff_, 2U * hSegmentBytes);
        vector_.hPreBuff_ =
            vector_.tmpBuff_.template GetWithOffset<P>(static_cast<uint32_t>(vector_.V1_BASE_T * vector_.N_), 0);
        vector_.hPostBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(
            static_cast<uint32_t>(vector_.V1_BASE_T * vector_.N_), hSegmentBytes);
    }

    __aicore__ inline void ConfigureDirectUbRowPartition()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        uint64_t firstRows = (vector_.curSingleT_ + 1U) / 2U;
        if (vector_.subBlockIdx_ == 0) {
            vector_.vectorOffset_.singleCoreM = firstRows;
            vector_.vectorOffset_.offsetMStart = 0;
            vector_.vectorOffset_.offsetMEnd = firstRows;
        } else {
            vector_.vectorOffset_.singleCoreM = vector_.curSingleT_ - firstRows;
            vector_.vectorOffset_.offsetMStart = firstRows;
            vector_.vectorOffset_.offsetMEnd = vector_.curSingleT_;
        }
    }

    __aicore__ inline bool IsSequentialPartialEnd(uint32_t offsetNd, uint32_t currentK) const
    {
        uint32_t endK = offsetNd + currentK;
        return endK >= vector_.matrixInfo_.nD || endK % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
    }

    __aicore__ inline void AICProcessBasicApi(uint32_t offsetNd, uint32_t partialIndex)
    {
        if (offsetNd + vector_.mnConfig_.singleCoreK > vector_.mnConfig_.k) {
            vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.k - offsetNd;
        }

        uint32_t currentK = vector_.mnConfig_.curSingleCoreK;
        uint64_t xOffset = vector_.chunkTSize_ * vector_.ND_LENGTH *
                           (vector_.coreIdx_ + (vector_.cubeCount_ % vector_.PARALLEL_NUM) * vector_.coreNum_);
        uint8_t aL1BufferId = vector_.cubeCount_ & 1;
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();

        LocalTensor<float> currentAL1 = aL1_[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        LocalTensor<float> currentBL1 = bL1_[bL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        mmService_.CopyInA1Nd2Nz(vector_.mnConfig_.curSingleCoreM, currentK, vector_.xFloatGm_[xOffset], currentAL1);
        mmService_.CopyInB1Nd2Nz(vector_.mnConfig_.k, currentK, vector_.mnConfig_.n, vector_.phiGm_[offsetNd],
                                 currentBL1);

        bool isFirstKL1 = offsetNd % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
        bool isLastKL1 = IsSequentialPartialEnd(offsetNd, currentK);
        uint64_t mAlign = BasicApiAlign(vector_.mnConfig_.curSingleCoreM, AscendC::BLOCK_CUBE);
        uint64_t nAlign = BasicApiAlign(vector_.mnConfig_.n, AscendC::BLOCK_CUBE);
        // Fill available L0 from the live M/N footprint instead of hard-coding baseK for one shape.
        uint64_t baseK = (256 / AscendC::Std::max(mAlign, nAlign)) * 32;
        mmService_.Process(vector_.mnConfig_.curSingleCoreM, vector_.mnConfig_.n, vector_.mnConfig_.curSingleCoreM,
                           baseK, isFirstKL1, isLastKL1, currentAL1, currentBL1);

        if (isLastKL1) {
            // Emit ordered K=1024 partials. Later partials atomically accumulate into the same compact
            // hMix rows, avoiding a separate vector reduction kernel while retaining the required order.
            if (partialIndex != 0U) {
                AscendC::SetAtomicAdd<float>();
            }
            uint64_t outputOffset = vector_.globalOffsetM_ * vector_.mnConfig_.n;
            mmService_.CopyOut(vector_.hMixGm_[outputOffset], vector_.mnConfig_.n);
            if (offsetNd + currentK >= vector_.matrixInfo_.nD) {
                AscendC::DisableDmaAtomic();
            }
        }
        vector_.cubeCount_++;
    }

    __aicore__ inline void AIV1ProcessDirect()
    {
        if (vector_.vectorOffset_.singleCoreM == 0) {
            return;
        }

        uint64_t localOffset = 0;
        for (uint64_t offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;
             offsetT += vector_.V1_BASE_T) {
            uint64_t lenT =
                AscendC::Std::min(static_cast<uint64_t>(vector_.V1_BASE_T), vector_.vectorOffset_.offsetMEnd - offsetT);
            uint64_t hMixOffset = (vector_.globalOffsetM_ + offsetT) * vector_.mnConfig_.n;
            vector_.HMixCopyIn(hMixOffset, lenT);
            LocalTensor<P> hMixLocal = vector_.xInQueue_.template DeQue<P>();
            vector_.AIV1PostProcessTile(hMixLocal, offsetT, lenT, static_cast<uint32_t>(localOffset),
                                        vector_.mnConfig_.n);
            vector_.xInQueue_.FreeTensor(hMixLocal);
            localOffset += lenT;
        }
    }

    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    MhcPreCubeCompute mmService_;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
};

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_H

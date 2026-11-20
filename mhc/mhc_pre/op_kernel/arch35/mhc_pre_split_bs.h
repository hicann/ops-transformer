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
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        vector_.InitFromTilingData(initParams.tilingData);
        stageRows_ = initParams.tilingData->bsStageRows;
        vector_.InitMNConfig();
        vector_.InitHMixBuffer(initParams);
        vector_.InitPipeAndCoreIdx(initParams.tPipeIn);
        alignedMixWidth_ = initParams.tilingData->fusionAlign;
        xQueueBytes_ = initParams.tilingData->bsXQueueBytes;

        // Fixpipe addresses the same UB offset on the AIC and both paired AIVs.
        uint32_t mixBytes = MhcPreAlign(((vector_.chunkTSize_ + 1U) / 2U) * alignedMixWidth_ * sizeof(P),
                                        MHC_PRE_BASIC_API_VL_FP32 * sizeof(P));
        vector_.pipe_->InitBuffer(partialBuffer_, mixBytes);
        partialLocal_ = partialBuffer_.Get<P>();
        if ASCEND_IS_AIV {
            vector_.pipe_->InitBuffer(accumBuffer_, mixBytes);
            accumLocal_ = accumBuffer_.Get<P>();
        }
        InitUbBuffersBasicApi();
        vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<float>();
        bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        if ASCEND_IS_AIV {
            if (stageRows_ == MHC_PRE_BS_DEFAULT_STAGE_ROWS) {
                vector_.pipe_->InitBuffer(nzBuffer_,
                                          (MHC_PRE_BS_DEFAULT_STAGE_ROWS + 1U) * vector_.ND_LENGTH * sizeof(P));
                nzLocal_ = nzBuffer_.Get<P>();
            }
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            vector_.coreIdx_ = GetBlockIdx() / vector_.kDoubleBufferCount;
            vector_.AIVPreLoad();
        }
        if ASCEND_IS_AIC {
            mmService_.Init(vector_.implMode_);
        }
    }
#else
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.InitHMixBuffer(initParams);
        vector_.InitPipeAndCoreIdx(initParams.tPipeIn);

        InitUbBuffersBasicApi();
        SyncAll<false>();

        if ASCEND_IS_AIV {
            vector_.coreIdx_ = GetBlockIdx() / MHC_PRE_AIV_PER_AIC;
            vector_.AIVPreLoad();
        }
        if ASCEND_IS_AIC {
            vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
            aL1_ = l1Buffer_.Get<float>();
            bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
            mmService_.Init(vector_.implMode_);
        }
    }
#endif

    __aicore__ inline void Process()
#if defined(MHC_PRE_ENABLE_ASCEND950)
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
                    if (vector_.hasGamma_) {
                        V0PrologueL1<true>(curNdLen, offsetNd);
                    } else {
                        V0PrologueL1<false>(curNdLen, offsetNd);
                    }
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
                    vector_.vectorCount_++;
                    if (isPartialEnd) {
                        ConsumePartial(partialIndex == 0U);
                    }
                }

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG);
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG +
                                                                                        MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    AICProcessBasicApi(offsetNd);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                              MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                }
                if (isPartialEnd) {
                    partialIndex++;
                }
            }
            vector_.mmCount_++;
            if ASCEND_IS_AIV {
                vector_.V0PostProcess(curblock, tBlockNum);
                AIV1ProcessDirect();
                vector_.vec1Count_++;
            }
        }

        if ASCEND_IS_AIV {
            vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
            vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
        } else {
            if (partialSent_) {
                WaitPartialConsumed();
            }
            mmService_.End(vector_.implMode_);
        }
    }
#else
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
                    if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                        AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
                            MHC_PRE_X_READY_FLAG + MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    }
                    AICProcessBasicApi(offsetNd, partialIndex);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_CONSUMED_FLAG);
                    if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_CONSUMED_FLAG +
                                                                                  MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    }
                }
                if (isPartialEnd) {
                    partialIndex++;
                }
            }
            vector_.mmCount_++;
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG);
                if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG +
                                                                             MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                }
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
            mmService_.End();
        }
    }
#endif

private:
#if defined(MHC_PRE_ENABLE_ASCEND950)
    uint32_t stageRows_ = MHC_PRE_BS_DEFAULT_STAGE_ROWS;
    uint32_t alignedMixWidth_ = 0U;
    uint32_t xQueueBytes_ = 0U;
    uint32_t rmsCorrectionOffset_ = 0U;
    bool partialSent_ = false;
#endif

    __aicore__ inline void InitUbBuffersBasicApi()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, xQueueBytes_);
        vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, MHC_PRE_BS_OUT_QUEUE_BYTES);

        uint32_t invRmsRows = (vector_.curSingleT_ + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
        uint32_t invRmsBytes = MhcPreAlign(invRmsRows, vector_.kAlignmentBytes / sizeof(P)) * sizeof(P);
        rmsCorrectionOffset_ = invRmsBytes / sizeof(P);
        vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, vector_.kSingleBufferCount, 2U * invRmsBytes);

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
#else
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, vector_.kXInQueueBufferBytes);
        vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, vector_.kOutQueueBufferBytes);

        uint32_t invRmsRows = MHC_PRE_AIV_PER_AIC == 1U ?
                                  vector_.curSingleT_ :
                                  (vector_.curSingleT_ + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
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
#endif

    __aicore__ inline void ConfigureDirectUbRowPartition()
#if defined(MHC_PRE_ENABLE_ASCEND950)
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
#else
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if constexpr (MHC_PRE_AIV_PER_AIC == 1U) {
            vector_.vectorOffset_.singleCoreM = vector_.curSingleT_;
            vector_.vectorOffset_.offsetMStart = 0U;
            vector_.vectorOffset_.offsetMEnd = vector_.curSingleT_;
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
#endif

    __aicore__ inline bool IsSequentialPartialEnd(uint32_t offsetNd, uint32_t currentK) const
    {
        uint32_t endK = offsetNd + currentK;
        return endK >= vector_.matrixInfo_.nD || endK % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline void AICProcessBasicApi(uint32_t offsetNd)
    {
        if (offsetNd + vector_.mnConfig_.singleCoreK > vector_.mnConfig_.k) {
            vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.k - offsetNd;
        }

        uint32_t currentK = vector_.mnConfig_.curSingleCoreK;
        uint8_t aL1BufferId = vector_.cubeCount_ & 1;
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();

        LocalTensor<float> currentAL1 = aL1_[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        LocalTensor<float> currentBL1 = bL1_[bL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        // X has already arrived in this core's L1 through its paired AIVs.
        mmService_.CopyInB1Nd2Nz(vector_.mnConfig_.k, currentK, vector_.mnConfig_.n, vector_.phiGm_[offsetNd],
                                 currentBL1);

        bool isFirstKL1 = offsetNd % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
        bool isLastKL1 = IsSequentialPartialEnd(offsetNd, currentK);
        uint64_t mAlign = BasicApiAlign(vector_.mnConfig_.curSingleCoreM, AscendC::BLOCK_CUBE);
        uint64_t nAlign = BasicApiAlign(vector_.mnConfig_.n, AscendC::BLOCK_CUBE);
        // Fill available L0 from the live M/N footprint instead of hard-coding baseK for one shape.
        uint64_t baseK = (256 / AscendC::Std::max(mAlign, nAlign)) * 32;
        mmService_.template Process<true>(vector_.mnConfig_.curSingleCoreM, vector_.mnConfig_.n,
                                          vector_.mnConfig_.curSingleCoreM, baseK, isFirstKL1, isLastKL1, currentAL1,
                                          currentBL1);

        if (isLastKL1) {
            if (partialSent_) {
                WaitPartialConsumed();
            }
            mmService_.CopyOut(partialLocal_);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_READY_FLAG);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_READY_FLAG +
                                                                     MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            partialSent_ = true;
        }
        vector_.cubeCount_++;
    }

    __aicore__ inline void WaitPartialConsumed()
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_CONSUMED_FLAG);
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_CONSUMED_FLAG +
                                                                  MHC_PRE_SUBBLOCK_FLAG_OFFSET);
    }

    __aicore__ inline void ConsumePartial(bool first)
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_V>(MHC_PRE_BS_PARTIAL_READY_FLAG);
        const uint32_t elements = vector_.vectorOffset_.singleCoreM * alignedMixWidth_;
        if (elements != 0U) {
            if (first) {
                MhcPreAccumulateBsPartial<true>(accumLocal_, partialLocal_, elements);
            } else {
                MhcPreAccumulateBsPartial<false>(accumLocal_, partialLocal_, elements);
            }
        }
        // The paired Cube can overwrite the slot after both AIVs finish reading it.
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_V>(MHC_PRE_BS_PARTIAL_CONSUMED_FLAG);
    }
#else
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
#endif

    __aicore__ inline void AIV1ProcessDirect()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        if (vector_.vectorOffset_.singleCoreM == 0) {
            return;
        }

        if (vector_.outFlag_) {
            auto ready = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            const uint32_t mixWidth = vector_.mnConfig_.n;
            const uint64_t row = vector_.globalOffsetM_ + vector_.vectorOffset_.offsetMStart;
            DataCopyExtParams copy{static_cast<uint16_t>(vector_.vectorOffset_.singleCoreM),
                                   mixWidth * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
            DataCopyPad(vector_.hMixGm_[row * mixWidth], accumLocal_, copy);
        }
        uint64_t localOffset = 0;
        for (uint64_t offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;
             offsetT += vector_.V1_BASE_T) {
            uint64_t lenT =
                AscendC::Std::min(static_cast<uint64_t>(vector_.V1_BASE_T), vector_.vectorOffset_.offsetMEnd - offsetT);
            vector_.AIV1PostProcessTile(accumLocal_[localOffset * alignedMixWidth_], offsetT, lenT,
                                        static_cast<uint32_t>(localOffset), alignedMixWidth_, xQueueBytes_,
                                        MHC_PRE_BS_OUT_QUEUE_BYTES);
            localOffset += lenT;
        }
        if (vector_.outFlag_) {
            auto done = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(done);
            WaitFlag<HardEvent::MTE3_V>(done);
        }
    }

    __aicore__ inline LocalTensor<P> GetNzTile(const LocalTensor<T>& x)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            if (stageRows_ == MHC_PRE_BS_EXTENDED_STAGE_ROWS) {
                // X and NZ share a queue slot, which stays live until MTE3 finishes reading NZ.
                using VectorCompute = MhcPreVectorCompute<T, P, RESI_MODE>;
                constexpr uint32_t xElements = MHC_PRE_BS_EXTENDED_STAGE_ROWS * VectorCompute::ND_LENGTH;
                static_assert(xElements * sizeof(T) +
                                  (MHC_PRE_BS_EXTENDED_STAGE_ROWS + 1U) * VectorCompute::ND_LENGTH * sizeof(P) <=
                              MHC_PRE_BS_X_QUEUE_MAX_BYTES);
                return x[xElements].template ReinterpretCast<P>();
            }
        }
        return nzLocal_;
    }

    template <bool HAS_GAMMA>
    __aicore__ inline void V0PrologueL1(uint32_t currentK, uint32_t offsetK)
    {
        __ubuf__ P* gamma = nullptr;
        if constexpr (HAS_GAMMA) {
            vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
            vector_.DataCopyGamma(currentK, offsetK);
            vector_.gammaUb_ = vector_.gammaInQueue_.template DeQue<P>();
            gamma = (__ubuf__ P*)vector_.gammaUb_.GetPhyAddr();
        }
        // This core owns complete rows and traverses all K. Paired AIVs split only those rows,
        // preserving the per-row operation order when B/S and the number of M blocks change.
        const uint32_t begin = vector_.vectorOffset_.offsetMStart;
        const uint32_t end = vector_.vectorOffset_.offsetMEnd;
        if (begin < end) {
            vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
            vector_.DataCopyX(MhcPreMin(uint32_t(stageRows_), end - begin), currentK, begin, offsetK);
        }
        for (uint32_t offsetM = begin; offsetM < end; offsetM += stageRows_) {
            uint32_t rows = MhcPreMin(uint32_t(stageRows_), end - offsetM);
            auto x = vector_.xInQueue_.template DeQue<T>();
            if (offsetM + rows < end) {
                vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
                vector_.DataCopyX(MhcPreMin(uint32_t(stageRows_), end - offsetM - rows), currentK, offsetM + rows,
                                  offsetK);
            }
            auto nd = GetNzTile(x);
            auto rms = vector_.invRmsUb_[offsetM - begin];
            auto correction = (__ubuf__ P*)vector_.invRmsUb_[rmsCorrectionOffset_ + offsetM - begin].GetPhyAddr();
            bool packed = false;
            if constexpr (IsSameType<T, bfloat16_t>::value && !HAS_GAMMA) {
                if (currentK % 128U == 0U) {
                    if (offsetK == 0U) {
                        MhcPreMkVFPackNz<true, true>(nd, rms, x, rows, currentK, nullptr, correction);
                    } else {
                        MhcPreMkVFPackNz<false, true>(nd, rms, x, rows, currentK, nullptr, correction);
                    }
                    packed = true;
                }
            }
            if (!packed) {
                auto ndAddr = (__ubuf__ P*)nd.GetPhyAddr();
                auto rmsAddr = (__ubuf__ P*)rms.GetPhyAddr();
                auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
                if (offsetK == 0U) {
                    vector_.template VFDoV0ProcessXIn<HAS_GAMMA, true, true, true, true>(
                        ndAddr, rmsAddr, xAddr, gamma, rows, currentK, 1U, correction);
                } else {
                    vector_.template VFDoV0ProcessXIn<HAS_GAMMA, false, true, true, true>(
                        ndAddr, rmsAddr, xAddr, gamma, rows, currentK, 1U, correction);
                }
            }
            auto ready = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            DataCopyParams copy;
            copy.blockCount = currentK / 8U;
            copy.blockLen = rows;
            copy.srcStride = BasicApiAlign(rows, 8U) + 1U - rows;
            copy.dstStride = BasicApiAlign(vector_.curSingleT_, AscendC::BLOCK_CUBE) - rows;
            BasicApiCopyToL1(nd, aL1_[(vector_.vectorCount_ % 2U) * MHC_PRE_BASIC_API_L1_BUF_OFFSET + offsetM * 8U],
                             copy);
            auto free = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(free);
            WaitFlag<HardEvent::MTE3_V>(free);
            vector_.xInQueue_.FreeTensor(x);
        }
        if constexpr (HAS_GAMMA) {
            vector_.gammaInQueue_.FreeTensor(vector_.gammaUb_);
        }
    }

    TBuf<TPosition::VECCALC> partialBuffer_;
    TBuf<TPosition::VECCALC> accumBuffer_;
    LocalTensor<P> partialLocal_;
    LocalTensor<P> accumLocal_;
    TBuf<TPosition::VECCALC> nzBuffer_;
    LocalTensor<P> nzLocal_;
#else
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
#endif

    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    MhcPreCubeCompute mmService_;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
};

#if defined(MHC_PRE_ENABLE_ASCEND950)
template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitBSWave {
public:
    __aicore__ inline void Init(InitParams params)
    {
        front_.BindGlobalTensors(params);
        front_.InitFromTilingData(params.tilingData);
        front_.InitMNConfig();
        front_.InitHMixBuffer(params);
        front_.InitPipeAndCoreIdx(params.tPipeIn);
        front_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        post_.BindGlobalTensors(params);
        post_.InitFromTilingData(params.tilingData);
        post_.InitMNConfig();
        post_.InitHMixBuffer(params);
        post_.InitPipeAndCoreIdx(params.tPipeIn);
        post_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        front_.pipe_->InitBuffer(l1_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1_.template Get<P>();
        bL1_ = aL1_[2U * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        // These first UB allocations have identical offsets on the AIC and both paired AIVs.
        front_.pipe_->InitBuffer(receive_, 2U * kMixElements * sizeof(P));
        recvLocal_ = receive_.template Get<P>();
        front_.pipe_->InitBuffer(accum_, 2U * kMixElements * sizeof(P));
        accLocal_ = accum_.template Get<P>();
        if ASCEND_IS_AIC {
            cube_.Init(front_.implMode_);
        }
        if ASCEND_IS_AIV {
            front_.coreIdx_ /= 2U;
            post_.coreIdx_ /= 2U;
            front_.pipe_->InitBuffer(front_.xInQueue_, 2U, kRows * 256U * sizeof(T));
            if (front_.hasGamma_) {
                front_.pipe_->InitBuffer(front_.gammaInQueue_, 1U, MHC_PRE_RMS_K_BLOCK * sizeof(P));
            }
            front_.pipe_->InitBuffer(nz_, (kRows + 1U) * 256U * sizeof(P));
            nzLocal_ = nz_.template Get<P>();
            front_.pipe_->InitBuffer(rms_, 4U * kRmsStride * sizeof(P));
            rmsLocal_ = rms_.template Get<P>();
            post_.curSingleT_ = kRows;
            post_.pipe_->InitBuffer(post_.xInQueue_, 2U, 4U * front_.D_ * sizeof(T));
            post_.pipe_->InitBuffer(post_.outQueue_, 2U, front_.D_ * sizeof(T));
            post_.pipe_->InitBuffer(post_.invRmsOutQueue_, 1U, kRmsStride * sizeof(P));
            post_.pipe_->InitBuffer(post_.biasInQue_, 1U, 96U);
            post_.pipe_->InitBuffer(post_.alphaBuf_, 96U);
            post_.alphaInUb_ = post_.alphaBuf_.template Get<P>();
            post_.pipe_->InitBuffer(post_.tmpBuff_, 2U * kCoeffStride * sizeof(P));
            post_.hPreBuff_ = post_.tmpBuff_.template Get<P>();
            post_.hPostBuff_ = post_.hPreBuff_[kCoeffStride];
            post_.pipe_->InitBuffer(mix_, (kRows * 24U + 64U) * sizeof(P));
            mixLocal_ = mix_.template Get<P>();
            post_.AIVPreLoad();
            nzEvent_ = static_cast<event_t>(front_.pipe_->template AllocEventID<HardEvent::MTE3_V>());
            rmsEvent_ = static_cast<event_t>(front_.pipe_->template AllocEventID<HardEvent::MTE3_V>());
            drainEvent_ = static_cast<event_t>(front_.pipe_->template AllocEventID<HardEvent::MTE3_V>());
            mixFreeEvent_ = static_cast<event_t>(front_.pipe_->template AllocEventID<HardEvent::V_MTE2>());
            mixReadyEvent_ = static_cast<event_t>(front_.pipe_->template AllocEventID<HardEvent::MTE2_V>());
        }
        SyncAll<false>();
    }
    __aicore__ inline void Process()
    {
        uint32_t blocks = BasicApiCeilDiv(front_.totalLength_, front_.chunkTSize_);
        for (uint32_t block = front_.coreIdx_; block < blocks; block += front_.coreNum_) {
            uint32_t begin = block * front_.chunkTSize_;
            uint32_t count = MhcPreMin(uint32_t(front_.chunkTSize_), uint32_t(front_.totalLength_ - begin));
            if ASCEND_IS_AIC {
                ProcessCube(begin, count);
            }
            if ASCEND_IS_AIV {
                ProcessVector(begin, count);
            }
        }
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < MhcPreMin(partialSeq_, 2U); ++i) {
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(kPartialConsumed);
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(kPartialConsumed + 16U);
            }
            cube_.End(front_.implMode_);
        }
        if ASCEND_IS_AIV {
            SetFlag<HardEvent::MTE3_V>(drainEvent_);
            WaitFlag<HardEvent::MTE3_V>(drainEvent_);
            post_.invRmsOutQueue_.FreeTensor(post_.invRmsUb_);
            post_.biasInQue_.FreeTensor(post_.biasInUb_);
        }
    }

private:
    static constexpr uint32_t kWaveM = 48U;
    static constexpr uint32_t kRows = 24U;
    static constexpr uint32_t kMixElements = kRows * 24U;
    static constexpr uint16_t kPartialReady = 11U;
    static constexpr uint16_t kPartialConsumed = 12U;
    static constexpr uint32_t kRmsStride = 64U;
    static constexpr uint32_t kCoeffStride = 128U;
    __aicore__ inline void ProcessCube(uint32_t begin, uint32_t count)
    {
        for (uint32_t off = 0U; off < count; off += kWaveM) {
            uint32_t m = MhcPreMin(kWaveM, count - off);
            uint32_t partial = 0U;
            uint32_t baseK = 256U / AscendC::Std::max(BasicApiAlign(m, 16U), 32U) * 32U;
            for (uint32_t k = 0U; k < front_.matrixInfo_.nD; k += 256U) {
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG);
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG + 16U);
                auto a = aL1_[(sequence_ % 2U) * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
                auto b = bL1_[cube_.GetBL1BufferId() * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
                cube_.CopyInB1Nd2Nz(front_.matrixInfo_.nD, 256U, 24U, front_.phiGm_[k], b);
                bool first = k % 1024U == 0U;
                bool last = (k + 256U) % 1024U == 0U || k + 256U >= front_.matrixInfo_.nD;
                cube_.template Process<true>(m, 24U, m, baseK, first, last, a, b);
                if (last) {
                    if (partialSeq_ >= 2U) {
                        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(kPartialConsumed);
                        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(kPartialConsumed + 16U);
                    }
                    CopyPartialToUb(partialSeq_ % 2U);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(kPartialReady);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(kPartialReady + 16U);
                    ++partialSeq_;
                    ++partial;
                }
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG + 16U);
                ++sequence_;
            }
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG + 16U);
        }
    }
    __aicore__ inline void CopyPartialToUb(uint32_t slot)
    {
        cube_.CopyOut(recvLocal_[slot * kMixElements]);
    }

    __aicore__ inline void ReceivePartial(uint32_t wave, uint32_t rows, bool first)
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_V>(kPartialReady);
        if (rows != 0U) {
            auto input = recvLocal_[(partialSeq_ % 2U) * kMixElements];
            auto acc = accLocal_[(wave % 2U) * kMixElements];
            if (first) {
                MhcPreAccumulateBsPartial<true>(acc, input, rows * 24U);
            } else {
                MhcPreAccumulateBsPartial<false>(acc, input, rows * 24U);
            }
        }
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_V>(kPartialConsumed);
        ++partialSeq_;
    }
    __aicore__ inline void ExportMix(uint32_t wave, uint32_t row, uint32_t rows)
    {
        if (rows == 0U) {
            return;
        }
        auto e = static_cast<event_t>(front_.pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
        DataCopyExtParams copy{1U, rows * 24U * uint32_t(sizeof(P)), 0U, 0U, 0U};
        DataCopyPad(front_.hMixGm_[row * 24U], accLocal_[(wave % 2U) * kMixElements], copy);
        // Subsequent NZ-copy fences, PreparePost, and the final drain protect this store before bank reuse.
    }
    __aicore__ inline void PrefetchFront(uint32_t rows, uint32_t rowStart, uint32_t k)
    {
        if (rows == 0U) {
            return;
        }
        front_.xLocal_ = front_.xInQueue_.template AllocTensor<T>();
        front_.DataCopyX(rows, 256U, rowStart, k);
    }
    __aicore__ inline void Normalize(uint32_t wave, uint32_t row, uint32_t rows)
    {
        if (rows == 0U) {
            return;
        }
        auto rms = rmsLocal_[(wave % 2U) * kRmsStride];
        front_.VFDoV0ProcessInvRms((__ubuf__ P*)rms.GetPhyAddr(), rows, front_.scaleMean_, front_.matrixInfo_.normEps);
        auto e = static_cast<event_t>(front_.pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
        DataCopyExtParams copy{1U, rows * uint32_t(sizeof(P)), 0U, 0U, 0U};
        DataCopyPad(front_.invRmsGm_[row], rms, copy);
        SetFlag<HardEvent::MTE3_V>(rmsEvent_);
        WaitFlag<HardEvent::MTE3_V>(rmsEvent_);
    }
    __aicore__ inline void PreparePost(uint32_t wave, uint32_t row, uint32_t rows)
    {
        if (rows == 0U) {
            return;
        }
        SetFlag<HardEvent::MTE3_V>(drainEvent_);
        WaitFlag<HardEvent::MTE3_V>(drainEvent_);
        post_.globalOffsetM_ = row;
        postRows_ = rows;
        DataCopy(post_.invRmsUb_, rmsLocal_[(wave % 2U) * kRmsStride], kRmsStride);
        mixLocal_ = accLocal_[(wave % 2U) * kMixElements];
        PipeBarrier<PIPE_V>();
        auto res = post_.outQueue_.template AllocTensor<P>();
        MhcPreVFPostSegments<P, 4U, true>(post_.hPreBuff_, post_.hPostBuff_, res, mixLocal_, post_.invRmsUb_,
                                          post_.alphaInUb_, post_.biasInUb_, rows, 0U, 24U);
        PipeBarrier<PIPE_V>();
        post_.outQueue_.template EnQue<P>(res);
        res = post_.outQueue_.template DeQue<P>();
        DataCopyExtParams rcopy{1U, rows * 16U * uint32_t(sizeof(P)), 0U, 0U, 0U};
        DataCopyPad(post_.hResGm_[row * 16U], res, rcopy);
        post_.outQueue_.FreeTensor(res);
        post_.AIV1ProcessHPost(0U, rows);
        post_.AIV1ProcessHPre(0U, rows);
        PostLoad(0U);
    }
    __aicore__ inline void PostLoad(uint32_t ordinal)
    {
        auto x = post_.xInQueue_.template AllocTensor<T>();
        DataCopyExtParams copy{1U, uint32_t(4U * front_.D_ * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, 0U};
        DataCopyPad(x, post_.xGm_[(post_.globalOffsetM_ + ordinal) * 4U * front_.D_], copy, pad);
        post_.xInQueue_.template EnQue<T>(x);
    }
    __aicore__ inline void PostTick(uint32_t ordinal)
    {
        auto x = post_.xInQueue_.template DeQue<T>();
        if (ordinal + 1U < postRows_) {
            PostLoad(ordinal + 1U);
        }
        auto out = post_.outQueue_.template AllocTensor<T>();
        post_.template VFDoV1ProcessHInRows<4U>((__ubuf__ T*)x.GetPhyAddr(), (__ubuf__ T*)out.GetPhyAddr(),
                                                (__ubuf__ P*)post_.hPreBuff_[ordinal * 4U].GetPhyAddr(), 1U, front_.D_,
                                                front_.D_);
        post_.xInQueue_.FreeTensor(x);
        post_.outQueue_.template EnQue<T>(out);
        out = post_.outQueue_.template DeQue<T>();
        DataCopyExtParams copy{1U, uint32_t(front_.D_ * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(post_.hinGm_[(post_.globalOffsetM_ + ordinal) * front_.D_], out, copy);
        post_.outQueue_.FreeTensor(out);
    }
    __aicore__ inline void ProcessVector(uint32_t begin, uint32_t count)
    {
        uint32_t previousRow = 0U;
        uint32_t previousRows = 0U;
        uint32_t wave = 0U;
        for (uint32_t off = 0U; off < count; off += kWaveM, ++wave) {
            uint32_t m = MhcPreMin(kWaveM, count - off);
            uint32_t first = (m + 1U) / 2U;
            uint32_t rowStart = GetSubBlockIdx() == 0U ? 0U : first;
            uint32_t rows = GetSubBlockIdx() == 0U ? first : m - first;
            front_.globalOffsetM_ = begin + off;
            auto rms = rmsLocal_[(wave % 2U) * kRmsStride];
            auto correction = (__ubuf__ P*)rmsLocal_[(2U + wave % 2U) * kRmsStride].GetPhyAddr();
            PrefetchFront(rows, rowStart, 0U);
            uint32_t postDone = 0U;
            uint32_t tile = 0U;
            uint32_t received = 0U;
            uint32_t postEvery =
                previousRows ?
                    MhcPreMin(4U, AscendC::Std::max(1U, (uint32_t(front_.matrixInfo_.nD) / 256U - 2U) / previousRows)) :
                    1U;
            for (uint32_t k = 0U; k < front_.matrixInfo_.nD; k += 256U, ++tile) {
                if (sequence_ >= 2U) {
                    CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);
                }
                if (rows != 0U) {
                    auto x = front_.xInQueue_.template DeQue<T>();
                    if (k + 256U < front_.matrixInfo_.nD) {
                        PrefetchFront(rows, rowStart, k + 256U);
                    }
                    LocalTensor<P> gamma;
                    __ubuf__ P* gammaAddr = nullptr;
                    if (front_.hasGamma_) {
                        gamma = front_.LoadGammaTile(k);
                        gammaAddr = (__ubuf__ P*)gamma.GetPhyAddr();
                    }
                    if (k == 0U) {
                        MhcPreMkVFPackNz<true, true>(nzLocal_, rms, x, rows, 256U, gammaAddr, correction);
                    } else {
                        MhcPreMkVFPackNz<false, true>(nzLocal_, rms, x, rows, 256U, gammaAddr, correction);
                    }
                    front_.xInQueue_.FreeTensor(x);
                    auto e = static_cast<event_t>(front_.pipe_->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(e);
                    WaitFlag<HardEvent::V_MTE3>(e);
                    if (front_.hasGamma_) {
                        front_.gammaInQueue_.FreeTensor(gamma);
                    }
                    DataCopyParams copy{32U, static_cast<uint16_t>(rows),
                                        static_cast<uint16_t>(BasicApiAlign(rows, 8U) + 1U - rows),
                                        static_cast<uint16_t>(BasicApiAlign(m, 16U) - rows)};
                    BasicApiCopyToL1(nzLocal_, aL1_[(sequence_ % 2U) * MHC_PRE_BASIC_API_L1_BUF_OFFSET + rowStart * 8U],
                                     copy);
                    SetFlag<HardEvent::MTE3_V>(nzEvent_);
                    WaitFlag<HardEvent::MTE3_V>(nzEvent_);
                }
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
                ++sequence_;
                // Feed two more X tiles before consuming the previous K1024 partial.
                if (tile >= 5U && tile % 4U == 1U) {
                    ReceivePartial(wave, rows, received == 0U);
                    ++received;
                }
                if (wave != 0U && tile == 1U) {
                    CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_MM_READY_FLAG);
                    // Keep the next wave's first two X tiles ahead of the final partial wait.
                    ReceivePartial(wave - 1U, previousRows, false);
                    ExportMix(wave - 1U, previousRow, previousRows);
                    PreparePost(wave - 1U, previousRow, previousRows);
                }
                if (wave != 0U && tile >= 2U && (tile - 2U) % postEvery == 0U && postDone < previousRows) {
                    PostTick(postDone++);
                }
            }
            // RMS depends on X only; complete it before waiting for the final Cube partial.
            Normalize(wave, begin + off + rowStart, rows);
            bool moreWaves = off + kWaveM < count;
            uint32_t receiveLimit = BasicApiCeilDiv(front_.matrixInfo_.nD, 1024U) - (moreWaves ? 1U : 0U);
            while (received < receiveLimit) {
                ReceivePartial(wave, rows, received == 0U);
                ++received;
            }
            if (!moreWaves) {
                ExportMix(wave, begin + off + rowStart, rows);
            }
            if (wave != 0U) {
                while (postDone < previousRows) {
                    PostTick(postDone++);
                }
            }
            previousRow = begin + off + rowStart;
            previousRows = rows;
        }
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_MM_READY_FLAG);
        PreparePost(wave - 1U, previousRow, previousRows);
        for (uint32_t i = 0U; i < previousRows; ++i) {
            PostTick(i);
        }
    }
    MhcPreVectorCompute<T, P, RESI_MODE> front_, post_;
    MhcPreCubeCompute cube_;
    TBuf<TPosition::A1> l1_;
    TBuf<TPosition::VECCALC> nz_;
    TBuf<TPosition::VECCALC> rms_;
    TBuf<TPosition::VECCALC> mix_;
    TBuf<TPosition::VECCALC> receive_;
    TBuf<TPosition::VECCALC> accum_;
    LocalTensor<P> recvLocal_;
    LocalTensor<P> accLocal_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    LocalTensor<P> nzLocal_;
    LocalTensor<P> rmsLocal_;
    LocalTensor<P> mixLocal_;
    event_t nzEvent_;
    event_t rmsEvent_;
    event_t drainEvent_;
    event_t mixFreeEvent_;
    event_t mixReadyEvent_;
    uint32_t sequence_ = 0U;
    uint32_t postRows_ = 0U;
    uint32_t partialSeq_ = 0U;
};
#endif

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_H

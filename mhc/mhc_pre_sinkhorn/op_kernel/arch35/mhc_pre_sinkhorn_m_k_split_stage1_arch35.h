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
 * \file mhc_pre_sinkhorn_m_k_split_stage1_arch35.h
 * \brief Shared stage-1 implementation.
 */

#ifndef MHC_PRE_SINKHORN_M_K_SPLIT_STAGE1_ARCH35_H
#define MHC_PRE_SINKHORN_M_K_SPLIT_STAGE1_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_base_arch35.h"
#include "mhc_pre_sinkhorn_cube_compute_arch35.h"

namespace MhcPreSinkhornNs {
using namespace AscendC;

template <typename T, uint64_t CV_RATIO = 2>
class MhcPreSinkhornMKSplitCorePart1 {
public:
    __aicore__ inline MhcPreSinkhornMKSplitCorePart1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR preMix, GM_ADDR y, GM_ADDR workspace,
                                const MhcPreSinkhornRegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        premixStage1_ = tilingData->premixStage1 != 0;
        if (premixStage1_) {
            preMixGm.SetGlobalBuffer((__gm__ float*)preMix);
            yGm.SetGlobalBuffer((__gm__ T*)y);
            ubRowGapBlocks_ = UbRowGapBlocks(tilingData->hcMult, tilingData->hcMix);
        }
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcFnGm.SetGlobalBuffer((__gm__ float*)hcFn);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];

        // InQue
        pipe->InitBuffer(xQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));

        // OutQue
        pipe->InitBuffer(rmsQue, 2, RoundUp<float>(CeilDiv(tilingData->mL1Size, CV_RATIO)) * sizeof(float));

        // Calc Buf
        pipe->InitBuffer(castBuf,
                         tilingData->mUbSize * (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
        pipe->InitBuffer(nd2NzBuf, CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize) *
                                       sizeof(float) * DOUBLE_BUFFER);
        if (premixStage1_) {
            const int64_t vectorMRows = CeilDiv(tilingData->mL1Size, CV_RATIO);
            const int64_t dPerCore = tilingData->multCoreSplitKSize / tilingData->hcMult;
            const int64_t maxGroupK = AscendC::Std::min((int64_t)tilingData->kL1Size, dPerCore);
            pipe->InitBuffer(yQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
            pipe->InitBuffer(preMixQue, 1, vectorMRows * RoundUp<float>(tilingData->hcMix) * sizeof(float));
            pipe->InitBuffer(yAccBuf, vectorMRows * RoundUp<float>(tilingData->kUbSize) * sizeof(float) *
                                          CeilDiv(maxGroupK, tilingData->kUbSize));
            yAccLocal = yAccBuf.Get<float>();
        }

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            if constexpr (CV_RATIO == 2) {
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            }
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            if constexpr (CV_RATIO == 2) {
                CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            }
        }
        xCastLocal = castBuf.Get<float>();
        xNd2NzLocal = nd2NzBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if (premixStage1_) {
            ProcessPremix();
            return;
        }
        int64_t curBlockIdx = GetBlockIdx();
        int64_t totalBlockNum = GetBlockNum();

        uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t singleCoreMaxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t mainCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t singleCoreRound =
            (mainCoreCount == 0 || curBlockIdx < mainCoreCount) ? singleCoreMaxRound : singleCoreMaxRound - 1;
        uint64_t mGmOffset = 0;
        uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
        if ASCEND_IS_AIC {
            mGmOffset = (curBlockIdx / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size;
        } else {
            mGmOffset =
                ((curBlockIdx / CV_RATIO) / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size;
        }
        int64_t xGmBaseOffset = 0;
        int64_t rmsGmBaseOffset = 0;
        if ASCEND_IS_AIV {
            int64_t aivCurBlockIdx = GetBlockIdx();
            xGmBaseOffset = ((aivCurBlockIdx / CV_RATIO) / tilingData->cubeBlockDimK) * singleCoreMaxRound *
                                tilingData->mL1Size * tilingData->hcMult * tilingData->d +
                            ((aivCurBlockIdx / CV_RATIO) % tilingData->cubeBlockDimK) * tilingData->multCoreSplitKSize;
            rmsGmBaseOffset =
                ((aivCurBlockIdx / CV_RATIO) / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size;
        }

        int64_t xSplitOffset = 0;
        int64_t rmsSplitOffset = 0;
        // m轴切分 按照0 0 1 1..分核
        int64_t bufferIdx = 0;
        int64_t curAicBlockIdx = 0;
        if ASCEND_IS_AIC {
            curAicBlockIdx = curBlockIdx;
        } else {
            curAicBlockIdx = curBlockIdx / CV_RATIO;
        }

        if (curAicBlockIdx < tilingData->cubeBlockDimK * tilingData->cubeBlockDimM) {
            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
            for (uint64_t roundIdx = 0; roundIdx < singleCoreRound; mGmOffset += tilingData->mL1Size, ++roundIdx) {
                uint64_t mL1RealSize = AscendC::Std::min(tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
                uint64_t kGmStartOffset = 0;
                uint64_t kGmEndOffset = AscendC::Std::min(
                    tilingData->multCoreSplitKSize,
                    tilingData->k - (curAicBlockIdx % tilingData->cubeBlockDimK) * tilingData->multCoreSplitKSize);
                if ASCEND_IS_AIV {
                    if (GetBlockIdx() % CV_RATIO != 0) {
                        xSplitOffset = (mL1RealSize / CV_RATIO) * tilingData->hcMult * tilingData->d;
                        rmsSplitOffset = mL1RealSize / CV_RATIO;
                    }
                    rmsNormLocal = rmsQue.template AllocTensor<float>();
                }

                int64_t curRowFactor = 0;
                for (int64_t kGmOffset = kGmStartOffset; kGmOffset < kGmEndOffset; kGmOffset += tilingData->kL1Size) {
                    uint64_t kL1RealSize = AscendC::Std::min(kGmEndOffset - kGmOffset, (uint64_t)tilingData->kL1Size);
                    if ASCEND_IS_AIC {
                        bool isFirstKL1 = kGmOffset == kGmStartOffset;
                        bool isLastKL1 = (kGmOffset + tilingData->kL1Size) >= kGmEndOffset;
                        mmService_.CopyInB1Nd2Nz(tilingData->hcMult * tilingData->d, kL1RealSize, tilingData->hcMix,
                                                 hcFnGm[kGmOffset + kBlkDimIdx * tilingData->multCoreSplitKSize],
                                                 wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        if constexpr (CV_RATIO == 2) {
                            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                        }
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                        uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                        uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);

                        mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize,
                                           (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32, isFirstKL1,
                                           isLastKL1, xL1_[aL1BufferID_ * L1_BUF_OFFSET],
                                           wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        if (isLastKL1) {
                            mmService_.CopyOut(
                                mmGm[mBlkDimIdx * tilingData->mL1Size * singleCoreMaxRound * tilingData->hcMix +
                                     kBlkDimIdx * tilingData->bs * tilingData->hcMix + roundIdx * tilingData->mL1Size]);
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(
                            SYNC_AIC_AIV_FLAG); // 写出ub搬出，cv流水同步比较复杂，暂不讨论
                        if constexpr (CV_RATIO == 2) {
                            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                        }
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                        int64_t rowFactor = mL1RealSize / CV_RATIO;
                        int64_t tailRowFactor = mL1RealSize - rowFactor;
                        curRowFactor = rowFactor;
                        int64_t mL1SizeAlign = CeilAlign(mL1RealSize, AscendC::BLOCK_CUBE);
                        if (curBlockIdx % CV_RATIO != 0) {
                            curRowFactor = tailRowFactor;
                        }
                        uint64_t cvLoopKSize = CeilDiv(kL1RealSize, tilingData->kUbSize);
                        uint64_t kReminderSize = kL1RealSize - (cvLoopKSize - 1) * tilingData->kUbSize;
                        float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
                        // With cvRatio 1 the AIV owns the whole L1 M tile; split it into mUbSize
                        // chunks so the stage-1 UB buffers stay bounded.
                        int64_t mUbTileSize = curRowFactor;
                        if constexpr (CV_RATIO == 1) {
                            mUbTileSize = tilingData->mUbSize;
                        }
                        for (int64_t cvLoopIdx = 0; cvLoopIdx < cvLoopKSize; cvLoopIdx++) {
                            uint64_t kRealSize = (cvLoopIdx == cvLoopKSize - 1) ? kReminderSize : tilingData->kUbSize;
                            for (int64_t mUbOffset = 0; mUbOffset < curRowFactor; mUbOffset += mUbTileSize) {
                                int64_t curMUbSize = AscendC::Std::min(curRowFactor - mUbOffset, mUbTileSize);
                                xLocal = xQue.template AllocTensor<T>();
                                CopyIn(xGm[xGmBaseOffset + xSplitOffset +
                                           roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d +
                                           mUbOffset * tilingData->hcMult * tilingData->d + kGmOffset +
                                           cvLoopIdx * tilingData->kUbSize],
                                       xLocal, curMUbSize, kRealSize, tilingData->hcMult * tilingData->d - kRealSize);
                                xQue.template EnQue(xLocal);
                                xLocal = xQue.template DeQue<T>();
                                if (kGmOffset == kGmStartOffset && cvLoopIdx == 0) {
                                    VFProcessCastAndInvRmsPart1<T, false>(rmsNormLocal[mUbOffset], xCastLocal, xLocal,
                                                                          coeff, curMUbSize, kRealSize);
                                } else {
                                    VFProcessCastAndInvRmsPart1<T, true>(rmsNormLocal[mUbOffset], xCastLocal, xLocal,
                                                                         coeff, curMUbSize, kRealSize);
                                }
                                xQue.template FreeTensor(xLocal);

                                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                                VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastLocal, curMUbSize,
                                             kRealSize);
                                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                                DataCopyParams dataCopyXParams;
                                dataCopyXParams.blockCount = CeilDiv(kRealSize, C0_SIZE);
                                dataCopyXParams.blockLen = curMUbSize * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                                dataCopyXParams.srcStride = CeilAlign(curMUbSize, C0_SIZE) - curMUbSize;
                                dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curMUbSize;
                                int64_t l1MOffset = mUbOffset * (BLOCK_SIZE / sizeof(float));
                                if (curBlockIdx % CV_RATIO != 0) {
                                    l1MOffset += rowFactor * (BLOCK_SIZE / sizeof(float));
                                }
                                CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)],
                                         xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + l1MOffset +
                                              cvLoopIdx * tilingData->kUbSize * mL1SizeAlign],
                                         dataCopyXParams);
                                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                                bufferIdx++;
                            }
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    }
                    aL1BufferID_ ^= 1;
                }
                if ASCEND_IS_AIV {
                    int64_t kBaseOffset = (GetBlockIdx() / CV_RATIO) % tilingData->kBlockFactor * tilingData->bs;
                    rmsQue.template EnQue(rmsNormLocal);
                    rmsNormLocal = rmsQue.template DeQue<float>();
                    CopyOut(rmsNormLocal,
                            rmsGm[kBaseOffset + rmsGmBaseOffset + rmsSplitOffset + roundIdx * tilingData->mL1Size], 1,
                            curRowFactor);
                    rmsQue.template FreeTensor(rmsNormLocal);
                }
            }
            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
        }
        if ASCEND_IS_AIV {
            // Drain the two double-buffer credits seeded by the paired AIC.
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        }
        if ASCEND_IS_AIC {
            mmService_.End();
        }
        SyncAll<false>();
    }

    __aicore__ inline void ProcessPremix()
    {
        int64_t curBlockIdx = GetBlockIdx();
        uint64_t logicalBlockIdx = static_cast<uint64_t>(curBlockIdx);
        if ASCEND_IS_AIV {
            logicalBlockIdx = static_cast<uint64_t>(curBlockIdx / CV_RATIO);
        }
        uint64_t activeBlockNum = (uint64_t)tilingData->cubeBlockDimM * tilingData->cubeBlockDimK;
        if (logicalBlockIdx >= activeBlockNum) {
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            } else {
                mmService_.End();
            }
            SyncAll<false>();
            return;
        }

        uint64_t mBlockIdx = logicalBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlockIdx = logicalBlockIdx % tilingData->cubeBlockDimK;
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t maxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t longCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t coreRound = (longCoreCount == 0 || mBlockIdx < longCoreCount) ? maxRound : maxRound - 1;
        uint64_t mGmBase =
            (longCoreCount == 0 || mBlockIdx < longCoreCount) ?
                mBlockIdx * maxRound * tilingData->mL1Size :
                (longCoreCount * maxRound + (mBlockIdx - longCoreCount) * (maxRound - 1)) * tilingData->mL1Size;

        uint64_t dPerCore = tilingData->multCoreSplitKSize / tilingData->hcMult;
        uint64_t dCoreStart = kBlockIdx * dPerCore;
        uint64_t dCoreSize = AscendC::Std::min(dPerCore, (uint64_t)tilingData->d - dCoreStart);
        uint64_t dGroupCount = CeilDiv(dCoreSize, (uint64_t)tilingData->kL1Size);
        uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
        uint64_t vectorMRows = CeilDiv(tilingData->mL1Size, CV_RATIO);
        uint64_t yAccSliceSize = vectorMRows * RoundUp<float>(tilingData->kUbSize);
        uint64_t hcMixAlign = RoundUp<float>(tilingData->hcMix);
        int64_t bufferIdx = 0;
        if ASCEND_IS_AIV {
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
        }

        for (uint64_t roundIdx = 0; roundIdx < coreRound; ++roundIdx) {
            uint64_t mGmOffset = mGmBase + roundIdx * tilingData->mL1Size;
            uint64_t mReal = AscendC::Std::min((uint64_t)tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
            uint64_t firstHalfRows = CeilDiv(mReal, CV_RATIO);
            uint64_t localRowOffset = 0;
            uint64_t localRows = firstHalfRows;
            if constexpr (CV_RATIO == 2) {
                if (curBlockIdx & 1) {
                    localRowOffset = firstHalfRows;
                    localRows = mReal - firstHalfRows;
                }
            }
            if ASCEND_IS_AIV {
                rmsNormLocal = rmsQue.template AllocTensor<float>();
                preMixLocal = preMixQue.template AllocTensor<float>();
                CopyInWithUbStride(preMixGm[(mGmOffset + localRowOffset) * tilingData->hcMult], preMixLocal,
                                   static_cast<uint16_t>(localRows), static_cast<uint32_t>(tilingData->hcMult), 0,
                                   ubRowGapBlocks_);
                preMixQue.EnQue(preMixLocal);
                preMixLocal = preMixQue.template DeQue<float>();
            }

            uint64_t mAlign = Align(mReal, AscendC::BLOCK_CUBE);
            for (uint64_t groupIdx = 0; groupIdx < dGroupCount; ++groupIdx) {
                uint64_t dGroupStart = dCoreStart + groupIdx * tilingData->kL1Size;
                uint64_t currentGroupK =
                    AscendC::Std::min((uint64_t)tilingData->kL1Size, dCoreStart + dCoreSize - dGroupStart);
                uint64_t subCount = CeilDiv(currentGroupK, (uint64_t)tilingData->kUbSize);
                for (uint64_t hcIdx = 0; hcIdx < (uint64_t)tilingData->hcMult; ++hcIdx) {
                    uint64_t sourceGroupKOffset = hcIdx * tilingData->d + dGroupStart;
                    uint64_t groupHcIdx = groupIdx * tilingData->hcMult + hcIdx;
                    bool isFirstK = groupHcIdx == 0;
                    bool isLastK = groupHcIdx + 1 == dGroupCount * tilingData->hcMult;
                    if ASCEND_IS_AIC {
                        mmService_.CopyInB1Nd2Nz(tilingData->hcMult * tilingData->d, currentGroupK, tilingData->hcMix,
                                                 hcFnGm[sourceGroupKOffset],
                                                 wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        if constexpr (CV_RATIO == 2) {
                            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                        }
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                        uint64_t nAlign = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                        mmService_.Process(tilingData->bs, tilingData->hcMix, mReal,
                                           (256 / AscendC::Std::max(mAlign, nAlign)) * 32, isFirstK, isLastK,
                                           xL1_[aL1BufferID_ * L1_BUF_OFFSET],
                                           wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        if (isLastK) {
                            mmService_.CopyOut(
                                mmGm[kBlockIdx * tilingData->bs * tilingData->hcMix + mGmOffset * tilingData->hcMix]);
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
                        if constexpr (CV_RATIO == 2) {
                            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                        }
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                        for (uint64_t mInner = 0; mInner < localRows; mInner += tilingData->mUbSize) {
                            uint64_t currentM = AscendC::Std::min((uint64_t)tilingData->mUbSize, localRows - mInner);
                            LocalTensor<float> rmsChunk = rmsNormLocal[mInner];
                            LocalTensor<float> preMixChunk = preMixLocal[mInner * hcMixAlign];
                            for (uint64_t subIdx = 0; subIdx < subCount; ++subIdx) {
                                uint64_t dOffset = dGroupStart + subIdx * tilingData->kUbSize;
                                uint64_t currentK = AscendC::Std::min((uint64_t)tilingData->kUbSize,
                                                                      dGroupStart + currentGroupK - dOffset);
                                uint64_t sourceKOffset = hcIdx * tilingData->d + dOffset;
                                LocalTensor<float> yAccSlice =
                                    yAccLocal[subIdx * yAccSliceSize + mInner * RoundUp<float>(tilingData->kUbSize)];
                                xLocal = xQue.template AllocTensor<T>();
                                CopyIn(xGm[(mGmOffset + localRowOffset + mInner) * tilingData->hcMult * tilingData->d +
                                           sourceKOffset],
                                       xLocal, currentM, currentK, tilingData->hcMult * tilingData->d - currentK);
                                xQue.template EnQue(xLocal);
                                xLocal = xQue.template DeQue<T>();
                                float coeff = 1.0f / static_cast<float>(tilingData->hcMult * tilingData->d);
                                bool isFirstSub = groupIdx == 0 && hcIdx == 0 && subIdx == 0;
                                if (isFirstSub) {
                                    VFProcessCastInvRmsAndPremixYPart1<T, false, false>(
                                        rmsChunk, yAccSlice, xCastLocal, preMixChunk, xLocal, coeff,
                                        static_cast<uint16_t>(hcIdx), static_cast<uint16_t>(hcMixAlign),
                                        static_cast<uint16_t>(RoundUp<float>(tilingData->kUbSize)),
                                        static_cast<uint16_t>(currentM), static_cast<uint16_t>(currentK));
                                } else if (hcIdx == 0) {
                                    VFProcessCastInvRmsAndPremixYPart1<T, true, false>(
                                        rmsChunk, yAccSlice, xCastLocal, preMixChunk, xLocal, coeff,
                                        static_cast<uint16_t>(hcIdx), static_cast<uint16_t>(hcMixAlign),
                                        static_cast<uint16_t>(RoundUp<float>(tilingData->kUbSize)),
                                        static_cast<uint16_t>(currentM), static_cast<uint16_t>(currentK));
                                } else {
                                    VFProcessCastInvRmsAndPremixYPart1<T, true, true>(
                                        rmsChunk, yAccSlice, xCastLocal, preMixChunk, xLocal, coeff,
                                        static_cast<uint16_t>(hcIdx), static_cast<uint16_t>(hcMixAlign),
                                        static_cast<uint16_t>(RoundUp<float>(tilingData->kUbSize)),
                                        static_cast<uint16_t>(currentM), static_cast<uint16_t>(currentK));
                                }
                                xQue.template FreeTensor(xLocal);

                                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                                VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastLocal, currentM,
                                             currentK);
                                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                                DataCopyParams copyParams;
                                copyParams.blockCount = CeilDiv(currentK, C0_SIZE);
                                copyParams.blockLen = currentM * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                                copyParams.srcStride = CeilAlign(currentM, C0_SIZE) - currentM;
                                copyParams.dstStride = mAlign - currentM;
                                uint64_t l1RowOffset = (localRowOffset + mInner) * (BLOCK_SIZE / sizeof(float));
                                uint64_t l1SubOffset = subIdx * tilingData->kUbSize * mAlign;
                                CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)],
                                         xL1_[aL1BufferID_ * L1_BUF_OFFSET + l1SubOffset + l1RowOffset], copyParams);
                                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                                ++bufferIdx;
                            }
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                        // Let the paired Cube consume x before exporting the independent hIn output.
                        if (hcIdx + 1 == (uint64_t)tilingData->hcMult) {
                            CopyOutPremixY(mGmOffset + localRowOffset, localRows, dGroupStart, currentGroupK);
                        }
                    }
                    aL1BufferID_ ^= 1;
                }
            }

            if ASCEND_IS_AIV {
                rmsQue.EnQue(rmsNormLocal);
                rmsNormLocal = rmsQue.template DeQue<float>();
                CopyOut(rmsNormLocal, rmsGm[kBlockIdx * tilingData->bs + mGmOffset + localRowOffset], 1, localRows);
                rmsQue.FreeTensor(rmsNormLocal);
                preMixQue.FreeTensor(preMixLocal);
            }
        }
        FinishPremix();
        SyncAll<false>();
    }

    __aicore__ inline void FinishPremix()
    {
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        } else {
            mmService_.End();
        }
    }

private:
    __aicore__ inline void CopyOutPremixY(uint64_t rowStart, uint64_t rows, uint64_t dStart, uint64_t cols)
    {
        const uint64_t subCount = CeilDiv(cols, (uint64_t)tilingData->kUbSize);
        const uint64_t yAccSliceSize = CeilDiv(tilingData->mL1Size, CV_RATIO) * RoundUp<float>(tilingData->kUbSize);
        for (uint64_t rowOffset = 0; rowOffset < rows; rowOffset += tilingData->mUbSize) {
            const uint64_t currentRows = AscendC::Std::min((uint64_t)tilingData->mUbSize, rows - rowOffset);
            for (uint64_t subIdx = 0; subIdx < subCount; ++subIdx) {
                const uint64_t dOffset = dStart + subIdx * tilingData->kUbSize;
                const uint64_t currentK = AscendC::Std::min((uint64_t)tilingData->kUbSize, dStart + cols - dOffset);
                LocalTensor<float> yAccSlice =
                    yAccLocal[subIdx * yAccSliceSize + rowOffset * RoundUp<float>(tilingData->kUbSize)];
                yLocal = yQue.template AllocTensor<T>();
                VFProcessPremixYCast<T>(yLocal, yAccSlice, static_cast<uint32_t>(RoundUp<float>(tilingData->kUbSize)),
                                        static_cast<uint16_t>(currentRows), static_cast<uint16_t>(currentK));
                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal, yGm[(rowStart + rowOffset) * tilingData->d + dOffset], currentRows, currentK,
                        tilingData->d - currentK);
                yQue.template FreeTensor(yLocal);
            }
        }
    }

    TPipe* pipe;
    const MhcPreSinkhornRegbaseTilingData* tilingData;
    // (M, K) * (N, K)

    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> preMixGm;
    bool premixStage1_ = false;
    uint32_t ubRowGapBlocks_ = 0;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> rmsQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECIN, 1> preMixQue;
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;
    TBuf<QuePosition::VECCALC> yAccBuf;

    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> mmXLocal;
    LocalTensor<float> rmsNormLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> xNd2NzLocal;
    LocalTensor<float> preMixLocal;
    LocalTensor<float> yAccLocal;

    MhcPreSinkhornCubeCompute<(CV_RATIO == 2)> mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIV_AIC_PRE_POST_FLAG = 10;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 11;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint64_t cvLoopIdx_ = 0;
    uint8_t aL1BufferID_ = 0;
};

} // namespace MhcPreSinkhornNs

#endif

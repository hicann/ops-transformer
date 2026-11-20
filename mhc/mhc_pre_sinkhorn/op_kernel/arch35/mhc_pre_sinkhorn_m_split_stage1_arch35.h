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
 * \file mhc_pre_sinkhorn_m_split_stage1_arch35.h
 * \brief Shared stage-1 implementation.
 */

#ifndef MHC_PRE_SINKHORN_M_SPLIT_STAGE1_ARCH35_H
#define MHC_PRE_SINKHORN_M_SPLIT_STAGE1_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_base_arch35.h"

namespace MhcPreSinkhornNs {
using namespace AscendC;

template <typename T, uint64_t CV_RATIO>
class MhcPreSinkhornMSplitStage1 {
public:
    template <typename Core>
    __aicore__ inline static void InitBuffers(Core& core)
    {
        const auto* tilingData = core.tilingData;
        const uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
        core.tbufPool1.Reset();
        core.tbufPool1.InitBuffer(core.xQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
        core.tbufPool1.InitBuffer(core.castBuf, (core.premixCastSegments_ ? tilingData->hcMult : 1) *
                                                    tilingData->mUbSize *
                                                    (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
        core.tbufPool1.InitBuffer(core.nd2NzBuf, nd2NzBufSize * sizeof(float) * DOUBLE_BUFFER);
        if (core.hasPremix_) {
            // External weights and either cast slices or hIn accumulators remain
            // resident until all four hc segments of a window have been processed.
            core.tbufPool1.InitBuffer(core.yQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
            if (!core.premixCastSegments_) {
                core.tbufPool1.InitBuffer(core.yAccBuf, RoundUp<float>(CeilDiv(tilingData->mL1Size, CV_RATIO)) *
                                                            RoundUp<float>(tilingData->kUbSize) * sizeof(float));
                core.yAccLocal = core.yAccBuf.template Get<float>();
            }
            core.tbufPool1.InitBuffer(core.preMixQue, 1,
                                      RoundUp<float>(CeilDiv(tilingData->mL1Size, CV_RATIO)) *
                                          RoundUp<float>(tilingData->hcMix) * sizeof(float));
        }

        core.xCastLocal = core.castBuf.template Get<float>();
        core.xNd2NzLocal = core.nd2NzBuf.template Get<float>();
        core.rmsNormLocal = core.rmsNormBuf.template Get<float>();
    }

    // Share the stage-1 schedule while retaining each kernel's existing UB resources.
    template <typename Core>
    __aicore__ inline static void ProcessPremix(Core& core, uint64_t mGmOffset, uint64_t mL1RealSize)
    {
        const auto* tilingData = core.tilingData;
        const int64_t curBlockIdx = GetBlockIdx();
        const uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
        int64_t bufferIdx = 0;
        const int64_t kUbSize = tilingData->kUbSize;
        const int64_t chunkCountPerSeg = CeilDiv(tilingData->d, kUbSize);
        int64_t rowFactor = CeilDiv(mL1RealSize, CV_RATIO);
        int64_t curRowFactor = rowFactor;
        if (curBlockIdx % CV_RATIO != 0) {
            curRowFactor = mL1RealSize - rowFactor;
        }
        const int64_t localRowOffset = curBlockIdx % CV_RATIO != 0 ? rowFactor : 0;
        if ASCEND_IS_AIV {
            core.preMixLocal = core.preMixQue.template AllocTensor<float>();
            CopyInWithUbStride(core.preMixGm[(mGmOffset + localRowOffset) * tilingData->hcMult], core.preMixLocal,
                               static_cast<uint16_t>(curRowFactor), static_cast<uint32_t>(tilingData->hcMult), 0,
                               core.ubRowGapBlocks_);
            core.preMixQue.EnQue(core.preMixLocal);
            core.preMixLocal = core.preMixQue.template DeQue<float>();
        }
        float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
        for (int64_t chunkInSeg = 0; chunkInSeg < chunkCountPerSeg; ++chunkInSeg) {
            const int64_t chunkCol = chunkInSeg * kUbSize;
            const int64_t chunkColSize = kUbSize;
            const bool isLastWindow = chunkInSeg + 1 == chunkCountPerSeg;
            for (int64_t segIdx = 0; segIdx < tilingData->hcMult; ++segIdx) {
                const int64_t kGmOffset = segIdx * tilingData->d + chunkCol;
                const bool isFirstChunk = chunkInSeg == 0 && segIdx == 0;
                const bool isLastChunk = isLastWindow && segIdx + 1 == tilingData->hcMult;
                if ASCEND_IS_AIC {
                    core.mmService_.CopyInB1Nd2Nz(tilingData->multCoreSplitKSize, chunkColSize, tilingData->hcMix,
                                                  core.hcFnGm[kGmOffset],
                                                  core.wL1_[core.mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    if constexpr (CV_RATIO == 2) {
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(Core::SYNC_AIV_AIC_FLAG + Core::FLAG_ID_MAX);
                    }
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(Core::SYNC_AIV_AIC_FLAG);
                    uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                    uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                    core.mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize,
                                            (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32, isFirstChunk,
                                            isLastChunk, core.xL1_[core.aL1BufferID_ * L1_BUF_OFFSET],
                                            core.wL1_[core.mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    if (isLastChunk) {
                        core.mmService_.CopyOut(core.mmXLocal);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(Core::SYNC_AIC_AIV_PRE_POST_FLAG);
                        if constexpr (CV_RATIO == 2) {
                            CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(Core::SYNC_AIC_AIV_PRE_POST_FLAG +
                                                                   Core::FLAG_ID_MAX);
                        }
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(Core::SYNC_AIC_AIV_FLAG);
                    if constexpr (CV_RATIO == 2) {
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(Core::SYNC_AIC_AIV_FLAG + Core::FLAG_ID_MAX);
                    }
                } else {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(Core::SYNC_AIC_AIV_FLAG);
                    for (int64_t mUbOffset = 0; mUbOffset < curRowFactor; mUbOffset += tilingData->mUbSize) {
                        int64_t curMUbSize = AscendC::Std::min(curRowFactor - mUbOffset, (int64_t)tilingData->mUbSize);
                        core.xLocal = core.xQue.template AllocTensor<T>();
                        CopyIn(core.xGm[(mGmOffset + localRowOffset + mUbOffset) * tilingData->hcMult * tilingData->d +
                                        kGmOffset],
                               core.xLocal, curMUbSize, chunkColSize,
                               tilingData->hcMult * tilingData->d - chunkColSize);
                        core.xQue.template EnQue(core.xLocal);
                        core.xLocal = core.xQue.template DeQue<T>();
                        LocalTensor<float> xCastChunk = GetPremixCastSlice(core, segIdx, mUbOffset);
                        ProcessPremixInput(core, xCastChunk, mUbOffset, curMUbSize, chunkColSize, segIdx, isFirstChunk,
                                           coeff);
                        core.xQue.template FreeTensor(core.xLocal);

                        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        VFTransND2NZ(core.xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastChunk, curMUbSize,
                                     chunkColSize);
                        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                        DataCopyParams dataCopyXParams;
                        dataCopyXParams.blockCount = CeilDiv(chunkColSize, C0_SIZE);
                        dataCopyXParams.blockLen = curMUbSize * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                        dataCopyXParams.srcStride = CeilAlign(curMUbSize, C0_SIZE) - curMUbSize;
                        dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curMUbSize;
                        int64_t l1MOffset = mUbOffset * (BLOCK_SIZE / sizeof(float));
                        if (curBlockIdx % CV_RATIO != 0) {
                            l1MOffset += rowFactor * (BLOCK_SIZE / sizeof(float));
                        }
                        CopyToL1(core.xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)],
                                 core.xL1_[(core.aL1BufferID_ * L1_BUF_OFFSET) + l1MOffset], dataCopyXParams);
                        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        bufferIdx++;
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(Core::SYNC_AIV_AIC_FLAG);

                    if (segIdx + 1 == tilingData->hcMult) {
                        CopyOutPremixY(core, (mGmOffset + localRowOffset) * tilingData->d + chunkCol, curRowFactor,
                                       chunkColSize);
                    }
                }
                core.aL1BufferID_ ^= 1;
            }
        }
        if ASCEND_IS_AIV {
            core.preMixQue.FreeTensor(core.preMixLocal);
        }
    }

private:
    template <typename Core>
    __aicore__ inline static LocalTensor<float> GetPremixCastSlice(Core& core, int64_t hcIndex, int64_t rowOffset)
    {
        const auto* tilingData = core.tilingData;
        if (!core.premixCastSegments_) {
            return core.xCastLocal;
        }
        const int64_t rowStride = RoundUp<float>(tilingData->kUbSize) + BLOCK_SIZE / sizeof(float);
        const int64_t segStride = tilingData->mUbSize * rowStride;
        return core.xCastLocal[hcIndex * segStride + rowOffset * rowStride];
    }

    template <typename Core>
    __aicore__ inline static void ProcessPremixInput(Core& core, const LocalTensor<float>& castSlice, int64_t rowOffset,
                                                     int64_t rows, int64_t cols, int64_t hcIndex, bool isFirstChunk,
                                                     float coeff)
    {
        const auto* tilingData = core.tilingData;
        if (core.premixCastSegments_) {
            if (isFirstChunk) {
                VFProcessCastAndInvRmsPart1<T, false>(core.rmsNormLocal[rowOffset], castSlice, core.xLocal, coeff, rows,
                                                      cols);
            } else {
                VFProcessCastAndInvRmsPart1<T, true>(core.rmsNormLocal[rowOffset], castSlice, core.xLocal, coeff, rows,
                                                     cols);
            }
            return;
        }
        const uint16_t hcMixAlign = RoundUp<float>(tilingData->hcMix);
        const uint16_t yAccRowStride = RoundUp<float>(tilingData->kUbSize);
        LocalTensor<float> yAccChunk = core.yAccLocal[rowOffset * yAccRowStride];
        LocalTensor<float> preMixChunk = core.preMixLocal[rowOffset * hcMixAlign];
        if (isFirstChunk) {
            VFProcessCastInvRmsAndPremixYPart1<T, false, false>(core.rmsNormLocal[rowOffset], yAccChunk, castSlice,
                                                                preMixChunk, core.xLocal, coeff, hcIndex, hcMixAlign,
                                                                yAccRowStride, rows, cols);
        } else if (hcIndex == 0) {
            VFProcessCastInvRmsAndPremixYPart1<T, true, false>(core.rmsNormLocal[rowOffset], yAccChunk, castSlice,
                                                               preMixChunk, core.xLocal, coeff, hcIndex, hcMixAlign,
                                                               yAccRowStride, rows, cols);
        } else {
            VFProcessCastInvRmsAndPremixYPart1<T, true, true>(core.rmsNormLocal[rowOffset], yAccChunk, castSlice,
                                                              preMixChunk, core.xLocal, coeff, hcIndex, hcMixAlign,
                                                              yAccRowStride, rows, cols);
        }
    }

    template <typename Core>
    __aicore__ inline static void CopyOutPremixY(Core& core, int64_t outputOffset, int64_t rows, int64_t cols)
    {
        const auto* tilingData = core.tilingData;
        const int64_t hcMixAlign = RoundUp<float>(tilingData->hcMix);
        const int64_t yAccRowStride = RoundUp<float>(tilingData->kUbSize);
        const int64_t castRowStride = yAccRowStride + BLOCK_SIZE / sizeof(float);
        for (int64_t rowOffset = 0; rowOffset < rows; rowOffset += tilingData->mUbSize) {
            const int64_t currentRows = AscendC::Std::min(rows - rowOffset, (int64_t)tilingData->mUbSize);
            core.yLocal = core.yQue.template AllocTensor<T>();
            if (core.premixCastSegments_) {
                VFProcessPremixYFromCastSegments<T>(
                    core.yLocal, core.preMixLocal[rowOffset * hcMixAlign], core.xCastLocal[rowOffset * castRowStride],
                    static_cast<uint16_t>(currentRows), static_cast<uint16_t>(cols),
                    static_cast<uint16_t>(tilingData->hcMult), static_cast<uint16_t>(hcMixAlign),
                    static_cast<uint16_t>(tilingData->mUbSize));
            } else {
                VFProcessPremixYCast<T>(core.yLocal, core.yAccLocal[rowOffset * yAccRowStride],
                                        static_cast<uint32_t>(yAccRowStride), static_cast<uint16_t>(currentRows),
                                        static_cast<uint16_t>(cols));
            }
            core.yQue.template EnQue(core.yLocal);
            core.yLocal = core.yQue.template DeQue<T>();
            CopyOut(core.yLocal, core.yGm[outputOffset + rowOffset * tilingData->d], currentRows, cols,
                    tilingData->d - cols);
            core.yQue.template FreeTensor(core.yLocal);
        }
    }
};

} // namespace MhcPreSinkhornNs

#endif

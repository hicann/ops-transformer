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
 * \file mhc_pre_sinkhorn_m_k_split_core_arch35.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_M_K_SPLIT_CORE_ARCH35_H
#define MHC_PRE_SINKHORN_M_K_SPLIT_CORE_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_base_arch35.h"
#include "mhc_pre_sinkhorn_m_k_split_stage1_arch35.h"

namespace MhcPreSinkhornNs {
using namespace AscendC;

class MhcPreSinkhornMPMDSinkhorn {
public:
    __aicore__ inline void Init(GM_ADDR scale, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
                                const MhcPreSinkhornRegbaseTilingData* data, TPipe* pipe)
    {
        td = data;
        scaleGm.SetGlobalBuffer((__gm__ float*)scale);
        biasGm.SetGlobalBuffer((__gm__ float*)bias);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + td->kBlockFactor * td->bs * td->hcMix);
        mmLocalSize = td->kBlockFactor * td->sinkhornRowFactor * MATRIX_ELEMENTS;
        int64_t rmsLocalSize = td->kBlockFactor * RoundUp<float>(td->sinkhornRowFactor);
        pipe->InitBuffer(partialQue, 2, (mmLocalSize + rmsLocalSize) * sizeof(float));
        pipe->InitBuffer(outputQue, 2, td->sinkhornRowFactor * MATRIX_ELEMENTS * sizeof(float));
        pipe->InitBuffer(biasBuf, td->hcMult * td->hcMultAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t core = GetBlockIdx() - td->secondUsedCoreNum;
        const int64_t rowsPerCore = CeilDiv(td->bs, td->sinkhornCoreNum);
        const int64_t begin = core * rowsPerCore;
        const int64_t end = AscendC::Std::min(begin + rowsPerCore, td->bs);
        if (begin >= end) {
            return;
        }
        LocalTensor<float> bias = biasBuf.Get<float>();
        CopyIn(biasGm[2 * td->hcMult], bias, td->hcMult, td->hcMult);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        const float scale = scaleGm.GetValue(2);
        for (int64_t row = begin; row < end; row += td->sinkhornRowFactor) {
            const int64_t rows = AscendC::Std::min(td->sinkhornRowFactor, end - row);
            LocalTensor<float> partial = partialQue.AllocTensor<float>();
            // Compact [K, rows, 16], retaining the GM token pitch of 24.
            CopyInWithLoopMode(mmGm[row * td->hcMix + 2 * td->hcMult], partial, td->kBlockFactor, rows, MATRIX_ELEMENTS,
                               td->bs * td->hcMix, td->hcMix - MATRIX_ELEMENTS);
            CopyIn(rmsGm[row], partial[mmLocalSize], td->kBlockFactor, rows, td->bs - rows);
            partialQue.EnQue(partial);
            partial = partialQue.DeQue<float>();
            LocalTensor<float> output = outputQue.AllocTensor<float>();
            VFProcessInvRmsPart3WithGroupReduce(output, partial, partial[mmLocalSize], td->normEps, td->kBlockFactor,
                                                rows, MATRIX_ELEMENTS);
            VFProcessCombFragElementMajor<false>(output, output, bias, scale, td->hcEps, td->normEps, td->iterTimes - 1,
                                                 rows);
            partialQue.FreeTensor(partial);
            outputQue.EnQue(output);
            output = outputQue.DeQue<float>();
            CopyOut(output, outputGm[row * MATRIX_ELEMENTS], 1, rows * MATRIX_ELEMENTS);
            outputQue.FreeTensor(output);
        }
    }

private:
    static constexpr int64_t MATRIX_ELEMENTS = 16;
    const MhcPreSinkhornRegbaseTilingData* td;
    int64_t mmLocalSize;
    GlobalTensor<float> scaleGm, biasGm, outputGm, mmGm, rmsGm;
    TQue<QuePosition::VECIN, 1> partialQue;
    TQue<QuePosition::VECOUT, 1> outputQue;
    TBuf<QuePosition::VECCALC> biasBuf;
};

template <typename T>
class MhcPreSinkhornMKSplitCorePart2 {
public:
    __aicore__ inline MhcPreSinkhornMKSplitCorePart2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR preMix, GM_ADDR y, GM_ADDR post,
                                GM_ADDR combFrag, GM_ADDR hPre, GM_ADDR workspace,
                                const MhcPreSinkhornRegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        hasPremix_ = tilingData->hasPremix != 0;
        premixStage1_ = tilingData->premixStage1 != 0;

        if ASCEND_IS_AIV {
            if (tilingData->sinkhornCoreNum > 0 && GetBlockIdx() >= tilingData->secondUsedCoreNum) {
                if (GetBlockIdx() < tilingData->secondUsedCoreNum + tilingData->sinkhornCoreNum) {
                    sinkhorn.Init(hcScale, hcBase, combFrag, workspace, tilingData, pipe);
                }
                return;
            }
        }

        if (hasPremix_) {
            preMixGm.SetGlobalBuffer((__gm__ float*)preMix);
            hPreGm.SetGlobalBuffer((__gm__ float*)hPre);
        }
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);

        // InQue
        if (!premixStage1_) {
            pipe->InitBuffer(
                xQue, 2,
                tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
            pipe->InitBuffer(yQue, 2, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
            if (hasPremix_) {
                pipe->InitBuffer(preMixQue, 1, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
            }
        }
        int64_t rmsAndmmQueSize =
            tilingData->kBlockFactor * RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float) +
            tilingData->kBlockFactor * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float);
        pipe->InitBuffer(rmsAndmmQue, 2, rmsAndmmQueSize);

        // OutQue
        pipe->InitBuffer(postQue, 2, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        if (tilingData->sinkhornCoreNum == 0) {
            pipe->InitBuffer(
                combFragQue, 2,
                tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        }

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        if (tilingData->sinkhornCoreNum == 0) {
            pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        }
        pipe->InitBuffer(mixesBuf, tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        if (tilingData->sinkhornCoreNum == 0) {
            hcBase2Local = hcBaseBuf2.Get<float>();
        }
        if (hasPremix_) {
            pipe->InitBuffer(hPreQue, 2, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        }

        mixesLocal = mixesBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t curBlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = tilingData->secondUsedCoreNum;
            if (curBlockIdx >= stage2UsedCoreNum) {
                if (curBlockIdx < stage2UsedCoreNum + tilingData->sinkhornCoreNum) {
                    sinkhorn.Process();
                }
                return;
            }
            int64_t rowOuterLoop = (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->rowLoopOfTailBlock :
                                                                            tilingData->rowLoopOfFormerBlock;
            int64_t tailRowFactor = (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                             tilingData->tailRowFactorOfFormerBlock;

            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            if (tilingData->sinkhornCoreNum == 0) {
                CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            }
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            int64_t mmGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
            int64_t rmsGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock;
            int64_t mmLocalSize = stage1UsedCoreNum * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix);
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->stage2RowFactor;
                const int64_t rowStart =
                    curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->stage2RowFactor;
                rmsAndmmLocal = rmsAndmmQue.AllocTensor<float>();
                CopyInWithLoopMode(mmGm[mmGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMix],
                                   rmsAndmmLocal, tilingData->kBlockFactor, curRowFactor, tilingData->hcMix,
                                   tilingData->bs * tilingData->hcMix);
                CopyIn(rmsGm[rmsGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor], rmsAndmmLocal[mmLocalSize],
                       tilingData->kBlockFactor, curRowFactor, tilingData->bs - curRowFactor);

                rmsAndmmQue.EnQue(rmsAndmmLocal);
                rmsAndmmLocal = rmsAndmmQue.DeQue<float>();

                if (hasPremix_ && !premixStage1_) {
                    // Overlap the small weight load with the pre/post normalization work.
                    preMixLocal = preMixQue.template AllocTensor<float>();
                    CopyIn(preMixGm[rowStart * tilingData->hcMult], preMixLocal, curRowFactor, tilingData->hcMult);
                    preMixQue.EnQue(preMixLocal);
                }

                VFProcessInvRmsPart3WithGroupReduce(mixesLocal, rmsAndmmLocal, rmsAndmmLocal[mmLocalSize],
                                                    tilingData->normEps, tilingData->kBlockFactor, curRowFactor,
                                                    tilingData->hcMix);

                VFProcessPre(mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                             curRowFactor, tilingData->hcMult, tilingData->hcMix);
                if (hasPremix_) {
                    int64_t preMixGmOffset =
                        (curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->stage2RowFactor) *
                        tilingData->hcMult;
                    // Export the learned hPre independently of the external hIn weights.
                    hPreLocal = hPreQue.template AllocTensor<float>();
                    CopyOut(mixesLocal, hPreLocal, curRowFactor, tilingData->hcMult, 0,
                            UbRowGapBlocks(tilingData->hcMult, tilingData->hcMix));
                    hPreQue.EnQue(hPreLocal);
                    hPreLocal = hPreQue.template DeQue<float>();
                    CopyOut(hPreLocal, hPreGm[preMixGmOffset], curRowFactor, tilingData->hcMult);
                    hPreQue.FreeTensor(hPreLocal);
                }
                if (!premixStage1_) {
                    ProcessY(rowStart, curRowFactor);
                }

                // post
                postLocal = postQue.AllocTensor<float>();
                VFProcessPost(postLocal, mixesLocal[tilingData->hcMult], hcBase1Local, hcScaleGm.GetValue(1),
                              tilingData->hcEps, curRowFactor, tilingData->hcMult, tilingData->hcMix);

                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal,
                        postGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                               rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                if (tilingData->sinkhornCoreNum == 0) {
                    // combFrag
                    combFragLocal = combFragQue.AllocTensor<float>();
                    VFProcessCombFragCompact(combFragLocal, mixesLocal[tilingData->hcMult * 2], hcBase2Local,
                                             hcScaleGm.GetValue(2), tilingData->hcEps, tilingData->iterTimes - 1,
                                             curRowFactor, tilingData->hcMix);

                    combFragQue.EnQue(combFragLocal);
                    combFragLocal = combFragQue.DeQue<float>();
                    CopyOut(
                        combFragLocal,
                        combFragGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult *
                                       tilingData->hcMult +
                                   rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMult],
                        curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    combFragQue.FreeTensor(combFragLocal);
                }
                rmsAndmmQue.FreeTensor(rmsAndmmLocal);
            }
        }
    }

private:
    __aicore__ inline void ProcessY(int64_t rowStart, int64_t rows)
    {
        LocalTensor<float> weights = mixesLocal;
        int64_t weightStride = tilingData->hcMix;
        if (hasPremix_) {
            // Reuse the prefetched weights for every D window in this row tile.
            preMixLocal = preMixQue.template DeQue<float>();
            weights = preMixLocal;
            weightStride = tilingData->hcMultAlign;
        }
        for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; ++dLoopIdx) {
            const int64_t currentD = dLoopIdx + 1 == tilingData->dLoop ? tilingData->tailDFactor : tilingData->dFactor;
            const int64_t dOffset = dLoopIdx * tilingData->dFactor;
            xLocal = xQue.template AllocTensor<T>();
            CopyIn(xGm[rowStart * tilingData->hcMult * tilingData->d + dOffset], xLocal, rows * tilingData->hcMult,
                   currentD, tilingData->d - currentD);
            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T>();
            yLocal = yQue.template AllocTensor<T>();
            VFProcessY(yLocal, weights, xLocal, rows, tilingData->hcMult, currentD, weightStride);
            xQue.template FreeTensor(xLocal);
            yQue.template EnQue(yLocal);
            yLocal = yQue.template DeQue<T>();
            CopyOut(yLocal, yGm[rowStart * tilingData->d + dOffset], rows, currentD, tilingData->d - currentD);
            yQue.template FreeTensor(yLocal);
        }
        if (hasPremix_) {
            preMixQue.FreeTensor(preMixLocal);
        }
    }

    MhcPreSinkhornMPMDSinkhorn sinkhorn;
    TPipe* pipe;
    const MhcPreSinkhornRegbaseTilingData* tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    GlobalTensor<float> preMixGm;
    bool hasPremix_ = false;
    bool premixStage1_ = false;
    GlobalTensor<float> hPreGm;

    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> rmsAndmmQue;
    TQue<QuePosition::VECIN, 1> xQue;

    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    TQue<QuePosition::VECIN, 1> preMixQue;
    LocalTensor<float> preMixLocal;
    TQue<QuePosition::VECOUT, 1> hPreQue;
    LocalTensor<float> hPreLocal;
};
} // namespace MhcPreSinkhornNs
#endif

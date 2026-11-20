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
 * \file mhc_pre_sinkhorn_m_k_split_core_gradout_arch35.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_M_K_SPLIT_CORE_GRADOUT_ARCH35_H
#define MHC_PRE_SINKHORN_M_K_SPLIT_CORE_GRADOUT_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_base_arch35.h"
#include "mhc_pre_sinkhorn_m_k_split_stage1_arch35.h"

namespace MhcPreSinkhornNs {
using namespace AscendC;

template <typename T>
class MhcPreSinkhornMKSplitCorePart2Gradout {
public:
    __aicore__ inline MhcPreSinkhornMKSplitCorePart2Gradout() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR preMix, GM_ADDR y, GM_ADDR post,
                                GM_ADDR combFrag, GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
                                GM_ADDR normOut, GM_ADDR workspace,
                                const MhcPreSinkhornRegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        hasPremix_ = tilingData->hasPremix != 0;

        if (hasPremix_) {
            preMixGm.SetGlobalBuffer((__gm__ float*)preMix);
        }
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);
        // 新增反向输出
        hPreGm.SetGlobalBuffer((__gm__ float*)hPre);
        hcBeforeNormGm.SetGlobalBuffer((__gm__ float*)hcBeforeNorm);
        invRmsGm.SetGlobalBuffer((__gm__ float*)invRms);
        sumOutGm.SetGlobalBuffer((__gm__ float*)sumOut);
        normOutGm.SetGlobalBuffer((__gm__ float*)normOut);

        // InQue
        if (!hasPremix_) {
            pipe->InitBuffer(
                xQue, 2,
                tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
            pipe->InitBuffer(yQue, 2, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        }
        int64_t rmsAndmmQueSize =
            tilingData->kBlockFactor * RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float) +
            tilingData->kBlockFactor * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float);
        pipe->InitBuffer(rmsAndmmQue, 2, rmsAndmmQueSize);

        // OutQue
        pipe->InitBuffer(postQue, 2, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        // normOut 和 sumOut 复用que
        pipe->InitBuffer(
            combFragQue, 2,
            tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float) +
                tilingData->iterTimes * 2 * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign *
                    sizeof(float) +
                tilingData->iterTimes * 2 * tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        // invRms copyout Que
        pipe->InitBuffer(invRmsOutQue, 2, RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float));
        // hcBeforeNorm copyout Que
        pipe->InitBuffer(hcBeforeNormQue, 2,
                         tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));
        // hPre copyout Que
        pipe->InitBuffer(hPreQue, 2, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixesBuf, tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        mixesLocal = mixesBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t curBlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = tilingData->secondUsedCoreNum;
            if (curBlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t rowOuterLoop = (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->rowLoopOfTailBlock :
                                                                            tilingData->rowLoopOfFormerBlock;
            int64_t tailRowFactor = (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                             tilingData->tailRowFactorOfFormerBlock;

            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            int64_t mmGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
            int64_t rmsGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock;
            int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;
            // 新增反向输出的GM baseOffset
            int64_t hPreGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult;
            int64_t hcBeforeNormGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
            int64_t invRmsGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock;
            int64_t sumOutGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult;
            int64_t normOutGmBaseOffset =
                curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult;
            int64_t mmLocalSize = stage1UsedCoreNum * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix);
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->stage2RowFactor;
                rmsAndmmLocal = rmsAndmmQue.AllocTensor<float>();
                CopyInWithLoopMode(mmGm[mmGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMix],
                                   rmsAndmmLocal, tilingData->kBlockFactor, curRowFactor, tilingData->hcMix,
                                   tilingData->bs * tilingData->hcMix);
                CopyIn(rmsGm[rmsGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor], rmsAndmmLocal[mmLocalSize],
                       tilingData->kBlockFactor, curRowFactor, tilingData->bs - curRowFactor);

                rmsAndmmQue.EnQue(rmsAndmmLocal);
                rmsAndmmLocal = rmsAndmmQue.DeQue<float>();

                invRmsOutLocal = invRmsOutQue.template AllocTensor<float>();
                hcBeforeNormLocal = hcBeforeNormQue.template AllocTensor<float>();

                VFProcessInvRmsPart3WithGroupReduceGradout(mixesLocal, hcBeforeNormLocal, invRmsOutLocal, rmsAndmmLocal,
                                                           rmsAndmmLocal[mmLocalSize], tilingData->normEps,
                                                           tilingData->kBlockFactor, curRowFactor, tilingData->hcMix);

                // copyout invRms
                invRmsOutQue.EnQue(invRmsOutLocal);
                invRmsOutLocal = invRmsOutQue.template DeQue<float>();
                CopyOut(invRmsOutLocal, invRmsGm[invRmsGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor], 1,
                        curRowFactor);
                invRmsOutQue.FreeTensor(invRmsOutLocal);
                // copyout hcBeforeNorm(matmul result)
                hcBeforeNormQue.EnQue(hcBeforeNormLocal);
                hcBeforeNormLocal = hcBeforeNormQue.template DeQue<float>();
                CopyOut(hcBeforeNormLocal,
                        hcBeforeNormGm[hcBeforeNormGmBaseOffset +
                                       rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMix],
                        curRowFactor, tilingData->hcMix);
                hcBeforeNormQue.FreeTensor(hcBeforeNormLocal);

                VFProcessPre(mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                             curRowFactor, tilingData->hcMult, tilingData->hcMix);
                // copy out h_pre
                hPreLocal = hPreQue.template AllocTensor<float>();
                CopyOut(mixesLocal, hPreLocal, curRowFactor, tilingData->hcMult, 0, 2);
                hPreQue.EnQue(hPreLocal);
                hPreLocal = hPreQue.template DeQue<float>();
                CopyOut(hPreLocal,
                        hPreGm[hPreGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                hPreQue.FreeTensor(hPreLocal);

                if (!hasPremix_) {
                    for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                        int64_t curDFactor =
                            (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                        xLocal = xQue.template AllocTensor<T>();
                        int64_t xGmOffset =
                            xGmBaseOffset +
                            rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->d +
                            dLoopIdx * tilingData->dFactor;
                        CopyIn(xGm[xGmOffset], xLocal, curRowFactor * tilingData->hcMult, curDFactor,
                               tilingData->d - curDFactor);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();

                        yLocal = yQue.template AllocTensor<T>();
                        VFProcessY(yLocal, mixesLocal, xLocal, curRowFactor, tilingData->hcMult, curDFactor,
                                   tilingData->hcMix);
                        xQue.template FreeTensor(xLocal);
                        yQue.template EnQue(yLocal);
                        yLocal = yQue.template DeQue<T>();
                        CopyOut(yLocal,
                                yGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d +
                                    rowOuterIdx * tilingData->stage2RowFactor * tilingData->d +
                                    dLoopIdx * tilingData->dFactor],
                                curRowFactor, curDFactor, tilingData->d - curDFactor);
                        yQue.template FreeTensor(yLocal);
                    }
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

                // combFrag (cacl combFrag, normOut, sumOut)
                combFragLocal = combFragQue.AllocTensor<float>();
                VFProcessCombFragRLessVLUseFourUnfoldGradout(
                    combFragLocal, mixesLocal[tilingData->hcMult * 2], hcBase2Local, hcScaleGm.GetValue(2),
                    tilingData->hcEps, tilingData->iterTimes, curRowFactor, tilingData->hcMult, tilingData->hcMult,
                    tilingData->hcMix, tilingData->stage2RowFactor);
                rmsAndmmQue.FreeTensor(rmsAndmmLocal);

                combFragQue.EnQue(combFragLocal);
                combFragLocal = combFragQue.DeQue<float>();
                CopyOut(
                    combFragLocal,
                    combFragGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                               rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMult],
                    curRowFactor * tilingData->hcMult, tilingData->hcMult);
                // copyout normOut
                int64_t normOutLocalOffset = tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign;
                CopyOutWithLoopMode(combFragLocal[normOutLocalOffset],
                                    normOutGm[normOutGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor *
                                                                        tilingData->hcMult * tilingData->hcMult],
                                    2 * tilingData->iterTimes, tilingData->bs * tilingData->hcMult * tilingData->hcMult,
                                    curRowFactor, tilingData->hcMult * tilingData->hcMult, tilingData->hcMult,
                                    tilingData->hcMult);
                // copyout sumOut
                int64_t sumOutLocalOffset = normOutLocalOffset + tilingData->iterTimes * 2 *
                                                                     tilingData->stage2RowFactor * tilingData->hcMult *
                                                                     tilingData->hcMultAlign;
                CopyOutWithLoopMode(
                    combFragLocal[sumOutLocalOffset],
                    sumOutGm[sumOutGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult], 1, 0,
                    2 * tilingData->iterTimes, tilingData->bs * tilingData->hcMult, curRowFactor, tilingData->hcMult);
                combFragQue.FreeTensor(combFragLocal);
            }
        }
    }

private:
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
    GlobalTensor<float> hPreGm;
    GlobalTensor<float> hcBeforeNormGm;
    GlobalTensor<float> invRmsGm;
    GlobalTensor<float> sumOutGm;
    GlobalTensor<float> normOutGm;

    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> rmsAndmmQue;
    TQue<QuePosition::VECIN, 1> xQue;

    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;
    TQue<QuePosition::VECOUT, 1> invRmsOutQue;
    TQue<QuePosition::VECOUT, 1> hcBeforeNormQue;
    TQue<QuePosition::VECOUT, 1> hPreQue;

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
    LocalTensor<float> invRmsOutLocal;
    LocalTensor<float> hcBeforeNormLocal;
    LocalTensor<float> hPreLocal;
};
} // namespace MhcPreSinkhornNs
#endif

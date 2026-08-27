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
 * \file mhc_pre_m_split_core.h
 * \brief
 */

#ifndef MHC_PRE_M_SPLIT_A3_CORE_H
#define MHC_PRE_M_SPLIT_A3_CORE_H

#include "kernel_operator.h"
#include "mhc_pre_base.h"
#include "mhc_pre_cube_compute.h"

namespace MhcPre {
using namespace AscendC;
template <typename T, bool isFac, bool hasResi>
class MhcPreStage1 {
public:
    __aicore__ inline MhcPreStage1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR gamma, GM_ADDR invRms, GM_ADDR hcBeforeNorm,
                                GM_ADDR workspace, const MhcPreMembaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        InitTilingParams(tilingDataPtr);
        hasGamma_ = (tilingData->hasGamma != 0);

        // 获取当前核心 ID
        uint64_t blockIdx = GetBlockIdx();
        uint64_t cubeCoreId = blockIdx;
        uint64_t vecCoreId = blockIdx;

        // 计算 workspace 偏移
        int64_t ncSize = hcMult * d;

        // 计算 Cube Core ID（Vector Core ID / 2）
        if ASCEND_IS_AIV {
            cubeCoreId = blockIdx / CV_RATIO;
            int64_t maxBsPerLoop = 32;
            int64_t xQueSize = maxBsPerLoop * stage1NcFactor;
            pipe->InitBuffer(xQue, NUM_TWO, xQueSize * sizeof(T));
            pipe->InitBuffer(xCastQue, NUM_TWO, xQueSize * sizeof(float));
            pipe->InitBuffer(tmpQue, maxBsPerLoop * stage1NcFactor * sizeof(float));
            pipe->InitBuffer(sumQue, maxBsPerLoop * sizeof(float));
            LocalTensor<float> sumLocal = sumQue.Get<float>();
            pipe->InitBuffer(invRmsQue, NUM_TWO, maxBsPerLoop * sizeof(float));
            if (hasGamma_) {
                pipe->InitBuffer(gammaQue, 1, stage1NcFactor * sizeof(float));
            }
        }

        // 设置 GM 地址
        xGm.SetGlobalBuffer((__gm__ T *)x);
        phiGm.SetGlobalBuffer((__gm__ float *)phi);
        if (hasGamma_) {
            gammaGm.SetGlobalBuffer((__gm__ float *)gamma);
        }
        xCastWsGm.SetGlobalBuffer((__gm__ float *)workspace);

        // 设置额外的 workspace 地址（needGrad=false 时使用）
        if (needGrad_) {
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
        } else {
            int64_t xCastWsTotalSize = stage1XCastWsSize / sizeof(float);
            invRmsWsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize);
            int64_t invRmsWsTotalSize = bs * sizeof(float) / sizeof(float);
            hcBeforeNormWsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize + invRmsWsTotalSize);
        }

        // 初始化 Cube Compute
        if ASCEND_IS_AIC {
            // 根据 needGrad 决定输出地址
            if (needGrad_) {
                cubeCompute_.Init(xCastWsGm, phiGm, hcBeforeNormGm, stage1BsFactor, hcMult, d, stage1VecCoreNum);
            } else {
                cubeCompute_.Init(xCastWsGm, phiGm, hcBeforeNormWsGm, stage1BsFactor, hcMult, d, stage1VecCoreNum);
            }
            return;
        }
    }

    __aicore__ inline void InitTilingParams(const MhcPreMembaseTilingData *tilingDataPtr)
    {
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;
        hcMult = tilingData->hcMult;
        stage1XCastWsSize = tilingData->stage1XCastWsSize;
        bs = tilingData->bs;
        d = tilingData->d;
        stage1NcFactor = tilingData->stage1NcFactor;
        stage1BsFactor = tilingData->stage1BsFactor;
        stage1VecCoreNum = tilingData->stage1VecCoreNum;
        stage1NcLoop = tilingData->stage1NcLoop;
    }

    __aicore__ inline void Process()
    {
        uint64_t blockIdx = GetBlockIdx();
        uint64_t vecCoreId = blockIdx;
        uint64_t cubeCoreId = blockIdx;

        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
        }

        // 循环处理 BS（每轮最多 32 个）
        // ASCEND_IS_AIC 直接根据总BS数，计算当前处理的BS数
        int32_t totalTasksAligned_ = AlignUp(bs, stage1VecCoreNum * stage1BsFactor);

        // Cube Core 流程
        if ASCEND_IS_AIC {
            for (int32_t taskOffset = blockIdx * 2 * stage1BsFactor; taskOffset < totalTasksAligned_;
                 taskOffset += tilingData->stage1CubeCoreNum * 2 * stage1BsFactor) {
                int32_t tileTaskCount =
                    min(static_cast<int32_t>(2 * stage1BsFactor), static_cast<int32_t>(bs - taskOffset));
                // 等待 Vector Core 完成 Cast
                CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG);
                // 执行矩阵乘
                if (tileTaskCount > 0) {
                    cubeCompute_.ProcessMatmulXPhi(taskOffset, tileTaskCount);
                }
                // 通知 Vector Core 完成
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            }
        }

        if ASCEND_IS_AIV {
            int64_t ncSize = hcMult * d;
            for (int32_t taskOffset = blockIdx * stage1BsFactor; taskOffset < totalTasksAligned_;
                 taskOffset += stage1VecCoreNum * stage1BsFactor) {
                CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
                int32_t curBs = min(static_cast<int32_t>(stage1BsFactor), static_cast<int32_t>(bs - taskOffset));
                if (curBs > 0) {
                    int64_t curBsOffset = taskOffset;
                    // 初始化 sum 为 0
                    LocalTensor<float> invRmsLocal = invRmsQue.AllocTensor<float>();
                    LocalTensor<float> squareLocal = tmpQue.Get<float>();
                    LocalTensor<float> sumLocal = sumQue.Get<float>();
                    Duplicate(sumLocal, 0.0f, curBs);

                    // 分段 ReduceSum
                    for (int64_t ncIdx = 0; ncIdx < stage1NcLoop; ncIdx++) {
                        int64_t curNcSize =
                            (ncIdx == stage1NcLoop - 1) ? tilingData->stage1TailNcFactor : stage1NcFactor;
                        int64_t ncOffset = ncIdx * stage1NcFactor;
                        int64_t curNcSizeAlign = AlignUp(static_cast<uint64_t>(curNcSize), ELEMENTS_SIZE_PER_BLOCK);

                        // 1. CopyIn X (BF16)
                        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
                        CopyIn(xGm[curBsOffset * ncSize + ncOffset], xLocal, curBs, curNcSize, ncSize - curNcSize);
                        xQue.EnQue(xLocal);

                        // 2. Cast X (BF16) → X_cast (FP32)
                        xLocal = xQue.DeQue<T>();
                        LocalTensor<float> xCastLocal = xCastQue.AllocTensor<float>();
                        Cast(xCastLocal, xLocal, RoundMode::CAST_NONE, curBs * curNcSizeAlign);
                        xQue.FreeTensor(xLocal);

                        // 3. 计算 inv_rms (BEFORE gamma, uses original x²)
                        // 3.1 Square
                        PipeBarrier<PIPE_V>();
                        Mul(squareLocal, xCastLocal, xCastLocal, curBs * curNcSizeAlign);
                        PipeBarrier<PIPE_V>();

                        // 对当前段进行 ReduceSum
                        uint32_t srcShape[2] = {static_cast<uint32_t>(curBs), static_cast<uint32_t>(curNcSize)};
                        AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(squareLocal, squareLocal, srcShape, true);
                        PipeBarrier<PIPE_V>();

                        // 累加到总和
                        Add(sumLocal, sumLocal, squareLocal, curBs);
                        PipeBarrier<PIPE_V>();

                        // 4. gamma (AFTER square, BEFORE CopyOut)
                        if (hasGamma_) {
                            LocalTensor<float> gammaLocal = gammaQue.AllocTensor<float>();
                            CopyIn(gammaGm[ncOffset], gammaLocal, 1, curNcSize);
                            gammaQue.EnQue(gammaLocal);
                            gammaLocal = gammaQue.DeQue<float>();
                            for (int64_t bsIdx = 0; bsIdx < curBs; bsIdx++) {
                                Mul(xCastLocal[bsIdx * curNcSizeAlign], xCastLocal[bsIdx * curNcSizeAlign], gammaLocal,
                                    curNcSizeAlign);
                            }
                            PipeBarrier<PIPE_V>();
                            gammaQue.FreeTensor(gammaLocal);
                        }

                        // 5. CopyOut X_cast to workspace (gamma*x if hasGamma, else original x)
                        // EnQue after the optional gamma Mul so the VECOUT event covers the final producer.
                        xCastQue.EnQue(xCastLocal);
                        xCastLocal = xCastQue.DeQue<float>();
                        int64_t xCastoffset =
                            blockIdx * stage1BsFactor * ncSize + ping4vec * stage1VecCoreNum * stage1BsFactor * ncSize;
                        CopyOut(xCastLocal, xCastWsGm[xCastoffset + ncOffset], curBs, curNcSize, ncSize - curNcSize);
                        xCastQue.FreeTensor(xCastLocal);
                    }
                    // 4.3 Compute inv_rms
                    // inv_rms = 1 / sqrt(sum / (n*c) + eps)
                    float invNc = 1.0f / static_cast<float>(ncSize);
                    Muls(invRmsLocal, sumLocal, invNc, curBs);
                    PipeBarrier<PIPE_V>();
                    Adds(invRmsLocal, invRmsLocal, tilingData->normEps, curBs);
                    PipeBarrier<PIPE_V>();
                    Sqrt(invRmsLocal, invRmsLocal, curBs);
                    PipeBarrier<PIPE_V>();
                    Duplicate(sumLocal, 1.0f, curBs);
                    PipeBarrier<PIPE_V>();
                    Div(invRmsLocal, sumLocal, invRmsLocal, curBs);
                    invRmsQue.EnQue(invRmsLocal);

                    // 5. CopyOut inv_rms to output
                    invRmsLocal = invRmsQue.DeQue<float>();
                    if (needGrad_) {
                        CopyOut(invRmsLocal, invRmsGm[curBsOffset], 1, curBs);
                    } else {
                        CopyOut(invRmsLocal, invRmsWsGm[curBsOffset], 1, curBs);
                    }
                    invRmsQue.FreeTensor(invRmsLocal);
                    ping4vec = 1 - ping4vec;
                }
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_TO_AIC_FLAG);
            }
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
        }
    }

public:
    TPipe *pipe;
    const MhcPreMembaseTilingData *tilingData;

    // GM 地址
    GlobalTensor<T> xGm;                  // 输入 X (BF16)
    GlobalTensor<float> phiGm;            // phi 矩阵 (FP32)
    GlobalTensor<float> invRmsGm;         // inv_rms 输出 (FP32)
    GlobalTensor<float> hcBeforeNormGm;   // hcBeforeNorm 输出 (FP32)
    GlobalTensor<float> xCastWsGm;        // X_cast workspace
    GlobalTensor<float> invRmsWsGm;       // invRms workspace (needGrad=false)
    GlobalTensor<float> hcBeforeNormWsGm; // hcBeforeNorm workspace (needGrad=false)
    GlobalTensor<float> gammaGm;          // gamma (FP32, 可选)

    // Queue
    TQue<QuePosition::VECIN, NUM_TWO> xQue;      // X 输入队列
    TQue<QuePosition::VECOUT, NUM_TWO> xCastQue; // X_cast 输出队列
    TBuf<QuePosition::VECCALC> tmpQue;
    TBuf<QuePosition::VECCALC> sumAddQue;         // sum 队列
    TBuf<QuePosition::VECCALC> sumQue;            // sum 队列
    TQue<QuePosition::VECOUT, NUM_TWO> invRmsQue; // inv_rms 队列
    TQue<QuePosition::VECIN, 1> gammaQue;         // gamma UB buffer (per nc segment)

    // Cube Compute
    MhcPreCubeCompute<float, isFac, hasResi> cubeCompute_;

    // 标志
    bool needGrad_ = true;
    bool hasGamma_ = false;
    uint64_t ping4vec = 1;
    int64_t hcMult = 0;
    int64_t stage1XCastWsSize = 0;
    int64_t bs = 0;
    int64_t d = 0;
    int64_t stage1NcFactor = 0;
    int64_t stage1BsFactor = 0;
    int64_t stage1VecCoreNum = 0;
    int64_t stage1NcLoop = 0;

    // 同步标志
    static constexpr uint64_t SYNC_AIV_TO_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_MODE2 = NUM_TWO;
    static constexpr uint64_t CV_RATIO = 2;
};

template <typename T, bool isFac, bool hasResi>
class MhcPreStage2 {
public:
    __aicore__ inline MhcPreStage2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y, GM_ADDR post, GM_ADDR hRes,
                                GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR workspace,
                                const MhcPreMembaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        InitTilingParams(tilingDataPtr);

        resSize = hcMult * hcMultAlign;
        if constexpr (isFac) {
            resSize = Factorial(hcMult);
        }
        if constexpr (!hasResi) {
            resSize = 0;
        }
        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        if constexpr (hasResi) {
            hResGm.SetGlobalBuffer((__gm__ float *)hRes);
        }

        if (needGrad_) {
            hPreGm.SetGlobalBuffer((__gm__ float *)hPre);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms);
        } else {
            int64_t xCastWsTotalSize = stage1XCastWsSize / sizeof(float);
            invRmsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize);
            int64_t invRmsWsTotalSize = bs * sizeof(float) / sizeof(float);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize + invRmsWsTotalSize);
        }

        // InQue
        int64_t mixesQue01Size = stage2RowFactor * hcMultAlign * NUM_TWO * sizeof(float);
        pipe->InitBuffer(mixesQue01, NUM_TWO, mixesQue01Size);
        pipe->InitBuffer(squareSumQue, NUM_TWO, stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));
        int64_t xQueNum2 = stage2RowFactor * hcMult * RoundUp<T>(dFactor);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum2 * sizeof(T));

        pipe->InitBuffer(yQue, NUM_TWO, stage2RowFactor * RoundUp<T>(dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO, stage2RowFactor * hcMultAlign * sizeof(float));
        // TBuf
        pipe->InitBuffer(hcBaseBuf0, hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(stage2RowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(stage2RowFactor * hcMultAlign * NUM_TWO) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, stage2RowFactor * hcMultAlign * sizeof(float));
        pipe->InitBuffer(mxies01ReduceBuf, stage2RowFactor * hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueNum2 * sizeof(float));
        pipe->InitBuffer(yCastBuf, stage2RowFactor * RoundUp<T>(dFactor) * sizeof(float));

        if constexpr (hasResi) {
            pipe->InitBuffer(mixesQue2, NUM_TWO, stage2RowFactor * hcMult * hcMultAlign * sizeof(float));
            pipe->InitBuffer(combFragQue, NUM_TWO, stage2RowFactor * resSize * sizeof(float));
            pipe->InitBuffer(hcBaseBuf2, resSize * sizeof(float));
            pipe->InitBuffer(mxies02ReduceBuf, stage2RowFactor * resSize * sizeof(float));
        }

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        mxies01ReduceLocal = mxies01ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        if constexpr (hasResi) {
            hcBase2Local = hcBaseBuf2.Get<float>();
            mxies02ReduceLocal = mxies02ReduceBuf.Get<float>();
        }
    }

    __aicore__ inline void InitTilingParams(const MhcPreMembaseTilingData *tilingDataPtr)
    {
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;
        hcMult = tilingData->hcMult;
        hcMultAlign = tilingData->hcMultAlign;
        stage1XCastWsSize = tilingData->stage1XCastWsSize;
        bs = tilingData->bs;
        stage2RowFactor = tilingData->stage2RowFactor;
        dFactor = tilingData->dFactor;
        hcMix = tilingData->hcMix;
        d = tilingData->d;
    }

    __aicore__ inline void Process(bool isTailBsLoop = false)
    {
        SyncAll();
        isTailBsLoop_ = isTailBsLoop;

        int64_t curRowOfFormerBlock = isTailBsLoop ? tilingData->tailBsRowOfFormerBlock : tilingData->rowOfFormerBlock;
        int64_t curRowLoopOfFormerBlock =
            isTailBsLoop ? tilingData->tailBsRowLoopOfFormerBlock : tilingData->rowLoopOfFormerBlock;
        int64_t curRowLoopOfTailBlock =
            isTailBsLoop ? tilingData->tailBsRowLoopOfTailBlock : tilingData->rowLoopOfTailBlock;
        int64_t curSecondUsedCoreNum = isTailBsLoop ? tilingData->tailBsUsedCoreNum : tilingData->secondUsedCoreNum;
        int64_t curStage2RowFactor = isTailBsLoop ? tilingData->tailBsRowFactor : stage2RowFactor;
        int64_t curTailRowFactorOfFormerBlock =
            isTailBsLoop ? tilingData->tailBsTailRowFactorOfFormerBlock : tilingData->tailRowFactorOfFormerBlock;
        int64_t curTailRowFactorOfTailBlock =
            isTailBsLoop ? tilingData->tailBsTailRowFactorOfTailBlock : tilingData->tailRowFactorOfTailBlock;
        int64_t curML1Size = isTailBsLoop ? tilingData->tailBsML1Size : tilingData->mL1Size;

        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t stage2BlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = curSecondUsedCoreNum;
            if (stage2BlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t mmLastAxisSize = CeilAlign(hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
            int64_t xCastFp32BufSize =
                curML1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float)) * sizeof(float);
            int64_t workspaceSize1 = (tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize) / sizeof(float);
            int64_t workspaceSize2 =
                CeilAlign(stage1UsedCoreNum * bs * mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) /
                sizeof(float);
            CopyIn(hcBaseGm, hcBase0Local, 1, hcMult);
            CopyIn(hcBaseGm[hcMult], hcBase1Local, 1, hcMult);
            if constexpr (hasResi) {
                if constexpr (isFac) {
                    CopyIn(hcBaseGm[hcMult * NUM_TWO], hcBase2Local, 1, resSize);
                } else {
                    CopyIn(hcBaseGm[hcMult * NUM_TWO], hcBase2Local, hcMult, hcMult);
                }
            }
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            int64_t rowOuterLoop =
                (stage2BlockIdx == stage2UsedCoreNum - 1) ? curRowLoopOfTailBlock : curRowLoopOfFormerBlock;
            int64_t tailRowFactor =
                (stage2BlockIdx == stage2UsedCoreNum - 1) ? curTailRowFactorOfTailBlock : curTailRowFactorOfFormerBlock;
            int64_t xGmBlockBaseOffsetPart2 = stage2BlockIdx * curRowOfFormerBlock * hcMult * d;
            uint64_t mixBaseOffset = 0;
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t xGmBsBaseOffsetPart2 = rowOuterIdx * curStage2RowFactor * hcMult * d;
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : curStage2RowFactor;

                squareSumOutLocal = squareSumQue.AllocTensor<float>();
                Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);

                CopyIn(invRmsGm[stage2BlockIdx * curRowOfFormerBlock + rowOuterIdx * curStage2RowFactor],
                       squareSumOutLocal, 1, curRowFactor);

                squareSumQue.EnQue(squareSumOutLocal);
                squareSumOutLocal = squareSumQue.DeQue<float>();
                // 搬运矩阵乘的前两段结果--> 内存格式的变更： 连续地址改为 bs *n ; bs *n ,n为4
                // 非对齐，则需要pad出对齐的8
                mxies01Local = mixesQue01.AllocTensor<float>();

                mixBaseOffset = stage2BlockIdx * curRowOfFormerBlock * hcMix + rowOuterIdx * curStage2RowFactor * hcMix;
                CopyInWithOuterFor(hcBeforeNormGm[mixBaseOffset], mxies01Local, 1, curRowFactor, hcMult, bs, hcMix);
                CopyInWithOuterFor(hcBeforeNormGm[mixBaseOffset + hcMult], mxies01Local[curRowFactor * hcMultAlign], 1,
                                   curRowFactor, hcMult, bs, hcMix);
                // // wk:[2, kcorenum, bs, n^2 +2n]
                // // mx0[2,kcorenum, curRowfator, n]
                mixesQue01.EnQue(mxies01Local);
                mxies01Local = mixesQue01.DeQue<float>();

                ProcessPre(mxies01ReduceLocal, mxies01Local, hcBase0Local, squareSumOutLocal, rowBrcbLocal0,
                           hcBrcbLocal1, hcScaleGm.GetValue(0), tilingData->hcEps, curRowFactor, hcMult);

                if (needGrad_) {
                    int64_t hPreBaseOffset =
                        stage2BlockIdx * curRowOfFormerBlock * hcMult + rowOuterIdx * curStage2RowFactor * hcMult;
                    VToMTE3Sync();
                    CopyOut(mxies01ReduceLocal, hPreGm[hPreBaseOffset], curRowFactor, hcMult);
                    MTE3ToVSync();
                }
                int64_t dLoop = tilingData->dLoop;
                // --- pre --
                for (int64_t dLoopIdx = 0; dLoopIdx < dLoop; dLoopIdx++) {
                    int64_t curDFactor = (dLoopIdx == dLoop - 1) ? tilingData->tailDFactor : dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(xGm[xGmBlockBaseOffsetPart2 + xGmBsBaseOffsetPart2 + dLoopIdx * dFactor], xLocal,
                           curRowFactor * hcMult, curDFactor, d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    yLocal = yQue.template AllocTensor<T>();

                    ProcessY(yLocal, xLocal, mxies01ReduceLocal, hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                             hcMult, curDFactor);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();

                    CopyOut(yLocal,
                            yGm[stage2BlockIdx * curRowOfFormerBlock * d + rowOuterIdx * curStage2RowFactor * d +
                                dLoopIdx * dFactor],
                            curRowFactor, curDFactor, d - curDFactor);
                    yQue.template FreeTensor(yLocal);
                }
                // post
                postLocal = postQue.AllocTensor<float>();
                ProcessPost(postLocal, mxies01Local[curRowFactor * hcMultAlign], hcBase1Local, squareSumOutLocal,
                            rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1), curRowFactor, hcMult);
                mixesQue01.FreeTensor(mxies01Local); // 这里对应的申请在上面
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();

                CopyOut(
                    postLocal,
                    postGm[stage2BlockIdx * curRowOfFormerBlock * hcMult + rowOuterIdx * curStage2RowFactor * hcMult],
                    curRowFactor, hcMult);
                postQue.FreeTensor(postLocal);

                // hRes
                if constexpr (hasResi) {
                    mixes2Local = mixesQue2.AllocTensor<float>();
                    mixBaseOffset =
                        stage2BlockIdx * curRowOfFormerBlock * hcMix + rowOuterIdx * curStage2RowFactor * hcMix;
                    for (int64_t j = 0; j < curRowFactor; ++j) {
                        if constexpr (isFac) {
                            CopyIn(hcBeforeNormGm[mixBaseOffset + j * hcMix + hcMult * NUM_TWO],
                                   mixes2Local[j * resSize], 1, resSize);
                        } else {
                            CopyIn(hcBeforeNormGm[mixBaseOffset + j * hcMix + hcMult * NUM_TWO],
                                   mixes2Local[j * hcMult * hcMultAlign], hcMult, hcMult);
                        }
                    }
                    mixesQue2.EnQue(mixes2Local);
                    mixes2Local = mixesQue2.DeQue<float>();

                    combFragLocal = combFragQue.AllocTensor<float>();
                    MulABLastDimBrcInline<float, false>(mxies02ReduceLocal, mixes2Local, squareSumOutLocal,
                                                        rowBrcbLocal0, curRowFactor, resSize);
                    mixesQue2.FreeTensor(mixes2Local);
                    Muls(mxies02ReduceLocal, mxies02ReduceLocal, hcScaleGm.GetValue(NUM_TWO), curRowFactor * resSize);
                    PipeBarrier<PIPE_V>();
                    AddBAFirstDimBrcInline<float>(mxies02ReduceLocal, mxies02ReduceLocal, hcBase2Local, curRowFactor,
                                                  resSize);
                    Copy(combFragLocal, mxies02ReduceLocal, curRowFactor * resSize, 1, {1, 1, 0, 0});
                    PipeBarrier<PIPE_V>();
                    squareSumQue.template FreeTensor(squareSumOutLocal);

                    combFragQue.EnQue(combFragLocal);
                    combFragLocal = combFragQue.DeQue<float>();
                    if constexpr (isFac) {
                        CopyOut(combFragLocal,
                                hResGm[stage2BlockIdx * curRowOfFormerBlock * resSize +
                                       rowOuterIdx * curStage2RowFactor * resSize],
                                curRowFactor, resSize);
                    } else {
                        CopyOut(combFragLocal,
                                hResGm[stage2BlockIdx * curRowOfFormerBlock * hcMult * hcMult +
                                       rowOuterIdx * curStage2RowFactor * hcMult * hcMult],
                                curRowFactor * hcMult, hcMult);
                    }
                    combFragQue.FreeTensor(combFragLocal);
                } else {
                    squareSumQue.template FreeTensor(squareSumOutLocal);
                }
            }
        }
    }

private:
    TPipe *pipe;
    const MhcPreMembaseTilingData *tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> hResGm;

    GlobalTensor<float> hPreGm;
    GlobalTensor<float> hcBeforeNormGm;
    GlobalTensor<float> invRmsGm;

    bool needGrad_ = false;
    bool isTailBsLoop_ = false;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> mxies01ReduceBuf;
    TBuf<QuePosition::VECCALC> mxies02ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;

    LocalTensor<float> mxies01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> rowBrcbLocal0;
    LocalTensor<float> hcBrcbLocal1;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> mxies01ReduceLocal;
    LocalTensor<float> mxies02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;

    int64_t resSize = 0;
    int64_t hcMult = 0;
    int64_t hcMultAlign = 0;
    int64_t stage1XCastWsSize = 0;
    int64_t bs = 0;
    int64_t stage2RowFactor = 0;
    int64_t dFactor = 0;
    int64_t hcMix = 0;
    int64_t d = 0;
};

} // namespace MhcPre

#endif

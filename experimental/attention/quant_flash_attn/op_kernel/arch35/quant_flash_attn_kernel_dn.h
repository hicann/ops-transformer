/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_kernel_dn.h
 * \brief
 */

#ifndef QFA_KERNEL_DN_H
#define QFA_KERNEL_DN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attn_template_tiling_key.h"
#include "quant_flash_attn_common_def.h"
#include "../../common/op_kernel/const_def.h"

using namespace optiling;
using namespace AscendC;
using namespace AttentionCommon;

namespace QFA_KERNEL {

template <typename QFAT, typename CubeBlock, typename VectorBlock>
class QuantFlashAttnKernelDn {
public:
    using QUANT_T = typename QFAT::quantType;
    using SCALE_T = typename QFAT::scaleType;
    using OUT_T = typename QFAT::outputType;
    using SEQLEN_T = uint32_t;
    static constexpr bool SOFTMAX_DN = true;
    static constexpr bool PAGE_ATTENTION = QFAT::pageAttention;
    static constexpr bool HAS_MASK = QFAT::hasMask;
    static constexpr QFA_LAYOUT LAYOUT_Q = QFAT::qLayout;
    static constexpr QFA_LAYOUT LAYOUT_KV = QFAT::kvLayout;

    static constexpr uint32_t mBaseSize = 128;
    static constexpr uint32_t s2BaseSize = 256;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;

    static constexpr uint8_t SYNC_MODE_2 = 2;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {9, 10};
    static constexpr uint16_t CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_BUF1_GMAX_UB_TO_L1 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB = 7;
    static constexpr uint16_t CROSS_CORE_SYNC_BUF1_GMAX_L1_TO_UB = 8;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_0 = 3;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_1 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 2;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 1;
    static constexpr uint16_t CROSS_CORE_SYNC_UB_L1 = 0;

    static constexpr uint32_t PRELOAD_N = 20;
    static constexpr uint32_t DELAY_P_SCALE_N = 3;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;
    static constexpr uint32_t TILE_N = 16;
    static constexpr uint32_t SUB_S2_BASE_SIZE = s2BaseSize / 2;

    ConstInfo constInfo;
    const FlashAttnTilingData *__restrict tilingData;

    // metadata
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    uint32_t sectionNum_ = 0;
    // fa metadata
    uint32_t bN2Start_ = 0;
    uint32_t bN2End_ = 0;
    uint32_t gS1OStart_ = 0;
    uint32_t gS1OEnd_ = 0;
    uint32_t s2OStart_ = 0;
    uint32_t s2OEnd_ = 0;
    uint32_t coreFirstTmpOutWsPos_ = 0;
    uint32_t s2FirstStartVecCore = 0;
    uint32_t tileLoopIdx = 1;
    uint32_t tileMaxIdx = 3;
    uint32_t updateScaleNum = 3;
    // fd metadata
    FDparams fdParams_;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;
    uint32_t mloop = 0;
    uint32_t pscaleNum = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    SeqLensTool<LAYOUT_Q, SEQLEN_T> qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> kvSeqLensTool;

    CubeBlock cubeBlock;
    VectorBlock vectorBlock;

    __aicore__ inline QuantFlashAttnKernelDn()
        : cubeBlock(constInfo, qSeqLensTool, kvSeqLensTool),
          vectorBlock(constInfo, qSeqLensTool, kvSeqLensTool){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensKv, __gm__ uint8_t *seqUsedQ,
                                __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                __gm__ uint8_t *fiaMetaData, const FlashAttnTilingData *__restrict tiling)
    {
        this->tilingData = tiling;

        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);
        fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * sectionNum_);

        InitConstInfo();

        qSeqLensTool.Init(cuSeqlensQ, constInfo.qCuSeqLensSize, seqUsedQ, constInfo.qSeqUsedSize, constInfo.maxSeqlenQ);
        kvSeqLensTool.Init(cuSeqlensKv, constInfo.kvCuSeqLensSize, seqUsedKv, constInfo.kvSeqUsedSize,
                           constInfo.maxSeqlenKv);

        if ASCEND_IS_AIC {
            cubeBlock.InitInput(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable);
        } else {
            vectorBlock.InitInput(attentionOut);
            vectorBlock.ClearOutput();
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        auto fiaBaseParams = this->tilingData->flashAttnBaseParams;
        auto fiaAttenMaskParams = this->tilingData->flashAttnAttenMaskParams;
        auto fiaPageAttentionParams = this->tilingData->flashAttnPageAttentionParams;
        auto fiaWorkspaceParams = this->tilingData->flashAttnWorkspaceParams;
        auto fiaEmptyTensorParams = this->tilingData->flashAttnEmptyTensorParams;

        constInfo.bSize = fiaBaseParams.bSize;
        constInfo.t1Size = fiaBaseParams.t1Size;
        constInfo.t2Size = fiaBaseParams.t2Size;
        constInfo.n2Size = fiaBaseParams.n2Size;
        constInfo.gSize = fiaBaseParams.gSize;
        constInfo.s1Size = fiaBaseParams.s1Size;
        constInfo.s2Size = fiaBaseParams.s2Size;
        constInfo.dSize = fiaBaseParams.dSize;
        constInfo.dSizeV = fiaBaseParams.dSizeV;
        constInfo.qCuSeqLensSize = fiaBaseParams.qCuSeqLensSize;
        constInfo.kvCuSeqLensSize = fiaBaseParams.kvCuSeqLensSize;
        constInfo.qSeqUsedSize = fiaBaseParams.qSeqUsedSize;
        constInfo.kvSeqUsedSize = fiaBaseParams.kvSeqUsedSize;

        constInfo.maxSeqlenQ =
            (fiaBaseParams.maxSeqlenQ >= 0) ? static_cast<uint64_t>(fiaBaseParams.maxSeqlenQ) : constInfo.s1Size;
        constInfo.maxSeqlenKv =
            (fiaBaseParams.maxSeqlenKv >= 0) ? static_cast<uint64_t>(fiaBaseParams.maxSeqlenKv) : constInfo.s2Size;
        constInfo.scaleValue = static_cast<float>(fiaBaseParams.scaleValue);
        constInfo.coreNum = fiaBaseParams.coreNum;

        constInfo.sparseMode = fiaAttenMaskParams.sparseMode;
        constInfo.preTokens = fiaAttenMaskParams.winLefts;
        constInfo.nextTokens = fiaAttenMaskParams.winRights;
        constInfo.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;

        constInfo.accumOutSize = fiaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = fiaWorkspaceParams.logSumExpSize;
        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = fiaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = fiaPageAttentionParams.blockSize;
            constInfo.paLayoutType = fiaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = fiaBaseParams.isSoftMaxLseEnable;
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * sectionIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
        uint32_t createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        uint32_t validTaskCount = 0;
        mloop = 0;
        headS2Split = false;
        tailS2Split = false;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        while (shouldDispatchTask || validTaskCount) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        // vectorBlock.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            if (validTaskCount) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
                if (executedTaskCount > PRELOAD_N) {
                    validTaskCount--;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (bN2Cur != bN2End_) {
            return bN2Cur < bN2End_;
        }
        if (gS1Cur != gS1OEnd_) {
            return gS1Cur < gS1OEnd_;
        }
        return s2Cur < s2OEnd_;
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;

        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split = s2OStart_ != 0U;
            curS2Start = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split = s2OEnd_ != 0U;
            curS2End = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur) {}

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            actSeqLensQ = qSeqLensTool.GetActualSeqLength(bIdx);
            actSeqLensKv = kvSeqLensTool.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (s2Cur < curS2Start && curS2Start < curS2End) {
            return TASK_DEAL_MODE::NOT_START;
        }

        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }
        if (s2Cur == curS2Start) {
            mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfo &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        RunInfo &runInfo3 = taskRunInfo[(loop - DELAY_P_SCALE_N) % PRELOAD_TASK_CACHE_SIZE];
        RunInfo &runInfo20 = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];

        if (loop >= PRELOAD_N && runInfo20.isValid) {
            if ASCEND_IS_AIC {
                if (runInfo20.isC2Sync) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo20.pscaleNum);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo20.pscaleNum + 16);
                }
                if (runInfo20.isUpdatePScale) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                }
                ComputeMm2(runInfo20);
                if (runInfo20.isUpdatePScale) {
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                }
            } else {
                if (runInfo20.isUpdatePScale) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                    ComputeVec2(runInfo20);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                }
            }
            runInfo20.isValid = false;
        }

        if (loop >= DELAY_P_SCALE_N && runInfo3.isValid) {
            if (runInfo3.isUpdatePScale) {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                              runInfo3.tileMaxIdx / 2);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                              runInfo3.tileMaxIdx / 2 + 16);
                    CopyGMaxL1ToUb(runInfo3);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB +
                                                             runInfo3.tileMaxIdx / 2);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB +
                                                             runInfo3.tileMaxIdx / 2 + 16);
                    if (runInfo3.tileMaxIdx == 3) {
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1);
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1 + 16);
                    }
                } else {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB +
                                                           runInfo3.tileMaxIdx / 2);
                    UpdatePScale(runInfo3);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo3.pscaleNum);
                }
            }
        }

        if (runInfo0.isValid) {
            uint32_t mm1ResBufId = (runInfo0.loop / 2) % 2;
            uint32_t subBlockIdx = runInfo0.loop % 2;
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                ComputeMm1(runInfo0);
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
            } else {
                if (subBlockIdx == constInfo.subBlockIdx) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    ComputeVec1(runInfo0);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                }
                if (runInfo0.isUpdatePScale) {
                    if (runInfo0.tileMaxIdx == 0) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_UB_L1);
                    }
                    CopyGMaxUbToL1(runInfo0);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                             runInfo0.tileMaxIdx / 2);
                }
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfo &runInfo) { cubeBlock.ComputeMm1(runInfo); }

    __aicore__ inline void ComputeMm2(RunInfo &runInfo) { cubeBlock.ComputeMm2(runInfo); }

    __aicore__ inline void ComputeVec1(RunInfo &runInfo) { vectorBlock.ComputeVec1(runInfo); }

    __aicore__ inline void CopyGMaxUbToL1(RunInfo &runInfo) { vectorBlock.CopyGMaxUbToL1(runInfo); }

    __aicore__ inline void CopyGMaxL1ToUb(RunInfo &runInfo) { cubeBlock.CopyGMaxL1ToUb(runInfo); }

    __aicore__ inline void UpdatePScale(RunInfo &runInfo) { vectorBlock.UpdatePScale(runInfo); }

    __aicore__ inline void ComputeVec2(RunInfo &runInfo) { vectorBlock.ComputeVec2(runInfo); }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfo &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == QFA_LAYOUT::BSND || LAYOUT_Q == QFA_LAYOUT::TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.curS2LoopIdx = s2Cur - curS2Start;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            uint64_t tailM = (gS1Size > gS1Cur * mBaseSize) ? (gS1Size - gS1Cur * mBaseSize) : mBaseSize;
            info.actMSize = (tailM < mBaseSize) ? static_cast<uint32_t>(tailM) : mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size =
                (info.actS2Size > s2Cur * s2BaseSize) ? (info.actS2Size - s2Cur * s2BaseSize) : 0;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)AttentionCommon::BYTE_BLOCK);      // 统一对齐到32
        info.actSingleLoopS2SizeAlign16 = Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)16); // 统一对齐到16
        info.actSingleLoopS2SizeAlign64 =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)BUFFER_SIZE_BYTE_64B); // 统一对齐到64

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        info.isUpdatePScale = (info.isLastS2Loop || ((info.curS2LoopIdx + 1) % TILE_N == 0));
        info.isC2Sync = (info.curS2LoopIdx % TILE_N == 0);
        if (info.isFirstS2Loop) {
            s2FirstStartVecCore = loop % 2;
        }
        info.s2FirstStartVecCore = s2FirstStartVecCore;
        if (info.isFirstS2Loop || info.isC2Sync) {
            tileLoopIdx = (tileLoopIdx + 1) % 2;
            tileMaxIdx = (tileMaxIdx + 1) % 4;
            pscaleNum = (pscaleNum + 1) % 20;
        }
        if (info.isUpdatePScale && info.curS2LoopIdx / 16 != 0) {
            updateScaleNum = (updateScaleNum + 1) % 4;
        }
        info.updateScaleNum = updateScaleNum;
        info.tileBuffIdx = tileLoopIdx;
        info.tileMaxIdx = tileMaxIdx;
        info.isS2FirstTilePerCore = (info.curS2LoopIdx % TILE_N / 2 == 0);
        info.pscaleNum = pscaleNum / 10;

        if constexpr (SOFTMAX_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actMSizeAlign128 = (info.actMSize + 127) >> 7 << 7;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actVecMSize = (info.actMSize + 1) >> 1;
        }
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if (bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[0]);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);

                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    vectorBlock.InitTensors();
                } else {
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1 + 16);
                    cubeBlock.InitTensors();
                }
                FlashAttention(sectionIdx);
                if ASCEND_IS_AIV {
                    vectorBlock.ReleaseTensors();
                } else {
                    cubeBlock.ReleaseTensors();
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0]);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0] + 16);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1]);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1] + 16);
                }
            }
        }
    }
};

} // namespace QFA_KERNEL

#endif

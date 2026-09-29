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
#include "quant_flash_attn_tiling_data.h"
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
    const FlashAttnTilingData* __restrict tilingData;

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
    uint32_t s2FirstStartVecCore = 0;
    uint32_t tileMaxIdx = 3;
    uint32_t updateScaleNum = 3;
    // fd metadata
    FDparams fdParams_;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint64_t gS1Size_ = 0;
    uint64_t s2LoopTimes_ = 0;
    uint64_t gS1LoopTimes_ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t pscaleNum = 0;
    uint32_t incBIdx_ = 0;
    uint32_t curN2Idx_ = 0;
    uint32_t curKvHeadIdx_ = 0;
    uint32_t kvHeadCnt_ = 0;

    SeqLensTool<LAYOUT_Q, SEQLEN_T> qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> kvSeqLensTool;

    CubeBlock cubeBlock;
    VectorBlock vectorBlock;

    __aicore__ inline QuantFlashAttnKernelDn()
        : cubeBlock(constInfo, qSeqLensTool, kvSeqLensTool),
          vectorBlock(constInfo, qSeqLensTool, kvSeqLensTool){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensKv, __gm__ uint8_t* seqUsedQ,
                                __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* attenMask, __gm__ uint8_t* learnableSink,
                                __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                __gm__ uint8_t* fiaMetaData, const FlashAttnTilingData* __restrict tiling)
    {
        this->tilingData = tiling;

        sectionNum_ = ((__gm__ uint32_t*)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t*)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);
        fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t*)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
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
        constInfo.gRealSize = (fiaBaseParams.gRealSize > 0) ? fiaBaseParams.gRealSize : 1;
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
        if ASCEND_IS_AIC {
            CubeRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
            FlashAttentionImpl<CubeRunInfo, true>(taskRunInfo);
        } else {
            VecRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
            FlashAttentionImpl<VecRunInfo, false>(taskRunInfo);
        }
    }

    template <typename RUNINFO_T, bool IS_AIC>
    __aicore__ inline void FlashAttentionImpl(RUNINFO_T taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        ICachePreLoad(2);
        uint64_t taskLoop = 0;
        incBIdx_ = bN2Start_ / constInfo.n2Size;
        curN2Idx_ = bN2Start_ % constInfo.n2Size;
        curKvHeadIdx_ = curN2Idx_ / constInfo.gRealSize;
        kvHeadCnt_ = curN2Idx_ % constInfo.gRealSize;
        for (uint32_t bN2 = bN2Start_;; ++bN2) {
            if (bN2 == bN2Start_ || curN2Idx_ == 0) {
                actSeqLensQ = qSeqLensTool.GetActualSeqLength(incBIdx_);
                actSeqLensKv = kvSeqLensTool.GetActualSeqLength(incBIdx_);
                gS1Size_ = actSeqLensQ * constInfo.gSize;
                s2LoopTimes_ = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
                gS1LoopTimes_ = (gS1Size_ + mBaseSize - 1) / mBaseSize;
            }
            if (s2LoopTimes_ != 0 && gS1LoopTimes_ != 0) {
                uint32_t gS1Begin = (bN2 == bN2Start_) ? gS1OStart_ : 0U;
                uint32_t gS1Last = (bN2 == bN2End_) ? gS1OEnd_ : static_cast<uint32_t>(gS1LoopTimes_) - 1U;
                for (uint32_t gS1 = gS1Begin; gS1 <= gS1Last; ++gS1) {
                    curS2Start = (bN2 == bN2Start_ && gS1 == gS1OStart_) ? s2OStart_ : 0U;
                    curS2End = (bN2 == bN2End_ && gS1 == gS1OEnd_) ? s2OEnd_ : static_cast<uint32_t>(s2LoopTimes_);
                    for (uint32_t s2 = curS2Start; s2 < curS2End; ++s2) {
                        CreateTask(taskLoop, bN2, gS1, s2, taskRunInfo);
                        ExecuteTask<RUNINFO_T, IS_AIC>(taskLoop, taskRunInfo);
                        taskLoop++;
                    }
                }
            }
            if (bN2 >= bN2End_) {
                break;
            }
            if (++curN2Idx_ == constInfo.n2Size) {
                curN2Idx_ = 0;
                incBIdx_++;
            }
            if (curN2Idx_ == 0) {
                curKvHeadIdx_ = 0;
                kvHeadCnt_ = 0;
            } else if (++kvHeadCnt_ == constInfo.gRealSize) {
                kvHeadCnt_ = 0;
                curKvHeadIdx_++;
            }
        }
        if (taskLoop != 0) {
            uint32_t drainCnt = 0;
            while (drainCnt < PRELOAD_N) {
                ExecuteTask<RUNINFO_T, IS_AIC>(taskLoop, taskRunInfo);
                taskLoop++;
                drainCnt++;
            }
        }
    }

    template <typename RUNINFO_T, bool IS_AIC>
    __aicore__ inline void ExecuteTask(uint64_t loop, RUNINFO_T taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RUNINFO_T& runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];

        if (runInfo0.isValid) {
            if constexpr (IS_AIC) {
                uint32_t mm1ResBufId = (runInfo0.loop / 2) % 2;
                uint32_t subBlockIdx = runInfo0.loop % 2;
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                cubeBlock.ComputeMm1(runInfo0);
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
            }
        }

        if (loop >= PRELOAD_N) {
            RUNINFO_T& runInfo20 = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfo20.isValid) {
                if constexpr (IS_AIC) {
                    if (unlikely(runInfo20.isC2Sync)) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo20.pscaleNum);
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo20.pscaleNum +
                                                                  16);
                    }
                    if (unlikely(runInfo20.isUpdatePScale)) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    }
                    cubeBlock.ComputeMm2(runInfo20);
                    if (unlikely(runInfo20.isUpdatePScale)) {
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                    }
                } else {
                    if (unlikely(runInfo20.isUpdatePScale)) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                        vectorBlock.ComputeVec2(runInfo20);
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    }
                }
                runInfo20.isValid = false;
            }
        }

        if (loop >= DELAY_P_SCALE_N) {
            RUNINFO_T& runInfo3 = taskRunInfo[(loop - DELAY_P_SCALE_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfo3.isValid && unlikely(runInfo3.isUpdatePScale)) {
                if constexpr (IS_AIC) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                              runInfo3.tileMaxIdx / 2);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                              runInfo3.tileMaxIdx / 2 + 16);
                    cubeBlock.CopyGMaxL1ToUb(runInfo3);
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
                    vectorBlock.UpdatePScale(runInfo3);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo3.pscaleNum);
                }
            }
        }

        if (runInfo0.isValid) {
            if constexpr (!IS_AIC) {
                uint32_t mm1ResBufId = (runInfo0.loop / 2) % 2;
                uint32_t subBlockIdx = runInfo0.loop % 2;
                if (subBlockIdx == constInfo.subBlockIdx) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    vectorBlock.ComputeVec1(runInfo0);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                }
                if (runInfo0.isUpdatePScale) {
                    if (runInfo0.tileMaxIdx == 0) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_UB_L1);
                    }
                    vectorBlock.CopyGMaxUbToL1(runInfo0);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                             runInfo0.tileMaxIdx / 2);
                }
            }
        }
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      CubeRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        CubeRunInfo& runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcCubeParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      VecRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        VecRunInfo& runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcVecParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcCubeParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                          CubeRunInfo& info)
    {
        info.loop = loop;
        info.bIdx = incBIdx_;
        info.n2Idx = curN2Idx_;
        info.kvHeadIdx = curKvHeadIdx_;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.s2Idx = s2Cur * s2BaseSize;

        info.actMSize = mBaseSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size_) {
            uint64_t tailM = (gS1Size_ > gS1Cur * mBaseSize) ? (gS1Size_ - gS1Cur * mBaseSize) : mBaseSize;
            info.actMSize = (tailM < mBaseSize) ? static_cast<uint32_t>(tailM) : mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size =
                (actSeqLensKv > (uint64_t)s2Cur * s2BaseSize) ? (actSeqLensKv - (uint64_t)s2Cur * s2BaseSize) : 0;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)AttentionCommon::BYTE_BLOCK);      // 统一对齐到32
        info.actSingleLoopS2SizeAlign16 = Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)16); // 统一对齐到16
        info.actSingleLoopS2SizeAlign64 =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)BUFFER_SIZE_BYTE_64B); // 统一对齐到64

        info.prefetched = (((s2Cur - curS2Start) & 1) == 1);
        info.pairKVCopyS2Size = info.actSingleLoopS2Size;
        if (!info.prefetched && (s2Cur + 1) < curS2End) {
            uint64_t nextSize = actSeqLensKv - (uint64_t)(s2Cur + 1) * s2BaseSize;
            info.pairKVCopyS2Size = s2BaseSize + static_cast<uint32_t>(nextSize > s2BaseSize ? s2BaseSize : nextSize);
        }

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        uint32_t curS2LoopIdx = s2Cur - curS2Start;
        info.isUpdatePScale = (info.isLastS2Loop || ((curS2LoopIdx + 1) % TILE_N == 0));
        info.isC2Sync = (curS2LoopIdx % TILE_N == 0);
        if (info.isC2Sync) {
            tileMaxIdx = (tileMaxIdx + 1) % 4;
            pscaleNum = (pscaleNum + 1) % 20;
        }
        info.actMSizeAlign128 = (info.actMSize + 127) >> 7 << 7;
        info.tileMaxIdx = tileMaxIdx;
        info.pscaleNum = pscaleNum / 10;
    }

    __aicore__ inline void CalcVecParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                         VecRunInfo& info)
    {
        info.loop = loop;
        info.bIdx = incBIdx_;
        info.n2Idx = curN2Idx_;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.curS2LoopIdx = s2Cur - curS2Start; // 在当前核处理的S2的循环下标

        info.actMSize = mBaseSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size_) {
            uint64_t tailM = (gS1Size_ > gS1Cur * mBaseSize) ? (gS1Size_ - gS1Cur * mBaseSize) : mBaseSize;
            info.actMSize = (tailM < mBaseSize) ? static_cast<uint32_t>(tailM) : mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size =
                (actSeqLensKv > (uint64_t)s2Cur * s2BaseSize) ? (actSeqLensKv - (uint64_t)s2Cur * s2BaseSize) : 0;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)AttentionCommon::BYTE_BLOCK); // 统一对齐到32
        info.actSingleLoopS2SizeAlign64 =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)BUFFER_SIZE_BYTE_64B); // 统一对齐到64

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        info.isUpdatePScale = (info.isLastS2Loop || ((info.curS2LoopIdx + 1) % TILE_N == 0));
        info.isC2Sync = (info.curS2LoopIdx % TILE_N == 0);
        if (info.isFirstS2Loop) {
            s2FirstStartVecCore = loop % 2;
        }
        info.s2FirstStartVecCore = s2FirstStartVecCore;
        if (info.isC2Sync) {
            tileMaxIdx = (tileMaxIdx + 1) % 4;
            pscaleNum = (pscaleNum + 1) % 20;
        }
        if (info.isUpdatePScale && info.curS2LoopIdx / 16 != 0) {
            updateScaleNum = (updateScaleNum + 1) % 4;
        }
        info.updateScaleNum = updateScaleNum;
        info.tileMaxIdx = tileMaxIdx;
        info.isS2FirstTilePerCore = (info.curS2LoopIdx % TILE_N / 2 == 0);
        info.pscaleNum = pscaleNum / 10;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
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
                if ASCEND_IS_AIC {
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

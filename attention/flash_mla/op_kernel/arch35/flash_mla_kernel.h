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
 * \file flash_mla_kernel.h
 * \brief FlashMla kernel for arch35.
 */

#ifndef FLASH_MLA_KERNEL_H_
#define FLASH_MLA_KERNEL_H_

#include "flash_mla_public_define_arch35.h"
#if __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202
#include "../arch92/flash_mla_block_cube_arch92.h"
#include "../arch92/flash_mla_block_vec_arch92.h"
#else
#include "flash_mla_block_cube.h"
#include "flash_mla_block_vec.h"
#endif
#include "../utils/flash_mla_memory_copy_arch35.h"
#include "flash_mla_block_vec_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_mla_tiling_data.h"

namespace FlashMlaKernel {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantMlaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;
    static constexpr uint32_t PRELOAD_N = 2U;                            // C1 C1 C2（对齐 fia，bmm2 落后 1 轮）
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N << 1U; // 4 (power-of-2 for bitmask)

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
    static constexpr FLASH_MLA_LAYOUT LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr FLASH_MLA_LAYOUT LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    AscendC::GlobalTensor<uint32_t> faMetaDataGm_;
    AscendC::GlobalTensor<uint32_t> fdMetaDataGm_;
    uint32_t sectionNum_ = 0U;
    uint32_t metaAicCoreNum_ = 0U;
    uint32_t metaAivCoreNum_ = 0U;
    bool enableFlashDecode_ = false;
    FlashMlaFdParamsX fdParams_ = {};
    static_assert(FlashMlaCommon::FA_METADATA_STRIDE == 16 && FlashMlaCommon::FD_METADATA_STRIDE == 16,
                  "flash_mla: metadata per-core stride must be 16 to match flash_mla_metadata AICPU layout");

    ConstInfoX constInfo_;
    FlashMlaSeqLensTool<Q_MODE, KV_MODE> seqLensTool_;

    const optiling::FlashMlaNoQuantTilingArch35* __restrict tilingData_;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    int64_t validTaskNum_ = 0;
    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    uint64_t cachedS2LoopTimes_ = 0;
    uint64_t cachedG1S1LoopTimes_ = 0;
    uint32_t curS2Start_ = 0;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_ = 0;
    uint32_t prevBN2Idx_ = 0;
    uint32_t prevGS1Idx_ = 0;
    uint32_t mloop_ = 0;
    bool headS2Split_ = false;
    bool tailS2Split_ = false;

    __aicore__ inline FlashAttentionNoQuantMlaKernel()
        : cubeBlock_(constInfo_, seqLensTool_),
          vecFaBlock_(constInfo_, seqLensTool_),
          vecFdBlock_(constInfo_, seqLensTool_){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* kCache, __gm__ uint8_t* sinks,
                                __gm__ uint8_t* attenMask, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                __gm__ uint8_t* cacheSeqlens, __gm__ uint8_t* blockTable, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace, __gm__ uint8_t* metadata,
                                const optiling::FlashMlaNoQuantTilingArch35* __restrict tiling)
    {
        tilingData_ = tiling;

        sectionNum_ = ((__gm__ uint32_t*)metadata)[0];
        enableFlashDecode_ = static_cast<bool>(((__gm__ uint32_t*)metadata)[1U]);
        metaAicCoreNum_ = ((__gm__ uint32_t*)metadata)[optiling::FLASH_MLA_METADATA_AIC_NUM_INDEX];
        metaAivCoreNum_ = ((__gm__ uint32_t*)metadata)[optiling::FLASH_MLA_METADATA_AIV_NUM_INDEX];
        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t*)(metadata + optiling::FA_METADATA_HEADER_OFFSET),
                                      metaAicCoreNum_ * FlashMlaCommon::FA_METADATA_STRIDE * sectionNum_);
        fdMetaDataGm_.SetGlobalBuffer(
            (__gm__ uint32_t*)(metadata + optiling::FA_METADATA_HEADER_OFFSET +
                               FlashMlaCommon::FA_METADATA_STRIDE * metaAicCoreNum_ * sectionNum_ * sizeof(uint32_t)),
            metaAivCoreNum_ * FlashMlaCommon::FD_METADATA_STRIDE * sectionNum_);

        InitConstInfo();
        constInfo_.learnableSinkFlag = (sinks != nullptr);

        seqLensTool_.InitQ(cuSeqlensQ, sequsedQ, constInfo_.actualSeqLenSize, constInfo_.s1Size, constInfo_.bSize);
        seqLensTool_.InitKv(cacheSeqlens, constInfo_.actualSeqLenKVSize, constInfo_.s2Size);

        if ASCEND_IS_AIV {
            vecFaBlock_.InitVecBlock(sinks, attenMask, softmaxLse, attentionOut, workspace);
            vecFaBlock_.ClearOutput(tilingData_->flashMlaEmptyTensorParams.needInit != 0);
        }

        if ASCEND_IS_AIC {
            cubeBlock_.InitCubeBlock(query, kCache, blockTable);
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock_.InitBlock(softmaxLse, attentionOut);
                vecFdBlock_.InitGlobalTensor(vecFaBlock_.softmaxFDMaxGm_, vecFaBlock_.softmaxFDSumGm_,
                                             vecFaBlock_.accumOutGm_);
            }
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = AscendC::GetBlockIdx();
        } else {
            constInfo_.aivIdx = AscendC::GetBlockIdx();
            constInfo_.aicIdx = constInfo_.aivIdx / AscendC::GetSubBlockNum();
            constInfo_.subBlockIdx = AscendC::GetSubBlockIdx();
        }

        const auto& flashMlaBaseParams = tilingData_->flashMlaBaseParams;
        const auto& flashMlaAttenMaskParams = tilingData_->flashMlaAttenMaskParams;
        const auto& flashMlaPageAttentionParams = tilingData_->flashMlaPageAttentionParams;
        const auto& flashMlaWorkspaceParams = tilingData_->flashMlaWorkspaceParams;

        constInfo_.bSize = flashMlaBaseParams.bSize;
        constInfo_.t1Size = flashMlaBaseParams.t1Size;
        constInfo_.n2Size = flashMlaBaseParams.n2Size;
        constInfo_.gSize = flashMlaBaseParams.gSize;
        constInfo_.s1Size = flashMlaBaseParams.s1Size;
        constInfo_.s2Size = flashMlaBaseParams.s2Size;
        constInfo_.dSize = flashMlaBaseParams.dSize;
        constInfo_.dSizeV = flashMlaBaseParams.dSizeV;
        constInfo_.dSizeRope = flashMlaBaseParams.dSizeRope;
        constInfo_.actualSeqLenSize = flashMlaBaseParams.actualSeqLengthsQSize;
        constInfo_.actualSeqLenKVSize = flashMlaBaseParams.actualSeqLengthsKVSize;
        constInfo_.scaleValue = flashMlaBaseParams.scaleValue;
        constInfo_.coreNum = flashMlaBaseParams.coreNum;

        constInfo_.keyStrides.bnStride = flashMlaBaseParams.keyStrides.bnStride;
        constInfo_.keyStrides.n2Stride = flashMlaBaseParams.keyStrides.n2Stride;

        constInfo_.sparseMode = flashMlaAttenMaskParams.sparseMode;
        constInfo_.winLeft = flashMlaAttenMaskParams.winLeft;
        constInfo_.winRight = flashMlaAttenMaskParams.winRight;
        if constexpr (HAS_MASK) {
            constInfo_.attenMaskBatch = flashMlaAttenMaskParams.attenMaskBatch;
            constInfo_.attenMaskS1Size = flashMlaAttenMaskParams.attenMaskS1Size;
            constInfo_.attenMaskS2Size = flashMlaAttenMaskParams.attenMaskS2Size;
        }
        constInfo_.l2CacheOffFlag = flashMlaBaseParams.l2CacheOffFlag;
        constInfo_.accumOutSize = flashMlaWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = flashMlaWorkspaceParams.logSumExpSize;
        constInfo_.maxBlockNumPerBatch = flashMlaPageAttentionParams.maxBlockNumPerBatch;
        constInfo_.blockSize = flashMlaPageAttentionParams.blockSize;
        constInfo_.isSoftmaxLseEnable = flashMlaBaseParams.isSoftMaxLseEnable;
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FlashMlaCommon::FA_METADATA_STRIDE * metaAicCoreNum_ * sectionIdx +
               FlashMlaCommon::FA_METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FlashMlaCommon::FD_METADATA_STRIDE * metaAivCoreNum_ * sectionIdx +
               FlashMlaCommon::FD_METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        constInfo_.bN2Start = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        constInfo_.gS1OStart = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_M_START_INDEX, sectionIdx));
        constInfo_.s2OStart = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_S2_START_INDEX, sectionIdx));
        constInfo_.bN2End = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        constInfo_.gS1OEnd =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_M_END_INDEX, sectionIdx));
        constInfo_.s2OEnd = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_S2_END_INDEX, sectionIdx));
        constInfo_.coreFirstTmpOutWsPos = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, optiling::FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, optiling::FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, optiling::FA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, optiling::FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx = fdMetaDataGm_.GetValue(
            GetFDMetaDataIndex(constInfo_.aivIdx, optiling::FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum = fdMetaDataGm_.GetValue(
            GetFDMetaDataIndex(constInfo_.aivIdx, optiling::FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, optiling::FA_FD_M_START_INDEX, sectionIdx));
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        // 跳过稀疏任务时也需重置 FD 状态，避免复用上一 section 的 workspace 槽。
        headS2Split_ = false;
        tailS2Split_ = false;

        FlashMlaRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        uint32_t bN2Cur = constInfo_.bN2Start;
        uint32_t gS1Cur = constInfo_.gS1OStart;
        uint32_t s2Cur = constInfo_.s2OStart;
        prevBN2Idx_ = bN2Cur;
        prevGS1Idx_ = gS1Cur;

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        uint32_t createdTaskCount = 0U;
        uint32_t executedTaskCount = 0U;
        while (shouldDispatchTask || shouldExecuteTask) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo, shouldDispatchTask);
            if (shouldExecuteTask) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (bN2Cur != constInfo_.bN2End) {
            return bN2Cur < constInfo_.bN2End;
        }
        if (gS1Cur != constInfo_.gS1OEnd) {
            return gS1Cur < constInfo_.gS1OEnd;
        }
        return s2Cur < constInfo_.s2OEnd;
    }

    __aicore__ inline bool ShouldExecuteTask(FlashMlaRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE],
                                             bool stillDispatchTask)
    {
        if (validTaskNum_ <= 0) {
            return false;
        }
        // 预取需要下一任务已创建；排空流水时不再等待后续任务。
        if (stillDispatchTask && validTaskNum_ < 2) {
            return false;
        }
        return true;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart) && (s2Cur == constInfo_.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo_.n2Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            actSeqLensKv_ = seqLensTool_.kvActSeqLensParser_.GetActualSeqLength(bIdx);
            actSeqLensQ_ = seqLensTool_.qActSeqLensParser_.GetActualSeqLength(bIdx);
            cachedS2LoopTimes_ = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
            uint64_t gS1Size = actSeqLensQ_ * constInfo_.gSize;
            cachedG1S1LoopTimes_ = (gS1Size + mBaseSize - 1) / mBaseSize;
        }

        if (cachedS2LoopTimes_ == 0 || cachedG1S1LoopTimes_ == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        if (curS2Start_ >= curS2End_) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur < curS2Start_) {
            return TASK_DEAL_MODE::NOT_START;
        }

        if (s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetWindowFromTopLeft(int64_t actSeqLensQ_, int64_t actSeqLensKv_,
                                                int64_t& winLeftFromTopLeft, int64_t& winRightFromTopLeft)
    {
        winLeftFromTopLeft = constInfo_.winLeft;
        winRightFromTopLeft = constInfo_.winRight;
        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, winLeftFromTopLeft, winRightFromTopLeft,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            winLeftFromTopLeft += static_cast<int64_t>(actSeqLensQ_) - static_cast<int64_t>(actSeqLensKv_);
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            winRightFromTopLeft = static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            winRightFromTopLeft += static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart)) {
            headS2Split_ = constInfo_.s2OStart != 0U;
            curS2Start_ = constInfo_.s2OStart;
        }

        if ((bN2Cur == constInfo_.bN2End) && (gS1Cur == constInfo_.gS1OEnd)) {
            tailS2Split_ = constInfo_.s2OEnd != 0U;
            curS2End_ = constInfo_.s2OEnd;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        int64_t winLeftFromTopLeft = 0;
        int64_t winRightFromTopLeft = 0;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;

        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ_) * static_cast<int64_t>(constInfo_.gSize)) -
            1;

        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo_.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo_.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ_) ==
                s1GLastToken / static_cast<int64_t>(actSeqLensQ_)) {
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ_);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ_);
            } else {
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
                s1FirstToken = 0;
            }
        }
        GetWindowFromTopLeft(actSeqLensQ_, actSeqLensKv_, winLeftFromTopLeft, winRightFromTopLeft);
        int64_t s2FirstToken = s1FirstToken - winLeftFromTopLeft;
        int64_t s2LastToken = s1LastToken + winRightFromTopLeft;
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            return;
        }
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));

        curS2Start_ = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        curS2End_ = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        if (bN2Cur == constInfo_.bN2Start && gS1Cur == constInfo_.gS1OStart) { // first line
            headS2Split_ = constInfo_.s2OStart > curS2Start_ ? true : false;
            curS2Start_ = AttentionCommon::Max(curS2Start_, constInfo_.s2OStart);
        }
        if (bN2Cur == constInfo_.bN2End && gS1Cur == constInfo_.gS1OEnd) { // last line
            tailS2Split_ = constInfo_.s2OEnd > 0U ? true : false;
            curS2End_ = constInfo_.s2OEnd > 0U ? AttentionCommon::Min(curS2End_, constInfo_.s2OEnd) : curS2End_;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, FlashMlaRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        FlashMlaRunInfoX& runInfo0 = taskRunInfo[loop & (PRELOAD_TASK_CACHE_SIZE - 1)]; // 本轮任务
        FlashMlaRunInfoX& runInfoNegN =
            taskRunInfo[(loop - (PRELOAD_N - 1)) &
                        (PRELOAD_TASK_CACHE_SIZE - 1)]; // 上 PRELOAD_N-1 轮任务（对齐 fia，bmm2 落后 1 轮）
        FlashMlaRunInfoX& runInfoNext =
            taskRunInfo[(loop + 1) & (PRELOAD_TASK_CACHE_SIZE - 1)]; // 下一轮任务（仅用于提前搬运 KV）

        if ASCEND_IS_AIC {
            ComputeMm1(runInfo0, runInfoNext, loop == 0U);
        } else {
            if (runInfo0.isValid) {
                ComputeVec1(runInfo0);
            }
        }

        if (loop >= (PRELOAD_N - 1)) {
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    ComputeMm2(runInfoNegN);
                } else {
                    ComputeVec2(runInfoNegN);
                }
                DisableTask(runInfoNegN);
            }
        }
    }

    __aicore__ inline void ComputeMm1(FlashMlaRunInfoX& runInfo, FlashMlaRunInfoX& nextRunInfo, bool isFirstLoop)
    {
        cubeBlock_.PreloadMte2(runInfo, nextRunInfo, isFirstLoop);
        if (runInfo.isValid) {
            cubeBlock_.IterateBmm1(runInfo);
        }
    }

    __aicore__ inline void ComputeMm2(FlashMlaRunInfoX& runInfo)
    {
        cubeBlock_.IterateBmm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(FlashMlaRunInfoX& runInfo)
    {
        vecFaBlock_.ProcessVec1(runInfo);
    }

    __aicore__ inline void ComputeVec2(FlashMlaRunInfoX& runInfo)
    {
        vecFaBlock_.ProcessVec2(runInfo);
    }

    __aicore__ inline void EnableTask(FlashMlaRunInfoX& runInfo)
    {
        runInfo.isValid = true;
        validTaskNum_++;
    }

    __aicore__ inline void DisableTask(FlashMlaRunInfoX& runInfo)
    {
        runInfo.isValid = false;
        validTaskNum_--;
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      FlashMlaRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        FlashMlaRunInfoX& runInfo = taskRunInfo[loop & (PRELOAD_TASK_CACHE_SIZE - 1)]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        EnableTask(runInfo);
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      FlashMlaRunInfoX& info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / constInfo_.n2Size;
        info.n2Idx = bN2Cur % constInfo_.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ_;
        info.actS2Size = actSeqLensKv_;
        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo_.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }

        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }

        GetWindowFromTopLeft(actSeqLensQ_, actSeqLensKv_, info.winLeftFromTopLeft, info.winRightFromTopLeft);

        info.isFirstFdBlock = !(headS2Split_ && bN2Cur == constInfo_.bN2Start && gS1Cur == constInfo_.gS1OStart);
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo_.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);
        info.actVecMSize = (info.actMSize + CV_RATIO - 1) / CV_RATIO;
        info.vecMbaseIdx = 0;
        if (CV_RATIO == 2 && constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((constInfo_.bN2Start == constInfo_.bN2End && constInfo_.gS1OStart == constInfo_.gS1OEnd)) {
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart)) {
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == constInfo_.bN2End) && (gS1Cur == constInfo_.gS1OEnd)) {
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t& bN2Cur, uint32_t& gS1Cur,
                                          uint32_t& s2Cur)
    {
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        } else if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
            s2Cur++;
            return;
        }

        s2Cur = 0;

        if (gS1Cur + 1 < cachedG1S1LoopTimes_) {
            gS1Cur++;
            return;
        }

        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        // 所有 AIV 必须执行相同次数的 SyncAll，包括没有 FD 任务的核。
        if (!enableFlashDecode_) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock_.template InitBuffers<VecFaBlockType::GetFdBaseOffset()>();
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        vecFdBlock_.FlashDecode(fdParams_);
        AscendC::SyncAll();
    }

    __aicore__ inline void Process()
    {
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.InitBuffers();
                vecFaBlock_.InitCrossCoreSync();
                vecFaBlock_.AllocEventID();
            } else {
                cubeBlock_.InitCrossCoreSync();
                cubeBlock_.InitBuffers();
            }
        }

        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                FlashAttention(sectionIdx);
            }

            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(sectionIdx);
                }
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
                vecFaBlock_.UnInitCrossCoreSync();
            } else {
                cubeBlock_.UnInitCrossCoreSync();
            }
        }
    }
}; // FlashAttentionNoQuantMlaKernel

} // namespace FlashMlaKernel

#endif // FLASH_MLA_KERNEL_H_

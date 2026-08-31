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
 * \file flash_attn_kernel_noquant_gqa_dn.h
 * \brief FlashAttentionNoQuantGqaKernelDn —— Dn 路径专用 kernel 模板（独立类）。
 */

#ifndef FLASH_ATTN_KERNEL_NOQUANT_GQA_DN_H_
#define FLASH_ATTN_KERNEL_NOQUANT_GQA_DN_H_

#include "adv_api/activation/softmax.h" // SoftmaxConfig / SoftmaxMode，fia_public_define.h 依赖
#if __has_include("../../../common/op_kernel/fia_public_define.h")
#include "../../../common/op_kernel/fia_public_define.h"
#else
#include "../../common/fia_public_define.h"
#endif
#include "kernel_operator_list_tensor_intf.h" // TensorDesc
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_utils.h"

#include "flash_attn_block_cube_noquant_gqa_dn.h"
#include "flash_attn_block_vec_noquant_gqa_dn.h"
#if __has_include("../../../common/op_kernel/memory_copy_arch35.h")
#include "../../../common/op_kernel/memory_copy_arch35.h"
#else
#include "../../common/memory_copy_arch35.h"
#endif
#include "flash_attn_block_vec_noquant_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename FA_T, typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernelDn {
public:
    using T = float;
    using SEQLEN_T = uint32_t;
    using INPUT_T = typename FA_T::inputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    // task ring buffer 至少需要 PRELOAD_N+1(=3) 个slot；为了用位掩码(loop & MASK)代替取模(loop % SIZE)，
    // 将容量向上取整到最近的2的幂(=4)。代价仅为多1个RunInfoX slot，换取消除内层循环的取模scalar bound。
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1, "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    ConstInfoX constInfo;

    SeqLensTool<LAYOUT_T, SEQLEN_T> qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> kvSeqLensTool;

    __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tilingData;

    // block define
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    // 分核信息
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    // metadata
    uint32_t sectionNum_;
    // fa metadata
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;
    // fd metadata
    FDparamsX fdParams_;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start;
    uint32_t curS2End = 0;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantGqaKernelDn()
        : cubeBlock(constInfo, qSeqLensTool, kvSeqLensTool), vecFaBlock(constInfo, qSeqLensTool, kvSeqLensTool),
          vecFdBlock(constInfo, qSeqLensTool, kvSeqLensTool){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *fiaMetaData, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tiling)
    {
        tilingData = tiling;

        InitConstInfo();

        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];
        constInfo.enableFlashDecode = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);
        constInfo.learnableSinkFlag = (learnableSink != nullptr);

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);

        qSeqLensTool.Init(cuSeqLensQ, constInfo.cuSeqLensQSize, seqUsedQ, constInfo.seqUsedQSize, constInfo.s1Size);
        kvSeqLensTool.Init(cuSeqLensKv, constInfo.cuSeqLensKVSize, seqUsedKv, constInfo.seqUsedKvSize,
                           constInfo.s2Size);

        if ASCEND_IS_AIC {
            cubeBlock.InitBlock(query, key, value, blockTable);
        }
        if ASCEND_IS_AIV {
            vecFaBlock.InitBlock(attenMask, learnableSink, softmaxLse, attentionOut, workspace);
            vecFaBlock.ClearOutput();
            if (constInfo.enableFlashDecode) {
                fdMetaDataGm.SetGlobalBuffer(
                    (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                        FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
                    FA_AIV_CORE_NUM * 16U * sectionNum_);
                vecFdBlock.InitBlock(learnableSink, softmaxLse, attentionOut);
                vecFdBlock.InitGlobalTensor(vecFaBlock.softmaxFDMaxGm, vecFaBlock.softmaxFDSumGm,
                                            vecFaBlock.accumOutGm);
            }
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
        constInfo.bSize = tilingData->flashAttnBaseParams.bSize;
        constInfo.t1Size = tilingData->flashAttnBaseParams.t1Size;
        constInfo.t2Size = tilingData->flashAttnBaseParams.t2Size;
        constInfo.n2Size = tilingData->flashAttnBaseParams.n2Size;
        constInfo.gSize = tilingData->flashAttnBaseParams.gSize;
        constInfo.s1Size = tilingData->flashAttnBaseParams.s1Size;
        constInfo.s2Size = tilingData->flashAttnBaseParams.s2Size;
        constInfo.dSize = tilingData->flashAttnBaseParams.dSize;
        constInfo.dSizeV = tilingData->flashAttnBaseParams.dSizeV;
        constInfo.cuSeqLensQSize = tilingData->flashAttnBaseParams.cuSeqLensQSize;
        constInfo.cuSeqLensKVSize = tilingData->flashAttnBaseParams.cuSeqLensKVSize;
        constInfo.seqUsedQSize = tilingData->flashAttnBaseParams.seqUsedQSize;
        constInfo.seqUsedKvSize = tilingData->flashAttnBaseParams.seqUsedKvSize;
        constInfo.scaleValue = static_cast<float>(tilingData->flashAttnBaseParams.scaleValue);
        constInfo.coreNum = tilingData->flashAttnBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(tilingData->flashAttnBaseParams.outputLayout);
        constInfo.needInitOutput = tilingData->flashAttnBaseParams.needInitOutput;

        constInfo.sparseMode = tilingData->flashAttnAttenMaskParams.sparseMode;
        constInfo.preTokens = tilingData->flashAttnAttenMaskParams.winLefts;
        constInfo.nextTokens = tilingData->flashAttnAttenMaskParams.winRights;
        constInfo.attenMaskBatch = tilingData->flashAttnAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = tilingData->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = tilingData->flashAttnAttenMaskParams.attenMaskS2Size;
        constInfo.isExistRowInvalid = tilingData->flashAttnAttenMaskParams.isExistRowInvalid;

        constInfo.accumOutSize = tilingData->flashAttnWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = tilingData->flashAttnWorkspaceParams.logSumExpSize;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = tilingData->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = tilingData->flashAttnPageAttentionParams.blockSize;
            constInfo.paLayoutType = tilingData->flashAttnPageAttentionParams.paLayoutType;
            // kvcache非连续场景的strides, 透传给offsetCalculator
            constInfo.keyBnStride = tilingData->flashAttnBaseParams.keyBnStride;
            constInfo.keyN2Stride = tilingData->flashAttnBaseParams.keyN2Stride;
            constInfo.valueBnStride = tilingData->flashAttnBaseParams.valueBnStride;
            constInfo.valueN2Stride = tilingData->flashAttnBaseParams.valueN2Stride;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = tilingData->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
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
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
        uint32_t createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        mloop = 0;
        headS2Split = false;
        tailS2Split = false;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        uint32_t validTaskCount = 0; // 未执行(有效)的任务数
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
        // S2有效块区间为[curS2Start, curS2End), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start继续计算, 不能跳行 (BAND等sparse模式curS2Start常>0)
        if (s2Cur < curS2Start && curS2Start < curS2End) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start>=curS2End)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start) {
            mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo.preTokens;
        nextTokenLeftUp = constInfo.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo.sparseMode);

        if (constInfo.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
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

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) -
            1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_T>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start, curS2End
        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            headS2Split = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            tailS2Split = s2OEnd_ > 0U ? true : false;
            curS2End = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务

        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
                ComputeVec1(runInfo0);
            }
        }
        if (loop >= PRELOAD_N) {
            RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK]; // 上PRELOAD_N轮任务
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    ComputeMm2(runInfoNegN);
                } else {
                    ComputeVec2(runInfoNegN);
                }
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        cubeBlock.IterateBmm1(runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        cubeBlock.IterateBmm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        vecFaBlock.ProcessVec1(runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        vecFaBlock.ProcessVec2(runInfo);
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
        info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
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

        // 尚未到达有效区间, 快进s2Cur到curS2Start, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前S2未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }
        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }
        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!constInfo.enableFlashDecode) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock.InitBuffers();
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        vecFdBlock.FlashDecode(fdParams_);
        AscendC::SyncAll();
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

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_START_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            vecFaBlock.InitBuffers();
            vecFaBlock.InitCrossCoreSync();
            vecFaBlock.AllocEventID();
        } else {
            cubeBlock.InitBuffers();
            cubeBlock.InitCrossCoreSync();
            cubeBlock.AllocEventID();
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                FlashAttention(sectionIdx);
            }
            if ASCEND_IS_AIV {
                FlashDecode(sectionIdx);
            }
        }
        if ASCEND_IS_AIV {
            vecFaBlock.FreeEventID();
            vecFaBlock.UnInitCrossCoreSync();
        } else {
            cubeBlock.FreeEventID();
            cubeBlock.UnInitCrossCoreSync();
        }
    }
}; // FlashAttentionNoQuantGqaKernelDn

} // namespace BaseApi

#endif // FLASH_ATTN_KERNEL_NOQUANT_GQA_DN_H_

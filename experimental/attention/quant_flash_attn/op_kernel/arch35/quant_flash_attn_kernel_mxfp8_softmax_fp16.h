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
 * \file quant_flash_attn_kernel_mxfp8_softmax_fp16.h
 * \brief MxFP8 Softmax FP16 kernel 主框架：三层任务调度（参考主线 MxFP8）+ C1/V1/C2/V2 核间流水
 *        item 级 flag 编排（参考 MxFP4 DN）。计算已全部下沉 block 层（cube/vector），
 *        本层只负责任务调度、跨核同步与 GM 指针接线。
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_KERNEL_MXFP8_SOFTMAX_FP16_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attn_common_def_mxfp8_softmax_fp16.h"
#include "quant_flash_attn_tiling_data.h"
#include "../../common/op_kernel/memcopy/attn/const_def.h"
#include "../../common/op_kernel/memcopy/attn/parser.h"

// 核间流水 printf 调试开关（置 0 关闭，printf 会拖慢流水）
#ifndef QFA_SF16_DEBUG_PRINTF
#define QFA_SF16_DEBUG_PRINTF 0
#endif
#include "quant_flash_attn_block_cube_mxfp8_softmax_fp16.h"
#include "quant_flash_attn_block_vector_mxfp8_softmax_fp16.h"

using namespace optiling;
using namespace AscendC;
using namespace AttentionCommon;

namespace QFA_KERNEL {

// ===== 类型别名（单模板组合：Q/K/V mxfp8 + softmax fp16 + 输出 bf16）=====
using QUANT_T = fp8_e4m3fn_t; // Q/K/V
using SCALE_T = fp8_e8m0_t;   // QScale/KScale/VScale/pScale
using SOFTMAX_T = half;       // mm1Res / softmax 中间类型（FP16 方案核心）
using OUT_T = bfloat16_t;     // 最终输出

template <typename CubeBlockType, typename VectorBlockType>
class QuantFlashAttnKernelMxfp8SoftmaxFp16 {
public:
    __aicore__ inline QuantFlashAttnKernelMxfp8SoftmaxFp16()
        : cubeBlock_(constInfo_),
          vectorBlock_(constInfo_)
    {}

    ConstInfo constInfo_;
    CubeBlockType cubeBlock_;
    VectorBlockType vectorBlock_;
    const __gm__ QuantFlashAttnTilingData* __restrict tilingData_ = nullptr;

    // ============================== GM 输入暂存 ==============================
    __gm__ uint8_t* queryPtr_ = nullptr;
    __gm__ uint8_t* keyPtr_ = nullptr;
    __gm__ uint8_t* valuePtr_ = nullptr;
    __gm__ uint8_t* dequantScaleQPtr_ = nullptr;
    __gm__ uint8_t* dequantScaleKPtr_ = nullptr;
    __gm__ uint8_t* dequantScaleVPtr_ = nullptr;
    __gm__ uint8_t* attnOutPtr_ = nullptr;
    __gm__ uint8_t* softmaxLsePtr_ = nullptr;
    __gm__ uint8_t* workspacePtr_ = nullptr;

    // ============================== 调度状态 ==============================
    uint64_t actSeqLensQ_ = 0;
    uint64_t actSeqLensKv_ = 0;
    uint32_t curS2Start_ = 0;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_ = 0;
    uint32_t prevBN2Idx_ = 0;
    uint32_t prevGS1Idx_ = 0;
    uint32_t mloop_ = 0; // [bN2, gS1] 行计数，isFirstS2Loop 时递增，供 softmaxStateSlot 索引
    bool headS2Split_ = false;
    bool tailS2Split_ = false;

    // ============================== metadata ==============================
    uint32_t sectionNum_ = 0;
    uint32_t metadataAicNum_ = 0;
    uint32_t metadataAivNum_ = 0;
    GlobalTensor<uint32_t> faMetaDataGm_;
    uint32_t bN2Start_ = 0;
    uint32_t bN2End_ = 0;
    uint32_t gS1OStart_ = 0;
    uint32_t gS1OEnd_ = 0;
    uint32_t s2OStart_ = 0;
    uint32_t s2OEnd_ = 0;

    // ============================== seqlen 解析（BNSD 无 PA → BY_BATCH seqUsed）==============================
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser_;

    // ============================== 函数 ==============================
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable, __gm__ uint8_t* pScale,
                                __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
                                __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* workspace, const __gm__ QuantFlashAttnTilingData* __restrict tiling)
    {
        this->tilingData_ = tiling;
        InitConstInfo();

        // GM 指针暂存（本阶段不消费）
        queryPtr_ = query;
        keyPtr_ = key;
        valuePtr_ = value;
        dequantScaleQPtr_ = dequantScaleQuery;
        dequantScaleKPtr_ = dequantScaleKey;
        dequantScaleVPtr_ = dequantScaleValue;
        attnOutPtr_ = attnOut;
        softmaxLsePtr_ = softmaxLse;
        workspacePtr_ = workspace;

        cubeBlock_.InitInput(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, qSeqUsedParser_,
                             kvSeqUsedParser_);

        // metadata 头（AICPU 生成）
        constInfo_.needInitOutput = ((__gm__ uint32_t*)metadata)[QFA_HEAD_NEED_INIT_OUTPUT_INDEX] != 0;
        sectionNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadataAicNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIC_NUM_INDEX];
        metadataAivNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIV_NUM_INDEX];
        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET),
                                      sectionNum_ * metadataAicNum_ * METADATA_STRIDE);

        // seqlen 解析（BNSD：seqUsed 按 batch 索引，未传入时回退 shape）
        qSeqUsedParser_.Init(sequsedQ, static_cast<uint32_t>(constInfo_.seqUsedQSize), constInfo_.s1Size);
        // 向量块 GM 输出初始化（必须在 parser Init 之后：OffsetCalculator 内部为拷贝语义）
        vectorBlock_.InitInput(attnOut, softmaxLse, pScale, qSeqUsedParser_);
        kvSeqUsedParser_.Init(sequsedKv, static_cast<uint32_t>(constInfo_.seqUsedKvSize), constInfo_.s2Size);

#if QFA_SF16_DEBUG_PRINTF
        if ASCEND_IS_AIC {
            AscendC::printf("[Init][AIC%u] coreNum=%u sectionNum=%u\n", constInfo_.aicIdx, constInfo_.coreNum,
                            sectionNum_);
        } else {
            AscendC::printf("[Init][AIV%u] aic=%u subBlk=%u coreNum=%u\n", constInfo_.aivIdx, constInfo_.aicIdx,
                            constInfo_.subBlockIdx, constInfo_.coreNum);
        }
#endif
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx();
            constInfo_.subBlockIdx = 0;
        } else {
            constInfo_.aivIdx = GetBlockIdx();
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }

        const auto& qfaBaseParams = this->tilingData_->quantTiling.quantFlashAttnBaseParams;
        constInfo_.bSize = qfaBaseParams.bSize;
        constInfo_.t1Size = qfaBaseParams.t1Size;
        constInfo_.t2Size = qfaBaseParams.t2Size;
        constInfo_.n2Size = qfaBaseParams.n2Size;
        constInfo_.gSize = qfaBaseParams.gSize;
        constInfo_.s1Size = qfaBaseParams.s1Size;
        constInfo_.s2Size = qfaBaseParams.s2Size;
        constInfo_.dSize = qfaBaseParams.dSize;
        constInfo_.dSizeV = qfaBaseParams.dSizeV;
        constInfo_.cuSeqLensQSize = qfaBaseParams.cuSeqLensQSize;
        constInfo_.cuSeqLensKVSize = qfaBaseParams.cuSeqLensKVSize;
        constInfo_.seqUsedQSize = qfaBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = qfaBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = qfaBaseParams.scaleValue;
        constInfo_.coreNum = qfaBaseParams.coreNum;
        constInfo_.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;

        // DN 语义（不合轴）：realN2 = n2 * g，realG = 1
        constInfo_.realN2Size = constInfo_.n2Size * constInfo_.gSize;
        constInfo_.realGSize = 1;
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAicNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_END_INDEX, sectionIdx));
    }

    // ============================== 三层任务分派主循环 ==============================
    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoMxfp8SoftmaxFp16 taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // 每 section 重置流水状态，避免跨 section 残留
        uint32_t createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        uint32_t validTaskCount = 0;
        mloop_ = 0;
        headS2Split_ = false;
        tailS2Split_ = false;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx_ = bN2Cur;
        prevGS1Idx_ = gS1Cur;

#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[FA] sec=%u bN2[%u,%u) gS1[%u,%u) s2[%u,%u)\n", sectionIdx, bN2Start_, bN2End_, gS1OStart_,
                        gS1OEnd_, s2OStart_, s2OEnd_);
#endif

        bool shouldDispatchTask = true;
        while (shouldDispatchTask || validTaskCount) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        if (constInfo_.isSoftmaxLseEnable && actSeqLensQ_ > 0) {
                            vectorBlock_.WriteLseNegInf(bN2Cur / constInfo_.realN2Size, bN2Cur % constInfo_.realN2Size,
                                                        static_cast<uint32_t>(actSeqLensQ_));
                        }
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

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo_.realN2Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            actSeqLensKv_ = kvSeqUsedParser_.GetActualSeqLength(bIdx);
            actSeqLensQ_ = qSeqUsedParser_.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv_ + S2_BASE_SIZE - 1) / S2_BASE_SIZE;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + S1_BASE_SIZE - 1) / S1_BASE_SIZE;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur 更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        // S2 有效块区间为 [curS2Start_, curS2End_)，尚未到达且该行有有效块时快进，不跳行
        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块或 s2Cur 已越过有效区间，跳过当前行剩余 S2
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + S2_BASE_SIZE - 1) / S2_BASE_SIZE;

        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split_ = s2OStart_ != 0U;
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split_ = s2OEnd_ != 0U;
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t& bN2Cur, uint32_t& gS1Cur,
                                          uint32_t& s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + S2_BASE_SIZE - 1) / S2_BASE_SIZE;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + S1_BASE_SIZE - 1) / S1_BASE_SIZE;

        // 尚未到达有效区间，快进 s2Cur 到 curS2Start_，不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前 S2 未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        // 当前 BN2 未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前 BN2 已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoMxfp8SoftmaxFp16 taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoMxfp8SoftmaxFp16& runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoMxfp8SoftmaxFp16& info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / constInfo_.realN2Size;
        info.n2Idx = (bN2Cur / (constInfo_.realN2Size / constInfo_.n2Size)) % constInfo_.n2Size;
        info.realN2Idx = bN2Cur % constInfo_.realN2Size;
        info.gS1Idx = gS1Cur * S1_BASE_SIZE;
        // BNSD 为 GS1 layout（S1 与 G 交错）
        info.s1Idx = info.gS1Idx % actSeqLensQ_;
        info.s2Idx = s2Cur * S2_BASE_SIZE;
        info.actS1Size = actSeqLensQ_;
        info.actS2Size = actSeqLensKv_;

        info.actMSize = S1_BASE_SIZE;
        uint64_t gS1Size = info.actS1Size * constInfo_.realGSize;
        if (((gS1Cur + 1) * S1_BASE_SIZE) > gS1Size) {
            info.actMSize = static_cast<uint32_t>(gS1Size - gS1Cur * S1_BASE_SIZE);
        }
        info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;

        info.actSingleLoopS2Size = S2_BASE_SIZE;
        if (((s2Cur + 1) * S2_BASE_SIZE) > info.actS2Size) {
            info.actSingleLoopS2Size = static_cast<uint32_t>(info.actS2Size - s2Cur * S2_BASE_SIZE);
        }
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align(info.actSingleLoopS2Size, static_cast<uint32_t>(BYTE_BLOCK / sizeof(QUANT_T)));

        // S1 切半：V0 处理前半 [0, 128)，V1 处理后半 [128, 256)
        info.vecS1BaseIdx = constInfo_.subBlockIdx * S1_SUB_BLOCK_SIZE;
        if (constInfo_.subBlockIdx == 0) {
            info.actVecS1Size = AttentionCommon::Min(info.actMSize, S1_SUB_BLOCK_SIZE);
        } else {
            info.actVecS1Size = info.actMSize > S1_SUB_BLOCK_SIZE ? info.actMSize - S1_SUB_BLOCK_SIZE : 0;
        }

        // 情况1: loop 不等于 0 时，第一个 S2 inner 循环就是第一个 S2 outer 循环，即 s2Cur=0
        // 情况2/3: loop=0 时，首个有效任务一定是某行 [bN2, gS1] 的首个 S2 分块
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);

        // V1 softmax 在线状态槽位（accMax ring，mloop % 3）
        info.softmaxStateSlot = mloop_ % PRELOAD_TASK_CACHE_SIZE;

        // P L1 ring 槽位（loop % 3，同 [bN2,gS1] 不同 S2 块用不同槽，防覆盖）
        info.pSlot = loop % PRELOAD_TASK_CACHE_SIZE;

#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[Task] loop=%llu b=%u n2=%u gS1=%u s2=%u actS1=%u actS2=%u actM=%u s2Loop=%u "
                        "vecS1Base=%u actVecS1=%u first=%d last=%d slot=%u pSlot=%u\n",
                        static_cast<unsigned long long>(loop), info.bIdx, info.n2Idx, info.gS1Idx, info.s2Idx,
                        static_cast<uint32_t>(info.actS1Size), static_cast<uint32_t>(info.actS2Size), info.actMSize,
                        info.actSingleLoopS2Size, info.vecS1BaseIdx, info.actVecS1Size,
                        static_cast<int32_t>(info.isFirstS2Loop), static_cast<int32_t>(info.isLastS2Loop),
                        info.softmaxStateSlot, info.pSlot);
#endif
    }

    // ============================== ExecuteTask：C1/V1/C2/V2 核间流水 ==============================
    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoMxfp8SoftmaxFp16 taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoMxfp8SoftmaxFp16& runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        RunInfoMxfp8SoftmaxFp16& runInfo2 =
            taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上 PRELOAD_N 轮任务

        // ====== Phase 1: C1/V1（本任务）======
        if (runInfo0.isValid) {
            uint32_t c1v1Loop = CeilDiv(runInfo0.actSingleLoopS2Size, S2_SUB_LOOP_SIZE);
            for (uint32_t subLoopIdx = 0; subLoopIdx < c1v1Loop; subLoopIdx++) {
                for (uint32_t subBlockId = 0; subBlockId < 2; subBlockId++) {
                    if ASCEND_IS_AIC {
                        // mm1Res UB 槽位空闲（对应 V 核已消费完）
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1 + subBlockId * 16);
                        if (subBlockId == 0) {
                            ComputeMm1(runInfo0, subLoopIdx, subLoopIdx + 1 == c1v1Loop);
                        }
                        FixpipeMm1(runInfo0, subLoopIdx, subBlockId); // 单目标搬 [128,128] 给 V(subBlockId)
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1 + subBlockId * 16);
                    } else {
                        if (subBlockId == constInfo_.subBlockIdx) {
                            CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C1_V1);
                            ComputeVec1(runInfo0, subLoopIdx); // 在线 flash softmax，P/pScale 写 AIC L1
                            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C1_V1);
                        }
                    }
                }
            }
            if ASCEND_IS_AIV {
                // P 就绪（MTE3 通道与 P 数据写入 L1 的搬运管道对齐）
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_V1_P_READY);
            }
        }

        // ====== Phase 2: C2/V2（延迟 PRELOAD_N 任务）======
        if (loop >= PRELOAD_N && runInfo2.isValid) {
            for (uint32_t vCore = 0; vCore < 2; vCore++) {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_V1_P_READY + vCore * 16);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + vCore * 16);
                    ComputeMm2(runInfo2, vCore); // V_ext[144,128]×P[128,128] → L0C[144,128]
                    FixpipeMm2(runInfo2, vCore); // 裁 [129,128] 单目标搬给 V(vCore)
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + vCore * 16);
                } else {
                    if (vCore == constInfo_.subBlockIdx) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                        ComputeVec2(runInfo2); // 跨 S2 分块 rescale 累积，末块除+cast+输出
                        if (runInfo2.isLastS2Loop) {
                            CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_C2_V2);
                        } else {
                            CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                        }
                    }
                }
            }
            runInfo2.isValid = false;
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx, bool isLastSubLoop)
    {
        cubeBlock_.ComputeMm1(runInfo, subLoopIdx, isLastSubLoop);
    }

    __aicore__ inline void FixpipeMm1(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx, uint32_t subBlockId)
    {
        cubeBlock_.FixpipeMm1(runInfo, subLoopIdx, subBlockId);
    }

    __aicore__ inline void ComputeVec1(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t subLoopIdx)
    {
#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[V1][AIV%u] loop=%u subLoop=%u slot=%u\n", constInfo_.aivIdx, runInfo.loop, subLoopIdx,
                        runInfo.softmaxStateSlot);
#endif
        vectorBlock_.ComputeVec1(runInfo, subLoopIdx);
    }

    __aicore__ inline void ComputeMm2(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t vCore)
    {
        cubeBlock_.ComputeMm2(runInfo, vCore);
    }

    __aicore__ inline void FixpipeMm2(RunInfoMxfp8SoftmaxFp16& runInfo, uint32_t vCore)
    {
        cubeBlock_.FixpipeMm2(runInfo, vCore);
    }

    __aicore__ inline void ComputeVec2(RunInfoMxfp8SoftmaxFp16& runInfo)
    {
#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[V2][AIV%u] loop=%u slot=%u\n", constInfo_.aivIdx, runInfo.loop, runInfo.softmaxStateSlot);
#endif
        vectorBlock_.ComputeVec2(runInfo);
    }

    // ============================== Process 入口 ==============================
    __aicore__ inline void Process()
    {
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                // 初始释放 mm1Res / mm2Res 槽位，避免首次循环死锁
                // 注意：CROSS_CORE_SYNC_V1_P_READY 不预置——它是「P/pscale 数据就绪」信号（非槽位释放），
                // 预置会让 cube Phase 2 在 V1 实际写入 L1 之前就读 PScale（读全零）。
                // Phase 2 仅在 loop >= PRELOAD_N 时执行，此时对应 Phase 1 已 Set，无需预置。
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C1_V1);
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
            }
        }

        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                if ASCEND_IS_AIC {
                    cubeBlock_.InitTensors();
                } else if ASCEND_IS_AIV {
                    vectorBlock_.InitTensorsVec();
                    if (sectionIdx == 0U) {
                        vectorBlock_.ClearOutput();
                    }
                }
                FlashAttention(sectionIdx);
                if ASCEND_IS_AIC {
                    cubeBlock_.ReleaseTensors();
                } else if ASCEND_IS_AIV {
                    vectorBlock_.ReleaseTensorsVec();
                }
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIC {
                // 收尾 Wait：配平两侧 V 核初始 Set 的 4 个 flag，防止 flag 计数泄漏
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1);
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1 + 16);
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
#if QFA_SF16_DEBUG_PRINTF
                AscendC::printf("[Process][AIC%u] pipeline done\n", constInfo_.aicIdx);
#endif
            }
        }
    }
};

} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_KERNEL_MXFP8_SOFTMAX_FP16_H_

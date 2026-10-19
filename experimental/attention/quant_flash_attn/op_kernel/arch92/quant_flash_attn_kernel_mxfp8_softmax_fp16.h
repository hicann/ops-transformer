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
 * \brief quant_mode=3 (QFA_MXFP8_SOFTMAX_FP16): MXFP8 量化 + softmax FP16 低精度方案 kernel 编排层
 *        仅支持 prefill / BNSD layout / mask_mode=0 / LSE, 不支持 FlashDecode
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_KERNEL_MXFP8_SOFTMAX_FP16_H_

#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_block_cube_mxfp8_softmax_fp16.h"
#include "quant_flash_attn_block_vec_mxfp8_softmax_fp16.h"
#include "utils/memory_copy_arch35.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "quant_flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {

template <typename CubeBlockType, typename VecFaBlockType>
class QuantFlashAttnKernelMxfp8SoftmaxFp16 {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;
    static constexpr uint32_t s2SplitSize = 256U; // 仅支持d=128, 无需按d=256缩小split

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    BufferManager<BufferType::UB> ubBufferManager_;
    BufferManager<BufferType::L1> l1BufferManager_;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers_;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers_;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers_;

    GlobalTensor<int32_t> seqUsedGmQ_;
    GlobalTensor<int32_t> seqUsedGmKv_;
    GlobalTensor<uint32_t> faMetaDataGm_;

    ConstInfoX constInfo_;

    const __gm__ QuantFlashAttnTilingData* __restrict tilingData_;
    TPipe* pipe_ = nullptr;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;

    uint32_t createdTaskCount_ = 0U;

    // schduler params
    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    uint32_t curS2Start_ = 0;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_ = 0;
    uint32_t prevBN2Idx_ = 0;
    uint32_t prevGS1Idx_ = 0;
    uint32_t mloop_ = 0;

    // metadata
    uint32_t sectionNum_;
    uint32_t metadataAicNum_;
    // fa metadata
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser_;

    __aicore__ inline QuantFlashAttnKernelMxfp8SoftmaxFp16()
        : cubeBlock_(constInfo_),
          vecFaBlock_(constInfo_){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* cuSeqLensQ,
                                __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* pScale, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* attnOut, __gm__ uint8_t* workspace, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedKv,
                                const __gm__ QuantFlashAttnTilingData* __restrict tiling, TPipe* tPipe)
    {
        this->pipe_ = tPipe;
        this->tilingData_ = tiling;

        InitConstInfo();
        constInfo_.needInitOutput = ((__gm__ uint32_t*)metadata)[QFA_HEAD_NEED_INIT_OUTPUT_INDEX] != 0;

        seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo_.seqUsedQSize);
        seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo_.seqUsedKvSize);
        sectionNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadataAicNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIC_NUM_INDEX];

        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET),
                                      sectionNum_ * metadataAicNum_ * METADATA_STRIDE);

        InitSeqUsedParsers();

        InitMMResBuf();

        if ASCEND_IS_AIV {
            vecFaBlock_.InitVecBlock(tPipe, sequsedQ, sequsedKv, pScale, attnMask, softmaxLse, attnOut, workspace);
            vecFaBlock_.SetCuSeqLensParser(qSeqUsedParser_);
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock_.InitCubeBlock(tPipe, &l1BufferManager_, query, key, value, blockTable, dequantScaleQuery,
                                     dequantScaleKey, dequantScaleValue, qSeqUsedParser_, kvSeqUsedParser_);
        }
    }

    __aicore__ inline void InitMMResBuf()
    {
        constexpr uint32_t mm1OutDtype = sizeof(half);
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype / 2;
        constexpr uint32_t mm2ResultSize = (dVBaseSize + 32U) * mBaseSize / CV_RATIO * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T) + mBaseSize * s2BaseSize / 32;
        l1BufferManager_.Init(pipe_, 524288);

        l1PBuffers_.Init(l1BufferManager_, mm2LeftSize);
        ubBufferManager_.Init(pipe_, mm1ResultSize * 2 + mm2ResultSize);
        bmm2Buffers_.Init(ubBufferManager_, mm2ResultSize);
        bmm1Buffers_.Init(ubBufferManager_, mm1ResultSize);
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx();
        } else {
            constInfo_.aivIdx = GetBlockIdx();
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }

        const auto& qfaBaseParams = this->tilingData_->baseTiling.quantFlashAttnBaseParams;

        constInfo_.bSize = qfaBaseParams.bSize;
        constInfo_.n2Size = qfaBaseParams.n2Size;
        constInfo_.gSize = qfaBaseParams.gSize;
        constInfo_.s1Size = qfaBaseParams.s1Size;
        constInfo_.s2Size = qfaBaseParams.s2Size;
        constInfo_.dSize = qfaBaseParams.dSize;
        constInfo_.dSizeV = qfaBaseParams.dSizeV;
        constInfo_.realN2Size = constInfo_.n2Size * constInfo_.gSize;
        constInfo_.realGSize = 1;
        constInfo_.seqUsedQSize = qfaBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = qfaBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = static_cast<float>(qfaBaseParams.scaleValue);
        constInfo_.coreNum = qfaBaseParams.coreNum;

        // strides
        constInfo_.keyStrides.bnStride = qfaBaseParams.keyStrides.bnStride;
        constInfo_.keyStrides.n2Stride = qfaBaseParams.keyStrides.n2Stride;
        constInfo_.valueStrides.bnStride = qfaBaseParams.valueStrides.bnStride;
        constInfo_.valueStrides.n2Stride = qfaBaseParams.valueStrides.n2Stride;

        // LSE
        constInfo_.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;
    }

    __aicore__ inline void InitSeqUsedParsers()
    {
        qSeqUsedParser_.Init(seqUsedGmQ_, constInfo_.seqUsedQSize, constInfo_.s1Size);
        kvSeqUsedParser_.Init(seqUsedGmKv_, constInfo_.seqUsedKvSize, constInfo_.s2Size);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAicNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm2Buffers_.Get().SetCrossCore();
        }

        if ASCEND_IS_AIV {
            bmm1Buffers_.Get().SetCrossCore();
            bmm1Buffers_.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers_.Get().WaitCrossCore();
            bmm1Buffers_.Get().WaitCrossCore();
        }
        if ASCEND_IS_AIC {
            bmm2Buffers_.Get().WaitCrossCore();
        }
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        createdTaskCount_ = 0;
        uint32_t executedTaskCount = 0;
        mloop_ = 0;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx_ = bN2Cur;
        prevGS1Idx_ = gS1Cur;

        bool shouldDispatchTask = true;
        uint32_t validTaskCount = 0;
        while (shouldDispatchTask || validTaskCount) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTask(createdTaskCount_, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount_++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        if (constInfo_.isSoftmaxLseEnable && actSeqLensQ_ > 0) {
                            vecFaBlock_.WriteLseNegInf(bN2Cur / constInfo_.realN2Size, bN2Cur % constInfo_.realN2Size,
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
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            CalcCurS2StartEnd(bN2Cur, gS1Cur);
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        // S2有效块区间为[curS2Start_, curS2End_), s2Cur尚未到达有效区间且该行有有效块,
        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start_>=curS2End_)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    // 不支持mask: S2有效块区间为[0, ceil(actSeqLensKv/s2BaseSize)), 仅按section边界裁剪
    __aicore__ inline void CalcCurS2StartEnd(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX& runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX& runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            uint32_t c1v1Loop = CeilDiv(runInfo0.actSingleLoopS2Size, s2SplitSize);
            for (uint32_t subLoop = 0; subLoop < c1v1Loop; ++subLoop) {
                if ASCEND_IS_AIC {
                    ComputeMm1(runInfo0, subLoop);
                } else {
                    ComputeVec1(runInfo0, subLoop);
                }
            }
        }

        if (loop >= PRELOAD_N) {
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

    __aicore__ inline void ComputeMm1(RunInfoX& runInfo, uint32_t subLoop)
    {
        cubeBlock_.IterateBmm1(this->bmm1Buffers_.Get(), runInfo, subLoop);
    }

    __aicore__ inline void ComputeMm2(RunInfoX& runInfo)
    {
        cubeBlock_.IterateBmm2(this->bmm2Buffers_.Get(), this->l1PBuffers_, runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX& runInfo, uint32_t subLoop)
    {
        if (subLoop % 2 == 0) {
            vecFaBlock_.ProcessVec1(this->l1PBuffers_.Get(), this->bmm1Buffers_.Get(), runInfo, subLoop);
        } else {
            vecFaBlock_.ProcessVec1(this->l1PBuffers_.GetPre(), this->bmm1Buffers_.Get(), runInfo, subLoop);
        }
    }

    __aicore__ inline void ComputeVec2(RunInfoX& runInfo)
    {
        this->vecFaBlock_.ProcessVec2(this->bmm2Buffers_.Get(), runInfo);
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX& runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX& info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / (constInfo_.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo_.realN2Size / constInfo_.n2Size)) % constInfo_.n2Size;
        info.realN2Idx = bN2Cur % constInfo_.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ_;
        info.actS2Size = actSeqLensKv_;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo_.realGSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);

        info.actVecMSize = info.actMSize;
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t& bN2Cur, uint32_t& gS1Cur,
                                          uint32_t& s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        // 尚未到达有效区间, 快进s2Cur到curS2Start_, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
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

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_END_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            CrossCoreBufferInit();
            if ASCEND_IS_AIV {
                vecFaBlock_.InitBuffers();
                vecFaBlock_.AllocEventID();
            } else {
                cubeBlock_.InitBuffers();
                cubeBlock_.AllocEventID();
            }
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                FlashAttention(sectionIdx);
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
            } else {
                cubeBlock_.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }
    }
}; // QuantFlashAttnKernelMxfp8SoftmaxFp16

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_KERNEL_MXFP8_SOFTMAX_FP16_H_

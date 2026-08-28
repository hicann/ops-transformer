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
 * \file fia_kernel_noquant_mla.h
 * \brief arch35 FIA 非量化 MLA kernel
 */

#ifndef FIA_KERNEL_NOQUANT_MLA_H_
#define FIA_KERNEL_NOQUANT_MLA_H_

#include "fia_public_define_arch35.h"
#include "fia_block_cube_noquant_mla.h"
#include "fia_block_vec_noquant_mla.h"
#include "memory_copy_arch35_fused_infer.h"
#include "fia_block_vec_flashdecode_mla.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "fia_tiling_data_noquant_gqa.h"

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantMlaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    // BufferManager<BufferType::GM> gmBufferManager_;
    BufferManager<BufferType::UB> ubBufferManager_;
    BufferManager<BufferType::L1> l1BufferManager_;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers_;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers_;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers_;
    // mm1和mm2右矩阵，在L1上复用，其中K_rope内存空间与bmm2的左矩阵p复用
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1KvPBuffers_;

    AscendC::GlobalTensor<uint32_t> fiaMetaDataGm_;
    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    ConstInfoX constInfo_;

    const optiling::NoQuantTilingArch35 *__restrict tilingData_;
    AscendC::TPipe *pipe_ = nullptr;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    // schduler params
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

    ActualSeqLensParser<Q_MODE> qActSeqLensParser_;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser_;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantMlaKernel()
        : cubeBlock_(constInfo_),
          vecFaBlock_(constInfo_),
          vecFdBlock_(constInfo_){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                const optiling::NoQuantTilingArch35 *__restrict tiling, AscendC::TPipe *tPipe)
    {
        this->pipe_ = tPipe;
        this->tilingData_ = tiling;

        AscendC::GlobalTensor<uint64_t> actualSeqLengthsGmQ;
        AscendC::GlobalTensor<uint64_t> actualSeqLengthsGmKv;

        fiaMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t *)fiaMetaData,
                                       NPU_AIC_CORE_NUM * FA_METADATA_SIZE + NPU_AIV_CORE_NUM * FD_METADATA_SIZE);

        InitConstInfo();

        keyPtr_ = key;
        valuePtr_ = value;

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo_.actualSeqLenSize);
        qActSeqLensParser_.Init(actualSeqLengthsGmQ, constInfo_.actualSeqLenSize, constInfo_.s1Size);

        actualSeqLengthsGmKv.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsKv, constInfo_.actualSeqLenKVSize);
        kvActSeqLensParser_.Init(actualSeqLengthsGmKv, constInfo_.actualSeqLenKVSize, constInfo_.s2Size);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            vecFaBlock_.InitVecBlock(tPipe, actualSeqLengths, actualSeqLengthsKv, attenMask, softmaxLse, attentionOut,
                                     workspace);
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock_.InitCubeBlock(tPipe, &l1BufferManager_, query, key, value, blockTable, queryRope, keyRope,
                                     actualSeqLengths, actualSeqLengthsKv);
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock_.InitParams();
                vecFdBlock_.InitGlobalTensor(this->vecFaBlock_.softmaxFDMaxGm, this->vecFaBlock_.softmaxFDSumGm,
                                             this->vecFaBlock_.accumOutGm, this->vecFaBlock_.attentionOutGm,
                                             actualSeqLengthsGmQ, actualSeqLengthsGmKv, keyPtr_);
                if (unlikely(constInfo_.isSoftmaxLseEnable)) {
                    AscendC::GlobalTensor<float> softmaxLseGm;
                    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vecFdBlock_.InitSoftmaxLseGm(softmaxLseGm);
                }
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        uint32_t mm12RightSize = s2BaseSize * 576 * sizeof(INPUT_T);
        l1BufferManager_.Init(pipe_, 524288); // 512 * 1024
        // 3Buffer
        l1KvPBuffers_.Init(l1BufferManager_, mm12RightSize); // L1: 144k * 3 = 432k
        ubBufferManager_.Init(pipe_, mm1ResultSize * 2 + mm2ResultSize * 2);
        bmm2Buffers_.Init(ubBufferManager_, mm2ResultSize);
        bmm1Buffers_.Init(ubBufferManager_, mm1ResultSize);
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

        const auto &fiaBaseParams = this->tilingData_->fiaBaseParams;
        const auto &fiaAttenMaskParams = this->tilingData_->fiaAttenMaskParams;
        const auto &fiaPageAttentionParams = this->tilingData_->fiaPageAttentionParams;
        const auto &fiaWorkspaceParams = this->tilingData_->fiaWorkspaceParams;
        const auto &fiaEmptyTensorParams = this->tilingData_->fiaEmptyTensorParams;
        // 清零开关: 短kv/空kv/短q 等整行无任务写回场景由 host 置1, vecFaBlock_.ClearOutput() 消费
        constInfo_.needInit = fiaEmptyTensorParams.needInit;

        constInfo_.bSize = fiaBaseParams.bSize;
        constInfo_.t1Size = fiaBaseParams.t1Size;
        constInfo_.t2Size = fiaBaseParams.t2Size;
        constInfo_.n2Size = fiaBaseParams.n2Size;
        constInfo_.gSize = fiaBaseParams.gSize;
        constInfo_.s1Size = fiaBaseParams.s1Size;
        constInfo_.s2Size = fiaBaseParams.s2Size;
        constInfo_.dSize = fiaBaseParams.dSize;
        constInfo_.dSizeV = fiaBaseParams.dSizeV;
        constInfo_.dSizeRope = 64; // todo 修改成传入值
        constInfo_.actualSeqLenSize = fiaBaseParams.actualSeqLengthsQSize;
        constInfo_.actualSeqLenKVSize = fiaBaseParams.actualSeqLengthsKVSize;
        constInfo_.scaleValue = fiaBaseParams.scaleValue;
        constInfo_.l2CacheOffFlag = fiaBaseParams.l2CacheOffFlag;
        constInfo_.coreNum = fiaBaseParams.coreNum;
        constInfo_.outputLayout = static_cast<FIA_LAYOUT>(fiaBaseParams.outputLayout);

        constInfo_.keyStrides.bnStride = fiaBaseParams.keyStrides.bnStride;
        constInfo_.keyStrides.n2Stride = fiaBaseParams.keyStrides.n2Stride;
        constInfo_.valueStrides.bnStride = fiaBaseParams.valueStrides.bnStride;
        constInfo_.valueStrides.n2Stride = fiaBaseParams.valueStrides.n2Stride;

        constInfo_.sparseMode = fiaAttenMaskParams.sparseMode;
        constInfo_.preTokens = fiaAttenMaskParams.preTokens;
        constInfo_.nextTokens = fiaAttenMaskParams.nextTokens;
        if constexpr (HAS_MASK) {
            constInfo_.attenMaskBatch = fiaAttenMaskParams.attenMaskBatch;
            constInfo_.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
            constInfo_.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;
        }
        constInfo_.isRowInvalidOpen = fiaAttenMaskParams.isRowInvalidOpen;
        constInfo_.isExistRowInvalid = fiaAttenMaskParams.isExistRowInvalid;
        constInfo_.accumOutSize = fiaWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = fiaWorkspaceParams.logSumExpSize;
        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo_.maxBlockNumPerBatch = fiaPageAttentionParams.maxBlockNumPerBatch;
            constInfo_.blockSize = fiaPageAttentionParams.blockSize;
            constInfo_.paLayoutType = fiaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo_.isSoftmaxLseEnable = fiaBaseParams.isSoftMaxLseEnable;

        constInfo_.bN2Start = fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_BN2_START_INDEX));
        constInfo_.gS1OStart = fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_M_START_INDEX));
        constInfo_.s2OStart = fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_S2_START_INDEX));
        constInfo_.bN2End = fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_BN2_END_INDEX));
        constInfo_.gS1OEnd = fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_M_END_INDEX));
        constInfo_.s2OEnd = fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_S2_END_INDEX));
        constInfo_.coreFirstTmpOutWsPos =
            fiaMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        constInfo_.dBasicBlock = AttentionCommon::Align(constInfo_.dSizeV, 64U);
        constInfo_.kRopeStrides.bnStride = fiaBaseParams.kRopeStrides.bnStride;
        constInfo_.kRopeStrides.n2Stride = fiaBaseParams.kRopeStrides.n2Stride;
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * NPU_AIC_CORE_NUM + FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm1Buffers_.Get().SetCrossCore();
            bmm1Buffers_.Get().SetCrossCore();
            bmm2Buffers_.Get().SetCrossCore();
            bmm2Buffers_.Get().SetCrossCore();
        } else {
            l1KvPBuffers_.Get().SetEventID();
            l1KvPBuffers_.Get().SetEventID();
            l1KvPBuffers_.Get().SetEventID();
            SetFlag<HardEvent::MTE1_MTE2>(l1KvPBuffers_.Get().GetEventID<HardEvent::MTE1_MTE2>());
            SetFlag<HardEvent::MTE1_MTE2>(l1KvPBuffers_.Get().GetEventID<HardEvent::MTE1_MTE2>());
            SetFlag<HardEvent::MTE1_MTE2>(l1KvPBuffers_.Get().GetEventID<HardEvent::MTE1_MTE2>());
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(l1KvPBuffers_.Get().GetEventID<HardEvent::MTE1_MTE2>());
            WaitFlag<HardEvent::MTE1_MTE2>(l1KvPBuffers_.Get().GetEventID<HardEvent::MTE1_MTE2>());
            WaitFlag<HardEvent::MTE1_MTE2>(l1KvPBuffers_.Get().GetEventID<HardEvent::MTE1_MTE2>());

            bmm1Buffers_.Get().WaitCrossCore();
            bmm1Buffers_.Get().WaitCrossCore();
            bmm2Buffers_.Get().WaitCrossCore();
            bmm2Buffers_.Get().WaitCrossCore();
        }
    }

    __aicore__ inline void FlashAttention()
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        RunInfoX taskRunInfo[PRELOAD_N] = {};
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
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        vecFaBlock_.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
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

    __aicore__ inline bool ShouldExecuteTask(RunInfoX taskRunInfo[PRELOAD_N])
    {
        return validTaskNum_ > 0;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart) && (s2Cur == constInfo_.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo_.n2Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            actSeqLensKv_ = kvActSeqLensParser_.GetActualSeqLength(bIdx);
            actSeqLensQ_ = qActSeqLensParser_.GetActualSeqLength(bIdx);
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

        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
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

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ_, int64_t actSeqLensKv_, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp += static_cast<int64_t>(actSeqLensQ_) - static_cast<int64_t>(actSeqLensKv_);
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp += static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
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
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
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
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ_);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ_);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
                s1FirstToken = 0;
            }
        }
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);
        // 3. trans index of token to index of block
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));

        // 4. Calc curS2Start_, curS2End_
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

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_N])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_N];                        // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - (PRELOAD_N - 1)) % PRELOAD_N]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
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

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm1(this->bmm1Buffers_.Get(), this->l1KvPBuffers_.Get(), runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm2(this->bmm2Buffers_.Get(), this->l1KvPBuffers_.GetReused(), runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        // c v都会独立get，所以mm1和vec1能一致
        vecFaBlock_.ProcessVec1(this->l1KvPBuffers_.Get(), this->bmm1Buffers_.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        this->vecFaBlock_.ProcessVec2(this->bmm2Buffers_.Get(), runInfo);
    }

    __aicore__ inline void EnableTask(RunInfoX &runInfo)
    {
        runInfo.isValid = true;
        validTaskNum_++;
    }

    __aicore__ inline void DisableTask(RunInfoX &runInfo)
    {
        runInfo.isValid = false;
        validTaskNum_--;
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_N])
    {
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_N]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        EnableTask(runInfo);
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / constInfo_.n2Size;
        info.n2Idx = bN2Cur % constInfo_.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo_.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ_;
        }
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
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));
        info.isChangeBatch = false;

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo_.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);
        info.actVecMSize = (info.actMSize + 1) >> 1;
        info.vecMbaseIdx = 0;
        if (constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((constInfo_.bN2Start == constInfo_.bN2End && constInfo_.gS1OStart == constInfo_.gS1OEnd)) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == constInfo_.bN2End) && (gS1Cur == constInfo_.gS1OEnd)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        } else if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
            s2Cur++;
            return;
        }

        // 当前BN2未处理完
        s2Cur = 0;

        uint64_t gS1LoopTimes = cachedG1S1LoopTimes_;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode()
    {
        vecFdBlock_.InitBuffers(this->pipe_);
        AscendC::ICachePreLoad(2);
        uint32_t fdCoreEnable = fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_CORE_ENABLE_INDEX));
        uint32_t fdBN2Idx = fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_BN2_IDX_INDEX));
        uint32_t fdMIdx = fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_M_IDX_INDEX));
        uint32_t fdS2SplitNum = fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_S2_SPLIT_NUM_INDEX));
        uint32_t mStart = fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_M_START_INDEX));
        uint32_t mLen = fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_M_NUM_INDEX));
        uint32_t fdWorkspaceIdx =
            fiaMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FD_WORKSPACE_IDX_INDEX));

        FDparamsX fdParams = {fdCoreEnable, fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, fdWorkspaceIdx};
        vecFdBlock_.AllocEventID();
        AscendC::SyncAll();
        vecFdBlock_.FlashDecode(fdParams);
        vecFdBlock_.FreeEventID();
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
            FlashAttention();

            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
            } else {
                cubeBlock_.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                FlashDecode();
            }
        }
    }
}; // FlashAttentionNoQuantMlaKernel

} // namespace BaseApi

#endif // FIA_KERNEL_NOQUANT_MLA_H_

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
 * \file quant_flash_attn_kernel_hif8.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_HIF8_H_
#define QUANT_FLASH_ATTN_KERNEL_HIF8_H_

#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_block_cube_hif8.h"
#include "quant_flash_attn_block_vec_hif8.h"
#include "quant_flash_attn_block_vec_flashdecode.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#if __has_include("../../../common/op_kernel/vector_common.h")
#include "../../../common/op_kernel/vector_common.h"
#else
#include "../../common/op_kernel/vector_common.h"
#endif

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
__aicore__ inline int64_t ClipSInnerTokenHif8(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class QuantFlashAttnKernelHif8 {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // 2 : C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

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
    BufferManager<BufferType::GM> gmBufferManager_;
    BufferManager<BufferType::UB> ubBufferManager_;
    BufferManager<BufferType::L1> l1BufferManager_;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers_;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers_;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers_;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers_;

    GlobalTensor<int32_t> cuSeqLensGmQ_;
    GlobalTensor<int32_t> cuSeqLensGmKv_;
    GlobalTensor<int32_t> seqUsedGmQ_;
    GlobalTensor<int32_t> seqUsedGmKv_;
    GlobalTensor<uint32_t> faMetaDataGm_;
    GlobalTensor<uint32_t> fdMetaDataGm_;
    GlobalTensor<float> softmaxLseGm_;
    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    ConstInfoX constInfo_;

    const __gm__ QuantFlashAttnTilingData *__restrict tilingData_;
    TPipe *pipe_ = nullptr;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

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
    bool headS2Split_ = false;
    bool tailS2Split_ = false;

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

    typename std::conditional<(LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                              ActualSeqLensParser<Q_MODE, int32_t>>::type qCuSeqLensParser_;

    typename std::conditional<(!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE, int32_t>>::type
        kvCuSeqLensParser_;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser_;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser_;

    // ==============================fuction=======================================================
    __aicore__ inline QuantFlashAttnKernelHif8()
        : cubeBlock_(constInfo_),
          vecFaBlock_(constInfo_),
          vecFdBlock_(constInfo_){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *cuSeqLensQ,
                                __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attnOut, __gm__ uint8_t *workspace, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
                                const __gm__ QuantFlashAttnTilingData *__restrict tiling, TPipe *tPipe)
    {
        this->pipe_ = tPipe;
        this->tilingData_ = tiling;

        InitConstInfo();

        keyPtr_ = key;
        valuePtr_ = value;

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ_.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfo_.cuSeqLensQSize);
            seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t *)sequsedQ, constInfo_.seqUsedQSize);
        } else {
            seqUsedGmQ_.SetGlobalBuffer((__gm__ int32_t *)sequsedQ, constInfo_.seqUsedQSize);
        }
        if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmKv_.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfo_.cuSeqLensKVSize);
            seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t *)sequsedKv, constInfo_.seqUsedKvSize);
        } else {
            seqUsedGmKv_.SetGlobalBuffer((__gm__ int32_t *)sequsedKv, constInfo_.seqUsedKvSize);
        }
        sectionNum_ = ((__gm__ uint32_t *)metadata)[0];

        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t *)(metadata + FA_METADATA_HEADER_OFFSET),
                                      QFA_AIC_CORE_NUM * 16U * sectionNum_);

        InitQCuSeqLensParser(cuSeqLensQ, sequsedQ);
        InitKvCuSeqLensParser(cuSeqLensKv, sequsedKv);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vecFaBlock_.InitVecBlock(tPipe, cuSeqLensQ, cuSeqLensKv, dequantScaleQuery, dequantScaleKey,
                                         dequantScaleValue, pScale, attnMask, softmaxLse, attnOut, workspace);
                vecFaBlock_.SetCuSeqLensParser(qCuSeqLensParser_);
            } else {
                vecFaBlock_.InitVecBlock(tPipe, sequsedQ, sequsedKv, dequantScaleQuery, dequantScaleKey,
                                         dequantScaleValue, pScale, attnMask, softmaxLse, attnOut, workspace);
                vecFaBlock_.SetCuSeqLensParser(qSeqUsedParser_);
            }
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cubeBlock_.InitCubeBlock(tPipe, &l1BufferManager_, query, key, value, blockTable, dequantScaleQuery,
                                         dequantScaleKey, dequantScaleValue, qCuSeqLensParser_, kvCuSeqLensParser_);
            } else {
                cubeBlock_.InitCubeBlock(tPipe, &l1BufferManager_, query, key, value, blockTable, dequantScaleQuery,
                                         dequantScaleKey, dequantScaleValue, qSeqUsedParser_, kvSeqUsedParser_);
            }
        }
        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                fdMetaDataGm_.SetGlobalBuffer(
                    (__gm__ uint32_t *)(metadata + FA_METADATA_HEADER_OFFSET +
                                        QFA_METADATA_SIZE * QFA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
                    QFA_AIV_CORE_NUM * 16U * sectionNum_);
                vecFdBlock_.InitParams();
                vecFdBlock_.InitGlobalTensor(this->vecFaBlock_.softmaxFDMaxGm_, this->vecFaBlock_.softmaxFDSumGm_,
                                             this->vecFaBlock_.accumOutGm_, this->vecFaBlock_.attentionOutGm_, keyPtr_);
                if (constInfo_.isSoftmaxLseEnable) {
                    softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vecFdBlock_.InitSoftmaxLseGm(softmaxLseGm_);
                }
                if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    vecFdBlock_.SetCuSeqLensParsers(qCuSeqLensParser_, kvCuSeqLensParser_);
                } else {
                    vecFdBlock_.SetCuSeqLensParsers(qSeqUsedParser_, kvSeqUsedParser_);
                }
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);

        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager_.Init(pipe_, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
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

        const auto &qfaBaseParams = this->tilingData_->baseTiling.quantFlashAttnBaseParams;
        const auto &qfaAttenMaskParams = this->tilingData_->baseTiling.quantFlashAttnAttenMaskParams;
        const auto &qfaPageAttentionParams = this->tilingData_->baseTiling.quantFlashAttnPageAttentionParams;
        const auto &qfaWorkspaceParams = this->tilingData_->baseTiling.quantFlashAttnWorkspaceParams;

        constInfo_.bSize = qfaBaseParams.bSize;
        constInfo_.t1Size = qfaBaseParams.t1Size;
        constInfo_.t2Size = qfaBaseParams.t2Size;
        constInfo_.n2Size = qfaBaseParams.n2Size;
        constInfo_.gSize = qfaBaseParams.gSize;
        constInfo_.s1Size = qfaBaseParams.s1Size;
        constInfo_.s2Size = qfaBaseParams.s2Size;
        constInfo_.dSize = qfaBaseParams.dSize;
        constInfo_.dSizeV = qfaBaseParams.dSizeV;
        if constexpr (USE_DN) { // prefill不合轴
            constInfo_.realN2Size = constInfo_.n2Size * constInfo_.gSize;
            constInfo_.realGSize = 1;
        } else { // decode合轴
            constInfo_.realN2Size = constInfo_.n2Size;
            constInfo_.realGSize = constInfo_.gSize;
        }
        constInfo_.cuSeqLensQSize = qfaBaseParams.cuSeqLensQSize;
        constInfo_.cuSeqLensKVSize = qfaBaseParams.cuSeqLensKVSize;
        constInfo_.seqUsedQSize = qfaBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = qfaBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = static_cast<float>(qfaBaseParams.scaleValue);
        constInfo_.isKvContinuous = true;
        constInfo_.coreNum = qfaBaseParams.coreNum;
        constInfo_.needInitOutput = qfaBaseParams.needInitOutput;
        constInfo_.outputLayout = static_cast<FA_LAYOUT>(qfaBaseParams.outputLayout);
        constInfo_.sparseMode = qfaAttenMaskParams.sparseMode;
        constInfo_.preTokens = qfaAttenMaskParams.winLefts;
        constInfo_.nextTokens = qfaAttenMaskParams.winRights;
        constInfo_.attenMaskBatch = qfaAttenMaskParams.attenMaskBatch;
        constInfo_.attenMaskS1Size = qfaAttenMaskParams.attenMaskS1Size;
        constInfo_.attenMaskS2Size = qfaAttenMaskParams.attenMaskS2Size;

        constInfo_.accumOutSize = qfaWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = qfaWorkspaceParams.logSumExpSize;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo_.maxBlockNumPerBatch = qfaPageAttentionParams.maxBlockNumPerBatch;
            constInfo_.blockSize = qfaPageAttentionParams.blockSize;
            constInfo_.paLayoutType = qfaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo_.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;

        constInfo_.dBasicBlock = Align64Func((uint16_t)constInfo_.dSizeV);
    }

    __aicore__ inline void InitQCuSeqLensParser(__gm__ uint8_t *cuSeqLensQPtr, __gm__ uint8_t *sequsedQPtr)
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser_.Init(cuSeqLensQPtr, constInfo_.cuSeqLensQSize, sequsedQPtr, constInfo_.seqUsedQSize);
        } else {
            qSeqUsedParser_.Init(seqUsedGmQ_, constInfo_.seqUsedQSize, constInfo_.s1Size);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser(__gm__ uint8_t *cuSeqLensKvPtr, __gm__ uint8_t *sequsedKvPtr)
    {
        if constexpr (!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser_.Init(cuSeqLensKvPtr, constInfo_.cuSeqLensKVSize, sequsedKvPtr, constInfo_.seqUsedKvSize);
        } else if constexpr (PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser_.Init(sequsedKvPtr, constInfo_.seqUsedKvSize, constInfo_.s2Size);
        } else {
            kvSeqUsedParser_.Init(seqUsedGmKv_, constInfo_.seqUsedKvSize, constInfo_.s2Size);
        }
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return QFA_METADATA_SIZE * QFA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return QFA_FD_METADATA_SIZE * QFA_AIV_CORE_NUM * sectionIdx + QFA_FD_METADATA_SIZE * coreIdx + metaIdx;
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
        headS2Split_ = false;
        tailS2Split_ = false;

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
            if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensKv_ = kvCuSeqLensParser_.GetActualSeqLength(bIdx);
            } else {
                actSeqLensKv_ = kvSeqUsedParser_.GetActualSeqLength(bIdx);
            }
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensQ_ = qCuSeqLensParser_.GetActualSeqLength(bIdx);
            } else {
                actSeqLensQ_ = qSeqUsedParser_.GetActualSeqLength(bIdx);
            }
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
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        // S2有效块区间为[curS2Start_, curS2End_), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start_继续计算, 不能跳行 (BAND等sparse模式curS2Start_常>0)
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

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL ||
            constInfo_.sparseMode == fa_base_vector::TREE) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split_ = s2OStart_ != 0U;
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split_ = s2OEnd_ != 0U;
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ_) * static_cast<int64_t>(constInfo_.realGSize)) -
            1;

        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo_.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo_.realGSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ_) ==
                s1GLastToken / static_cast<int64_t>(actSeqLensQ_)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ_);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ_);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
            }
        }

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerTokenHif8(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipSInnerTokenHif8(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start_, curS2End_
        curS2Start_ = s2StartWithSparse;
        curS2End_ = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            headS2Split_ = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start_ = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            tailS2Split_ = s2OEnd_ > 0U ? true : false;
            curS2End_ = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
                ComputeVec1(runInfo0);
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

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm1(this->bmm1Buffers_.Get(), runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm2(this->bmm2Buffers_.Get(), this->l1PBuffers_, runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        vecFaBlock_.ProcessVec1(this->l1PBuffers_.Get(), this->bmm1Buffers_.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        this->vecFaBlock_.ProcessVec2(this->bmm2Buffers_.Get(), runInfo);
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / (constInfo_.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo_.realN2Size / constInfo_.n2Size)) % constInfo_.n2Size;
        info.realN2Idx = bN2Cur % constInfo_.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo_.realGSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ_;
        }
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
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(BYTE_BLOCK / sizeof(INPUT_T)));

        info.isChangeBatch = false;

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);

        if constexpr (USE_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = (info.actMSize + 1) >> 1;
        }
        info.vecMbaseIdx = 0;
        if (constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = USE_DN ? info.actVecMSize : (info.actMSizeAlign32 >> 1);
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_)) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
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

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        GetFDSectionInfo(sectionIdx);
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        vecFdBlock_.InitBuffers(this->pipe_);
        AscendC::ICachePreLoad(2);
        vecFdBlock_.AllocEventID();
        SyncAll();
        vecFdBlock_.FlashDecode(fdParams_);
        vecFdBlock_.FreeEventID();
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QFA_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, QFA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QFA_FD_M_START_INDEX, sectionIdx));
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
            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(sectionIdx);
                }
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
}; // QuantFlashAttnKernelHif8

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_KERNEL_HIF8_H_

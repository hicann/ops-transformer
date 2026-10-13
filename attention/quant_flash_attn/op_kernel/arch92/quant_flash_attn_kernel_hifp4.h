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
 * \file quant_flash_attn_kernel_hifp4.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_HIFP4_H_
#define QUANT_FLASH_ATTN_KERNEL_HIFP4_H_

#include "quant_flash_attn_common_def_arch92.h"
#include "quant_flash_attn_block_cube_hifp4.h"
#include "quant_flash_attn_block_vec_hifp4.h"
#include "memory_copy_arch92_quant_flash_attn.h"
#include "quant_flash_attn_block_vec_flashdecode_arch92.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "quant_flash_attn_tiling_data_arch92.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class QuantFlashAttnKernelHifp4 {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 1; // C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = half;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;

    ConstInfoX constInfo;

    const __gm__ QuantFlashAttnTilingData* __restrict tilingData;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    uint32_t createdTaskCount = 0U;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    // metadata
    uint32_t sectionNum_;
    uint32_t metadataAicNum_;
    uint32_t metadataAivNum_;
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
                              ActualSeqLensParser<Q_MODE, int32_t>>::type qCuSeqLensParser;

    typename std::conditional<(!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE, int32_t>>::type
        kvCuSeqLensParser;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser;

    // ==============================fuction=======================================================
    __aicore__ inline QuantFlashAttnKernelHifp4()
        : cubeBlock(constInfo),
          vecFaBlock(constInfo),
          vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* cuSeqLensQ,
                                __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* pScale, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* attnOut, __gm__ uint8_t* workspace, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedKv,
                                const __gm__ QuantFlashAttnTilingData* __restrict tiling)
    {
        this->tilingData = tiling;
        InitConstInfo();
        keyPtr = key;
        valuePtr = value;

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensQ, constInfo.cuSeqLensQSize + 1);
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo.seqUsedQSize);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t*)sequsedQ, constInfo.seqUsedQSize);
        }
        if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensKv, constInfo.cuSeqLensKVSize + 1);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo.seqUsedKvSize);
        } else {
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t*)sequsedKv, constInfo.seqUsedKvSize);
        }
        sectionNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadataAicNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIC_NUM_INDEX];
        metadataAivNum_ = ((__gm__ uint32_t*)metadata)[METADATA_HEADER_AIV_NUM_INDEX];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET),
                                     metadataAicNum_ * 16U * sectionNum_);

        InitQCuSeqLensParser(cuSeqLensQ, sequsedQ);
        InitKvCuSeqLensParser(cuSeqLensKv, sequsedKv);

        if ASCEND_IS_AIV { // 初始化Tbuf
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vecFaBlock.InitInput(sequsedQ, sequsedKv, pScale, attnMask, softmaxLse, attnOut, workspace);
                vecFaBlock.SetCuSeqLensParser(qCuSeqLensParser);
            } else {
                vecFaBlock.InitInput(sequsedQ, sequsedKv, pScale, attnMask, softmaxLse, attnOut, workspace);
                vecFaBlock.SetCuSeqLensParser(qSeqUsedParser);
            }
        }

        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cubeBlock.InitInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue, qCuSeqLensParser, kvCuSeqLensParser);
            } else {
                cubeBlock.InitInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue, qSeqUsedParser, kvSeqUsedParser);
            }
        }
        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                fdMetaDataGm.SetGlobalBuffer(
                    (__gm__ uint32_t*)(metadata + METADATA_HEADER_OFFSET +
                                       METADATA_STRIDE * metadataAicNum_ * sectionNum_ * sizeof(uint32_t)),
                    metadataAivNum_ * 16U * sectionNum_);
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm, keyPtr);
                if (constInfo.isSoftmaxLseEnable) {
                    softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
                    vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
                }
                if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    vecFdBlock.SetCuSeqLensParsers(qCuSeqLensParser, kvCuSeqLensParser);
                } else {
                    vecFdBlock.SetCuSeqLensParsers(qSeqUsedParser, kvSeqUsedParser);
                }
            }
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx();
            constInfo.subBlockIdx = 0;
        }

        const auto& qfaBaseParams = this->tilingData->baseTiling.quantFlashAttnBaseParams;
        const auto& qfaAttenMaskParams = this->tilingData->baseTiling.quantFlashAttnAttenMaskParams;
        const auto& qfaPageAttentionParams = this->tilingData->baseTiling.quantFlashAttnPageAttentionParams;
        const auto& qfaWorkspaceParams = this->tilingData->baseTiling.quantFlashAttnWorkspaceParams;

        constInfo.bSize = qfaBaseParams.bSize;
        constInfo.t1Size = qfaBaseParams.t1Size;
        constInfo.t2Size = qfaBaseParams.t2Size;
        constInfo.n2Size = qfaBaseParams.n2Size;
        constInfo.gSize = qfaBaseParams.gSize;
        constInfo.s1Size = qfaBaseParams.s1Size;
        constInfo.s2Size = qfaBaseParams.s2Size;
        constInfo.dSize = qfaBaseParams.dSize;
        constInfo.dSizeV = qfaBaseParams.dSizeV;
        if constexpr (USE_DN) { // prefill不合轴
            constInfo.realN2Size = constInfo.n2Size * constInfo.gSize;
            constInfo.realGSize = 1;
        } else { // decode合轴
            constInfo.realN2Size = constInfo.n2Size;
            constInfo.realGSize = constInfo.gSize;
        }
        constInfo.cuSeqLensQSize = qfaBaseParams.cuSeqLensQSize;
        constInfo.cuSeqLensKVSize = qfaBaseParams.cuSeqLensKVSize;
        constInfo.seqUsedQSize = qfaBaseParams.seqUsedQSize;
        constInfo.seqUsedKvSize = qfaBaseParams.seqUsedKvSize;
        constInfo.maxSeqlenQ =
            (qfaBaseParams.maxSeqlenQ >= 0) ? static_cast<uint64_t>(qfaBaseParams.maxSeqlenQ) : constInfo.s1Size;
        constInfo.maxSeqlenKv =
            (qfaBaseParams.maxSeqlenKv >= 0) ? static_cast<uint64_t>(qfaBaseParams.maxSeqlenKv) : constInfo.s2Size;
        constInfo.scaleValue = static_cast<float>(qfaBaseParams.scaleValue);
        constInfo.coreNum = qfaBaseParams.coreNum;
        constInfo.needInitOutput = qfaBaseParams.needInitOutput;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(qfaBaseParams.outputLayout);
        constInfo.sparseMode = qfaAttenMaskParams.sparseMode;
        constInfo.preTokens = qfaAttenMaskParams.winLefts;
        constInfo.nextTokens = qfaAttenMaskParams.winRights;
        constInfo.attenMaskBatch = qfaAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = qfaAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = qfaAttenMaskParams.attenMaskS2Size;

        constInfo.accumOutSize = qfaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = qfaWorkspaceParams.logSumExpSize;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = qfaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = qfaPageAttentionParams.blockSize;
            constInfo.paLayoutType = qfaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline void InitQCuSeqLensParser(__gm__ uint8_t* cuSeqLensQPtr, __gm__ uint8_t* sequsedQPtr)
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser.Init(cuSeqLensQPtr, constInfo.cuSeqLensQSize + 1, sequsedQPtr, constInfo.seqUsedQSize);
        } else {
            qSeqUsedParser.Init(seqUsedGmQ, constInfo.seqUsedQSize, constInfo.maxSeqlenQ);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser(__gm__ uint8_t* cuSeqLensKvPtr, __gm__ uint8_t* sequsedKvPtr)
    {
        if constexpr (!PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser.Init(cuSeqLensKvPtr, constInfo.cuSeqLensKVSize + 1, sequsedKvPtr,
                                   constInfo.seqUsedKvSize);
        } else if constexpr (PAGE_ATTENTION && LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser.Init(sequsedKvPtr, constInfo.seqUsedKvSize, constInfo.s2Size);
        } else {
            kvSeqUsedParser.Init(seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.maxSeqlenKv);
        }
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAicNum_ * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadataAivNum_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        createdTaskCount = 0;
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
        uint32_t validTaskCount = 0;
        while (shouldDispatchTask || validTaskCount) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
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
        uint32_t bIdx = bN2Cur / constInfo.realN2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensKv = kvCuSeqLensParser.GetActualSeqLength(bIdx);
            } else {
                actSeqLensKv = kvSeqUsedParser.GetActualSeqLength(bIdx);
            }
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                actSeqLensQ = qCuSeqLensParser.GetActualSeqLength(bIdx);
            } else {
                actSeqLensQ = qSeqUsedParser.GetActualSeqLength(bIdx);
            }
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
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

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t& preTokenLeftUp,
                                                 int64_t& nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo.preTokens;
        nextTokenLeftUp = constInfo.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo.sparseMode);

        if (constInfo.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL || constInfo.sparseMode == fa_base_vector::TREE) {
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
                                 static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.realGSize)) -
            1;

        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.realGSize);
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
        RunInfoX& runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX& runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            uint32_t c1v1Loop = CeilDiv(runInfo0.actSingleLoopS2Size, 128);
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
        cubeBlock.IterateBmm1(runInfo, subLoop);
    }

    __aicore__ inline void ComputeMm2(RunInfoX& runInfo)
    {
        cubeBlock.IterateBmm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX& runInfo, uint32_t subLoop)
    {
        vecFaBlock.ProcessVec1(runInfo, subLoop);
    }

    __aicore__ inline void ComputeVec2(RunInfoX& runInfo)
    {
        vecFaBlock.ProcessVec2(runInfo);
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
        info.mloop = mloop;
        info.bIdx = bN2Cur / (constInfo.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo.realN2Size / constInfo.n2Size)) % constInfo.n2Size;
        info.realN2Idx = bN2Cur % constInfo.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.realGSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.realGSize;
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

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        if constexpr (USE_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5; // 31: 对齐到32, 5: 2^5=32
            info.actVecMSize = info.actMSizeAlign32;
        } else {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5; // 31: 对齐到32, 5: 2^5=32
            info.actVecMSize = info.actMSize;
        }
        info.vecMbaseIdx = 0;

        if ((bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_)) {
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

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t& bN2Cur, uint32_t& gS1Cur,
                                          uint32_t& s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
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
        GetFDSectionInfo(sectionIdx);
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        AscendC::ICachePreLoad(2); // 2: PreLoad number
        vecFdBlock.AllocEventID();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams_);
        vecFdBlock.FreeEventID();
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, QFA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_M_START_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx < constInfo.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock.InitTensors();
            } else {
                cubeBlock.InitTensors();
                cubeBlock.InitConstTensor();
            }
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                FlashAttention(sectionIdx);
            }
        }

        if (constInfo.aicIdx < constInfo.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock.ReleaseTensors();
            } else {
                cubeBlock.ReleaseTensors();
            }
        }
    }
}; // FlashAttentionFullQuantMxKernel

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_KERNEL_HIFP4_H_

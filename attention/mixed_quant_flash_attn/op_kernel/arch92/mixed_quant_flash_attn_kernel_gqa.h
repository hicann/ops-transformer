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
 * \file flash_attention_noquant_kernel_base.h
 * \brief
 */

#ifndef FLASH_ATTENTION_ANTIQUANT_GQA_KERNEL_H_
#define FLASH_ATTENTION_ANTIQUANT_GQA_KERNEL_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "mqfa_public_define_arch92.h"
#include "mixed_quant_flash_attn_block_cube_gqa.h"
#include "mixed_quant_flash_attn_block_vec_gqa.h"
#include "mixed_quant_flash_attn_block_vec_flashdecode.h"
#include "memory_copy_arch35_mixed_quant_flash_attn.h"
#include "../../../common/op_kernel/const_def.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionAntiQuantGqaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;
    static constexpr uint8_t QUANT_COMPUTE_MODE = CubeBlockType::QUANT_COMPUTE_MODE;
    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT_Q;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT_Q; // todo 整改
    using Q_T = typename CubeBlockType::Q_T;
    using MM_T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;

    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;
    /* ===== 核间共享 buffer [静态 LocalTensor]===== */
    LocalTensor<uint8_t> l1PBuffers[CubeBlockType::L1_P_BUF_NUM]; // 2 buf, 按bank布局
    LocalTensor<uint8_t> l1QBuffers[CubeBlockType::L1_Q_BUF_NUM]; // 2 buf, 按bank布局
    LocalTensor<uint8_t> ubBmm1Buffers;                           // BMM1 result buf
    LocalTensor<uint8_t> ubBmm2Buffers;                           // BMM2 result buf

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<Q_T> sinkGm;

    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;

    ConstInfoX constInfo;

    __tiling_data_ptr__ MixedQuantFlashAttnTiling* tilingData = nullptr;

    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    static constexpr uint32_t faPrefetchLen = 6;
    static constexpr uint32_t fdPrefetchLen = 2;

    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start;
    uint32_t curS2End;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;
    bool isFd = false;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser;
    static constexpr uint32_t L1_Q_SLOT_BYTES = CubeBlockType::L1_Q_SLOT_BYTES;
    static constexpr uint32_t L1_P_SLOT_BYTES = CubeBlockType::L1_P_SLOT_BYTES;
    static constexpr uint32_t L1_Q_BUF_NUM = CubeBlockType::L1_Q_BUF_NUM;
    static constexpr uint32_t L1_P_BUF_NUM = CubeBlockType::L1_P_BUF_NUM;
    using L1BufView = typename CubeBlockType::L1_BUF_VIEW;

    static constexpr uint32_t UB_MM1RES_BYTE = VecFaBlockType::UB_MM1RES_BYTE;
    static constexpr uint32_t UB_MM2RES_BYTE = VecFaBlockType::UB_MM2RES_BYTE;
    static constexpr uint32_t UB_MM1RES_BUF_NUM = VecFaBlockType::UB_MM1RES_BUF_NUM;
    static constexpr uint32_t UB_MM2RES_BUF_NUM = VecFaBlockType::UB_MM2RES_BUF_NUM;
    static constexpr uint32_t mm1ResSize = UB_MM1RES_BYTE * UB_MM1RES_BUF_NUM; // 1buf
    static constexpr uint32_t mm2ResSize = UB_MM2RES_BYTE * UB_MM2RES_BUF_NUM; // 2buf

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionAntiQuantGqaKernel()
        : cubeBlock(constInfo),
          vecFaBlock(constInfo),
          vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* kDescale, __gm__ uint8_t* vDescale, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* seqUsedQ,
                                __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* learnableSink, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                __gm__ uint8_t* faMetaData, __tiling_data_ptr__ MixedQuantFlashAttnTiling* tiling)
    {
        this->tilingData = tiling;

        aicCoreNum = GetBlockNum();
        if ASCEND_IS_AIV {
            aicCoreNum = aicCoreNum / GetSubBlockNum();
        }

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t*)faMetaData);

        InitConstInfo();

        keyPtr = key;
        // valuePtr = value;

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensQ, constInfo.cuSeqLensQSize + 1);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t*)seqUsedQ, constInfo.seqUsedQSize);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t*)seqUsedKv, constInfo.seqUsedKvSize);
        }

        qSeqUsedParser.Init(seqUsedGmQ, constInfo.seqUsedQSize, constInfo.s1Size); // TODO parser需适配
        kvSeqUsedParser.Init(seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.s2Size);

        InitShareBuf();

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock();
            cubeBlock.InitCubeInput(query, key, value, attenMask, cuSeqLensQ, cuSeqLensKv, seqUsedQ, seqUsedKv,
                                    blockTable, kDescale, vDescale);
        }

        if ASCEND_IS_AIV {
            vecFaBlock.InitVecBlock();
            vecFaBlock.InitVecInput(query, seqUsedQ, seqUsedKv, attenMask, learnableSink, softmaxLse, attentionOut,
                                    workspace);
            if constexpr (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC) {
                vecFaBlock.InitQuant(kDescale, vDescale);
            }
            if (this->tilingData->mixedQuantFlashAttnBaseParams.needInitOutput) {
                vecFaBlock.ClearOutput();
                // if (constInfo.aicIdx == 0) {
                //     DumpTensor(this->vecFaBlock.attentionOutGm, __LINE__, 10 * 128);
                //     DumpTensor(ubBmm2Buffers, __LINE__, 10 * 128);
                // }
            }

            if (isFd) {
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm,
                                            this->seqUsedGmQ, this->seqUsedGmKv, keyPtr);
                if (learnableSink != nullptr) {
                    sinkGm.SetGlobalBuffer((__gm__ Q_T*)learnableSink);
                    constInfo.learnableSinkFlag = true;
                    vecFdBlock.InitLearnableSinkGm(sinkGm);
                }
                if (constInfo.isSoftmaxLseEnable) {
                    softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
                    vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
                }
            }
            AscendC::ICachePreLoad(faPrefetchLen);
        }
    }

    __aicore__ inline void InitShareBuf()
    {
        if constexpr (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC) {
            l1QBuffers[0] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1BufView, l1QBuffer0), L1_Q_SLOT_BYTES);
            l1QBuffers[1] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1BufView, l1QBuffer1), L1_Q_SLOT_BYTES);
        }
        l1PBuffers[0] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1BufView, l1PBuffer0), L1_P_SLOT_BYTES);
        l1PBuffers[1] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1BufView, l1PBuffer1), L1_P_SLOT_BYTES);

        uint32_t ubBaseAddr = 0U;
        // bmm2 必须放在bmm1前面，初始化buf从0开始8k，如果bmm1放在前面，与其复用，会引入c1的脏数据
        ubBmm2Buffers = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, mm2ResSize);
        ubBaseAddr += mm2ResSize;
        ubBmm1Buffers = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, mm1ResSize);
        ubBaseAddr += mm1ResSize;
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_1);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_1);
        } else {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CC_L1P_0);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CC_L1P_1);
        }
    }

    __aicore__ inline void FreeCrossCoreSync()
    {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_1);
        } else {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_1);
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        const auto& baseParams = this->tilingData->mixedQuantFlashAttnBaseParams;
        const auto& attenMaskParams = this->tilingData->mixedQuantFlashAttnAttenMaskParams;
        const auto& workspaceParams = this->tilingData->mixedQuantFlashAttnWorkspaceParams;
        const auto& pageAttentionParams = this->tilingData->mixedQuantFlashAttnPageAttentionParams;
        // 必须先设置核号，后续 GetFAMetaDataIndex(constInfo.aicIdx,...) 依赖它
        // spilit param
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
            // pageAttention
            if constexpr (PAGE_ATTENTION) {
                constInfo.maxBlockNumPerBatch = pageAttentionParams.maxBlockNumPerBatch;
                constInfo.blockSize = pageAttentionParams.blockSize;
                constInfo.paLayoutType = pageAttentionParams.paLayoutType;
            }
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();

            constInfo.attenMaskBatch = attenMaskParams.attenMaskBatch;
            constInfo.attenMaskS1Size = attenMaskParams.attenMaskS1Size;
            constInfo.attenMaskS2Size = attenMaskParams.attenMaskS2Size;
            constInfo.isRowInvalidOpen = false; // todo
            constInfo.isExistRowInvalid = attenMaskParams.isExistRowInvalid;
            constInfo.accumOutSize = workspaceParams.accumOutSize;
            constInfo.logSumExpSize = workspaceParams.logSumExpSize;
            // LSE
            constInfo.isSoftmaxLseEnable = baseParams.isSoftMaxLseEnable;
            // fd
            constInfo.isFd = faMetaDataGm.GetValue(MIXED_QUANT_FLASH_ATTN_IS_FD_INDEX);
            isFd = constInfo.isFd;
        }
        // share param
        constInfo.bN2Start =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_BN2_START_INDEX));
        constInfo.gS1OStart =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_M_START_INDEX));
        constInfo.s2OStart =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_S2_START_INDEX));
        constInfo.bN2End =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_BN2_END_INDEX));
        constInfo.gS1OEnd =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_M_END_INDEX));
        constInfo.s2OEnd =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_S2_END_INDEX));
        constInfo.coreFirstTmpOutWsPos = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));

        constInfo.bSize = baseParams.bSize;
        constInfo.t1Size = baseParams.t1Size;
        constInfo.t2Size = baseParams.t2Size;
        constInfo.n2Size = baseParams.n2Size;
        constInfo.gSize = baseParams.gSize;
        constInfo.s1Size = baseParams.s1Size;
        constInfo.s2Size = baseParams.s2Size;
        constInfo.dSize = baseParams.dSize;
        constInfo.dSizeV = baseParams.dSizeV;
        constInfo.cuSeqLensQSize = baseParams.cuSeqLensQSize;
        // constInfo.cuSeqLensKVSize = this->tilingData->mixedQuantFlashAttnBaseParams.cuSeqLensKVSize;
        constInfo.seqUsedQSize = baseParams.seqUsedQSize;
        constInfo.seqUsedKvSize = baseParams.seqUsedKvSize;
        constInfo.scaleValue = static_cast<float>(baseParams.scaleValue);
        constInfo.isKvContinuous = baseParams.isKvContinuous != 0;
        constInfo.coreNum = baseParams.coreNum;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(baseParams.outputLayout);
        constInfo.sparseMode = attenMaskParams.sparseMode;
        constInfo.preTokens = attenMaskParams.winLefts;
        constInfo.nextTokens = attenMaskParams.winRights;
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return MIXED_QUANT_FLASH_ATTN_METADATA_HEAD_SIZE + MIXED_QUANT_FLASH_ATTN_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return MIXED_QUANT_FLASH_ATTN_METADATA_HEAD_SIZE + MIXED_QUANT_FLASH_ATTN_METADATA_SIZE * aicCoreNum +
               MQFA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void FlashAttention()
    {
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE];

        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = constInfo.gS1OStart;
        uint32_t s2Cur = constInfo.s2OStart;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        uint64_t createdTaskCount = 0;
        uint64_t executedTaskCount = 0;

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
            if (likely(validTaskCount)) {
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
        return ((bN2Cur != constInfo.bN2End) || (gS1Cur != constInfo.gS1OEnd) || (s2Cur != constInfo.s2OEnd));
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart) && (s2Cur == constInfo.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            // TODO TND适配
            if (constInfo.seqUsedKvSize == 0 && !constInfo.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorList<LAYOUT_KV>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvSeqUsedParser.GetActualSeqLength(bIdx);
            }
            actSeqLensQ = qSeqUsedParser.GetActualSeqLength(bIdx);
        }

        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP;
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

        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur == curS2Start) { // 不应该在这里更新轴
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

        if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
            headS2Split = constInfo.s2OStart != 0U;
            curS2Start = constInfo.s2OStart;
        }

        if ((bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
            tailS2Split = constInfo.s2OEnd != 0U;
            curS2End = constInfo.s2OEnd;
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
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
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

        uint32_t s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        uint32_t s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start, curS2End
        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == constInfo.bN2Start && gS1Cur == constInfo.gS1OStart) { // first line
            headS2Split = constInfo.s2OStart > s2StartWithSparse ? true : false;
            curS2Start = AttentionCommon::Max(s2StartWithSparse, constInfo.s2OStart);
        }
        if (bN2Cur == constInfo.bN2End && gS1Cur == constInfo.gS1OEnd) { // last line
            tailS2Split = constInfo.s2OEnd > 0U ? true : false;
            curS2End =
                constInfo.s2OEnd > 0U ? AttentionCommon::Min(s2EndWithSparse, constInfo.s2OEnd) : s2EndWithSparse;
        }
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX& runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX& runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务

        if (runInfo0.isValid) {
            if ASCEND_IS_AIV {
                if constexpr (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC) {
                    ComputeVec0(runInfo0);
                }
            }

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

    __aicore__ inline void ComputeMm1(RunInfoX& runInfo)
    {
        uint32_t l1QBufId = runInfo.mloop % L1_Q_BUF_NUM;
        LocalTensor<Q_T> l1QBuf = l1QBuffers[l1QBufId].template ReinterpretCast<Q_T>();

        uint32_t bmm1BufId = runInfo.loop % UB_MM1RES_BUF_NUM;
        uint32_t bmm1BufOffset = bmm1BufId * UB_MM1RES_BYTE;
        LocalTensor<MM_T> bmm1Buf = ubBmm1Buffers[bmm1BufOffset].template ReinterpretCast<MM_T>();

        cubeBlock.IterateBmm1(l1QBuf, bmm1Buf, bmm1BufId, runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX& runInfo)
    {
        uint32_t l1PBufId = runInfo.loop % L1_P_BUF_NUM;
        LocalTensor<Q_T> l1PBuf = l1PBuffers[l1PBufId].template ReinterpretCast<Q_T>();

        uint32_t bmm2BufId = runInfo.loop % UB_MM2RES_BUF_NUM;
        uint32_t bmm2BufOffset = bmm2BufId * UB_MM2RES_BYTE;
        LocalTensor<MM_T> bmm2Buf = ubBmm2Buffers[bmm2BufOffset].template ReinterpretCast<MM_T>();

        cubeBlock.IterateBmm2(l1PBuf, bmm2Buf, l1PBufId, bmm2BufId, runInfo);
    }

    __aicore__ inline void ComputeVec0(RunInfoX& runInfo)
    {
        uint32_t l1QBufId = runInfo.mloop % L1_Q_BUF_NUM;
        LocalTensor<Q_T> l1QBuf = l1QBuffers[l1QBufId].template ReinterpretCast<Q_T>();

        uint32_t bmm2BufId = runInfo.loop % UB_MM2RES_BUF_NUM;
        uint32_t bmm2BufOffset = bmm2BufId * UB_MM2RES_BYTE;
        LocalTensor<Q_T> bmm2Buf = ubBmm2Buffers[bmm2BufOffset].template ReinterpretCast<Q_T>();

        vecFaBlock.ProcessVec0(l1QBuf, l1QBufId, bmm2Buf, runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX& runInfo)
    {
        uint32_t l1PBufId = runInfo.loop % L1_P_BUF_NUM;
        LocalTensor<Q_T> l1PBuf = l1PBuffers[l1PBufId].template ReinterpretCast<Q_T>();

        uint32_t bmm1BufId = runInfo.loop % UB_MM1RES_BUF_NUM;
        uint32_t bmm1BufOffset = bmm1BufId * UB_MM1RES_BYTE;
        LocalTensor<MM_T> bmm1Buf = ubBmm1Buffers[bmm1BufOffset].template ReinterpretCast<MM_T>();

        vecFaBlock.ProcessVec1(l1PBuf, l1PBufId, bmm1Buf, bmm1BufId, runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX& runInfo)
    {
        uint32_t bmm2BufId = runInfo.loop % UB_MM2RES_BUF_NUM;
        uint32_t bmm2BufOffset = bmm2BufId * UB_MM2RES_BYTE;
        LocalTensor<MM_T> bmm2Buf = ubBmm2Buffers[bmm2BufOffset].template ReinterpretCast<MM_T>();

        this->vecFaBlock.ProcessVec2(bmm2Buf, bmm2BufId, runInfo);
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
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
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
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(Q_T)));

        if (constInfo.isKvContinuous) {
            info.isChangeBatch = false;
        }

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        if constexpr (USE_DN) {
            // TODO
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actVecMSize = info.actMSize;
        }
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if (constInfo.bN2Start == constInfo.bN2End && constInfo.gS1OStart == constInfo.gS1OEnd) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
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
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

        // 当前S2未处理完
        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
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

    __aicore__ inline void FlashDecode()
    {
        vecFdBlock.InitBuffers();
        AscendC::ICachePreLoad(fdPrefetchLen);

        FDparamsX fdParams;
        fdParams.fdBN2Idx = faMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_BN2_IDX_INDEX));
        fdParams.fdMIdx = faMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_M_IDX_INDEX));
        fdParams.fdWorkspaceIdx =
            faMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_WORKSPACE_IDX_INDEX));
        fdParams.fdS2SplitNum =
            faMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_WORKSPACE_NUM_INDEX));
        fdParams.mStart = faMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_M_START_INDEX));
        fdParams.mLen = faMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_M_NUM_INDEX));
        fdParams.fdCoreEnable = fdParams.mLen > 0 ? 1U : 0U;

        vecFdBlock.AllocEventID();
        vecFdBlock.InitDecodeParams();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams);
        vecFdBlock.FreeEventID();
    }

    __aicore__ inline void Process()
    {
        FlashAttention();
        if ASCEND_IS_AIC {
            cubeBlock.FreeCube();
        } else {
            vecFaBlock.FreeVec();
        }
        FreeCrossCoreSync();
        if ASCEND_IS_AIV {
            if (isFd) {
                FlashDecode();
            }
        }
    }
}; // FlashAttentionAntiQuantGqaKernel
} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_

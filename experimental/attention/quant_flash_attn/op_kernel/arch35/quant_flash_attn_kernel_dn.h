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
#include "quant_flash_attn_block_vec_fd_dn.h"
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
    bool enableFlashDecode_ = false;
    uint32_t coreFirstTmpOutWsPos_ = 0;
    typename VectorBlock::VecFdBlockType vecFdBlock{constInfo};

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint64_t gS1Size_ = 0;
    uint64_t s2LoopTimes_ = 0;
    uint64_t gS1LoopTimes_ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    // m-tile 级常量: 只依赖 gS1/batch, 提到 s2 循环外算一次
    uint32_t actM_ = mBaseSize;
    uint32_t actMAlign128_ = mBaseSize;
    int32_t maskBase0_ = 0;
    // CAUSAL: tile 起始行号 = gS1*mBaseSize, 对 gSize 取余 ⇒ 逐行阈值的补偿 (S1G 合轴, gSize 不能整除时非 0)
    int32_t maskRem_ = 0;
    uint32_t pscaleNum = 0;
    uint32_t incBIdx_ = 0;
    uint32_t curN2Idx_ = 0;
    uint32_t curKvHeadIdx_ = 0;
    uint32_t kvHeadCnt_ = 0;

    // CTRL saturation-mode originals: saved in Init, restored at end of Process
    int64_t oriSatStrategy_ = 0;
    int64_t oriFloatSat_ = 0;

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
        // header[1]: 任一 section 存在 FD 归并任务时 AICPU 置 1
        enableFlashDecode_ = ((__gm__ uint32_t*)fiaMetaData)[1] != 0;

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t*)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);
        fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t*)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                               FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * sectionNum_);

        // ND 的 FA 段很小, FD B 区在 228K 以上, 用 UB 总容量(248K)作为上界

        InitConstInfo();

        qSeqLensTool.Init(cuSeqlensQ, constInfo.qCuSeqLensSize, seqUsedQ, constInfo.qSeqUsedSize, constInfo.maxSeqlenQ);
        kvSeqLensTool.Init(cuSeqlensKv, constInfo.kvCuSeqLensSize, seqUsedKv, constInfo.kvSeqUsedSize,
                           constInfo.maxSeqlenKv);
        // v_scale 段距取 max(seqused_kv): seqused 与 max_seqlen_kv 二选一,
        // 前者更大时会覆盖不全; 必须在 kvSeqLensTool.Init 之后取.
        if (constInfo.kvSeqUsedSize > 0) {
            uint64_t seqUsedMax = 0;
            for (uint32_t bIdx = 0; bIdx < constInfo.bSize; ++bIdx) {
                uint64_t len = kvSeqLensTool.seqUsedParser.GetActualSeqLength(bIdx);
                seqUsedMax = (len > seqUsedMax) ? len : seqUsedMax;
            }
            constInfo.maxSeqlenKv = seqUsedMax;
        }

        // fixpipe 饱和模式: s 超 fp16 上界时饱和到 ±MAX 而非 inf, 避免 softmax 产生 nan.
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
        // CTRL[60]=0 切 USE_API 使 per-mode 位生效, CTRL[48]=0 为 FLOAT 饱和
        oriSatStrategy_ = AscendC::GetCtrlSpr<60, 60>();
        oriFloatSat_ = AscendC::GetCtrlSpr<48, 48>();
        AscendC::SetCtrlSpr<60, 60>(0);
        AscendC::SetCtrlSpr<48, 48>(0);
#else
        AscendC::AscendCUtils::SetOverflow(0);
#endif

        if ASCEND_IS_AIC {
            cubeBlock.InitInput(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable);
        } else {
            vectorBlock.InitInput(attentionOut, workspace);
            vectorBlock.ClearOutput();
            vecFdBlock.InitGlobalTensor(workspace, attentionOut, qSeqLensTool);
            vecFdBlock.template InitBuffers<VectorBlock::FD_UB_A_OFFSET, VectorBlock::FD_UB_B_OFFSET>();
            vectorBlock.InitFdMaxStage(VectorBlock::UB_FD_MAX_STAGE_OFFSET);
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
        // 物理 s2 兜底; max(seqused_kv) 的覆盖在 parser Init 之后(未 Init 时恒返回 0).
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

        tileMaxIdx = 3;
        pscaleNum = 0;
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
        // S1G (gSize>1): 任务空间 B×N_kv, n2Size 即 N_kv, kvHeadIdx = n2Idx; 摊平按 gRealSize 折算
        curKvHeadIdx_ = (constInfo.gSize > 1) ? curN2Idx_ : curN2Idx_ / constInfo.gRealSize;
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
                    // m-tile 常量: 与 s2 无关, 每 loop 复用
                    actM_ = mBaseSize;
                    if (((gS1 + 1) * mBaseSize) > gS1Size_) {
                        uint64_t tailM = (gS1Size_ > gS1 * mBaseSize) ? (gS1Size_ - gS1 * mBaseSize) : mBaseSize;
                        actM_ = (tailM < mBaseSize) ? static_cast<uint32_t>(tailM) : mBaseSize;
                    }
                    // ND: n 只需对齐到 16(ND_M_ROWS 内), 不再按 128 列白算, 也没有 112 个垃圾 lane;
                    // DN: PV/Fixpipe 的 N 约束要求 128 对齐
                    actMAlign128_ = (actM_ + 127) >> 7 << 7;
                    if (unlikely(constInfo.sparseMode == 3)) {
                        uint32_t qTileStart =
                            (constInfo.gSize > 1) ? (gS1 * mBaseSize) / constInfo.gSize : gS1 * mBaseSize;
                        maskBase0_ = -(static_cast<int32_t>(qTileStart)) -
                                     (static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(actSeqLensQ));
                        // S1G 合轴: 逐行 s1 号 t = r/gSize (r = gS1*mBaseSize + lane), 故阈值需再减
                        // rem = (gS1*mBaseSize) % gSize; gSize 整除 tile 起点时 rem=0, 与旧口径一致.
                        maskRem_ =
                            static_cast<int32_t>((constInfo.gSize > 1) ? ((gS1 * mBaseSize) % constInfo.gSize) : 0U);
                    }
                    curS2Start = (bN2 == bN2Start_ && gS1 == gS1OStart_) ? s2OStart_ : 0U;
                    curS2End = (bN2 == bN2End_ && gS1 == gS1OEnd_) ? s2OEnd_ : static_cast<uint32_t>(s2LoopTimes_);
                    if (HAS_MASK && unlikely(constInfo.sparseMode == 3)) {
                        // CAUSAL: 行块最大可见 key 之后的 s2 tile 全被 mask, 截断(输出已由 ClearOutput 预置 0);
                        // 只收缩非空区间, 保证 FD 分核时每个核仍写出 partial
                        uint64_t rowLast = static_cast<uint64_t>(mBaseSize) * (gS1 + 1) - 1;
                        rowLast = (rowLast < gS1Size_) ? rowLast : (gS1Size_ - 1);
                        uint64_t tokenLast = (constInfo.gSize > 1) ? rowLast / constInfo.gSize : rowLast;
                        int64_t lastKey = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) +
                                          static_cast<int64_t>(tokenLast);
                        if (lastKey >= 0) {
                            uint32_t s2Need = static_cast<uint32_t>((lastKey + s2BaseSize) / s2BaseSize);
                            if (s2Need > curS2Start && s2Need < curS2End) {
                                curS2End = s2Need;
                            }
                        }
                    }
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
            } else if (constInfo.gSize > 1) {
                curKvHeadIdx_ = curN2Idx_; // S1G: 每任务直取对应 KV 头
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

        info.actMSize = actM_;
        info.actSingleLoopS2Size = s2BaseSize;
        info.actSingleLoopS2SizeAlign = s2BaseSize; // s2BaseSize 天然对齐 32/16/64
        info.actSingleLoopS2SizeAlign16 = s2BaseSize;
        info.actSingleLoopS2SizeAlign64 = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size =
                (actSeqLensKv > (uint64_t)s2Cur * s2BaseSize) ? (actSeqLensKv - (uint64_t)s2Cur * s2BaseSize) : 0;
            uint32_t actS2 = static_cast<uint32_t>(info.actSingleLoopS2Size);
            info.actSingleLoopS2SizeAlign = Align(actS2, (uint32_t)AttentionCommon::BYTE_BLOCK); // 统一对齐到32
            info.actSingleLoopS2SizeAlign16 = Align(actS2, (uint32_t)16);                        // 统一对齐到16
            info.actSingleLoopS2SizeAlign64 = Align(actS2, (uint32_t)BUFFER_SIZE_BYTE_64B);      // 统一对齐到64
        }
        // 按绝对 tile 奇偶配对(而不是相对起点): 让"2 块一次搬"的窗口对齐到 2*s2BaseSize,
        // PA 下 bs>=512 时窗口即整页, 避免跨页拆成两次 DMA. 首 tile 强制非预取, 保证配对交替成立.
        info.prefetched = ((s2Cur & 1U) == 1U) && (s2Cur != curS2Start);
        info.pairKVCopyS2Size = info.actSingleLoopS2Size;
        if (!info.prefetched && (s2Cur + 1) < curS2End) {
            uint64_t nextSize = actSeqLensKv - (uint64_t)(s2Cur + 1) * s2BaseSize;
            info.pairKVCopyS2Size = s2BaseSize + static_cast<uint32_t>(nextSize > s2BaseSize ? s2BaseSize : nextSize);
        }

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        uint32_t curS2LoopIdx = s2Cur - curS2Start;
        uint32_t phase = curS2LoopIdx & (TILE_N - 1); // TILE_N 为 2 的幂
        info.isUpdatePScale = (info.isLastS2Loop || (phase + 1 == TILE_N));
        info.isC2Sync = (phase == 0);
        if (info.isC2Sync) {
            tileMaxIdx = (tileMaxIdx + 1) % 4;
            pscaleNum = (pscaleNum + 1) % 20;
        }
        info.actMSizeAlign128 = actMAlign128_;
        info.tileMaxIdx = tileMaxIdx;
        info.pscaleNum = (pscaleNum >= 10) ? 1 : 0;
    }

    __aicore__ inline void CalcVecParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                         VecRunInfo& info)
    {
        info.loop = loop;
        info.bIdx = incBIdx_;
        info.n2Idx = curN2Idx_;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.curS2LoopIdx = s2Cur - curS2Start; // 在当前核处理的S2的循环下标

        info.actMSize = actM_;
        info.actSingleLoopS2Size = s2BaseSize;
        info.actSingleLoopS2SizeAlign = s2BaseSize; // s2BaseSize 天然对齐 32/64
        info.actSingleLoopS2SizeAlign64 = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size =
                (actSeqLensKv > (uint64_t)s2Cur * s2BaseSize) ? (actSeqLensKv - (uint64_t)s2Cur * s2BaseSize) : 0;
            uint32_t actS2 = static_cast<uint32_t>(info.actSingleLoopS2Size);
            info.actSingleLoopS2SizeAlign = Align(actS2, (uint32_t)AttentionCommon::BYTE_BLOCK); // 统一对齐到32
            info.actSingleLoopS2SizeAlign64 = Align(actS2, (uint32_t)BUFFER_SIZE_BYTE_64B);      // 统一对齐到64
        }

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        uint32_t phase = info.curS2LoopIdx & (TILE_N - 1); // TILE_N 为 2 的幂
        info.isUpdatePScale = (info.isLastS2Loop || (phase + 1 == TILE_N));
        info.isC2Sync = (phase == 0);
        if (unlikely(constInfo.sparseMode == 3)) { // CAUSAL: maskBase = k_tile - q_tile - (sk-sq)
            info.maskBase = static_cast<int32_t>(s2Cur * s2BaseSize) + maskBase0_;
            info.maskRem = maskRem_;
        } else {
            info.maskBase = 0;
            info.maskRem = 0;
        }
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
        info.isS2FirstTilePerCore = (phase < 2);
        info.pscaleNum = (pscaleNum >= 10) ? 1 : 0;
        // FD: 本核该行 S2 区间是完整 S2 的真子区间 ⇒ 该行 partial 写 workspace;
        // 头行用核首槽, 尾行在头行也被切时用第二槽
        if (enableFlashDecode_) {
            bool isHeadRow = (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_);
            bool isTailRow = (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_);
            bool headSplit = (s2OStart_ != 0);
            bool tailSplit = (isTailRow && (s2OEnd_ < s2LoopTimes_));
            info.isS2SplitCore = (isHeadRow && headSplit) || tailSplit;
            info.faTmpOutWsPos = coreFirstTmpOutWsPos_ + ((!isHeadRow && isTailRow && headSplit) ? 1U : 0U);
        }
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        // 本核首个 FD partial 槽号 (与 fdWorkspaceIdx 同一全局槽空间)
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

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!enableFlashDecode_) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        ICachePreLoad(2);
        vecFdBlock.FlashDecode(fdParams_, vectorBlock.GetFdTransUb());
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
            // FD 第二阶段: 仅 AIV 参与归并; SyncAll 仅同步 AIV, AIC 可提前进入下一 section
            if ASCEND_IS_AIV {
                if (enableFlashDecode_) {
                    SyncAll(); // 所有 AIV 的 partial 写 GM 完成后才能开始归并
                    FlashDecode(sectionIdx);
                    SyncAll(); // 归并读 workspace 完成后, 才能进入下一 section FA 覆写
                }
            }
        }
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
        // 恢复 CTRL 原值, 避免影响同核后续任务
        AscendC::SetCtrlSpr<48, 48>(oriFloatSat_);
        AscendC::SetCtrlSpr<60, 60>(oriSatStrategy_);
#endif
    }
};

} // namespace QFA_KERNEL

#endif

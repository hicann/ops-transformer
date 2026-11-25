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
 * \file flash_attn_block_cube_dn.h
 * \brief FANoQuantGqaBlockCubeDnUnmerged
 */
#ifndef FLASH_ATTN_BLOCK_CUBE_DN_UNMERGED_H_
#define FLASH_ATTN_BLOCK_CUBE_DN_UNMERGED_H_

#include "../utils/flash_attn_type.h"
#include "../../../common/op_kernel/arch_info.h"

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockCubeDnUnmerged {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    // Mmad/Fixpipe synchronize L0C in hardware through unitFlag.
    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = float;

    const ConstInfo_t& constInfo_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool_;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<LAYOUT_KV, PAGE_ATTENTION>();
    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, SEQLEN_T, IS_TND<LAYOUT_T>()>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, SEQLEN_T, IS_TND<LAYOUT_KV>()>;
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, InnerMLayout::S1_ONLY_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    GlobalTensor<int32_t> blockTableGm_;

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    // mmRes 槽数来自 ConfigValue 的 count 常量（同类 buffer 的 id 连续）：
    //   0 .. UB_MM1_RES_BUFCNT-1         -> bmm1 分数槽
    //   UB_MM1_RES_BUFCNT .. +UB_MM2-1   -> bmm2 结果槽（UB_MM2_RES_BUFCNT=0 时不存在，bmm2 复用分数槽）
    //   CROSSCORE_L1P_0 (= MM_0 + UB_MM_RES_FLAGCNT) 起 -> L1 P 三缓冲
    static constexpr uint32_t UB_MM1_RES_BUFCNT = (uint32_t)FA_T::UB_MM1_RES_BUFCNT;
    static constexpr uint32_t UB_MM2_RES_BUFCNT = (uint32_t)FA_T::UB_MM2_RES_BUFCNT;
    static constexpr bool MM_RES_SEPARATE_BMM2 = (UB_MM2_RES_BUFCNT != 0U);
    static constexpr uint32_t MM1_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(MM_T);
    static constexpr uint32_t MM2_BUF_BYTES =
        mBaseSize / CV_RATIO * ((((uint32_t)dVBaseSize + 15U) >> 4) << 4) * sizeof(MM_T);
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES =
        UB_MM1_RES_BUFCNT * MM1_BUF_BYTES + UB_MM2_RES_BUFCNT * MM2_BUF_BYTES;
    static constexpr uint32_t UB_MM_RES_FLAGCNT = UB_MM1_RES_BUFCNT + UB_MM2_RES_BUFCNT;

    static constexpr uint32_t CROSSCORE_MM_0 = 0U;
    static constexpr uint32_t CROSSCORE_MM_1 = 1U;
    static constexpr uint32_t CROSSCORE_MM_2 = 2U;
    static constexpr uint32_t CROSSCORE_MM_3 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = CROSSCORE_MM_0 + UB_MM_RES_FLAGCNT;
    static constexpr uint32_t CROSSCORE_L1P_1 = CROSSCORE_L1P_0 + 1U;
    static constexpr uint32_t CROSSCORE_L1P_2 = CROSSCORE_L1P_0 + 2U;

    // 核内同步ID
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 0U;
    static constexpr uint32_t Q_L1_BUFFER_ID1 = 1U;
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 2U;
    static constexpr uint32_t KV_L1_BUFFER_ID1 = 3U;
    static constexpr uint32_t KV_L1_BUFFER_ID2 = 4U;
    static constexpr uint32_t KV_L1_BUFFER_ID3 = 5U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 7U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 8U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 9U;
    static constexpr uint32_t L0C_BUFFER_ID0 = 10U;
    static constexpr uint32_t L0C_BUFFER_ID1 = 11U;
    static constexpr uint32_t L0C_BUFFER_ID2 = 12U;
    static constexpr uint32_t L0C_BUFFER_ID3 = 13U;

    // UB
    LocalTensor<uint8_t> ubMmResBuffers_;
    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_BUFCNT = (s2BaseSize == 256 && dBaseSize > 128) ? 2U : 4U;
    static constexpr uint32_t L1_KV_BUF_BYTES =
        (s2BaseSize == 256 && dBaseSize > 128) ? s2BaseSize * dBaseSize * sizeof(KV_T) : 64U * 1024U;
    static_assert(L1_P_BUFCNT * L1_P_BUF_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES + L1_KV_BUFCNT * L1_KV_BUF_BYTES <=
                      512U * 1024U,
                  "DN L1 buffers exceed device capacity");
    // buffer位置+用途+Buffers, 例如l1PBuffers; 使用时命名: 用途+buffer位置+Tensor, 例如pL1Tensor
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    // L0C
    static constexpr uint32_t L0C_BUFCNT = 4U;
    static constexpr uint32_t L0C_BUF_BYTES = 64U * 1024U;
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;
    uint32_t mmResBufId_ = 0U;
    // L0A
    static constexpr uint32_t L0A_BUFCNT = 2U;
    static constexpr uint32_t L0A_BUF_BYTES = 32U * 1024U;
    LocalTensor<uint8_t> l0ABuffers_;
    uint32_t l0aBufId_ = 0U;
    // L0B
    static constexpr uint32_t L0B_BUFCNT = 2U;
    static constexpr uint32_t L0B_BUF_BYTES = 32U * 1024U;
    LocalTensor<uint8_t> l0BBuffers_;
    uint32_t l0bBufId_ = 0U;

    __aicore__ inline FANoQuantGqaBlockCubeDnUnmerged(ConstInfo_t& constInfo,
                                                      SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool,
                                                      SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                     __gm__ uint8_t* blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyBnStride, constInfo_.keyN2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueBnStride, constInfo_.valueN2Stride);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM_RES_TOTAL_BYTES);

        /*--------------------------------------------L1--------------------------------------------*/
        uint32_t addrL1 = 0;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);
        addrL1 = L1_P_BUFCNT * L1_P_BUF_BYTES;
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
        addrL1 += L1_Q_BUFCNT * L1_Q_BUF_BYTES;
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * L1_KV_BUF_BYTES);

        /*--------------------------------------------L0A/B--------------------------------------------*/
        l0ABuffers_ = LocalTensor<uint8_t>(TPosition::A2, 0U, L0A_BUFCNT * L0A_BUF_BYTES);
        l0BBuffers_ = LocalTensor<uint8_t>(TPosition::B2, 0U, L0B_BUFCNT * L0B_BUF_BYTES);

        /*--------------------------------------------L0C--------------------------------------------*/
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ& qGmTensor, __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool_.seqUsedParser);
        } else {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV& kvGmTensor,
                                        __gm__ uint8_t* gm, uint64_t bnStride = 0, uint64_t n2Stride = 0)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, kvSeqLensTool_.seqUsedParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1 + AIV0_AIV1_OFFSET);
        if constexpr (UB_MM_RES_FLAGCNT > 2U) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2 + AIV0_AIV1_OFFSET);
        }
        if constexpr (UB_MM_RES_FLAGCNT > 3U) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3 + AIV0_AIV1_OFFSET);
        }
    }

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfo& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        // 不合轴: 任务为单 query 头(n1)的纯 s1 块, 按 (b, n1) 直接寻址(n1 = n2*g + g)
        GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .gIdx = runInfo.gIdx,
                              .s1Idx = runInfo.s1Idx,
                              .dIdx = dOffset,
                              .s1DealSize = runInfo.actMSize,
                              .dDealSize = dRealSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfo& runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfo& runInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void IterateBmm1(RunInfo& runInfo)
    {
        uint32_t mmResUbBufId = mmResBufId_;
        uint32_t mm1BufOffset = 0U;
        if constexpr (MM_RES_SEPARATE_BMM2) {
            // bmm2 有独立槽时, 分数槽按任务奇偶轮转 -> 复用距离 2 轮(bmm1 双缓冲)
            mmResUbBufId = (uint32_t)(runInfo.loop % UB_MM1_RES_BUFCNT);
            mm1BufOffset = mmResUbBufId * MM1_BUF_BYTES;
        } else {
            mmResBufId_ = (mmResBufId_ + 1) % UB_MM1_RES_BUFCNT;
            mm1BufOffset = mmResUbBufId * MM1_BUF_BYTES;
        }
        LocalTensor<MM_T> mm1ResUbTensor = ubMmResBuffers_[mm1BufOffset].template ReinterpretCast<MM_T>();
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm1Dn(mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<MM_T>& dstTensor, const LocalTensor<MM_T>& l0C,
                                        uint32_t mSize, RunInfo& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = mSize;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.unitFlag = 3U; // 读完本块后释放 L0C，供后续 Mmad 写入。
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    template <uint32_t kOffset, bool writeResult>
    __aicore__ inline void ComputeBmm1DnKTile(const LocalTensor<Q_T>& qL1Tensor, const LocalTensor<KV_T>& kL1Tensor,
                                              LocalTensor<MM_T>& mm1ResUbTensor, RunInfo& runInfo, uint32_t kSize)
    {
        constexpr uint32_t mTile = 128U;
        constexpr uint32_t blockRows = 16U;
        constexpr uint32_t blockKA = 32U / sizeof(KV_T);
        constexpr uint32_t blockKB = 32U / sizeof(Q_T);
        const uint32_t s2Size = runInfo.actSingleLoopS2Size;
        const uint32_t nSize = (static_cast<uint32_t>(runInfo.actMSize) + 31U) / 32U * 32U;
        static_assert(s2BaseSize <= mTile * L0C_BUFCNT, "Each S2 tile needs its own L0C slot across D slices");
        const uint32_t mTiles = CeilDiv(s2Size, mTile);

        LoadData2DParamsV2 loadB{};
        loadB.kStartPosition = kOffset / blockKB;
        loadB.kStep = CeilDiv(kSize, blockKB);
        loadB.mStep = nSize / blockRows;
        loadB.srcStride = loadB.mStep;
        loadB.dstStride = loadB.mStep;
        LoadData2DParamsV2 loadA{};
        loadA.kStartPosition = kOffset / blockKA;
        loadA.kStep = CeilDiv(kSize, blockKA);
        loadA.srcStride = CeilDiv(s2Size, blockRows);

        // Every S2 tile reads the same Q slice for this D range.
        auto l0BTensor = l0BBuffers_[l0bBufId_ * L0B_BUF_BYTES].template ReinterpretCast<Q_T>();
        const uint32_t bMutexId = L0B_BUFFER_ID0 + l0bBufId_;
        Mutex::Lock<PIPE_MTE1>(bMutexId);
        LoadData(l0BTensor, qL1Tensor, loadB);
        Mutex::Unlock<PIPE_MTE1>(bMutexId);

        MmadParams mm{};
        mm.n = nSize;
        mm.cmatrixSource = false;
        mm.k = kSize;
        mm.cmatrixInitVal = (kOffset == 0U);
        // unitFlag 取 2 时保留 L0C 部分和，取 3 时通知 Fixpipe 读取最终结果。
        mm.unitFlag = writeResult ? 3U : 2U;
        Mutex::Lock<PIPE_M>(bMutexId);
        for (uint32_t m = 0; m < mTiles; ++m) {
            const uint32_t mOffset = m * mTile;
            const uint32_t mSize = AttentionCommon::Min(mTile, s2Size - mOffset);
            loadA.mStartPosition = mOffset / blockRows;
            loadA.mStep = CeilDiv(mSize, blockRows);
            loadA.dstStride = loadA.mStep;
            auto l0ATensor = l0ABuffers_[l0aBufId_ * L0A_BUF_BYTES].template ReinterpretCast<KV_T>();
            const uint32_t aMutexId = L0A_BUFFER_ID0 + l0aBufId_;
            Mutex::Lock<PIPE_MTE1>(aMutexId);
            LoadData(l0ATensor, kL1Tensor, loadA);
            Mutex::Unlock<PIPE_MTE1>(aMutexId);

            const uint32_t cSlot = (l0cBufId_ + m) % L0C_BUFCNT;
            auto l0CTensor = l0CBuffers_[cSlot * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            mm.m = (mSize == 1U) ? blockRows : mSize; // Keep a one-row tail in matrix mode on arch35.
            Mutex::Lock<PIPE_M>(aMutexId);
            Mmad(l0CTensor, l0ATensor, l0BTensor, mm);
            Mutex::Unlock<PIPE_M>(aMutexId);
            l0aBufId_ = (l0aBufId_ + 1U) % L0A_BUFCNT;

            if constexpr (writeResult) {
                FixpipeMm1Dn(mm1ResUbTensor[mOffset * (nSize / 2U)], l0CTensor, mSize, runInfo);
            }
        }
        Mutex::Unlock<PIPE_M>(bMutexId);
        l0bBufId_ = (l0bBufId_ + 1U) % L0B_BUFCNT;
        if constexpr (writeResult) {
            l0cBufId_ = (l0cBufId_ + mTiles) % L0C_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm1Dn(LocalTensor<MM_T>& mm1ResUbTensor, RunInfo& runInfo)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            CopyQuerySlice(qL1Tensor, 0, constInfo_.dSize, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyKeySlice(kL1Tensor, 0, constInfo_.dSize, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        if constexpr (dBaseSize <= 128U) {
            // D=72 also uses the 128-wide template. Mmad must consume only the actual D.
            ComputeBmm1DnKTile<0U, true>(qL1Tensor, kL1Tensor, mm1ResUbTensor, runInfo, constInfo_.dSize);
        } else {
            static_assert(dBaseSize == 192U, "Split-K DN tiling expects D=192");
            constexpr uint32_t kTile = dBaseSize / 2U;
            ComputeBmm1DnKTile<0U, false>(qL1Tensor, kL1Tensor, mm1ResUbTensor, runInfo, kTile);
            ComputeBmm1DnKTile<kTile, true>(qL1Tensor, kL1Tensor, mm1ResUbTensor, runInfo, kTile);
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfo& runInfo)
    {
        uint32_t mmResUbBufId = mmResBufId_;
        uint32_t mm2BufOffset = 0U;
        if constexpr (MM_RES_SEPARATE_BMM2) {
            // 独立的 bmm2 结果槽, id 紧接在分数槽之后
            (void)mmResBufId_;
            mmResUbBufId = UB_MM1_RES_BUFCNT;
            mm2BufOffset = UB_MM1_RES_BUFCNT * MM1_BUF_BYTES;
        } else {
            mmResBufId_ = (mmResBufId_ + 1) % UB_MM1_RES_BUFCNT;
            mm2BufOffset = mmResUbBufId * MM1_BUF_BYTES;
        }
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;
        LocalTensor<Q_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<Q_T>();
        LocalTensor<MM_T> mm2ResUbTensor = ubMmResBuffers_[mm2BufOffset].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm2Dn(mm2ResUbTensor, pL1Tensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeBmm2Dn(const DST_TENSOR_T& dstTensor, const LocalTensor<MM_T>& l0C)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = mBaseSize;                             // N = m = 128，沿 N 切给两个 AIV
        fixpipeParams.mSize = (constInfo_.dSizeV + 15U) / 16U * 16U; // M = align16(dV)
        fixpipeParams.srcStride = fixpipeParams.mSize;
        fixpipeParams.dstStride = mBaseSize / CV_RATIO; // 每 AIV 64 列（DN 行距）
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.unitFlag = 3U; // 读完本块后释放 L0C，供后续 Mmad 写入。
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2Dn(LocalTensor<MM_T>& mm2ResUbTensor, LocalTensor<Q_T>& pL1Tensor,
                                         RunInfo& runInfo)
    {
        static_assert(dVBaseSize <= 128, "PV DN requires dVBaseSize <= 128");
        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(vL1Tensor, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        constexpr uint32_t kTile = 128U;
        constexpr uint32_t blockRows = 16U;
        constexpr uint32_t blockKA = 32U / sizeof(KV_T);
        constexpr uint32_t blockKB = 32U / sizeof(Q_T);
        const uint32_t s2Size = runInfo.actSingleLoopS2Size;
        const uint32_t dVSize = constInfo_.dSizeV;
        auto l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();

        // Both L1 operands have S2 as their row axis. Transpose each fractal for V^T * P^T.
        LoadData2DParamsV2 loadV{};
        loadV.ifTranspose = true;
        loadV.kStep = CeilDiv(dVSize, blockKA);
        const uint32_t s2Stride = CeilDiv(s2Size, blockRows);
        loadV.srcStride = s2Stride;
        loadV.dstStride = CeilDiv(dVSize, blockRows);
        LoadData2DParamsV2 loadP{};
        loadP.ifTranspose = true;
        loadP.kStep = CeilDiv(mBaseSize, blockKB);
        loadP.srcStride = s2Stride;
        loadP.dstStride = CeilDiv(mBaseSize, blockRows);

        MmadParams mm{};
        mm.m = (dVSize == 1U) ? blockRows : dVSize;
        mm.n = mBaseSize;
        mm.cmatrixSource = false;
        for (uint32_t kOffset = 0; kOffset < s2Size; kOffset += kTile) {
            const uint32_t kSize = AttentionCommon::Min(kTile, s2Size - kOffset);
            const uint32_t s2StartBlock = kOffset / blockRows;
            const uint32_t s2BlockCount = CeilDiv(kSize, blockRows);
            loadV.mStartPosition = s2StartBlock;
            loadV.mStep = s2BlockCount;
            loadP.mStartPosition = s2StartBlock;
            loadP.mStep = s2BlockCount;

            const uint32_t aMutexId = L0A_BUFFER_ID0 + l0aBufId_;
            const uint32_t bMutexId = L0B_BUFFER_ID0 + l0bBufId_;
            auto l0ATensor = l0ABuffers_[l0aBufId_ * L0A_BUF_BYTES].template ReinterpretCast<KV_T>();
            auto l0BTensor = l0BBuffers_[l0bBufId_ * L0B_BUF_BYTES].template ReinterpretCast<Q_T>();
            Mutex::Lock<PIPE_MTE1>(aMutexId);
            LoadData(l0ATensor, vL1Tensor, loadV);
            Mutex::Unlock<PIPE_MTE1>(aMutexId);
            Mutex::Lock<PIPE_MTE1>(bMutexId);
            LoadData(l0BTensor, pL1Tensor, loadP);
            Mutex::Unlock<PIPE_MTE1>(bMutexId);

            mm.k = kSize;
            mm.cmatrixInitVal = (kOffset == 0);
            // 最后一段 S2 用 3 通知 Fixpipe，其余段用 2 继续累加。
            mm.unitFlag = (kOffset + kSize == s2Size) ? 3U : 2U;
            Mutex::Lock<PIPE_M>(aMutexId);
            Mutex::Lock<PIPE_M>(bMutexId);
            Mmad(l0CTensor, l0ATensor, l0BTensor, mm);
            Mutex::Unlock<PIPE_M>(aMutexId);
            Mutex::Unlock<PIPE_M>(bMutexId);
            l0aBufId_ = (l0aBufId_ + 1U) % L0A_BUFCNT;
            l0bBufId_ = (l0bBufId_ + 1U) % L0B_BUFCNT;
        }
        FixpipeBmm2Dn(mm2ResUbTensor, l0CTensor);
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
    }
}; // FANoQuantGqaBlockCubeDnUnmerged

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockCubeDummyDnUnmerged {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockCubeDummyDnUnmerged(ConstInfo_t& constInfo,
                                                           SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool,
                                                           SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool){};
};

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_BLOCK_CUBE_DN_UNMERGED_H_

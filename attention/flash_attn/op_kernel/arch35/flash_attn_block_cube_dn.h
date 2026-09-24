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
 * \brief FANoQuantGqaBlockCubeDn
 */
#ifndef FLASH_ATTN_BLOCK_CUBE_DN_H_
#define FLASH_ATTN_BLOCK_CUBE_DN_H_

#include "../utils/flash_attn_type.h"
#include "../../../common/op_kernel/matmul.h"

using namespace fa_base_matmul;

namespace FlashAttnKernel {

template <typename FA_T>
class FANoQuantGqaBlockCubeDn {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = float;

    const ConstInfo_t &constInfoDn_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensToolDn_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensToolDn_;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<LAYOUT_KV, PAGE_ATTENTION>();
    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, SEQLEN_T, IS_TND<LAYOUT_T>()>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, SEQLEN_T, IS_TND<LAYOUT_KV>()>;
    FaGmTensorQ queryGmDn_;
    FaGmTensorKV keyGmDn_;
    FaGmTensorKV valueGmDn_;
    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1Dn_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1Dn_;
    GlobalTensor<int32_t> blockTableGmDn_;

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_MM_0 = 0U;
    static constexpr uint32_t CROSSCORE_MM_1 = 1U;
    static constexpr uint32_t CROSSCORE_MM_2 = 2U;
    static constexpr uint32_t CROSSCORE_MM_3 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;

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
    static constexpr uint32_t UB_MM_RES_BUFCNT = (dBaseSize > 128) ? 2U : 4U;
    static constexpr uint32_t UB_MM_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * (s2BaseSize > dVBaseSize ? s2BaseSize : dVBaseSize) * sizeof(MM_T);
    LocalTensor<uint8_t> ubMmResBuffersDn_;
    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t L1_KV_BUF_BYTES = 64U * 1024U;
    // buffer位置+用途+Buffers, 例如l1PBuffers; 使用时命名: 用途+buffer位置+Tensor, 例如pL1Tensor
    LocalTensor<uint8_t> l1PBuffersDn_;
    LocalTensor<uint8_t> l1QBuffersDn_;
    LocalTensor<uint8_t> l1KvBuffersDn_;
    uint32_t qL1BufIdDn_ = 0U;
    uint32_t kvL1BufIdDn_ = 0U;
    // L0C
    static constexpr uint32_t L0C_BUFCNT = 4U;
    static constexpr uint32_t L0C_BUF_BYTES = 64U * 1024U;
    LocalTensor<uint8_t> l0CBuffersDn_;
    uint32_t l0cBufIdDn_ = 0U;
    uint32_t mmResBufIdDn_ = 0U;
    // L0A/B
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManagerDn_;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManagerDn_;
    using L0APolicyType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    using L0BPolicyType =
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    L0APolicyType mmL0APolicyDn_;
    L0BPolicyType mmL0BPolicyDn_;

    __aicore__ inline FANoQuantGqaBlockCubeDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                              SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfoDn_(constInfo),
          qSeqLensToolDn_(qSeqLensTool),
          kvSeqLensToolDn_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGmDn_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBufferDn(constInfoDn_.bSize, constInfoDn_.n2Size, constInfoDn_.gSize, constInfoDn_.s1Size,
                      constInfoDn_.dSize, queryGmDn_, query);
        InitKVBufferDn(constInfoDn_.bSize, constInfoDn_.s2Size, constInfoDn_.n2Size, constInfoDn_.blockSize,
                       constInfoDn_.dSize, keyGmDn_, key, constInfoDn_.keyBnStride, constInfoDn_.keyN2Stride);
        InitKVBufferDn(constInfoDn_.bSize, constInfoDn_.s2Size, constInfoDn_.n2Size, constInfoDn_.blockSize,
                       constInfoDn_.dSizeV, valueGmDn_, value, constInfoDn_.valueBnStride, constInfoDn_.valueN2Stride);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffersDn_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES);

        /*--------------------------------------------L1--------------------------------------------*/
        uint32_t addrL1 = 0;
        l1PBuffersDn_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);
        addrL1 = L1_P_BUFCNT * L1_P_BUF_BYTES;
        l1QBuffersDn_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
        addrL1 += L1_Q_BUFCNT * L1_Q_BUF_BYTES;
        l1KvBuffersDn_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * L1_KV_BUF_BYTES);

        // /*--------------------------------------------L0A--------------------------------------------*/
        l0aBufferManagerDn_.Init(BUFFER_SIZE_BYTE_64K);

        // /*--------------------------------------------L0B--------------------------------------------*/
        l0bBufferManagerDn_.Init(BUFFER_SIZE_BYTE_64K);

        /*--------------------------------------------L0C--------------------------------------------*/
        l0CBuffersDn_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitQBufferDn(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                         uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensToolDn_.seqUsedParser);
        } else {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensToolDn_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBufferDn(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                          uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                          __gm__ uint8_t *gm, uint64_t bnStride = 0, uint64_t n2Stride = 0)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGmDn_,
                                             constInfoDn_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGmDn_,
                                             constInfoDn_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, kvSeqLensToolDn_.seqUsedParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensToolDn_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1 + AIV0_AIV1_OFFSET);
        if constexpr (dBaseSize <= 128) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2 + AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3 + AIV0_AIV1_OFFSET);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mmL0APolicyDn_.Init(l0aBufferManagerDn_, BUFFER_SIZE_BYTE_32K, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        mmL0BPolicyDn_.Init(l0bBufferManagerDn_, BUFFER_SIZE_BYTE_32K, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
    }

    __aicore__ inline void FreeEventID()
    {
        mmL0APolicyDn_.Uninit(l0aBufferManagerDn_);
        mmL0BPolicyDn_.Uninit(l0bBufferManagerDn_);
    }

    __aicore__ inline void CopyQuerySliceDn(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                            RunInfo &dnRunInfo)
    {
        uint32_t dstStride = (dnRunInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoord gmCoord{.bIdx = dnRunInfo.bIdx,
                        .n2Idx = dnRunInfo.n2Idx,
                        .gS1Idx = dnRunInfo.gS1Idx,
                        .dIdx = dOffset,
                        .gS1DealSize = dnRunInfo.actMSize,
                        .dDealSize = dRealSize};
        copyQueryGmToL1Dn_(l1Tensor, queryGmDn_, gmCoord);
    }

    __aicore__ inline void CopyKeySliceDn(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfo &dnRunInfo)
    {
        uint32_t dstStride = (dnRunInfo.actSingleLoopS2Size + 15) >> 4 << 4;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = dnRunInfo.bIdx,
                          .n2Idx = dnRunInfo.n2Idx,
                          .s2Idx = dnRunInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = dnRunInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1Dn_(l1Tensor, keyGmDn_, gmCoord);
    }

    __aicore__ inline void CopyValueSliceDn(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                            RunInfo &dnRunInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(dnRunInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = dnRunInfo.bIdx,
                          .n2Idx = dnRunInfo.n2Idx,
                          .s2Idx = dnRunInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = dnRunInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1Dn_(l1Tensor, valueGmDn_, gmCoord);
    }

    __aicore__ inline void IterateBmm1(RunInfo &dnRunInfo)
    {
        uint32_t mmResUbBufId = mmResBufIdDn_;
        mmResBufIdDn_ = (mmResBufIdDn_ + 1) % UB_MM_RES_BUFCNT;
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMmResBuffersDn_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm1Dn(mm1ResUbTensor, dnRunInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                        RunInfo &dnRunInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (dnRunInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = dnRunInfo.actSingleLoopS2Size;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1Dn(LocalTensor<MM_T> &mm1ResUbTensor, RunInfo &dnRunInfo)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffersDn_[qL1BufIdDn_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(dnRunInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufIdDn_);
            CopyQuerySliceDn(qL1Tensor, 0, constInfoDn_.dSize, dnRunInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufIdDn_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufIdDn_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffersDn_[kvL1BufIdDn_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        CopyKeySliceDn(kL1Tensor, 0, constInfoDn_.dSize, dnRunInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufIdDn_);
            LocalTensor<MM_T> l0CSubTensor =
                l0CBuffersDn_[l0cBufIdDn_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            // Mmad K=D 精确累加; LoadData 搬运宽度按分形粒度对齐(对齐D=原值, D=72→80与Nd2Nz写入组数一致)
            uint32_t ldK = ((constInfoDn_.dSize + 15) >> 4) << 4;
            MMParam param = MakeMMParam((uint32_t)dnRunInfo.actSingleLoopS2Size, (uint32_t)dnRunInfo.actMSize,
                                        constInfoDn_.dSize, false, true, true, true, 0, 0, ldK, 0);
            if constexpr (dBaseSize > 128) {
                MatmulK<KV_T, Q_T, MM_T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                    kL1Tensor, qL1Tensor, mmL0APolicyDn_, mmL0BPolicyDn_, l0CSubTensor, param);
            } else {
                MatmulBase<KV_T, Q_T, MM_T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                    kL1Tensor, qL1Tensor, mmL0APolicyDn_, mmL0BPolicyDn_, l0CSubTensor, param);
            }
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufIdDn_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufIdDn_);

            FixpipeMm1Dn(mm1ResUbTensor, l0CSubTensor, dnRunInfo);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufIdDn_);
            l0cBufIdDn_ = (l0cBufIdDn_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        kvL1BufIdDn_ = (kvL1BufIdDn_ + 1) % L1_KV_BUFCNT;

        if (unlikely(dnRunInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufIdDn_);
            qL1BufIdDn_ = (qL1BufIdDn_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfo &dnRunInfo)
    {
        uint32_t mmResUbBufId = mmResBufIdDn_;
        mmResBufIdDn_ = (mmResBufIdDn_ + 1) % UB_MM_RES_BUFCNT;
        uint32_t pL1BufId = dnRunInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;
        LocalTensor<Q_T> pL1Tensor = l1PBuffersDn_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<Q_T>();
        LocalTensor<MM_T> mm2ResUbTensor =
            ubMmResBuffersDn_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm2l0SplitDn(mm2ResUbTensor, pL1Tensor, dnRunInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialNDn(const DST_TENSOR_T &dstTensor, const LocalTensor<MM_T> &l0C,
                                                uint32_t realN, RunInfo &dnRunInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        fixpipeParams.dstStride = (dVBaseSize + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2l0SplitDn(LocalTensor<MM_T> &mm2ResUbTensor, LocalTensor<Q_T> &pL1Tensor,
                                                RunInfo &dnRunInfo)
    {
        LocalTensor<KV_T> vL1Tensor = l1KvBuffersDn_[kvL1BufIdDn_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        CopyValueSliceDn(vL1Tensor, 0, constInfoDn_.dSizeV, dnRunInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        {
            uint32_t nLoops = (constInfoDn_.dSizeV + 128 - 1) / 128;
            // Mmad N=真实尾块宽度精确计算; LoadData 搬运宽度按分形粒度对齐(D=72→80, 与Nd2Nz写入组数一致);
            // Fixpipe nSize=真实列 + dstStride=128(UB行距) → vec2 128宽读, 尾列脏但 D 轴逐元素隔离无害
            uint32_t lastTileN = constInfoDn_.dSizeV - (nLoops - 1) * 128;
            uint32_t ldN = ((lastTileN + 15) >> 4) << 4;
            for (uint32_t n = 0; n < nLoops; n++) {
                uint32_t tileN = (n == nLoops - 1) ? lastTileN : 128;
                uint32_t tileLdN = (n == nLoops - 1) ? ldN : 128;
                Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufIdDn_);
                LocalTensor<MM_T> l0CSubTensor =
                    l0CBuffersDn_[l0cBufIdDn_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
                MMParam param = {(uint32_t)mBaseSize,
                                 tileN,
                                 (uint32_t)dnRunInfo.actSingleLoopS2Size,
                                 true,
                                 false,
                                 true,
                                 true,
                                 0,
                                 0,
                                 0,
                                 tileLdN};

                uint32_t s2Aligned = AttentionCommon::Align(dnRunInfo.actSingleLoopS2Size, 16U);
                uint64_t vL1Offset = n * 128U / 16U * s2Aligned * 16U;
                LocalTensor<KV_T> vL1TileTensor = vL1Tensor[vL1Offset];

                if constexpr (dVBaseSize > 128) {
                    MatmulFull<Q_T, KV_T, MM_T, 128, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                        pL1Tensor, vL1TileTensor, mmL0APolicyDn_, mmL0BPolicyDn_, l0CSubTensor, param);
                } else {
                    if constexpr (s2BaseSize == 128) {
                        MatmulFull<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
                            pL1Tensor, vL1TileTensor, mmL0APolicyDn_, mmL0BPolicyDn_, l0CSubTensor, param);
                    } else {
                        MatmulBase<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
                            pL1Tensor, vL1TileTensor, mmL0APolicyDn_, mmL0BPolicyDn_, l0CSubTensor, param);
                    }
                }
                Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufIdDn_);
                Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufIdDn_);

                uint32_t dstOffset = n * 128U;
                FixpipeMm2PartialNDn(mm2ResUbTensor[dstOffset], l0CSubTensor, tileN, dnRunInfo);

                Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufIdDn_);
                l0cBufIdDn_ = (l0cBufIdDn_ + 1) % L0C_BUFCNT;
            }
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufIdDn_);
        kvL1BufIdDn_ = (kvL1BufIdDn_ + 1) % L1_KV_BUFCNT;
    }
}; // FANoQuantGqaBlockCubeDn

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockCubeDummyDn {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockCubeDummyDn(ConstInfo_t &constInfo,
                                                   SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                   SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_BLOCK_CUBE_DN_H_

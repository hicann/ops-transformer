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
 * \file quant_flash_attn_block_cube_fullquant_fp8.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_FULLQUANT_FP8_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_FULLQUANT_FP8_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "memory_copy_arch35_quant_flash_attn.h"

#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeGqaFp8 {
public:
    // 静态Tensor: L0A/L0B 使用 BufferManager + BuffersPolicy(LOCK_UNLOCK),
    // L1/L0C 使用裸 LocalTensor + 显式地址 + Mutex::Lock/Unlock, 不依赖 TPipe 分配 eventId。
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;
    static constexpr InnerMLayout Q_M_LAYOUT = USE_DN ? InnerMLayout::S1_ONLY_LAYOUT : InnerMLayout::GS1_MERGE_LAYOUT;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GmFormat::NGTD;
    static constexpr GmFormat KV_FORMAT = GmFormat::PA_BnNBsD;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using L0AType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    using L0BType = std::conditional_t<
        (s2BaseSize == 256 && dBaseSize > 128),
        BuffersPolicySingleBuffer<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK,
                                  IdSource::EXTERNAL>,
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>>;

    // L1/L0C 裸 LocalTensor 布局(字节): P 由 kernel 的跨核 policy 从基址0分配, Q/KV 紧随其后
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t MM_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(INPUT_T);
    static constexpr uint32_t MM_KV_BUF_BYTES = dBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * dVBaseSize * sizeof(T);

    // 跨核同步 id(参考 flash_attn arch35): AIV 只用基址, AIC 需同时访问 base 和 base + AIV0_AIV1_OFFSET
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CC_MM1_READY_0 = 0U; // AIC Fixpipe -> AIV, bmm1 slot0
    static constexpr uint32_t CC_MM1_READY_1 = 1U; // AIC Fixpipe -> AIV, bmm1 slot1
    static constexpr uint32_t CC_MM2_READY = 2U;   // AIC Fixpipe -> AIV, bmm2 slot
    static constexpr uint32_t CC_MM1_FREE_0 = 8U;  // AIV -> AIC, bmm1 slot0
    static constexpr uint32_t CC_MM1_FREE_1 = 9U;  // AIV -> AIC, bmm1 slot1
    static constexpr uint32_t CC_MM2_FREE = 10U;   // AIV -> AIC, bmm2 slot
    static constexpr uint32_t CC_L1P_0 = 5U;       // L1 P slot0
    static_assert(CC_L1P_0 + L1_P_BUFCNT <= CC_MM1_FREE_0,
                  "cross-core free flag ids must not overlap the L1 P flag id range");
    static constexpr bool L0C_IS_4BUFF = (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                          mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4);
    static constexpr uint32_t L0C_BUFCNT = L0C_IS_4BUFF ? 4U : 2U;
    static constexpr uint32_t L0C_BUF_BYTES = (L0C_SIZE / L0C_BUFCNT) * KB_TO_BYTES;
    static constexpr uint32_t L0A_MANAGER_BYTES = 64U * KB_TO_BYTES;
    static constexpr uint32_t L0B_MANAGER_BYTES = 64U * KB_TO_BYTES;
    static constexpr uint32_t L0A_MM_BUF_BYTES = 32U * KB_TO_BYTES;
    static constexpr uint32_t L0B_MM_BUF_BYTES = 32U * KB_TO_BYTES;

    // 核内 Mutex ID: 与 flash_attn arch35 一致(L1 0-5 / L0A 6-7 / L0B 8-9 / L0C 10-13)
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

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = true;
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_PARSER_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_PARSER_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<(!PAGE_ATTENTION && IS_TND_LAYOUT),
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqParserType* qSeqParserPtr_ = nullptr;
    KvSeqParserType* kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;

    /* =====================LocalBuffer变量====================*/
    // 跨核 buffer(裸 LocalTensor): bmm2 结果 / bmm1 结果 / L1 P, 与 vec 侧地址一致
    LocalTensor<uint8_t> ubBmm2ResBuf_;
    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> l1PBuffers_;
    uint32_t bmm1ResBufId_ = 0U;
    uint32_t l1PBufId_ = 0U;

    // L1: Q/KV 裸 LocalTensor 紧随 P 之后 + Mutex
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KVBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;

    // L0C: 裸 LocalTensor + Mutex::Lock/Unlock<PIPE_M/PIPE_FIX>
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;

    // L0A/L0B: BufferManager + BuffersPolicy(LOCK_UNLOCK)
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;

    __gm__ uint8_t* keyPtr_ = nullptr;
    __gm__ uint8_t* valuePtr_ = nullptr;

    const ConstInfoX& constInfo_;

    /* ============================================================================= */
    __aicore__ inline QuantFlashAttnBlockCubeGqaFp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, QSeqParserType& qParser, KvSeqParserType& kvParser)
    {
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        // 跨核 UB 区: bmm2 结果在基址0, bmm1 结果紧随其后, 与 vec 侧推导保持一致
        uint32_t addrUb = 0U;
        ubBmm2ResBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUF_BYTES;
        ubBmm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);

        // L1: P(3 slot) 从基址0分配, Q/KV 紧随其后, 与 vec 侧地址推导保持一致
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);
        uint32_t addrL1 = L1_P_BUFCNT * L1_P_BUF_BYTES;
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * MM_Q_BUF_BYTES);
        addrL1 += L1_Q_BUFCNT * MM_Q_BUF_BYTES;
        l1KVBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * MM_KV_BUF_BYTES);

        l0aBufferManager_.Init(L0A_MANAGER_BYTES);
        l0bBufferManager_.Init(L0B_MANAGER_BYTES);
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ& qGmTensor, __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV& kvGmTensor,
                                        __gm__ uint8_t* gm, uint64_t bnStrides, uint64_t n2Strides)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        // 等待 AIV 消费完所有 UB 结果后再退出。
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_1 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void AllocEventID()
    {
        // 只初始化 L0A/L0B 的 BuffersPolicy(Mutex ID); L1/L0C 直接使用 Mutex, 无需初始化
        mmL0ABuffers_.Init(l0aBufferManager_, L0A_MM_BUF_BYTES, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            mmL0BBuffers_.Init(l0bBufferManager_, L0B_MM_BUF_BYTES, L0B_BUFFER_ID0);
        } else {
            mmL0BBuffers_.Init(l0bBufferManager_, L0B_MM_BUF_BYTES, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX& runInfo)
    {
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        if constexpr (USE_DN) {
            // DN分支下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
            GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.realN2Idx,
                                  .gIdx = 0,
                                  .s1Idx = runInfo.gS1Idx,
                                  .dIdx = dOffset,
                                  .s1DealSize = runInfo.actMSize,
                                  .dDealSize = nopeDealSize};
            copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
        } else {
            GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                    .n2Idx = runInfo.realN2Idx,
                                    .gS1Idx = runInfo.gS1Idx,
                                    .dIdx = dOffset,
                                    .gS1DealSize = runInfo.actMSize,
                                    .dDealSize = nopeDealSize};
            copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T>& dstTensor, RunInfoX& runInfo)
    {
        CopyQuerySlice(dstTensor, 0, constInfo_.dSize, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T>& dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                        uint32_t dOffset, uint32_t dRealSize, RunInfoX& runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        uint32_t nopeDealSize = dRealSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo, uint32_t s2RealSize)
    {
        uint32_t s2Offset = runInfo.s2Idx;
        CopyKeySlice(dstTensor, s2Offset, s2RealSize, 0, constInfo_.dSize, runInfo);
    }

    // copy value with full s2
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T>& dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                          uint32_t dOffset, uint32_t dRealSize, RunInfoX& runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo)
    {
        CopyValueSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size, 0, constInfo_.dSizeV, runInfo);
    }

    __aicore__ inline void IterateBmm1(RunInfoX& runInfo)
    {
        IterateBmm1Dn(runInfo);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo,
                                        uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 64;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(RunInfoX& runInfo)
    {
        uint32_t bmm1ResBufId = bmm1ResBufId_;
        bmm1ResBufId_ = (bmm1ResBufId_ + 1U) % UB_MM1_RES_BUFCNT;
        uint32_t mmReadySyncIdx = CC_MM1_READY_0 + bmm1ResBufId;
        uint32_t mmFreeSyncIdx = CC_MM1_FREE_0 + bmm1ResBufId;
        LocalTensor<T> mmResUbTensor =
            ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * MM_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            CopyQueryTile(qL1Tensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KVBuffers_[kvL1BufId_ * MM_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyKeyTile(kL1Tensor, runInfo, s2CalcSize);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
            MMParam param =
                MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, (uint32_t)constInfo_.dSize, false, true);
            MatmulFull<Q_T, KV_T, T, 256, 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType>(
                kL1Tensor, qL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param);
            // Matmul 已读完 L1 中的 K/V, 先归还给 MTE2, 使下一轮 GM->L1 预取可与 Fixpipe 重叠。
            Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx + AIV0_AIV1_OFFSET);
            FixpipeMm1Dn(mmResUbTensor, l0CTensor, runInfo, s2CalcSize);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx + AIV0_AIV1_OFFSET);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfoX& runInfo)
    {
        if constexpr (USE_DN) {
            IterateBmm2Dn(runInfo);
        }
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (constInfo_.dSizeV + 7) >> 3 << 3;

        if constexpr (!USE_DN) {
            fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        }

        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2Dn(RunInfoX& runInfo)
    {
        uint32_t pL1BufId = l1PBufId_;
        l1PBufId_ = (l1PBufId_ + 1U) % L1_P_BUFCNT;
        uint32_t pSyncIdx = CC_L1P_0 + pL1BufId;
        uint32_t mmReadySyncIdx = CC_MM2_READY;
        uint32_t mmFreeSyncIdx = CC_MM2_FREE;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        LocalTensor<T> mmResUbTensor = ubBmm2ResBuf_.template ReinterpretCast<T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx + AIV0_AIV1_OFFSET);

        uint32_t realK = runInfo.actSingleLoopS2Size;
        LocalTensor<KV_T> vL1Tensor = l1KVBuffers_[kvL1BufId_ * MM_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(vL1Tensor, runInfo.s2Idx, realK, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo_.dSizeV, realK, USE_DN, false, true, true);
            MatmulFull<INPUT_T, KV_T, T, 128, 128, 256, ABLayout::MK, ABLayout::KN, L0AType, L0BType>(
                pL1Tensor, vL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param);
            // Matmul 已读完 L1 中的 V, 先归还给 MTE2, 使下一轮预取可与 Fixpipe 重叠。
            Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx + AIV0_AIV1_OFFSET);
            FixpipeMm2(mmResUbTensor, l0CTensor, runInfo);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx + AIV0_AIV1_OFFSET);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
    }
}; // QuantFlashAttnBlockCubeGqaFp8

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeGqaFp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockCubeGqaFp8Dummy(ConstInfoX& constInfo){};
    __aicore__ inline void InitCrossCoreSync() {}
    __aicore__ inline void UnInitCrossCoreSync() {}
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_FULLQUANT_FP8_H_

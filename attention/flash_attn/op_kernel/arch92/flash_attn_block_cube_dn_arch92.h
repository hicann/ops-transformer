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
 * \file flash_attn_block_cube_dn_arch92.h
 * \brief FANoQuantGqaBlockCubeDn
 */
#ifndef FLASH_ATTN_BLOCK_CUBE_DN_ARCH92_H_
#define FLASH_ATTN_BLOCK_CUBE_DN_ARCH92_H_

#include "../utils/flash_attn_type.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/arch_info.h"

using namespace fa_base_matmul;

namespace BaseApi {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockCubeDn {
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

    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = float;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    const ConstInfoX& constInfo_;

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
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM1_1 = 1U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
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
    static constexpr uint32_t UB_MM1_RES_BUFCNT = FA_T::UB_MM1_RES_BUFCNT;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(MM_T);
    static constexpr uint32_t UB_MM2_RES_BUFCNT = FA_T::UB_MM2_RES_BUFCNT;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(MM_T);
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    // L1
    static constexpr uint32_t L1_P_BUFCNT = FA_T::L1_P_BUFCNT;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = FA_T::L1_Q_BUFCNT;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_BUFCNT = FA_T::L1_KV_BUFCNT;
    static constexpr uint32_t L1_KV_BUF_BYTES = s2BaseSize * dBaseSize * sizeof(KV_T);
    // buffer位置+用途+Buffers, 例如l1PBuffers; 使用时命名: 用途+buffer位置+Tensor, 例如pL1Tensor
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    // L0A
    static constexpr uint32_t L0A_BUFCNT = 2U;
    static constexpr uint32_t L0A_BUF_BYTES = BUFFER_SIZE_BYTE_32K;
    LocalTensor<uint8_t> l0ABuffers_;
    uint32_t l0aBufId_ = 0U;
    // L0B
    static constexpr uint32_t L0B_BUFCNT = 2U;
    static constexpr uint32_t L0B_BUF_BYTES = BUFFER_SIZE_BYTE_32K;
    LocalTensor<uint8_t> l0BBuffers_;
    uint32_t l0bBufId_ = 0U;
    // L0C
    static constexpr uint32_t L0C_BUFCNT = 4U;
    static constexpr uint32_t L0C_BUF_BYTES = 64U * 1024U;
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;

    __aicore__ inline FANoQuantGqaBlockCubeDn(ConstInfoX& constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool,
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
        struct UbLayout {
            uint8_t mm2ResBuffers[UB_MM2_RES_BUFCNT][UB_MM2_RES_BUF_BYTES]; // CV通信BUF
            uint8_t mm1ResBuffers[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES]; // CV通信BUF
        };
        static_assert(sizeof(UbLayout) <= (CV_RATIO == 1 ? 376 * 1024 : 248 * 1024), "UB buffer too large");
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm2ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm2ResBuffers));
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm1ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm1ResBuffers));

        /*--------------------------------------------L1--------------------------------------------*/
        struct L1Layout {
            uint8_t pBuffers[L1_P_BUFCNT][L1_P_BUF_BYTES];
            uint8_t qBuffers[L1_Q_BUFCNT][L1_Q_BUF_BYTES];
            uint8_t kvBuffers[L1_KV_BUFCNT][L1_KV_BUF_BYTES];
        };
        static_assert(sizeof(L1Layout) <= 512 * 1024, "L1 buffer too large");
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, OFFSET_OF_MEMBER(L1Layout, pBuffers),
                                           SIZE_OF_MEMBER(L1Layout, pBuffers));
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, OFFSET_OF_MEMBER(L1Layout, qBuffers),
                                           SIZE_OF_MEMBER(L1Layout, qBuffers));
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, OFFSET_OF_MEMBER(L1Layout, kvBuffers),
                                            SIZE_OF_MEMBER(L1Layout, kvBuffers));

        /*--------------------------------------------L0A--------------------------------------------*/
        l0ABuffers_ = LocalTensor<uint8_t>(TPosition::A2, 0U, L0A_BUFCNT * L0A_BUF_BYTES);

        /*--------------------------------------------L0B--------------------------------------------*/
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
        for (int bufferId = 0; bufferId < UB_MM1_RES_BUFCNT; ++bufferId) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_0 + bufferId);
        }
        for (int bufferId = 0; bufferId < UB_MM2_RES_BUFCNT; ++bufferId) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_0 + bufferId);
        }
    }

    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoS1Only& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

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
                                        RunInfoS1Only& runInfo)
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
                                          RunInfoS1Only& runInfo)
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

    __aicore__ inline void IterateBmm1(RunInfoS1Only& runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
        IterateBmm1Dn(mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
    }

    __aicore__ inline void IterateBmm1Dn(LocalTensor<MM_T>& mm1ResUbTensor, RunInfoS1Only& runInfo)
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

        LocalTensor<Q_T> L0BTensor = l0BBuffers_[l0bBufId_ * L0B_BUF_BYTES].template ReinterpretCast<Q_T>();
        LoadData2DParamsV2 loadDataParamsB;
        loadDataParamsB.mStartPosition = 0;
        loadDataParamsB.kStartPosition = 0;
        loadDataParamsB.ifTranspose = false;
        loadDataParamsB.mStep = (runInfo.actMSize + 15) >> 4;
        loadDataParamsB.kStep = GetBlockNum<KV_T>(constInfo_.dSize);
        loadDataParamsB.srcStride = loadDataParamsB.mStep;
        loadDataParamsB.dstStride = loadDataParamsB.mStep;
        Mutex::Lock<PIPE_MTE1>(L0B_BUFFER_ID0 + l0bBufId_);
        LoadData(L0BTensor, qL1Tensor, loadDataParamsB);
        Mutex::Unlock<PIPE_MTE1>(L0B_BUFFER_ID0 + l0bBufId_);

        constexpr uint32_t dnSplitMBaseM = 128; // Dn模板mm1左矩阵为K，右矩阵为Q，M轴为S2方向
        const uint32_t mLoops = CeilDiv(runInfo.actSingleLoopS2Size, dnSplitMBaseM);
        for (uint32_t m = 0; m < mLoops; m++) {
            uint32_t mSize = dnSplitMBaseM;
            if (m == mLoops - 1) {
                mSize = (uint32_t)runInfo.actSingleLoopS2Size - m * dnSplitMBaseM;
            }

            LocalTensor<KV_T> L0ATensor = l0ABuffers_[l0aBufId_ * L0A_BUF_BYTES].template ReinterpretCast<KV_T>();
            LoadData2DParamsV2 loadDataParamsA;
            loadDataParamsA.mStartPosition = m * (dnSplitMBaseM / 16);
            loadDataParamsA.kStartPosition = 0;
            loadDataParamsA.ifTranspose = false;
            loadDataParamsA.mStep = (mSize + 15) >> 4;
            loadDataParamsA.kStep = GetBlockNum<KV_T>(constInfo_.dSize);
            loadDataParamsA.srcStride = (runInfo.actSingleLoopS2Size + 15) >> 4;
            loadDataParamsA.dstStride = loadDataParamsA.mStep;
            Mutex::Lock<PIPE_MTE1>(L0A_BUFFER_ID0 + l0aBufId_);
            LoadData(L0ATensor, kL1Tensor, loadDataParamsA);
            Mutex::Unlock<PIPE_MTE1>(L0A_BUFFER_ID0 + l0aBufId_);

            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            MmadParams mmadParams;
            mmadParams.m = mSize;
            mmadParams.n = (uint32_t)runInfo.actMSize;
            mmadParams.k = (uint32_t)(constInfo_.dSize);
            mmadParams.cmatrixInitVal = true;
            mmadParams.cmatrixSource = false;
            // 单次 Mmad 对应单次 Fixpipe, 使能 unitFlag: Mmad=3 (写后置位允许 Fixpipe 读)
            mmadParams.unitFlag = UNITFLAG_EN_OUTER_LAST;
            mmadParams.disableGemv = true;

            Mutex::Lock<PIPE_M>(L0A_BUFFER_ID0 + l0aBufId_);
            Mutex::Lock<PIPE_M>(L0B_BUFFER_ID0 + l0bBufId_);
            // unitFlag 使能后, Mmad→Fixpipe 由硬件 512B 细粒度同步接管, 去掉指令级Lock/UnLock同步
            Mmad(l0CSubTensor, L0ATensor, L0BTensor, mmadParams);
            Mutex::Unlock<PIPE_M>(L0A_BUFFER_ID0 + l0aBufId_);
            Mutex::Unlock<PIPE_M>(L0B_BUFFER_ID0 + l0bBufId_);
            l0aBufId_ = (l0aBufId_ + 1) % L0A_BUFCNT;

            uint32_t ubOffset = m * dnSplitMBaseM * mBaseSize;
            FixpipeMm1Dn(mm1ResUbTensor[ubOffset], l0CSubTensor, mSize, runInfo);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        l0bBufId_ = (l0bBufId_ + 1) % L0B_BUFCNT;

        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<MM_T>& dstTensor, const LocalTensor<MM_T>& l0C,
                                        uint32_t mSize, RunInfoS1Only& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = ((runInfo.actMSize + 7) >> 3) << 3; // 使能NZ2ND功能，nSize*sizeof(T)必须为32的倍数
        fixpipeParams.mSize = mSize;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        fixpipeParams.dstStride = mBaseSize;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST; // 读后置0, 允许后续 Mmad 写
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(RunInfoS1Only& runInfo)
    {
        uint32_t mm2ResUbBufId = runInfo.loop % UB_MM2_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t c2v2CrossCoreSyncIdx = CROSSCORE_BMM2_0 + mm2ResUbBufId;
        LocalTensor<Q_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<Q_T>();
        LocalTensor<MM_T> mm2ResUbTensor =
            ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c2v2CrossCoreSyncIdx);
        IterateBmm2l0Split(mm2ResUbTensor, pL1Tensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c2v2CrossCoreSyncIdx);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T& dstTensor, const LocalTensor<MM_T>& l0C,
                                              uint32_t realN, RunInfoS1Only& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = (mBaseSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = (dVBaseSize + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST; // 读后置0, 允许后续 Mmad 写
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2l0Split(LocalTensor<MM_T>& mm2ResUbTensor, LocalTensor<Q_T>& pL1Tensor,
                                              RunInfoS1Only& runInfo)
    {
        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(vL1Tensor, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);

        constexpr uint32_t bmm2BaseK = 128;
        const uint32_t bmm2KLoops = CeilDiv(AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U), bmm2BaseK);
        const uint32_t kTailSize =
            (runInfo.actSingleLoopS2Size % bmm2BaseK == 0) ? bmm2BaseK : (runInfo.actSingleLoopS2Size % bmm2BaseK);
        constexpr uint64_t bmm2L1AOffset = bmm2BaseK << 4; // 2048, half: 128*16
        constexpr uint64_t bmm2L1BOffset = bmm2BaseK << 4; // 2048

        MMParam param = {
            (uint32_t)mBaseSize,                   // singleM 128
            (uint32_t)constInfo_.dSizeV,           // singleN 128
            (uint32_t)runInfo.actSingleLoopS2Size, // singleK
            true,                                  // isLeftTranspose
            false                                  // isRightTranspose
        };
        uint32_t bmm2M = param.realM != 0 ? param.realM : param.singleM;

        LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
        for (uint32_t k = 0; k < bmm2KLoops; k++) {
            uint32_t tileK = (k == (bmm2KLoops - 1)) ? kTailSize : bmm2BaseK;
            LocalTensor<KV_T> l0bTensor = l0BBuffers_[l0bBufId_ * L0B_BUF_BYTES].template ReinterpretCast<KV_T>();
            Mutex::Lock<PIPE_MTE1>(L0B_BUFFER_ID0 + l0bBufId_);
            LoadDataToL0B<KV_T>(l0bTensor, vL1Tensor, param, k * bmm2L1BOffset, tileK, param.singleN);
            Mutex::Unlock<PIPE_MTE1>(L0B_BUFFER_ID0 + l0bBufId_);

            if (k == 0) {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P_0 + runInfo.loop % L1_P_BUFCNT);
            }

            LocalTensor<Q_T> l0aTensor = l0ABuffers_[l0aBufId_ * L0A_BUF_BYTES].template ReinterpretCast<Q_T>();
            Mutex::Lock<PIPE_MTE1>(L0A_BUFFER_ID0 + l0aBufId_);
            LoadDataToL0A<Q_T>(l0aTensor, pL1Tensor, param, k * bmm2L1AOffset, tileK, param.singleM);
            Mutex::Unlock<PIPE_MTE1>(L0A_BUFFER_ID0 + l0aBufId_);

            MmadParams mmadParams;
            mmadParams.m = bmm2M;
            mmadParams.n = param.singleN;
            mmadParams.k = tileK;
            mmadParams.cmatrixInitVal = (k == 0);
            mmadParams.cmatrixSource = false;
            mmadParams.unitFlag = (k < bmm2KLoops - 1) ? UNITFLAG_ENABLE : UNITFLAG_EN_OUTER_LAST;
            mmadParams.disableGemv = true;

            Mutex::Lock<PIPE_M>(L0A_BUFFER_ID0 + l0aBufId_);
            Mutex::Lock<PIPE_M>(L0B_BUFFER_ID0 + l0bBufId_);
            // unitFlag 使能后, Mmad→Fixpipe 由硬件 512B 细粒度同步接管, 去掉指令级Lock/UnLock同步
            Mmad(l0CSubTensor, l0aTensor, l0bTensor, mmadParams);

            // 当矩阵沿K轴累加时，连续两次Mmad间是否需要PipeBarrier(PIPE_M)取决于计算量：
            if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                AscendC::PipeBarrier<PIPE_M>(); // 计算量小于阈值，需同步
            }
            // 计算量大于阈值时，硬件自动处理依赖，无需同步

            Mutex::Unlock<PIPE_M>(L0A_BUFFER_ID0 + l0aBufId_);
            Mutex::Unlock<PIPE_M>(L0B_BUFFER_ID0 + l0bBufId_);

            l0aBufId_ = (l0aBufId_ + 1) % L0A_BUFCNT;
            l0bBufId_ = (l0bBufId_ + 1) % L0B_BUFCNT;
        }

        FixpipeMm2PartialN(mm2ResUbTensor, l0CSubTensor, constInfo_.dSizeV, runInfo);
        l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;

        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
    }
}; // FANoQuantGqaBlockCubeDn

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockCubeDummyDn {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockCubeDummyDn(ConstInfoX& constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool,
                                                   SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool){};
};

} // namespace BaseApi

#endif // FLASH_ATTN_BLOCK_CUBE_DN_H_

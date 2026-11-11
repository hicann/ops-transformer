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
 * \file flash_mla_block_cube_arch92.h
 * \brief arch92 flash_mla block cube MLA；
 *        静态 tensor buffer + Mutex/cross-core 显式同步。
 */
#ifndef FLASH_MLA_BLOCK_CUBE_ARCH92_H_
#define FLASH_MLA_BLOCK_CUBE_ARCH92_H_
#include "../../../common/op_kernel/const_def.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_mla_type.h"
using namespace AscendC;
using namespace AttentionCommon;

namespace FlashMlaKernel {

template <typename FA_T>
class FlashMlaNoQuantMlaBlockCube {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FLASH_MLA_LAYOUT LAYOUT = FA_T::qLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;

    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<FA_T::qLayout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<FA_T::qLayout, FA_T::kvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t<FlashMlaKernelType::NO_QUANT>;

    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4ULL;
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;

    static constexpr uint32_t L0C_BUFFER_ID0 = 0U;
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 2U; // Q 单槽，容量 mBaseSize*576*sizeof(Q_T)
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 4U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 9U;

    // M64/S2=128, CV=1. UB aliases the vector block's 160 KiB communication prefix.
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize * 256U * sizeof(MM_T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize * 128U * sizeof(MM_T);
    // Q: 72 KiB; three KV/P slots: 144 KiB each. P overwrites rope after QK.
    static constexpr uint32_t L1_Q_BUFCNT = 1U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * 576U * sizeof(Q_T);
    static constexpr uint32_t L1_KVP_BUFCNT = 3U;
    static constexpr uint32_t L1_KVP_BUF_BYTES = s2BaseSize * 576U * sizeof(INPUT_T);
    static constexpr uint32_t L0C_BUFCNT = 2U;
    static constexpr uint32_t L0C_BUF_BYTES = 128U * 1024U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768U;
    // Same Q-cache and alternating Q/P lifetime as arch35, scaled to M=64.
    static constexpr uint32_t QK_CACHE_K = 96U;
    static constexpr uint32_t QK_SHORT_K = 112U;
    static constexpr uint32_t QK_LONG_K = 128U;
    static constexpr uint32_t QK_STEPS = 5U;
    static constexpr uint32_t QK_CACHE_BYTES = mBaseSize * QK_CACHE_K * sizeof(Q_T);
    static constexpr uint32_t QK_TEMP_OFFSET = QK_CACHE_BYTES;
    static constexpr uint32_t QK_TEMP_BYTES = mBaseSize * QK_LONG_K * sizeof(Q_T);
    static constexpr uint32_t PV_L0A_OFFSET = QK_TEMP_OFFSET + QK_TEMP_BYTES;
    static constexpr uint32_t PV_L0A_BYTES = mBaseSize * s2BaseSize * sizeof(Q_T);
    static_assert(QK_CACHE_K + 2U * (QK_SHORT_K + QK_LONG_K) == dBaseSize, "QK must cover all 576 features");
    static_assert(PV_L0A_OFFSET + PV_L0A_BYTES <= BUFFER_SIZE_BYTE_64K, "Q-cache, temporary Q and P exceed L0A");
    static_assert(QK_SHORT_K <= s2BaseSize, "P slot must also hold the short Q slice");
    static_assert(s2BaseSize * QK_LONG_K * sizeof(KV_T) <= BUFFER_SIZE_BYTE_32K, "QK tile exceeds L0B slot");

    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvpBuffers_; // K/V/P 共用 ring（vec 写 P 的目标偏移注释见 vec 块）
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;
    LocalTensor<uint8_t> l0ABuffers_;
    LocalTensor<uint8_t> l0BBuffers_;
    uint32_t l0bBufId_ = 0U;
    bool lastMatmulWasPv_ = false;

    /* =====================GM变量(with layout)==================== */
    FaGmTensor<Q_T, Q_FORMAT, uint32_t, LAYOUT == FLASH_MLA_LAYOUT::TND> queryGm_;
    FaGmTensor<KV_T, KV_FORMAT> kCacheGm_;
    GlobalTensor<int32_t> blockTableGm_;
    SeqLensToolType& seqLensTool_;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;

    const ConstInfoX& constInfo_;

    /*============================================================================== */
    __aicore__ inline FlashMlaNoQuantMlaBlockCube(ConstInfoX& constInfo, SeqLensToolType& seqLensTool)
        : constInfo_(constInfo),
          seqLensTool_(seqLensTool){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t* query, __gm__ uint8_t* kCache, __gm__ uint8_t* blockTable)
    {
        InitCubeInput(query, kCache, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        static_assert(mBaseSize == 64 && s2BaseSize == 128 && CV_RATIO == 1, "arch92 requires M64/S2=128/CV=1");
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES);
        addrUb = UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);

        /*--------------------------------------------L1--------------------------------------------*/
        struct L1Layout {
            uint8_t qBuffers[L1_Q_BUFCNT][L1_Q_BUF_BYTES];
            uint8_t kvpBuffers[L1_KVP_BUFCNT][L1_KVP_BUF_BYTES];
        };
        static_assert(sizeof(L1Layout) <= 512 * 1024, "L1 buffer too large");
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, SIZE_OF_MEMBER(L1Layout, qBuffers));
        // P 复用 KV 槽的 rope 区，写入偏移必须与 Vector 侧一致。
        l1KvpBuffers_ =
            LocalTensor<uint8_t>(TPosition::A1, L1_Q_BUFCNT * L1_Q_BUF_BYTES, SIZE_OF_MEMBER(L1Layout, kvpBuffers));

        /*--------------------------------------------L0A/B--------------------------------------------*/
        l0ABuffers_ = LocalTensor<uint8_t>(TPosition::A2, 0U, BUFFER_SIZE_BYTE_64K);
        l0BBuffers_ = LocalTensor<uint8_t>(TPosition::B2, 0U, BUFFER_SIZE_BYTE_64K);
        l0bBufId_ = 0U;
        lastMatmulWasPv_ = false;

        /*--------------------------------------------L0C--------------------------------------------*/
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* kCache, __gm__ uint8_t* blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size,
                    constInfo_.dSize + constInfo_.dSizeRope, queryGm_, query);

        InitKVBuffer(constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize + constInfo_.dSizeRope, kCacheGm_,
                     kCache, constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);

        // 按 tiling 策略禁用 k_cache 的 L2 缓存。
        if (constInfo_.l2CacheOffFlag) {
#ifndef ASCENDC_OOM
            kCacheGm_.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim,
                                       FaGmTensor<Q_T, Q_FORMAT, uint32_t, LAYOUT == FLASH_MLA_LAYOUT::TND>& qGmTensor,
                                       __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                            seqLensTool_.qActSeqLensParser_);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, seqLensTool_.qActSeqLensParser_);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensor<KV_T, KV_FORMAT>& kvGmTensor, __gm__ uint8_t* gm, uint32_t bnStride,
                                        uint32_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32U / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        // 消耗剩余跨核标志，避免影响下一次 kernel 的 Wait。
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_1);
    }

    // 合并搬运 nope 和 rope，避免拆分读取降低 MTE2 效率。
    __aicore__ inline void CopyQueryAndRopeTile(const LocalTensor<Q_T>& dstTensor, FlashMlaRunInfoX& runInfo)
    {
        uint32_t dstStride = ((runInfo.actMSize + 15U) >> 4) << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.n2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = (uint32_t)(constInfo_.dSize + constInfo_.dSizeRope)};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyKeyAndRopeTile(const LocalTensor<KV_T>& dstTensor, FlashMlaRunInfoX& runInfo)
    {
        uint32_t dstStride = ((runInfo.actSingleLoopS2Size + 15U) >> 4) << 4;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = (uint32_t)(constInfo_.dSize + constInfo_.dSizeRope)};
        copyKvGmToL1_(l1Tensor, kCacheGm_, gmCoord);
    }

    __aicore__ inline void PreloadMte2(FlashMlaRunInfoX& runInfo, FlashMlaRunInfoX& nextRunInfo, bool isFirstLoop)
    {
        if (nextRunInfo.isValid) {
            IterateBmm1Load(nextRunInfo);
        }
        if (isFirstLoop && runInfo.isValid) {
            IterateBmm1Load(runInfo);
        }
    }

    __aicore__ inline void IterateBmm1Load(FlashMlaRunInfoX& runInfo)
    {
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        LocalTensor<KV_T> kvpL1Tensor = l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvpL1BufId);
        CopyKeyAndRopeTile(kvpL1Tensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvpL1BufId);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C,
                                      FlashMlaRunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = ((runInfo.actSingleLoopS2Size + 7U) >> 3) << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = ((runInfo.actMSize + 1U) >> 1) << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15U) >> 4) << 4;
        fixpipeParams.dstStride = 128U; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 0;   // One AIV receives all rows on arch92.
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void Mm1LoadL0A(LocalTensor<Q_T>& l0a, const LocalTensor<Q_T>& l1, uint64_t offset,
                                      uint32_t tileK, uint32_t actM)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = false;
        p.mStep = (actM + 15U) >> 4;
        p.kStep = (tileK + 15U) >> 4;
        p.srcStride = p.mStep;
        p.dstStride = p.mStep;
        LoadData(l0a, l1[offset], p);
    }
    __aicore__ inline void Mm1LoadL0B(LocalTensor<KV_T>& l0b, const LocalTensor<KV_T>& l1, uint64_t offset,
                                      uint32_t tileK, uint32_t actS2)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = false;
        p.mStep = (actS2 + 15U) >> 4;
        p.kStep = (tileK + 15U) >> 4;
        p.srcStride = p.mStep;
        p.dstStride = p.mStep;
        LoadData(l0b, l1[offset], p);
    }
    __aicore__ inline void Mm2LoadL0A(LocalTensor<KV_T>& l0a, const LocalTensor<KV_T>& l1, uint32_t actS2,
                                      uint32_t realM)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = false;
        p.mStep = (mBaseSize + 15U) >> 4;
        p.kStep = (actS2 + 15U) >> 4;
        p.srcStride = p.mStep;
        p.mStep = (realM + 15U) >> 4;
        p.dstStride = p.mStep;
        LoadData(l0a, l1[0], p);
    }
    __aicore__ inline void Mm2LoadL0B(LocalTensor<KV_T>& l0b, const LocalTensor<KV_T>& l1, uint64_t offset,
                                      uint32_t tileK, uint32_t actS2)
    {
        constexpr uint32_t nSplit = 256U;
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = true;
        p.mStep = (tileK + 15U) >> 4;
        p.kStep = nSplit >> 4;
        p.srcStride = (actS2 + 15U) >> 4;
        p.dstStride = nSplit >> 4;
        LoadData(l0b, l1[offset], p);
    }

    // Q 首块常驻；Q112/P 共用槽需等 QK 的倒数第二次 Mmad 读取结束。
    __aicore__ inline void Mm1MatmulK(const LocalTensor<Q_T>& qL1, const LocalTensor<KV_T>& kvL1,
                                      const LocalTensor<MM_T>& c, uint32_t actM, uint32_t actS2, bool loadQueryCache)
    {
        const uint32_t am = ((actM + 15U) >> 4) << 4;
        const uint32_t an = ((actS2 + 15U) >> 4) << 4;
        uint32_t koff = 0U;
#pragma unroll
        for (uint32_t k = 0; k < QK_STEPS; ++k) {
            const uint32_t tk = k == 0U ? QK_CACHE_K : ((k & 1U) ? QK_SHORT_K : QK_LONG_K);
            const uint32_t ao = k == 0U ? 0U : ((k & 1U) ? PV_L0A_OFFSET : QK_TEMP_OFFSET);
            const uint32_t l0bMutexId = L0B_BUFFER_ID0 + l0bBufId_;
            auto a = l0ABuffers_[ao].template ReinterpretCast<Q_T>();
            auto b = l0BBuffers_[l0bBufId_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<KV_T>();
            Mutex::Lock<PIPE_MTE1>(l0bMutexId);
            if (k != 0U || loadQueryCache) {
                Mm1LoadL0A(a, qL1, koff * am, tk, actM);
            }
            Mm1LoadL0B(b, kvL1, koff * an, tk, actS2);
            Mutex::Unlock<PIPE_MTE1>(l0bMutexId);
            Mutex::Lock<PIPE_M>(l0bMutexId);
            MmadParams mp;
            mp.m = actM;
            mp.disableGemv = (mp.m == 1U);
            mp.n = actS2;
            mp.k = tk;
            mp.cmatrixInitVal = k == 0U;
            mp.cmatrixSource = false;
            mp.unitFlag = 0;
            Mmad(c, a, b, mp);
            Mutex::Unlock<PIPE_M>(l0bMutexId);
            l0bBufId_ ^= 1U;
            koff += tk;
        }
        lastMatmulWasPv_ = false;
    }

    __aicore__ inline void IterateBmm1(FlashMlaRunInfoX& runInfo)
    {
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);

        LocalTensor<Q_T> qL1Tensor = l1QBuffers_.template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
            CopyQueryAndRopeTile(qL1Tensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0);
        }

        LocalTensor<KV_T> kvpL1Tensor = l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvpL1BufId);
        {
            const uint32_t l0cMutexId = L0C_BUFFER_ID0 + l0cBufId_;
            Mutex::Lock<PIPE_M>(l0cMutexId);
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();

            Mm1MatmulK(qL1Tensor, kvpL1Tensor, l0CSubTensor, (uint32_t)runInfo.actMSize,
                       (uint32_t)runInfo.actSingleLoopS2Size, runInfo.isFirstS2Loop);

            Mutex::Unlock<PIPE_M>(l0cMutexId);
            Mutex::Lock<PIPE_FIX>(l0cMutexId);
            FixpipeMm1(mm1ResUbTensor, l0CSubTensor, runInfo);
            Mutex::Unlock<PIPE_FIX>(l0cMutexId);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0);
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(c1v1CrossCoreSyncIdx);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T& dstTensor, const LocalTensor<T>& l0C, uint32_t realN,
                                              FlashMlaRunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = ((realN + 7U) >> 3) << 3;
        fixpipeParams.mSize = ((runInfo.actMSize + 1U) >> 1) << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15U) >> 4) << 4;
        fixpipeParams.dstStride = 256U;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(FlashMlaRunInfoX& runInfo)
    {
        uint32_t kvpL1BufId = runInfo.loop % L1_KVP_BUFCNT;
        auto kv = l1KvpBuffers_[kvpL1BufId * L1_KVP_BUF_BYTES].template ReinterpretCast<KV_T>();
        // The first B-slot lock also waits for the penultimate QK MMAD, the last
        // reader of Q112 in this shared Q/P slot. The final QK reads Q128 elsewhere.
        auto a = l0ABuffers_[PV_L0A_OFFSET].template ReinterpretCast<KV_T>();
        uint32_t am = (runInfo.actMSize + 15U) / 16U * 16U;
        uint32_t an = (runInfo.actSingleLoopS2Size + 15U) / 16U * 16U;
        uint32_t kl = (runInfo.actSingleLoopS2Size + 63U) / 64U;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P_0 + kvpL1BufId);
        if (lastMatmulWasPv_) {
            l0bBufId_ ^= 1U;
        }
        for (uint32_t n = 0; n < 2U; ++n) {
            uint32_t l0cMutexId = L0C_BUFFER_ID0 + l0cBufId_;
            uint32_t cflag = CROSSCORE_BMM2_0 + n;
            auto c = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            auto ub = ubMm2ResBuffers_[n * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
            Mutex::Lock<PIPE_M>(l0cMutexId);
            for (uint32_t k = 0; k < kl; ++k) {
                uint32_t tk = k + 1U < kl ? 64U : runInfo.actSingleLoopS2Size - k * 64U;
                uint32_t l0bMutexId = L0B_BUFFER_ID0 + l0bBufId_;
                auto b = l0BBuffers_[l0bBufId_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<KV_T>();
                Mutex::Lock<PIPE_MTE1>(l0bMutexId);
                if (n == 0U && k == 0U) {
                    Mm2LoadL0A(a, kv[s2BaseSize * dVBaseSize], runInfo.actSingleLoopS2Size, runInfo.actMSize);
                }
                Mm2LoadL0B(b, kv, n * 256U * an + k * 64U * 16U, tk, runInfo.actSingleLoopS2Size);
                Mutex::Unlock<PIPE_MTE1>(l0bMutexId);
                Mutex::Lock<PIPE_M>(l0bMutexId);
                MmadParams mp;
                mp.m = runInfo.actMSize;
                mp.disableGemv = (mp.m == 1U);
                mp.n = 256U;
                mp.k = tk;
                mp.cmatrixInitVal = k == 0U;
                mp.cmatrixSource = false;
                mp.unitFlag = 0;
                Mmad(c, a[k * 64U * am], b, mp);
                Mutex::Unlock<PIPE_M>(l0bMutexId);
                l0bBufId_ ^= 1U;
            }
            Mutex::Unlock<PIPE_M>(l0cMutexId);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(cflag);
            Mutex::Lock<PIPE_FIX>(l0cMutexId);
            FixpipeMm2PartialN(ub, c, 256U, runInfo);
            Mutex::Unlock<PIPE_FIX>(l0cMutexId);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(cflag);
            l0cBufId_ ^= 1U;
        }
        lastMatmulWasPv_ = true;
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvpL1BufId);
    }

}; // FlashMlaNoQuantMlaBlockCube

template <typename FA_T>
class FlashMlaNoQuantMlaBlockCubeDummy {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FLASH_MLA_LAYOUT LAYOUT = FA_T::qLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using ConstInfoX = ConstInfo_t<FlashMlaKernelType::NO_QUANT>;
    template <typename FlashMlaSeqLensToolT>
    __aicore__ inline FlashMlaNoQuantMlaBlockCubeDummy(ConstInfoX& constInfo, FlashMlaSeqLensToolT& seqLensTool){};
};

} // namespace FlashMlaKernel

#endif // FLASH_MLA_BLOCK_CUBE_ARCH92_H_

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
 * \file quant_flash_mla_block_cube_fp8.h
 * \brief QuantFlashMlaWithKvcache cube block（自FIA MLA block_cube裁剪适配, rope方案A: 576维含rope布局）
 *        静态张量编程: 所有buffer地址为constexpr偏移, 使用Mutex核内同步 + CrossCore核间同步
 */

#ifndef QUANT_FLASH_MLA_BLOCK_CUBE_FP8_H_
#define QUANT_FLASH_MLA_BLOCK_CUBE_FP8_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"

#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "quant_flash_mla_with_kvcache_public_def.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

/* ============静态张量编程所需的宏与常量============= */
#ifndef OFFSET_OF_MEMBER
#define OFFSET_OF_MEMBER(TYPE, MEMBER) ((uint64_t) & ((TYPE*)0)->MEMBER)
#endif
#ifndef SIZE_OF_MEMBER
#define SIZE_OF_MEMBER(TYPE, MEMBER) sizeof(((TYPE*)0)->MEMBER)
#endif
#ifndef BUFFER_SIZE_BYTE_64K
#define BUFFER_SIZE_BYTE_64K 65536U
#endif
#ifndef BUFFER_SIZE_BYTE_32K
#define BUFFER_SIZE_BYTE_32K 32768U
#endif

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false>
class QuantFlashMlaBlockCubeFp8 {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = true; // MLA固定PA场景
    // rope方案A: q/k_cache的D维=576(nope 512 + rope 64)拼接, bmm1单次576维matmul
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;

    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr TPosition bmm2OutPos =
        GetC2Position(dVTemplateType,
                      UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, PseTypeEnum::PSE_NONE_TYPE, false,
                                              false, false, mBaseSize == 64),
                      (s2BaseSize == 256 && mBaseSize == 64), true);
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, BMM2_TOUB};

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    // 新接口KvLayoutType编码: 0=PA_BBND 1=PA_BNBD 2=PA_NZ
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MLA_FULLQUANT_MM2_T = std::conditional_t<isInt8, int32_t, T>;

    // Q: TND, cu_seq(含前导0)+seq_used, int32
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;
    // KV: PA, cache_seqlens按batch, int32
    using KvSeqParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;

    static constexpr bool Q_NEEDS_WZH = true;
    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t>;

    using ConstInfoX = QmlaConstInfo;

    /* =====================核间同步ID==================== */
    // Cross-core sync (CROSS_CORE_SYNC_MODE = 4, AIV0_AIV1_OFFSET = 16)
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_MM_0 = 0U;  // bmm1 result slot 0
    static constexpr uint32_t CROSSCORE_MM_1 = 1U;  // bmm1 result slot 1
    static constexpr uint32_t CROSSCORE_MM_2 = 2U;  // bmm2 result slot 0
    static constexpr uint32_t CROSSCORE_MM_3 = 3U;  // bmm2 result slot 1
    static constexpr uint32_t CROSSCORE_L1P_0 = 4U; // L1 P slot 0 (AIV→AIC)
    static constexpr uint32_t CROSSCORE_L1P_1 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 6U;

    // Cube internal Mutex IDs
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

    /* =====================Buffer尺寸常量==================== */
    // UB (shared between AIC cube and AIV vec)
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);

    // L1 (shared between AIC cube and AIV vec)
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = 1U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_KV_BUFCNT = 3U;
    static constexpr uint32_t L1_KV_BUF_BYTES = dBaseSize * s2BaseSize * sizeof(INPUT_T);

    // L0C (static)
    static constexpr uint32_t L0C_BUFCNT = 2U;
    static constexpr uint32_t L0C_BUF_BYTES = 128U * 1024U;

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<float> deScaleQGm_;
    GlobalTensor<float> deScaleKGm_;
    GlobalTensor<float> deScaleVGm_;

    QSeqParserType* qSeqParserPtr_ = nullptr;
    KvSeqParserType* kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;

    __gm__ uint8_t* keyPtr_ = nullptr;
    __gm__ uint8_t* valuePtr_ = nullptr;

    const ConstInfoX& constInfo_;

    /* =====================静态LocalTensor变量==================== */
    // UB (shared with vec, same offsets)
    LocalTensor<uint8_t> ubMmResBuffers_;
    // L1 (shared with vec, same offsets)
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    // L0C (static)
    LocalTensor<uint8_t> l0CBuffers_;

    // Buffer ID trackers
    uint32_t qL1BufId_ = 0U;
    uint32_t l0cBufId_ = 0U;
    uint32_t mmResUbBufId_ = 0U; // bmm1 result UB slot

    // L0A: resident Q (36 KiB), then P (16 KiB). L0B: two 32 KiB slots.
    LocalTensor<uint8_t> l0ABuffers_;
    LocalTensor<uint8_t> l0BBuffers_;
    uint32_t l0ABSlot_ = 0U;

    __aicore__ inline QuantFlashMlaBlockCubeFp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                         __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue,
                                         QSeqParserType& qParser, KvSeqParserType& kvParser)
    {
        qSeqParserPtr_ = &qParser;
        kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------UB--------------------------------------------*/
        struct UbLayout {
            uint8_t bmm1Res[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES];
            uint8_t bmm2Res[UB_MM2_RES_BUFCNT][UB_MM2_RES_BUF_BYTES];
        };
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, 0, sizeof(UbLayout));

        /*--------------------------------------------L1--------------------------------------------*/
        struct L1Layout {
            uint8_t qBuffers[L1_Q_BUFCNT][L1_Q_BUF_BYTES];
            uint8_t kvBuffers[L1_KV_BUFCNT][L1_KV_BUF_BYTES];
        };
        static_assert(sizeof(L1Layout) <= 512 * 1024, "L1 buffer too large");
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, OFFSET_OF_MEMBER(L1Layout, kvBuffers),
                                           SIZE_OF_MEMBER(L1Layout, kvBuffers));
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, OFFSET_OF_MEMBER(L1Layout, qBuffers),
                                           SIZE_OF_MEMBER(L1Layout, qBuffers));
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, OFFSET_OF_MEMBER(L1Layout, kvBuffers),
                                            SIZE_OF_MEMBER(L1Layout, kvBuffers));

        /*--------------------------------------------L0A--------------------------------------------*/
        l0ABuffers_ = LocalTensor<uint8_t>(TPosition::A2, 0U, BUFFER_SIZE_BYTE_64K);

        /*--------------------------------------------L0B--------------------------------------------*/
        l0BBuffers_ = LocalTensor<uint8_t>(TPosition::B2, 0U, BUFFER_SIZE_BYTE_64K);

        /*--------------------------------------------L0C--------------------------------------------*/
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                         __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue)
    {
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);

        // rope方案A: q的D维=576(nope 512+rope 64拼接), 整维拷贝
        InitQBuffer(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSize, queryGm_, query);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize, keyGm_, key,
                     constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize, valueGm_, value,
                     constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);

        // MLA全量化dequantScale: Q per-token-head, KV per-tensor
        if (dequantScaleQuery != nullptr) {
            deScaleQGm_.SetGlobalBuffer((__gm__ float*)dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            deScaleKGm_.SetGlobalBuffer((__gm__ float*)dequantScaleKey);
        }
        if (dequantScaleValue != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float*)dequantScaleValue);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t n2Size, uint32_t gSize, uint32_t headDim, FaGmTensorQ& qGmTensor,
                                       __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
    }

    __aicore__ inline void InitKVBuffer(uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensorKV& kvGmTensor, __gm__ uint8_t* gm, uint64_t bnStrides,
                                        uint64_t n2Strides)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else { // GM_KV_PA_BBND
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void AllocEventID() {}
    __aicore__ inline void FreeEventID() {}

    // FP8 NZ uses 32-byte K blocks and 32-row source alignment.
    __aicore__ inline void LoadOperand(const LocalTensor<INPUT_T>& dst, const LocalTensor<INPUT_T>& src, uint32_t rows,
                                       uint32_t cols, bool transpose, uint32_t srcRows = 0U)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.ifTranspose = transpose;
        p.mStep = (rows + 15U) / 16U;
        p.kStep = (cols + 31U) / 32U;
        p.srcStride = ((srcRows == 0U ? rows : srcRows) + 31U) / 32U * 2U;
        p.dstStride = transpose ? (cols + 15U) / 16U : p.mStep;
        LoadData(dst, src, p);
    }

    // Full default tile: interleave two independent accumulators across K.
    __aicore__ inline void Mm1FullTile(const LocalTensor<INPUT_T>& qCache, const LocalTensor<INPUT_T>& kv,
                                       const LocalTensor<T>& c)
    {
#pragma unroll
        for (uint32_t k = 0; k < 3U; ++k) {
#pragma unroll
            for (uint32_t ni = 0; ni < 2U; ++ni) {
                const uint32_t id = L0B_BUFFER_ID0 + l0ABSlot_;
                auto a = qCache[k * 192U * 64U];
                auto b = l0BBuffers_[l0ABSlot_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<INPUT_T>();
                Mutex::Lock<PIPE_MTE1>(id);
                LoadOperand(b, kv[k * 192U * 256U + ni * 128U * 32U], 128U, 192U, false, 256U);
                Mutex::Unlock<PIPE_MTE1>(id);
                Mutex::Lock<PIPE_M>(id);
                MmadParams mp;
                mp.m = 64U;
                mp.n = 128U;
                mp.k = 192U;
                mp.cmatrixInitVal = k == 0U;
                mp.cmatrixSource = false;
                mp.unitFlag = 0;
                Mmad(c[ni * 128U * 64U], a, b, mp);
                Mutex::Unlock<PIPE_M>(id);
                l0ABSlot_ ^= 1U;
            }
        }
    }

    // QK: three equal K=192 blocks, Q resident across the full S2 loop.
    // MTE1 fills the alternate B slot while Cube consumes the current slot.
    __aicore__ inline void Mm1MatmulK(const LocalTensor<INPUT_T>& q, const LocalTensor<INPUT_T>& kv,
                                      const LocalTensor<T>& c, uint32_t m, uint32_t n, bool first, bool last)
    {
        static_assert(mBaseSize * 192U <= BUFFER_SIZE_BYTE_32K);
        static_assert(128U * 192U <= BUFFER_SIZE_BYTE_32K);
        auto qCache = l0ABuffers_.template ReinterpretCast<INPUT_T>();
        if (first) {
            Mutex::Lock<PIPE_MTE1>(L0A_BUFFER_ID0);
            LoadOperand(qCache, q, m, 576U, false);
            Mutex::Unlock<PIPE_MTE1>(L0A_BUFFER_ID0);
            Mutex::Lock<PIPE_M>(L0A_BUFFER_ID0);
        }
        if (likely(m == 64U && n == 256U)) {
            Mm1FullTile(qCache, kv, c);
        } else {
            for (uint32_t ni = 0; ni < (n + 127U) / 128U; ++ni) {
                const uint32_t tileN = n - ni * 128U > 128U ? 128U : n - ni * 128U;
#pragma unroll
                for (uint32_t k = 0; k < 3U; ++k) {
                    const uint32_t id = L0B_BUFFER_ID0 + l0ABSlot_;
                    auto a = qCache[k * 192U * ((m + 15U) / 16U * 16U)];
                    auto b = l0BBuffers_[l0ABSlot_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<INPUT_T>();
                    Mutex::Lock<PIPE_MTE1>(id);
                    LoadOperand(b, kv[k * 192U * ((n + 31U) / 32U * 32U) + ni * 128U * 32U], tileN, 192U, false, n);
                    Mutex::Unlock<PIPE_MTE1>(id);
                    Mutex::Lock<PIPE_M>(id);
                    MmadParams mp;
                    mp.m = m == 1U ? 16U : m;
                    mp.n = tileN;
                    mp.k = 192U;
                    mp.cmatrixInitVal = k == 0U;
                    mp.cmatrixSource = false;
                    mp.unitFlag = 0;
                    Mmad(c[ni * 128U * ((m + 15U) / 16U * 16U)], a, b, mp);
                    Mutex::Unlock<PIPE_M>(id);
                    l0ABSlot_ ^= 1U;
                }
            }
        }
        if (last) {
            Mutex::Unlock<PIPE_M>(L0A_BUFFER_ID0);
        }
    }

    // PV: four N=128 blocks at K=256 fill the 32 KiB B slots.
    // P is loaded once and protected across all four Mmad instructions.
    __aicore__ inline void Mm2MatmulN(const LocalTensor<INPUT_T>& p, const LocalTensor<INPUT_T>& v,
                                      const LocalTensor<MLA_FULLQUANT_MM2_T>& c, uint32_t k)
    {
        auto a = l0ABuffers_[mBaseSize * 576U].template ReinterpretCast<INPUT_T>();
        Mutex::Lock<PIPE_MTE1>(L0A_BUFFER_ID1);
        LoadOperand(a, p, mBaseSize, k, false);
        Mutex::Unlock<PIPE_MTE1>(L0A_BUFFER_ID1);
        Mutex::Lock<PIPE_M>(L0A_BUFFER_ID1);
#pragma unroll
        for (uint32_t n = 0; n < 4U; ++n) {
            const uint32_t id = L0B_BUFFER_ID0 + l0ABSlot_;
            auto b = l0BBuffers_[l0ABSlot_ * BUFFER_SIZE_BYTE_32K].template ReinterpretCast<INPUT_T>();
            Mutex::Lock<PIPE_MTE1>(id);
            LoadOperand(b, v[n * 128U * ((k + 31U) / 32U * 32U)], (k + 31U) / 32U * 32U, 128U, true);
            Mutex::Unlock<PIPE_MTE1>(id);
            Mutex::Lock<PIPE_M>(id);
            MmadParams mp;
            mp.m = mBaseSize;
            mp.n = 128U;
            mp.k = k;
            mp.cmatrixInitVal = true;
            mp.cmatrixSource = false;
            mp.unitFlag = 0;
            Mmad(c[n * mBaseSize * 128U], a, b, mp);
            Mutex::Unlock<PIPE_M>(id);
            l0ABSlot_ ^= 1U;
        }
        Mutex::Unlock<PIPE_M>(L0A_BUFFER_ID1);
    }

    // rope方案A: 整维拷贝q(含rope, D=576)到L1 NZ布局
    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = constInfo_.dSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    // rope方案A: 整维拷贝k(含rope, D=576)到L1 NZ布局
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo, uint32_t s2RealSize)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0,
                          .s2DealSize = s2RealSize,
                          .dDealSize = constInfo_.dSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // MLA中V = K_nope, bmm2复用K的L1 buffer, 无需独立搬运V

    // 提前一轮发起下一任务的K搬运(仅MTE2、不等落地): MAC算本轮(L)与上轮(L-1)时, MTE2后台搬L+1,
    // 避免bmm1现场等MTE2断流; buffer轮转取task.loop % L1_KV_BUFCNT
    __aicore__ inline void IterateBmm1Load(RunInfoX& runInfo)
    {
        if (!runInfo.isValid || runInfo.keyPrefetched) {
            return;
        }
        uint32_t kvBufId = runInfo.loop % L1_KV_BUFCNT;
        runInfo.kvL1BufId = kvBufId;
        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvBufId * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvBufId);
        CopyKeyTile(kL1Tensor, runInfo, runInfo.actSingleLoopS2Size);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvBufId);
        runInfo.keyPrefetched = true;
    }

    // Q预加载: 仅MTE2发起Q(含rope, D=576)拷贝到L1 NZ布局
    // 启动首任务(loop==0): 与首任务K同拍发出, 压缩启动期;
    // 换Q块边界: 在上块最后一次bmm1释放MTE1(Q)锁并轮转qL1BufId_之后调用, 预取写新槽与在途读无冲突
    __aicore__ inline void IterateQPreload(RunInfoX& runInfo)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
        CopyQueryTile(qL1Tensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
        runInfo.qPreloaded = true;
    }

    __aicore__ inline void IterateBmm1(RunInfoX& runInfo, bool qPreloaded = false)
    {
        uint32_t mmResUbBufId = mmResUbBufId_;
        mmResUbBufId_ = (mmResUbBufId_ + 1) % UB_MM1_RES_BUFCNT;
        LocalTensor<T> mm1ResUbTensor =
            ubMmResBuffers_[mmResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;

        // 等待AIV释放UB bmm1 result slot
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);

        IterateBmm1Internal(mm1ResUbTensor, runInfo, qPreloaded);

        // 通知AIV bmm1 result已就绪
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    // qPreloaded: Q已由IterateQPreload提前发起MTE2, 跳过重复拷贝, 仅锁MTE1
    __aicore__ inline void IterateBmm1Internal(LocalTensor<T>& mm1ResUbTensor, RunInfoX& runInfo, bool qPreloaded)
    {
        // MLA全量化bmm1(rope方案A): Q[64,576] @ K[576,256]单次matmul
        // Q(含rope)首次全载L1，后续s2循环复用
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            if (!qPreloaded) {
                Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
                CopyQueryTile(qL1Tensor, runInfo);
                Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            }
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        // 加载当前轮的K(含rope)到L1: 已由IterateBmm1Load预取则直接锁MTE1使用, 否则现场加载
        uint32_t kvBufId = runInfo.loop % L1_KV_BUFCNT;
        runInfo.kvL1BufId = kvBufId;
        uint32_t s2CurSize = runInfo.actSingleLoopS2Size;
        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvBufId * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        if (!runInfo.keyPrefetched) {
            Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvBufId);
            CopyKeyTile(kL1Tensor, runInfo, s2CurSize);
            Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvBufId);
        }
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvBufId);
        {
            // MatMul: Q[64,576] @ K[576,256], K维含rope(方案A单次matmul)
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
            Mm1MatmulK(qL1Tensor, kL1Tensor, l0CSubTensor, runInfo.actMSize, s2CurSize, runInfo.isFirstS2Loop,
                       runInfo.isLastS2Loop);

            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            FixpipeMm1(mm1ResUbTensor, l0CSubTensor, runInfo, s2CurSize);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        // 释放K的MTE1锁, 允许bmm2复用读取
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvBufId);

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(RunInfoX& runInfo)
    {
        // MLA全量化bmm2: P @ V_nope, V复用K的nope部分(MLA中V=K_nope)

        // bmm2 result UB双buffer轮转: 写slot n时vec并行消费slot n^1
        uint32_t mm2ResUbBufId = 0U;
        if constexpr (BMM2_TOUB) {
            mm2ResUbBufId = runInfo.loop % UB_MM2_RES_BUFCNT;
        }
        uint32_t mm2SyncIdx = CROSSCORE_MM_2 + mm2ResUbBufId;

        // a. 等待AIV vec1通知L1 P就绪
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
        LocalTensor<INPUT_T> pL1Tensor =
            l1PBuffers_[pL1BufId * L1_KV_BUF_BYTES + 512U * s2BaseSize].template ReinterpretCast<INPUT_T>();

        // b. 如果bmm2写UB: 等待AIV vec2释放UB bmm2 result对应slot
        if constexpr (BMM2_TOUB) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mm2SyncIdx);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mm2SyncIdx + AIV0_AIV1_OFFSET);
        }

        // c. K复用: MLA中V = K_nope, bmm2复用bmm1加载的K buffer
        //    直接使用bmm1加载/预取时记录的buffer id, 不做轮转推算
        uint32_t kvReuseBufId = runInfo.kvL1BufId;
        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvReuseBufId * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvReuseBufId);

        // d. L0C + Matmul
        Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
        LocalTensor<MLA_FULLQUANT_MM2_T> l0CSubTensor =
            l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MLA_FULLQUANT_MM2_T>();
        Mm2MatmulN(pL1Tensor, vL1Tensor, l0CSubTensor, runInfo.actSingleLoopS2Size);

        Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
        Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

        // e. 释放K的MTE1锁, 允许下一轮bmm1复用加载K
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvReuseBufId);

        // f. Fixpipe: 写bmm2 result到UB对应slot
        LocalTensor<MLA_FULLQUANT_MM2_T> mm2ResUbTensor =
            ubMmResBuffers_[UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES + mm2ResUbBufId * UB_MM2_RES_BUF_BYTES]
                .template ReinterpretCast<MLA_FULLQUANT_MM2_T>();
        FixpipeMm2(mm2ResUbTensor, l0CSubTensor, runInfo);

        // g. 释放L0C
        Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
        l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;

        // h. 通知AIV bmm2 result对应slot已就绪
        if constexpr (BMM2_TOUB) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mm2SyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mm2SyncIdx + AIV0_AIV1_OFFSET);
        }
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T& dstTensor, const LocalTensor<MLA_FULLQUANT_MM2_T>& l0C,
                                      RunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.nSize = ((uint32_t)constInfo_.dSizeV + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = constInfo_.dSizeV;
        }
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.dstStride = ((uint32_t)dVBaseSize + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo_.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MLA_FULLQUANT_MM2_T, MLA_FULLQUANT_MM2_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }
};

// AIV编译单元使用的Cube空壳: kernel在AIC分支外不会实例化Cube接口, 仅需类型/常量成员
template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false>
class QuantFlashMlaBlockCubeFp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = true;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;

    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MLA_FULLQUANT_MM2_T = std::conditional_t<isInt8, int32_t, T>;
    using ConstInfoX = QmlaConstInfo;
    __aicore__ inline QuantFlashMlaBlockCubeFp8Dummy(ConstInfoX& constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_BLOCK_CUBE_FP8_H_

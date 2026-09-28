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
 * \file quant_flash_attn_block_cube_mxfp8_nd.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_ND_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_ND_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../arch35/memory_copy_arch35_quant_flash_attn.h"

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
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool isDAligned = true>
class QuantFlashAttnBlockCubeMxfp8Nd {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t s2SplitSize = s2BaseSize >> 1;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    static constexpr bool IS_D256 = (dBaseSize == static_cast<uint32_t>(DTemplateType::Aligned256));
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;
    static constexpr InnerMLayout Q_M_LAYOUT = InnerMLayout::GS1_MERGE_LAYOUT;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = T;

    /* ============确定key scale的类型============= */
    template <uint8_t kvLayoutType>
    struct KeyScaleGmToL1Sel {
        using Type = std::conditional_t<(kvLayoutType == 3), // 3: PA_NZ
                                        CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::NO_TRANS>,
                                        CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::DN2NZ>>;
    };

    /* ============确定value scale的类型============= */
    template <uint8_t kvLayoutType>
    struct ValueScaleGmToL1Sel {
        using Type =
            std::conditional_t<(kvLayoutType == 3), // 3: PA_NZ
                               CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::NO_TRANS>,
                               CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::ND2NZ>>;
    };

    using L1KvType = BuffersPolicy4buff<BufferType::L1>;
    using KeyScaleGmToL1Type = typename KeyScaleGmToL1Sel<KvLayoutType>::Type;
    using ValueScaleGmToL1Type = typename ValueScaleGmToL1Sel<KvLayoutType>::Type;
    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
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
    using FaGmTensorQScale = FaGmTensor<SCALE_T, Q_SCALE_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKScale = FaGmTensor<SCALE_T, K_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorVScale = FaGmTensor<SCALE_T, V_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    FaGmTensorVScale valueScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqParserType *qSeqParserPtr_ = nullptr;
    KvSeqParserType *kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1_;
    KeyScaleGmToL1Type copyKeyScaleGmToL1_;
    ValueScaleGmToL1Type copyValueScaleGmToL1_;

    /* =====================LocalBuffer变量====================*/
    static constexpr uint32_t L1_P_SIZE = mBaseSize * s2BaseSize;
    static constexpr uint32_t L1_P_SCALE_SIZE = mBaseSize * s2BaseSize / MXFP_GROUP_SIZE;
    static constexpr uint32_t L1_P_BUFCNT = 3;

    static constexpr uint32_t L1_Q_SIZE = mBaseSize * dBaseSize;
    static constexpr uint32_t L1_Q_SCALE_SIZE = mBaseSize * dBaseSize / MXFP_GROUP_SIZE;
    static constexpr uint32_t L1_Q_BUFCNT = 2;
    // static constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
    static constexpr uint32_t L1_KV_SIZE = s2SplitSize * dBaseSize;
    static constexpr uint32_t L1_KV_SCALE_SIZE = s2SplitSize * dBaseSize / MXFP_GROUP_SIZE;
    // KV_EVENT 数量联动：dim=256 仅用 KV_EVENT0~3（EVENT_ID2~5），Set/Wait 配对周期 = L1_KV_BUFCNT
    static constexpr uint32_t L1_KV_BUFCNT = IS_D256 ? 4U : 6U;
    static constexpr uint64_t L0A_SIZE = 64;
    static constexpr uint64_t L0B_SIZE = 64;
    static constexpr uint32_t QK_L0A_SIZE = 128 * 256;
    static constexpr uint32_t QK_L0B_SIZE = 256 * 128;
    static constexpr uint32_t QK_L0C_SIZE = 128 * 256;
    static constexpr uint32_t PV_L0A_SIZE = 128 * 256;
    static constexpr uint32_t PV_L0B_SIZE = 256 * 128;
    static constexpr uint32_t PV_L0C_SIZE = 128 * 256;
    static constexpr uint32_t L0AB_BUFCNT = 2;
    static constexpr uint32_t L0C_BUFCNT = 2;

    static constexpr uint32_t UB_MM1_SIZE = mBaseSize / CV_RATIO * s2SplitSize;

    LocalTensor<INPUT_T> qL1Tensor;
    LocalTensor<fp8_e8m0_t> qScaleL1Tensor;
    LocalTensor<INPUT_T> kvL1Tensor;
    LocalTensor<fp8_e8m0_t> kvScaleL1Tensor;
    LocalTensor<INPUT_T> aL0Tensor;
    LocalTensor<INPUT_T> bL0Tensor;
    LocalTensor<float> cL0Tensor;

    // =================================Event&Buffer ID===========================
    // mte2 <> mte1 EventID
    static constexpr uint32_t Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID1;
    int qBufId = 0;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID2;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID3;
    static constexpr uint32_t KV_EVENT2 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT3 = EVENT_ID5;
    static constexpr uint32_t KV_EVENT4 = EVENT_ID6;
    static constexpr uint32_t KV_EVENT5 = EVENT_ID7;
    int kvBufId = 0;

    // mte1 <> mmad EventID
    static constexpr uint32_t QK_L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t QK_L0AB_EVENT1 = EVENT_ID4;
    int qkL0abBufId = 0;
    // uint32_t qkL0abBufId = 0;

    // mmad <> fixpipe EventID
    static constexpr uint32_t QK_L0C_EVENT0 = EVENT_ID2;
    static constexpr uint32_t QK_L0C_EVENT1 = EVENT_ID3;
    int qkL0cBufId = 0;

    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {7, 9};
    static constexpr uint16_t CROSS_CORE_SYNC_C1_V1[2] = {6, 8};
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_P_C2[3] = {1, 2, 3};

    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    const ConstInfoX &constInfo_;

    /*============================================================================== */
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8Nd(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery,
                                         __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue,
                                         QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT3);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT4);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT5);

        SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT1);

        SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT0);
        SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT3);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT4);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT5);

        WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT1);

        WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void InitBuffers()
    {
        // L1
        uint32_t addrL1Start =
            L1_P_SIZE * L1_P_BUFCNT * sizeof(fp8_e4m3fn_t) + L1_P_SCALE_SIZE * L1_P_BUFCNT * sizeof(SCALE_T);
        qL1Tensor = LocalTensor<Q_T>(TPosition::A1, addrL1Start, L1_Q_SIZE * L1_Q_BUFCNT); // 8K * 2 = 16K

        addrL1Start += L1_Q_SIZE * L1_Q_BUFCNT * sizeof(Q_T);
        qScaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_Q_SCALE_SIZE * L1_Q_BUFCNT); // 0.5K * 2 = 1K

        addrL1Start += L1_Q_SCALE_SIZE * L1_Q_BUFCNT * sizeof(SCALE_T);
        kvL1Tensor = LocalTensor<KV_T>(TPosition::A1, addrL1Start,
                                       L1_KV_SIZE * L1_KV_BUFCNT); // dim=128: 32K*6=192K; dim=256: 64K*4=256K

        addrL1Start += L1_KV_SIZE * L1_KV_BUFCNT * sizeof(KV_T);
        kvScaleL1Tensor = LocalTensor<SCALE_T>(TPosition::A1, addrL1Start,
                                               L1_KV_SCALE_SIZE * L1_KV_BUFCNT); // dim=128: 1K*6=6K; dim=256: 2K*4=8K

        // L0A
        uint32_t addrL0AStart = 0;
        aL0Tensor = LocalTensor<INPUT_T>(TPosition::A2, addrL0AStart, 128 * 512);

        // L0B
        uint32_t addrL0BStart = 0;
        bL0Tensor = LocalTensor<INPUT_T>(TPosition::B2, addrL0BStart, 128 * 512);

        // L0C
        uint32_t addrL0CStart = 0;
        cL0Tensor = LocalTensor<float>(TPosition::CO1, addrL0CStart, 128 * 512);
    }

    __aicore__ inline void ReleaseTensors()
    {
        FreeEventID();
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery,
                                         __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);

        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         dBaseSize / MXFP_GROUP_SIZE, queryScaleGm_, dequantScaleQuery);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize,
                         dBaseSize / MXFP_GROUP_SIZE, keyScaleGm_, dequantScaleKey);
        InitVScaleBuffer(constInfo_.bSize, constInfo_.s2Size / MXFP_DIVISOR_SIZE, constInfo_.n2Size,
                         constInfo_.blockSize / MXFP_DIVISOR_SIZE, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE,
                         valueScaleGm_, dequantScaleValue);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensorQScale &qScaleGmTensor, __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm, uint64_t bnStride, uint64_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (KV_NEEDS_WZH) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorKScale &kScaleGmTensor, __gm__ uint8_t *gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                                 constInfo_.maxBlockNumPerBatch, constInfo_.kDescaleStrides.bnStride,
                                                 constInfo_.kDescaleStrides.n2Stride);
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_K_SCALE_PA_NZ) {
            uint32_t bs0 = 32 / sizeof(KV_T);
            uint32_t kvCacheBlockSize1 = kvCacheBlockSize / bs0 * MXFP_MULTI_BASE_SIZE;
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize1, headDim / MXFP_MULTI_BASE_SIZE, bs0,
                                                 blockTableGm_, constInfo_.maxBlockNumPerBatch,
                                                 constInfo_.kDescaleStrides.bnStride,
                                                 constInfo_.kDescaleStrides.n2Stride);
        } else if constexpr (KV_NEEDS_WZH) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorVScale &vScaleGmTensor, __gm__ uint8_t *gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                                 constInfo_.maxBlockNumPerBatch, constInfo_.vDescaleStrides.bnStride,
                                                 constInfo_.vDescaleStrides.n2Stride);
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                                 constInfo_.maxBlockNumPerBatch, constInfo_.vDescaleStrides.bnStride,
                                                 constInfo_.vDescaleStrides.n2Stride);
        } else if constexpr (KV_NEEDS_WZH) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            vScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
        }
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint64_t queryL1BaseOffset = qBufId * (L1_Q_SIZE / sizeof(Q_T));
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = qL1Tensor[queryL1BaseOffset], .rowCount = dstStride};
        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = dOffset,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = nopeDealSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryTile(RunInfoX &runInfo)
    {
        CopyQuerySlice(0, constInfo_.dSize, runInfo);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQueryScaleSlice(uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t offset = qBufId * (L1_Q_SCALE_SIZE / sizeof(SCALE_T));
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = qScaleL1Tensor[offset], .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = dOffset,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dRealSize};
        copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(RunInfoX &runInfo)
    {
        // GM 上 Q scale 按对齐档组数存（dim=72→128/32=4 组），拷贝量须用 dBaseSize
        CopyQueryScaleSlice(0, dBaseSize / MXFP_GROUP_SIZE, runInfo);
    }

    // dim 非对齐时按 NZ 布局整块清零 L1 [k, n]（k×n 字节），随后 GM 拷贝仅覆盖真实 dim 列，保证非对齐列为 0
    // 模板化以同时支持 KV 数据区（KV_T）与 V scale 独立 buffer（SCALE_T）
    template <typename DT>
    __aicore__ inline void InitValueL1BufferNAxis(const LocalTensor<DT> &valueL1, const uint32_t k, const uint32_t n)
    {
        InitConstValueParams<half> initConstValueParams;
        initConstValueParams.repeatTimes = n / 32U;
        initConstValueParams.blockNum = k;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>(), initConstValueParams);
    }

    __aicore__ inline void InitValueL1BufferNoTrans(const LocalTensor<KV_T> &valueL1, const uint32_t realK,
                                                    const uint32_t curN)
    {
        InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        uint32_t curPadK = s2SplitSize;
        // nd2nz pading to k 64 align, (k,n)->(n1,k1,k0,n0)
        uint32_t movK = realK;
        if (curPadK == movK) {
            return;
        }
        offset = movK * 16U;
        initConstValueParams.repeatTimes = AttentionCommon::Align(curN, 16U) / 32U;
        initConstValueParams.blockNum = curPadK - movK;
        initConstValueParams.dstGap = movK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[offset], initConstValueParams);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(uint32_t s2Offset, uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfoX &runInfo)
    {
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(KV_T));
        constexpr uint32_t blockNumDtype = 32 / sizeof(KV_T);
        uint32_t dstStride = s2SplitSize;

        uint32_t nopeDealSize = dRealSize;

        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = kvL1Tensor[l1BaseOffset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(RunInfoX &runInfo, uint32_t s2RealSize, uint32_t subLoop)
    {
        uint32_t s2Offset = runInfo.s2Idx + subLoop * s2SplitSize;
        CopyKeySlice(s2Offset, s2RealSize, 0, constInfo_.dSize, runInfo);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKeyScaleSlice(uint32_t s2Offset, uint32_t s2RealSize, uint32_t dOffset,
                                             uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t offset = kvBufId * (L1_KV_SCALE_SIZE / sizeof(SCALE_T));
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = kvScaleL1Tensor[offset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKeyScaleGmToL1_(l1Tensor, keyScaleGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyScaleTile(RunInfoX &runInfo, uint32_t s2RealSize, uint32_t subLoop)
    {
        uint32_t s2Offset = runInfo.s2Idx + subLoop * s2SplitSize;
        // K scale GM 按对齐档组数存（dim=72→128/32=4 组），拷贝量须用 dBaseSize/32（对齐档，含 pad 区）
        CopyKeyScaleSlice(s2Offset, s2RealSize, 0, dBaseSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueSlice(uint32_t s2Offset, uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        uint32_t offset = kvBufId * L1_KV_SIZE;
        uint32_t dstStride = s2SplitSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = kvL1Tensor[offset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size, 0, constInfo_.dSizeV, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueScaleSlice(uint32_t s2Offset, uint32_t s2RealSize, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t offset = kvBufId * L1_KV_SCALE_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = kvScaleL1Tensor[offset],
                                                   .rowCount = s2SplitSize / MXFP_DIVISOR_SIZE};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyValueScaleGmToL1_(l1Tensor, valueScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyValueScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueScaleSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size / MXFP_DIVISOR_SIZE, 0,
                            constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr_));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr_, bIdx);
        keyGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSize);
        keyGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr_));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr_, bIdx);
        valueGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSizeV);
        valueGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }
    __aicore__ inline void LoadDataMxToL0AMm1(RunInfoX &runInfo)
    {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = false;
        loadData2DParamsA.mStep = ((runInfo.actMSize + 15) >> 4 << 4) >> 4;
        // kStep 须按对齐档（reduce 轴 = dBaseSize，dim=72 时按 128 档搬运，pad 列已清零）
        loadData2DParamsA.kStep = dBaseSize >> 5;
        loadData2DParamsA.srcStride = ((runInfo.actMSize + 31) >> 5 << 5) >> 4;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = ((runInfo.actMSize + 15) >> 4 << 4) >> 4;
        // yStep 同 kStep 基准：scale 沿 reduce 轴按对齐档全量加载（dim=72 时 yStep=2 与真实 dim
        // 恰同，取对齐档保持一致）
        loadData2DMxParamsA.yStep = (dBaseSize + 63) >> 5 >> 1;
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        uint32_t L0aOffset = qkL0abBufId * QK_L0A_SIZE;
        uint32_t qL1Offset = qBufId * L1_Q_SIZE;
        uint32_t qScaleL1Offset = qBufId * L1_Q_SCALE_SIZE;
        LoadData(aL0Tensor[L0aOffset].template ReinterpretCast<mx_fp8_e4m3_t>(),
                 qL1Tensor[qL1Offset].template ReinterpretCast<Q_T>(), qScaleL1Tensor[qScaleL1Offset],
                 loadData2DParamsA, loadData2DMxParamsA);
    }

    __aicore__ inline void LoadDataMxToL0BMm1(RunInfoX &runInfo, uint32_t s2RealSize)
    {
        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = false;
        loadData2DParamsB.mStep = ((s2SplitSize + 15) >> 4 << 4) >> 4;
        // kStep 须按对齐档（L0B 分形 n 轴 = dBaseSize，dim=72 时按 128 档搬运，pad 列已清零）
        loadData2DParamsB.kStep = dBaseSize >> 5;
        loadData2DParamsB.srcStride = ((s2SplitSize + 31) >> 5 << 5) >> 4;
        loadData2DParamsB.dstStride = loadData2DParamsB.mStep;

        LoadData2DMxParams loadData2DMxParamsB;
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = ((s2SplitSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsB.yStep = (dBaseSize + 63) >> 5 >> 1; // 与 A 侧一致，reduce 轴按对齐档基准
        loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
        loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;

        uint32_t L0bOffset = qkL0abBufId * QK_L0B_SIZE;
        uint32_t kvL1Offset = kvBufId * L1_KV_SIZE;
        uint32_t kvScaleL1Offset = kvBufId * L1_KV_SCALE_SIZE;
        LoadData(bL0Tensor[L0bOffset].template ReinterpretCast<mx_fp8_e4m3_t>(),
                 kvL1Tensor[kvL1Offset].template ReinterpretCast<KV_T>(), kvScaleL1Tensor[kvScaleL1Offset],
                 loadData2DParamsB, loadData2DMxParamsB);
    }

    __aicore__ inline void MatmulQK(const RunInfoX &runInfo, uint32_t s2RealSize)
    {
        MmadParams mmadParams;
        mmadParams.m = runInfo.actMSize;
        mmadParams.n = s2SplitSize;
        // Mmad reduce 轴须为对齐档（dim=72 时 k=72 非法），pad 列已清零参与累加无害
        mmadParams.k = dBaseSize;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;

        if (unlikely(mmadParams.m == 1)) {
            mmadParams.m = 16;
        }

        uint32_t qkL0AOffset = qkL0abBufId * QK_L0A_SIZE;
        uint32_t qkL0BOffset = qkL0abBufId * QK_L0B_SIZE;
        uint32_t qkL0COffset = qkL0cBufId * QK_L0C_SIZE;

        Mmad(cL0Tensor[qkL0COffset], aL0Tensor[qkL0AOffset].template ReinterpretCast<mx_fp8_e4m3_t>(),
             bL0Tensor[qkL0BOffset].template ReinterpretCast<mx_fp8_e4m3_t>(), mmadParams);
    }

    __aicore__ inline void IterateBmm1(const LocalTensor<T> &mm1ResUb, RunInfoX &runInfo, uint32_t subLoop)
    {
        IterateBmm1Nd(mm1ResUb, runInfo, subLoop);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, 左矩阵驻留. GS1<=80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Nd(const LocalTensor<T> &mm1ResUb, RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2CalcSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) { // unlikely告诉编译器大概率走不到
            s2CalcSize = AttentionCommon::Min((int32_t)s2SplitSize,
                                              (int32_t)(runInfo.actSingleLoopS2Size - subLoop * s2SplitSize));
        }
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
            if constexpr (!isDAligned) {
                // dim 非对齐（如 72）：先整块清零 L1 Q 的 n 轴对齐区域，GM 拷贝仅覆盖真实 dim 列
                InitValueL1BufferNAxis(qL1Tensor[qBufId * L1_Q_SIZE], mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(runInfo);
            CopyQueryScaleTile(runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
        }

        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(KV_T));
        if (unlikely(runInfo.isLastS2Loop && s2CalcSize < s2SplitSize)) {
            InitValueL1BufferNoTrans(kvL1Tensor[l1BaseOffset], s2CalcSize, (uint32_t)constInfo_.dSize);
        }
        if constexpr (!isDAligned) {
            // dim 非对齐（如 72）：先整块清零 L1 K 的 n 轴对齐区域（K scale GM 按对齐档组数存、拷贝全覆盖，无需清）
            InitValueL1BufferNAxis(kvL1Tensor[l1BaseOffset], s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(runInfo, s2CalcSize, subLoop);
        CopyKeyScaleTile(runInfo, s2CalcSize, subLoop);
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);

        WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT0 + qkL0cBufId);
        WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
        LoadDataMxToL0AMm1(runInfo);
        LoadDataMxToL0BMm1(runInfo, s2CalcSize);
        SetFlag<HardEvent::MTE1_M>(QK_L0AB_EVENT0 + qkL0abBufId);

        WaitFlag<HardEvent::MTE1_M>(QK_L0AB_EVENT0 + qkL0abBufId);
        MatmulQK(runInfo, s2CalcSize);
        SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);

        qkL0abBufId = (qkL0abBufId + 1) % (IS_D256 ? 1U : L0AB_BUFCNT);
        uint32_t c1v1Loop = (runInfo.actSingleLoopS2Size + s2SplitSize - 1) / s2SplitSize;
        if (unlikely(runInfo.isLastS2Loop && (subLoop == c1v1Loop - 1))) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
            qBufId = (qBufId + 1) % L1_Q_BUFCNT;
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;

        SetFlag<HardEvent::M_FIX>(QK_L0C_EVENT0 + qkL0cBufId);
        WaitFlag<HardEvent::M_FIX>(QK_L0C_EVENT0 + qkL0cBufId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[subLoop % 2]);
        FixpipeMm1(mm1ResUb, cL0Tensor[qkL0cBufId * QK_L0C_SIZE], runInfo, s2CalcSize);
        SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT0 + qkL0cBufId);
        qkL0cBufId = (qkL0cBufId + 1) % L0C_BUFCNT;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSS_CORE_SYNC_C1_V1[subLoop % 2]);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数，所以这里nsize向8对齐
        fixpipeParams.nSize = (s2SplitSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;使能NZ2ND，mSize*sizeof(T)必须为32的倍数，所以这里mSize应该向8对齐
        // 为什么向16对齐，是因为保证在V1阶段，处理的S1方向要满足16行（scale需要满足16*2分形）
        fixpipeParams.mSize = (runInfo.actMSize + 15) >> 4 << 4;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移，单位是16*sizeof(T)
        fixpipeParams.srcStride = (runInfo.actMSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = s2SplitSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 0;          // 单目标模式
        fixpipeParams.subBlockId = 0;          // 单目标模式。需要指定的目标UB的编号
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }
    __aicore__ inline void IterateBmm2(LocalTensor<T> &mm2ResUb, LocalTensor<fp8_e4m3fn_t> pL1Tensor,
                                       LocalTensor<fp8_e8m0_t> pScaleL1Tensor, RunInfoX &runInfo)
    {
        IterateBmm2Nd(mm2ResUb, pL1Tensor, pScaleL1Tensor, runInfo);
    }

    __aicore__ inline void LoadDataMxToL0AMm2(LocalTensor<fp8_e4m3fn_t> pL1Tensor,
                                              LocalTensor<fp8_e8m0_t> pScaleL1Tensor, RunInfoX &runInfo,
                                              uint32_t s2RealSize)
    {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.ifTranspose = false;
        loadData2DParamsA.mStep = ((runInfo.actMSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsA.kStep = ((s2SplitSize + 31) >> 5 << 5) >> 5;
        loadData2DParamsA.srcStride = ((mBaseSize + 31) >> 5 << 5) >> 4; // 这个竟然没问题？？？？
        loadData2DParamsA.dstStride = ((runInfo.actMSize + 15) >> 4 << 4) >> 4;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = ((runInfo.actMSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.yStep = (s2SplitSize + 63) >> 5 >> 1;
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        uint32_t L0aOffset = qkL0abBufId * QK_L0A_SIZE;
        LoadData(aL0Tensor[L0aOffset].template ReinterpretCast<mx_fp8_e4m3_t>(),
                 pL1Tensor.template ReinterpretCast<INPUT_T>(), pScaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);
    }

    __aicore__ inline void LoadDataMxToL0BMm2(RunInfoX &runInfo, uint32_t s2RealSize)
    {
        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.ifTranspose = true;
        loadData2DParamsB.mStep = ((s2SplitSize + 15) >> 4 << 4) >> 4;
        // kStep/dstStride/xStep 须按对齐档（L0B 分形 n 轴 = dVBaseSize，dim=72 时按 128 档搬运，pad 列已清零），与老
        // decode MatmulFullMX 一致
        loadData2DParamsB.kStep = dVBaseSize >> 5;
        loadData2DParamsB.srcStride = ((s2SplitSize + 31) >> 5 << 5) >> 4;
        loadData2DParamsB.dstStride = (dVBaseSize + 15) >> 4;

        LoadData2DMxParams loadData2DMxParamsB;
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = ((dVBaseSize + 15) >> 4 << 4) >> 4;
        // yStep：V scale 沿 reduce 轴（S2 方向）的组数，(s2SplitSize+63)>>5>>1 与 mm1 侧 (dBaseSize+63)>>5>>1 同构
        loadData2DMxParamsB.yStep = ((s2SplitSize + 63) >> 5 >> 1);

        loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
        loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;

        uint32_t L0bOffset = qkL0abBufId * QK_L0B_SIZE;
        uint32_t vL1Offset = kvBufId * L1_KV_SIZE;
        uint32_t vScaleL1Offset = kvBufId * L1_KV_SCALE_SIZE;
        LoadData(bL0Tensor[L0bOffset].template ReinterpretCast<mx_fp8_e4m3_t>(),
                 kvL1Tensor[vL1Offset].template ReinterpretCast<INPUT_T>(), kvScaleL1Tensor[vScaleL1Offset],
                 loadData2DParamsB, loadData2DMxParamsB);
    }

    __aicore__ inline void MatmulPV(const RunInfoX &runInfo, uint32_t s2RealSize, uint32_t k)
    {
        MmadParams mmadParams;
        mmadParams.m = runInfo.actMSize;
        // Mmad n 轴须为对齐档（dim=72 时 n=72 非法），pad 列已清零参与累加无害
        mmadParams.n = dVBaseSize;
        mmadParams.k = (s2SplitSize + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE;
        mmadParams.cmatrixInitVal = (k == 0);
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        if (unlikely(mmadParams.m == 1)) {
            mmadParams.m = 16;
        }

        uint32_t qkL0AOffset = qkL0abBufId * QK_L0A_SIZE;
        uint32_t qkL0BOffset = qkL0abBufId * QK_L0B_SIZE;
        uint32_t qkL0COffset = qkL0cBufId * QK_L0C_SIZE;

        Mmad(cL0Tensor[qkL0COffset], aL0Tensor[qkL0AOffset].template ReinterpretCast<mx_fp8_e4m3_t>(),
             bL0Tensor[qkL0BOffset].template ReinterpretCast<mx_fp8_e4m3_t>(), mmadParams);
    }

    __aicore__ inline void IterateBmm2Nd(LocalTensor<T> &mm2ResUb, LocalTensor<fp8_e4m3fn_t> pL1Tensor,
                                         LocalTensor<fp8_e8m0_t> pScaleL1Tensor, RunInfoX &runInfo)
    {
        WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT0 + qkL0cBufId);
        constexpr uint32_t baseK = s2SplitSize;
        uint64_t l1BaseKOffset = baseK * mBaseSize;
        uint64_t l1ScaleOffset = baseK / MXFP_GROUP_SIZE * runInfo.actMSizeAlign16;
        uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1) / baseK;
        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k == kLoops - 1) {
                realK = runInfo.actSingleLoopS2Size - k * baseK;
            }
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
            uint32_t offset = kvBufId * L1_KV_SIZE;
            if (unlikely(runInfo.isLastS2Loop && realK < s2SplitSize)) {
                InitValueL1BufferNoTrans(kvL1Tensor[offset], realK, (uint32_t)constInfo_.dSizeV);
            }
            if constexpr (!isDAligned) {
                // dim 非对齐（如 72）：先整块清零 L1 V 数据区与 scale 区（分离 buffer，各清一次），GM 拷贝仅覆盖真实
                // dim 列 scale 区布局：[s2SplitSize/64 行, dVBaseSize*2 字节]（64 行 S2 压缩 1 行，scale 按 16bit
                // 对齐存）
                InitValueL1BufferNAxis(kvL1Tensor[offset], s2SplitSize, dVBaseSize);
                InitValueL1BufferNAxis(kvScaleL1Tensor[kvBufId * L1_KV_SCALE_SIZE], s2SplitSize / MXFP_DIVISOR_SIZE,
                                       dVBaseSize * MXFP_MULTI_BASE_SIZE);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyValueSlice(runInfo.s2Idx + k * baseK, realK, 0, constInfo_.dSizeV, runInfo);
            CopyValueScaleSlice((runInfo.s2Idx + k * baseK) / MXFP_DIVISOR_SIZE,
                                (realK + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE, 0,
                                constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);

            WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
            uint32_t realKAligned64 = (realK + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE;
            LoadDataMxToL0AMm2(pL1Tensor[k * l1BaseKOffset], pScaleL1Tensor[k * l1ScaleOffset], runInfo,
                               realKAligned64);
            LoadDataMxToL0BMm2(runInfo, realKAligned64);
            SetFlag<HardEvent::MTE1_M>(QK_L0AB_EVENT0 + qkL0abBufId);

            WaitFlag<HardEvent::MTE1_M>(QK_L0AB_EVENT0 + qkL0abBufId);
            MatmulPV(runInfo, realK, k);
            SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
            // K→V 覆写由既有 WaitFlag(M_MTE1)→LoadData 时序保护（QK mmad 完成后 V 才进入）
            qkL0abBufId = (qkL0abBufId + 1) % (IS_D256 ? 1U : 2U);
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
            kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;
        }

        SetFlag<HardEvent::M_FIX>(QK_L0C_EVENT0 + qkL0cBufId);
        WaitFlag<HardEvent::M_FIX>(QK_L0C_EVENT0 + qkL0cBufId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
        FixpipeMm2(mm2ResUb, cL0Tensor[qkL0cBufId * QK_L0C_SIZE], runInfo);
        SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT0 + qkL0cBufId);
        qkL0cBufId = (qkL0cBufId + 1) % 2;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (constInfo_.dSizeV + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 15) >> 4 << 4;
        fixpipeParams.srcStride = (runInfo.actMSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 0; // 单目标模式
        fixpipeParams.subBlockId = 0; // 单目标模式。需要指定的目标UB的编号
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }
}; // QuantFlashAttnBlockCubeMxfp8Nd

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false>
class QuantFlashAttnBlockCubeMxfp8NdDummy {
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
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8NdDummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_

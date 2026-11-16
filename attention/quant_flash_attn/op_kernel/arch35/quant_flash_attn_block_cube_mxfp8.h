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
 * \file quant_flash_attn_block_cube_mxfp8.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (kvLayoutType == 0) {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) {
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

// Only the enabled specialization owns the BF16 V-tail addressing objects.
template <bool hasVTail, GmFormat kvFormat>
struct QuantFlashAttnCubeVTailState {};

template <GmFormat kvFormat>
struct QuantFlashAttnCubeVTailState<true, kvFormat> {
    CopyKvGmToL1<bfloat16_t, kvFormat> copyVTailGmToL1_;
    FaGmTensor<bfloat16_t, kvFormat, int32_t, GmLayoutParams<kvFormat>::CATEGORY == FormatCategory::GM_KV_TND> vTailGm_;
    GlobalTensor<int32_t> vTailBttGm_;
};

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool isDAligned = true, bool hasVTail = false>
class QuantFlashAttnBlockCubeMxfp8
    : private QuantFlashAttnCubeVTailState<hasVTail, GetKVGmFormat<layout, KvLayoutType, (KvLayoutType > 0)>()> {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t s2SplitSize =
        (dBaseSize == static_cast<uint32_t>(DTemplateType::Aligned256)) ? 128U : 256U;
    // K 维分片宽度: 保证 A[s2SplitSize x baseK] 能放进单个 L0A(32KB)
    static constexpr uint32_t dBaseK =
        (dBaseSize <= (32 * 1024) / s2SplitSize) ? dBaseSize : ((32 * 1024) / s2SplitSize);
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_V_TAIL = hasVTail;
    // D=192 / D_v=128(config5) 场景 L1 超预算, 定向回退为 3 槽;
    // P 槽深度与 D 无关, producer(AIV) 领先 consumer(AIC) 恒为 2, 3 槽为理论下限
    static constexpr bool USE_3_P_SLOTS = (dBaseSize == 192U && dVBaseSize == 128U);
    static constexpr InnerMLayout Q_M_LAYOUT = USE_DN ? InnerMLayout::S1_ONLY_LAYOUT : InnerMLayout::GS1_MERGE_LAYOUT;

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

    // 静态Tensor布局与原TPipe/BufferManager分配顺序一致: L1 P -> Q -> KV, UB bmm2 -> bmm1
    static constexpr uint32_t L1_P_BUFCNT = (HAS_V_TAIL && !USE_3_P_SLOTS) ? 4U : 3U;
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t L1_P_BUF_BYTES =
        mBaseSize * s2BaseSize * sizeof(INPUT_T) + mBaseSize * s2BaseSize / MXFP_GROUP_SIZE +
        (HAS_V_TAIL ? mBaseSize * MXFP8_V_TAIL_BOUNDARY * sizeof(bfloat16_t) : 0U);
    static constexpr uint32_t MM_Q_BUF_BYTES =
        mBaseSize * dBaseSize * sizeof(INPUT_T) + mBaseSize * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);
    static constexpr uint32_t MM_KV_BUF_BYTES =
        dBaseSize * ((dBaseSize == 256U) ? s2BaseSize : (s2BaseSize >> 1U)) * sizeof(INPUT_T) +
        ((dBaseSize == 256U) ? s2BaseSize : (s2BaseSize >> 1U)) * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);

    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * s2BaseSize * sizeof(T) / 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * dVBaseSize * sizeof(T);

    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CC_MM1_READY_0 = 0U;
    static constexpr uint32_t CC_MM1_READY_1 = 1U;
    static constexpr uint32_t CC_MM2_READY = 2U;
    static constexpr uint32_t CC_L1P_0 = 4U;
    static constexpr uint32_t CC_MM1_FREE_0 = 8U;
    static constexpr uint32_t CC_MM1_FREE_1 = 9U;
    static constexpr uint32_t CC_MM2_FREE = 10U;
    static_assert(CC_L1P_0 + L1_P_BUFCNT <= CC_MM1_FREE_0, "L1 P cross-core flag ids overlap UB free flag ids");

    // 原mxfp8 L0C始终为双buffer, 小规格仅缩小单槽为64KB, 不改为四槽
    static constexpr bool L0C_IS_SMALL_BUF =
        (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4);
    static constexpr uint32_t L0C_BUFCNT = 2U;
    static constexpr uint32_t L0C_BUF_BYTES = (L0C_SIZE / (L0C_IS_SMALL_BUF ? NUM_4 : NUM_2)) * KB_TO_BYTES;
    static_assert(L0C_BUFCNT * L0C_BUF_BYTES <= L0C_SIZE * KB_TO_BYTES, "static L0C buffers exceed 256KB");

    // 各HardEvent方向独立编号: L1 Q 0-1/KV 2-5, L0A 0-1/L0B 2-3, L0C 0-1
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 0U;
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 2U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 0U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 1U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 2U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 3U;
    static constexpr uint32_t L0C_BUFFER_ID0 = 0U;

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

    using L0AType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::SET_WAIT_FLAG, IdSource::EXTERNAL>;
    using L0BType =
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::SET_WAIT_FLAG, IdSource::EXTERNAL>;
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

    using ConstInfoX = std::conditional_t<HAS_V_TAIL, ConstInfoVTail_t, ConstInfo_t>;
    using RunInfoX = std::conditional_t<HAS_V_TAIL, RunInfoVTailX, ::RunInfoX>;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    FaGmTensorVScale valueScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqParserType* qSeqParserPtr_ = nullptr;
    KvSeqParserType* kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryScaleGmToL1_;
    KeyScaleGmToL1Type copyKeyScaleGmToL1_;
    ValueScaleGmToL1Type copyValueScaleGmToL1_;

    /* =====================LocalBuffer变量====================*/
    LocalTensor<uint8_t> ubBmm2ResBuf_;
    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KVBuffers_;
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t bmm1ResBufId_ = 0U;
    uint32_t l1PBufId_ = 0U;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    uint32_t l0cBufId_ = 0U;

    // 仅L0A/L0B保留BufferManager, 使用EXTERNAL eventId避免依赖TPipe
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;

    __gm__ uint8_t* keyPtr_ = nullptr;
    __gm__ uint8_t* valuePtr_ = nullptr;

    const ConstInfoX& constInfo_;

    /*============================================================================== */
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                         __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue,
                                         QSeqParserType& qParser, KvSeqParserType& kvParser,
                                         __gm__ uint8_t* vTail = nullptr, __gm__ uint8_t* blockTableTail = nullptr)
    {
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue, vTail,
                      blockTableTail);
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t addrUb = 0U;
        ubBmm2ResBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUF_BYTES;
        ubBmm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);

        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);
        uint32_t addrL1 = L1_P_BUFCNT * L1_P_BUF_BYTES;
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * MM_Q_BUF_BYTES);
        addrL1 += L1_Q_BUFCNT * MM_Q_BUF_BYTES;
        l1KVBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * MM_KV_BUF_BYTES);

        l0aBufferManager_.Init(65536U);
        l0bBufferManager_.Init(65536U);
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* blockTable, __gm__ uint8_t* dequantScaleQuery,
                                         __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue,
                                         __gm__ uint8_t* vTail = nullptr, __gm__ uint8_t* blockTableTail = nullptr)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
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
        InitVScaleBuffer(constInfo_.bSize, CeilDiv(constInfo_.s2Size, MXFP_DIVISOR_SIZE), constInfo_.n2Size,
                         CeilDiv(constInfo_.blockSize, MXFP_DIVISOR_SIZE), constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE,
                         valueScaleGm_, dequantScaleValue);
        if constexpr (HAS_V_TAIL) {
            InitVTailBuffer(vTail, blockTableTail);
        }
    }

    __aicore__ inline bool IsLastDnTask(const RunInfoX& runInfo) const
    {
        if constexpr (HAS_V_TAIL) {
            return runInfo.isLastDnTask;
        }
        return runInfo.isLastS2Loop;
    }

    __aicore__ inline void InitVTailBuffer(__gm__ uint8_t* vTail, __gm__ uint8_t* blockTableTail)
    {
        if (vTail == nullptr) {
            return;
        }
        this->vTailGm_.gmTensor.SetGlobalBuffer((__gm__ bfloat16_t*)vTail);
        if constexpr (PAGE_ATTENTION) {
            // v_tail与主KV同布局, btt=block_table_tail寻址; 尾窗<64落单块(blockIdxInBatch=0)
            this->vTailBttGm_.SetGlobalBuffer((__gm__ int32_t*)blockTableTail);
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                // dSizeV/nzD0整除: host侧vtailDimOk已保证PA_NZ下D%16==0(拒绝D=72), 此处无余数
                constexpr uint32_t nzD0 = 32 / sizeof(bfloat16_t);
                this->vTailGm_.offsetCalculator.Init(
                    constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV / nzD0, nzD0, this->vTailBttGm_,
                    constInfo_.tailMaxBlockNum, constInfo_.vTailStrides.bnStride, constInfo_.vTailStrides.n2Stride);
            } else {
                this->vTailGm_.offsetCalculator.Init(
                    constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV, this->vTailBttGm_,
                    constInfo_.tailMaxBlockNum, constInfo_.vTailStrides.bnStride, constInfo_.vTailStrides.n2Stride);
            }
        }
        // 非PA(TND等)不支持v_tail(checker入口拒绝), 无需初始化
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

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensorQScale& qScaleGmTensor, __gm__ uint8_t* gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
        if constexpr (Q_NEEDS_WZH) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV& kvGmTensor,
                                        __gm__ uint8_t* gm, uint64_t bnStride, uint64_t n2Stride)
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
        } else if constexpr (KV_NEEDS_WZH) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            // BNSD/BSND: GM_KV_BNSD 使用带 parser 的 5 参重载
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, *this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorKScale& kScaleGmTensor, __gm__ uint8_t* gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
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
            // BNSD/BSND: GM_KV_BNSD 使用带 parser 的 5 参重载
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, *this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorVScale& vScaleGmTensor, __gm__ uint8_t* gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T*)gm);
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
            // BNSD/BSND: GM_KV_BNSD 使用带 parser 的 5 参重载
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, *this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM1_FREE_1 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void AllocEventID()
    {
        // 与原Buffer::Init一致, 预置消费端归还给生产端的信用
        for (uint32_t i = 0; i < L1_Q_BUFCNT; ++i) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + i);
        }
        for (uint32_t i = 0; i < L1_KV_BUFCNT; ++i) {
            SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + i);
        }
        mmL0ABuffers_.Init(l0aBufferManager_, 32U * KB_TO_BYTES, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        mmL0BBuffers_.Init(l0bBufferManager_, 32U * KB_TO_BYTES, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
        for (uint32_t i = 0; i < L0C_BUFCNT; ++i) {
            SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + i);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        for (uint32_t i = 0; i < L1_Q_BUFCNT; ++i) {
            WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + i);
        }
        for (uint32_t i = 0; i < L1_KV_BUFCNT; ++i) {
            WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + i);
        }
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
        for (uint32_t i = 0; i < L0C_BUFCNT; ++i) {
            WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + i);
        }
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX& runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
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

    // copy query scale with full s1g
    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<SCALE_T>& dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX& runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        if constexpr (USE_DN) {
            // DN分支下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
            GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.realN2Idx,
                                  .gIdx = 0,
                                  .s1Idx = runInfo.gS1Idx,
                                  .dIdx = dOffset,
                                  .s1DealSize = runInfo.actMSize,
                                  .dDealSize = dRealSize};
            copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
        } else {
            GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                    .n2Idx = runInfo.realN2Idx,
                                    .gS1Idx = runInfo.gS1Idx,
                                    .dIdx = dOffset,
                                    .gS1DealSize = runInfo.actMSize,
                                    .dDealSize = dRealSize};
            copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<SCALE_T>& dstTensor, RunInfoX& runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, dBaseSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T>& dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                        uint32_t dOffset, uint32_t dRealSize, RunInfoX& runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(KV_T);
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;

        uint32_t nopeDealSize = dRealSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        // CopyKey的控制参数
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo, uint32_t s2RealSize,
                                       uint32_t subLoop)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / 2;
        CopyKeySlice(dstTensor, s2Offset, s2RealSize, 0, constInfo_.dSize, runInfo);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<SCALE_T>& dstTensor, uint32_t s2Offset,
                                             uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                             RunInfoX& runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKeyScaleGmToL1_(l1Tensor, keyScaleGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<SCALE_T>& dstTensor, RunInfoX& runInfo,
                                            uint32_t s2RealSize, uint32_t subLoop)
    {
        uint32_t s2Offset = subLoop % 2 == 0 ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / 2;
        CopyKeyScaleSlice(dstTensor, s2Offset, s2RealSize, 0, dBaseSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T>& dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                          uint32_t dOffset, uint32_t dRealSize, RunInfoX& runInfo)
    {
        uint32_t dstStride = (s2RealSize + 63) >> 6 << 6;
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

    // copy key with full s2
    __aicore__ inline void CopyValueScaleSlice(const LocalTensor<SCALE_T>& dstTensor, uint32_t s2Offset,
                                               uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                               RunInfoX& runInfo)
    {
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = s2RealSize};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyValueScaleGmToL1_(l1Tensor, valueScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyValueScaleTile(const LocalTensor<SCALE_T>& dstTensor, RunInfoX& runInfo)
    {
        CopyValueScaleSlice(dstTensor, runInfo.s2Idx / MXFP_DIVISOR_SIZE,
                            CeilDiv(runInfo.actSingleLoopS2Size, MXFP_DIVISOR_SIZE), 0,
                            constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void*)(this->keyPtr_));
        __gm__ uint8_t* key_ = (__gm__ uint8_t*)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr_, bIdx);
        keyGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T*)key_);
        keyGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSize);
        keyGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void*)(this->valuePtr_));
        __gm__ uint8_t* value_ = (__gm__ uint8_t*)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr_, bIdx);
        valueGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T*)value_);
        valueGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSizeV);
        valueGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void IterateBmm1(RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t bmm1ResBufId = bmm1ResBufId_;
        bmm1ResBufId_ = (bmm1ResBufId_ + 1U) % UB_MM1_RES_BUFCNT;
        if constexpr (HAS_V_TAIL) {
            if (unlikely(runInfo.isVtailTask)) {
                if constexpr (USE_DN) {
                    IterateBmm1DnTail(bmm1ResBufId, runInfo, subLoop);
                } else {
                    IterateBmm1NdTail(bmm1ResBufId, runInfo, subLoop);
                }
                return;
            }
        }
        if constexpr (USE_DN) {
            IterateBmm1Dn(bmm1ResBufId, runInfo, subLoop);
        } else {
            IterateBmm1Nd(bmm1ResBufId, runInfo, subLoop);
        }
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, 左矩阵驻留. GS1<=80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Nd(uint32_t bmm1ResBufId, RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2CurSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) {
            if (subLoop == 0) {
                s2CurSize = AttentionCommon::Min((int32_t)s2SplitSize, (int32_t)runInfo.actSingleLoopS2Size);
            } else {
                s2CurSize = runInfo.actSingleLoopS2Size - s2SplitSize;
            }
        }
        LocalTensor<uint8_t> mm1A;
        uint32_t qL1BufId = qL1BufId_;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            qL1BufId_ = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1A = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId);
            LocalTensor<Q_T> mm1ATensor = mm1A.ReinterpretCast<Q_T>();
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(mm1ATensor, mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(mm1ATensor, runInfo);

            LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.ReinterpretCast<SCALE_T>()[qScaleOffset];
            CopyQueryScaleTile(mm1AScaleTensor, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        } else {
            qL1BufId = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1A = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        }

        uint32_t kvL1BufId = kvL1BufId_;
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
        LocalTensor<uint8_t> mm1B = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
        WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
        LocalTensor<KV_T> mm1BTensor = mm1B.ReinterpretCast<KV_T>();
        if constexpr (!isDAligned) {
            InitValueL1BufferNAxis(mm1BTensor, s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(mm1BTensor, runInfo, s2CurSize, subLoop);

        uint32_t kScaleOffset = ((s2CurSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.ReinterpretCast<SCALE_T>()[kScaleOffset];
        CopyKeyScaleTile(mm1BScaleTensor, runInfo, s2CurSize, subLoop);
        SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);

        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm1ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();
        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CurSize, dBaseSize, false, true);
        if constexpr (dBaseSize == 192) {
            MatmulKMX<Q_T, KV_T, T, 128, (s2BaseSize >> 1), dBaseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                      SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[qScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[kScaleOffset]);
        } else {
            MatmulFullMX<Q_T, KV_T, T, 128, (s2BaseSize >> 1), dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[qScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[kScaleOffset]);
        }

        // Release Q at the last main task even when a tail task follows.
        if (unlikely(IsLastDnTask(runInfo) && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                               (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId);
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId); // 释放L1B
        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);        // 通知
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);       // 等待L0C

        LocalTensor<T> mmResUbTensor =
            ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmFreeSyncIdx = CC_MM1_FREE_0 + bmm1ResBufId;
        uint32_t mmReadySyncIdx = CC_MM1_READY_0 + bmm1ResBufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx + AIV0_AIV1_OFFSET);

        FixpipeMm1(mmResUbTensor, mm1ResL0C, runInfo, s2CurSize);

        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 31) >> 5 << 5;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2SplitSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    // The tail VF consumes scores with a 128-element row pitch.
    __aicore__ inline void FixpipeMm1Tail(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo,
                                          uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = (s2SplitSize == 256) ? (s2SplitSize >> 1) : s2SplitSize;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo,
                                        uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 64; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 2; // 双目标模式，按M维度拆分， M * N / 2写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    // DN tail score row pitch is half of the aligned M dimension.
    __aicore__ inline void FixpipeMm1DnTail(const LocalTensor<T>& dstTensor, const LocalTensor<T>& l0C,
                                            RunInfoX& runInfo, uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(uint32_t bmm1ResBufId, RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2CalcSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) { // unlikely告诉编译器大概率走不到
            if (subLoop == 0) {
                s2CalcSize = AttentionCommon::Min((int32_t)s2SplitSize, (int32_t)runInfo.actSingleLoopS2Size);
            } else {
                s2CalcSize = runInfo.actSingleLoopS2Size - s2SplitSize;
            }
        }
        LocalTensor<uint8_t> mm1B;
        uint32_t qL1BufId = qL1BufId_;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            qL1BufId_ = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1B = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId); // B矩阵常驻，两次Sub中，只有第一次搬B
            LocalTensor<Q_T> mm1BTensor = mm1B.ReinterpretCast<Q_T>();
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(mm1BTensor, mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(mm1BTensor, runInfo);

            LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.ReinterpretCast<SCALE_T>()[qScaleOffset];
            CopyQueryScaleTile(mm1BScaleTensor, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        } else {
            qL1BufId = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1B = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        }

        uint32_t kvL1BufId = kvL1BufId_;
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
        LocalTensor<uint8_t> mm1A = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
        WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
        LocalTensor<KV_T> mm1ATensor = mm1A.ReinterpretCast<KV_T>();
        if constexpr (!isDAligned) {
            InitValueL1BufferNAxis(mm1ATensor, s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(mm1ATensor, runInfo, s2CalcSize, subLoop);

        uint32_t kScaleOffset = ((s2CalcSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.ReinterpretCast<SCALE_T>()[kScaleOffset];
        CopyKeyScaleTile(mm1AScaleTensor, runInfo, s2CalcSize, subLoop);
        SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm1ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();
        MMParam param = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, dBaseSize, false, true);
        if constexpr (dBaseSize == 192) {
            MatmulKMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                      SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[kScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[qScaleOffset]);
        } else {
            MatmulFullMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[kScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[qScaleOffset]);
        }

        // Release Q at the last main task even when a tail task follows.
        if (unlikely(IsLastDnTask(runInfo) && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                               (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId);
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId); // 释放L1B
        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);        // 通知
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);       // 等待L0C

        LocalTensor<T> mmResUbTensor =
            ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmFreeSyncIdx = CC_MM1_FREE_0 + bmm1ResBufId;
        uint32_t mmReadySyncIdx = CC_MM1_READY_0 + bmm1ResBufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx + AIV0_AIV1_OFFSET);

        FixpipeMm1Dn(mmResUbTensor, mm1ResL0C, runInfo, s2CalcSize);

        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx + AIV0_AIV1_OFFSET);
    }

    // Compute tail QK and write scores using the dynamic DN row pitch.
    __aicore__ inline void IterateBmm1DnTail(uint32_t bmm1ResBufId, RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size; // == kvTailLen(纯尾task判定)
        LocalTensor<uint8_t> mm1B;
        uint32_t qL1BufId = qL1BufId_;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            qL1BufId_ = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1B = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId); // B矩阵常驻，两次Sub中，只有第一次搬B
            LocalTensor<Q_T> mm1BTensor = mm1B.ReinterpretCast<Q_T>();
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(mm1BTensor, mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(mm1BTensor, runInfo);

            LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.ReinterpretCast<SCALE_T>()[qScaleOffset];
            CopyQueryScaleTile(mm1BScaleTensor, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        } else {
            qL1BufId = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1B = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        }

        uint32_t kvL1BufId = kvL1BufId_;
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
        LocalTensor<uint8_t> mm1A = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
        WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
        LocalTensor<KV_T> mm1ATensor = mm1A.ReinterpretCast<KV_T>();
        if constexpr (!isDAligned) {
            InitValueL1BufferNAxis(mm1ATensor, s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(mm1ATensor, runInfo, s2CalcSize, subLoop);

        uint32_t kScaleOffset = ((s2CalcSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.ReinterpretCast<SCALE_T>()[kScaleOffset];
        CopyKeyScaleTile(mm1AScaleTensor, runInfo, s2CalcSize, subLoop);
        SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm1ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();
        MMParam param = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, dBaseSize, false, true);
        if constexpr (dBaseSize == 192) {
            MatmulKMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                      SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[kScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[qScaleOffset]);
        } else {
            MatmulFullMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[kScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[qScaleOffset]);
        }

        if (unlikely(IsLastDnTask(runInfo) && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                               (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId);
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId); // 释放L1B
        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);        // 通知
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);       // 等待L0C

        LocalTensor<T> mmResUbTensor =
            ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmFreeSyncIdx = CC_MM1_FREE_0 + bmm1ResBufId;
        uint32_t mmReadySyncIdx = CC_MM1_READY_0 + bmm1ResBufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx + AIV0_AIV1_OFFSET);

        FixpipeMm1DnTail(mmResUbTensor, mm1ResL0C, runInfo, s2CalcSize);

        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx + AIV0_AIV1_OFFSET);
    }

    // Compute tail QK with the ND score layout.
    __aicore__ inline void IterateBmm1NdTail(uint32_t bmm1ResBufId, RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2CurSize = runInfo.actSingleLoopS2Size; // == kvTailLen(纯尾task判定)
        LocalTensor<uint8_t> mm1A;
        uint32_t qL1BufId = qL1BufId_;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            qL1BufId_ = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1A = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId);
            LocalTensor<Q_T> mm1ATensor = mm1A.ReinterpretCast<Q_T>();
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(mm1ATensor, mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(mm1ATensor, runInfo);

            LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.ReinterpretCast<SCALE_T>()[qScaleOffset];
            CopyQueryScaleTile(mm1AScaleTensor, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        } else {
            qL1BufId = (qL1BufId_ + 1U) % L1_Q_BUFCNT;
            mm1A = l1QBuffers_[qL1BufId * MM_Q_BUF_BYTES];
            SetFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        }

        uint32_t kvL1BufId = kvL1BufId_;
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
        LocalTensor<uint8_t> mm1B = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
        WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
        LocalTensor<KV_T> mm1BTensor = mm1B.ReinterpretCast<KV_T>();
        if constexpr (!isDAligned) {
            InitValueL1BufferNAxis(mm1BTensor, s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(mm1BTensor, runInfo, s2CurSize, subLoop);

        uint32_t kScaleOffset = ((s2CurSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.ReinterpretCast<SCALE_T>()[kScaleOffset];
        CopyKeyScaleTile(mm1BScaleTensor, runInfo, s2CurSize, subLoop);
        SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);

        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm1ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();
        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CurSize, dBaseSize, false, true);
        if constexpr (dBaseSize == 192) {
            MatmulKMX<Q_T, KV_T, T, 128, (s2BaseSize >> 1), dBaseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                      SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[qScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[kScaleOffset]);
        } else {
            MatmulFullMX<Q_T, KV_T, T, 128, (s2BaseSize >> 1), dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm1A.ReinterpretCast<INPUT_T>(), mm1B.ReinterpretCast<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C, param, mm1A.ReinterpretCast<SCALE_T>()[qScaleOffset],
                mm1B.ReinterpretCast<SCALE_T>()[kScaleOffset]);
        }

        if (unlikely(IsLastDnTask(runInfo) && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                               (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId);
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId); // 释放L1B
        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);        // 通知
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);       // 等待L0C

        LocalTensor<T> mmResUbTensor =
            ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmFreeSyncIdx = CC_MM1_FREE_0 + bmm1ResBufId;
        uint32_t mmReadySyncIdx = CC_MM1_READY_0 + bmm1ResBufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmFreeSyncIdx + AIV0_AIV1_OFFSET);

        FixpipeMm1Tail(mmResUbTensor, mm1ResL0C, runInfo, s2CurSize);

        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmReadySyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void IterateBmm2(RunInfoX& runInfo)
    {
        uint32_t pL1BufId = l1PBufId_;
        l1PBufId_ = (l1PBufId_ + 1U) % L1_P_BUFCNT;
        if constexpr (USE_DN) {
            if constexpr (HAS_V_TAIL) {
                if (unlikely(runInfo.isVtailTask)) {
                    IterateBmm2DnTail(pL1BufId, runInfo);
                    return;
                }
            }
            IterateBmm2Dn(pL1BufId, runInfo);
        } else {
            if constexpr (HAS_V_TAIL) {
                if (unlikely(runInfo.isVtailTask)) {
                    IterateBmm2NdTail(pL1BufId, runInfo);
                    return;
                }
            }
            IterateBmm2Nd(pL1BufId, runInfo);
        }
    }

    __aicore__ inline void IterateBmm2Nd(uint32_t pL1BufId, RunInfoX& runInfo)
    {
        LocalTensor<uint8_t> mm2A = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        uint32_t pSyncIdx = CC_L1P_0 + pL1BufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx + AIV0_AIV1_OFFSET);

        uint32_t pScaleOffset = mBaseSize * s2BaseSize; // PScale offset in L1.
        LocalTensor<SCALE_T> mm2AScaleFakeTensor = mm2A.ReinterpretCast<SCALE_T>()[pScaleOffset];

        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm2ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();

        constexpr uint32_t baseK = s2SplitSize;
        uint64_t l1BaseKOffset = baseK * mBaseSize;
        uint64_t l1ScaleOffset = baseK / 32 * runInfo.actMSizeAlign32;
        // Only the task ending at the sequence boundary contains the tail.
        uint32_t tailLen = 0;
        if constexpr (HAS_V_TAIL) {
            if (runInfo.s2Idx + runInfo.actSingleLoopS2Size >= runInfo.actS2Size) {
                tailLen = runInfo.kvTailLen;
            }
        }
        uint32_t kMainSize = runInfo.actSingleLoopS2Size - tailLen;
        uint32_t kLoops = (kMainSize + baseK - 1) / baseK;
        uint32_t realK = baseK;
        for (uint32_t kIdx = 0; kIdx < kLoops; kIdx++) {
            uint32_t kvL1BufId = kvL1BufId_;
            kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
            LocalTensor<uint8_t> mm2B = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
            if (kIdx == kLoops - 1) {
                realK = kMainSize - kIdx * baseK;
            }
            WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
            LocalTensor<KV_T> mm2BTensor = mm2B.ReinterpretCast<KV_T>();
            if (unlikely(runInfo.isLastS2Loop)) {
                InitValueL1BufferNoTrans(mm2BTensor, realK, (uint32_t)constInfo_.dSizeV);
            }
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(mm2BTensor, baseK + baseK / MXFP_GROUP_SIZE, dVBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyValueSlice(mm2BTensor, runInfo.s2Idx + kIdx * baseK, realK, 0, constInfo_.dSizeV, runInfo);
            uint32_t vScaleOffset = (((realK) + 63) >> 6 << 6) * dVBaseSize;
            LocalTensor<SCALE_T> mm2BScaleTensor = mm2B.ReinterpretCast<SCALE_T>()[vScaleOffset];
            CopyValueScaleSlice(mm2BScaleTensor, (runInfo.s2Idx + kIdx * baseK) / MXFP_DIVISOR_SIZE,
                                (realK + 63) / MXFP_DIVISOR_SIZE, 0, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);

            WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, dVBaseSize, (realK + 63) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE,
                            false, false, kIdx == 0, kIdx == 0);
            if constexpr (!USE_DN) {
                param.realM = (uint32_t)runInfo.actMSize;
            }
            MatmulFullMX<INPUT_T, KV_T, T, 128, dVBaseSize, baseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm2A.ReinterpretCast<INPUT_T>()[kIdx * l1BaseKOffset], mm2BTensor, mmL0ABuffers_, mmL0BBuffers_,
                mm2ResL0C, param, mm2AScaleFakeTensor[kIdx * l1ScaleOffset], mm2BScaleTensor);
            SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
        }
        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mmResUbTensor = ubBmm2ResBuf_.template ReinterpretCast<T>();
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE + AIV0_AIV1_OFFSET);
        FixpipeMm2(mmResUbTensor, mm2ResL0C, runInfo); // 数据搬出
        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY + AIV0_AIV1_OFFSET);
    }

    // Tail PV uses BF16 operands; Vec2 merges its result with the main softmax state.
    __aicore__ inline void IterateBmm2DnTail(uint32_t pL1BufId, RunInfoX& runInfo)
    {
        LocalTensor<uint8_t> mm2A = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        uint32_t pSyncIdx = CC_L1P_0 + pL1BufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx + AIV0_AIV1_OFFSET);

        // P尾区(bf16)在l1P槽的偏移(与V1写侧同公式): 主区(fp8)+scale区之后
        constexpr uint64_t pTailOffset =
            (static_cast<uint64_t>(mBaseSize) * s2BaseSize + mBaseSize * s2BaseSize / 32) / sizeof(bfloat16_t);

        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm2ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();
        MMParam tailParam =
            MakeMMParam((uint32_t)mBaseSize, dVBaseSize, MXFP8_V_TAIL_BOUNDARY, USE_DN, false, true, true);
        // Compute the aligned M block; Vec2 discards rows beyond actM.
        {
            uint32_t kvL1BufId = kvL1BufId_;
            kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
            LocalTensor<uint8_t> mm2BTail = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
            WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
            {
                LocalTensor<bfloat16_t> mm2BTailTensor = mm2BTail.ReinterpretCast<bfloat16_t>();
                FaL1Tensor<bfloat16_t, L1Format::NZ> l1Tensor{.tensor = mm2BTailTensor,
                                                              .rowCount = MXFP8_V_TAIL_BOUNDARY};
                GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.n2Idx,
                                  .s2Idx = 0,
                                  .dIdx = 0,
                                  .s2DealSize = MXFP8_V_TAIL_BOUNDARY, // Padding is cleared before MTE1 reads L1.
                                  .dDealSize = constInfo_.dSizeV};
                this->copyVTailGmToL1_(l1Tensor, this->vTailGm_, gmCoord);
            }
            // Finish the GM copy before overwriting its padding in L1.
            AscendC::PipeBarrier<PIPE_MTE2>();
            // Clear unused V rows so zero P cannot multiply NaN/Inf padding.
            if (runInfo.kvTailLen < MXFP8_V_TAIL_BOUNDARY) {
                InitConstValueParams<half> vTailPadParams;
                vTailPadParams.repeatTimes = AttentionCommon::Align((uint32_t)constInfo_.dSizeV, 16U) / 16U;
                vTailPadParams.blockNum = MXFP8_V_TAIL_BOUNDARY - runInfo.kvTailLen;
                vTailPadParams.dstGap = runInfo.kvTailLen;
                vTailPadParams.initValue = 0;
                InitConstValue(mm2BTail.ReinterpretCast<bfloat16_t>().ReinterpretCast<half>()[runInfo.kvTailLen * 16U],
                               vTailPadParams);
            }
            // MTE1 must wait for both the GM copy and the padding clear.
            SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
            MatmulFull<bfloat16_t, bfloat16_t, T, 128, dVBaseSize, MXFP_DIVISOR_SIZE, ABLayout::MK, ABLayout::KN,
                       L0AType, L0BType>(mm2A.ReinterpretCast<bfloat16_t>()[pTailOffset],
                                         mm2BTail.ReinterpretCast<bfloat16_t>(), mmL0ABuffers_, mmL0BBuffers_,
                                         mm2ResL0C, tailParam);
            SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId); // 释放KV slot
        }

        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mmResUbTensor = ubBmm2ResBuf_.template ReinterpretCast<T>();
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE + AIV0_AIV1_OFFSET);
        FixpipeMm2(mmResUbTensor, mm2ResL0C, runInfo);
        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY + AIV0_AIV1_OFFSET);
    }

    // ND tail PV uses its own L0C result and the actual M size.
    __aicore__ inline void IterateBmm2NdTail(uint32_t pL1BufId, RunInfoX& runInfo)
    {
        LocalTensor<uint8_t> mm2A = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        uint32_t pSyncIdx = CC_L1P_0 + pL1BufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx + AIV0_AIV1_OFFSET);

        // P尾区(bf16)在l1P槽的偏移(与V1写侧同公式): 主区(fp8)+scale区之后
        constexpr uint64_t pTailOffset =
            (static_cast<uint64_t>(mBaseSize) * s2BaseSize + mBaseSize * s2BaseSize / 32) / sizeof(bfloat16_t);

        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm2ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();
        MMParam tailParam =
            MakeMMParam((uint32_t)mBaseSize, dVBaseSize, MXFP8_V_TAIL_BOUNDARY, USE_DN, false, true, true);
        tailParam.realM = (uint32_t)runInfo.actMSize;
        {
            uint32_t kvL1BufId = kvL1BufId_;
            kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
            LocalTensor<uint8_t> mm2BTail = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
            WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
            {
                LocalTensor<bfloat16_t> mm2BTailTensor = mm2BTail.ReinterpretCast<bfloat16_t>();
                FaL1Tensor<bfloat16_t, L1Format::NZ> l1Tensor{.tensor = mm2BTailTensor,
                                                              .rowCount = MXFP8_V_TAIL_BOUNDARY};
                GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.n2Idx,
                                  .s2Idx = 0,
                                  .dIdx = 0,
                                  .s2DealSize = MXFP8_V_TAIL_BOUNDARY, // Padding is cleared before MTE1 reads L1.
                                  .dDealSize = constInfo_.dSizeV};
                this->copyVTailGmToL1_(l1Tensor, this->vTailGm_, gmCoord);
            }
            // Finish the GM copy before overwriting its padding in L1.
            AscendC::PipeBarrier<PIPE_MTE2>();
            // Clear unused V rows so zero P cannot multiply NaN/Inf padding.
            if (runInfo.kvTailLen < MXFP8_V_TAIL_BOUNDARY) {
                InitConstValueParams<half> vTailPadParams;
                vTailPadParams.repeatTimes = AttentionCommon::Align((uint32_t)constInfo_.dSizeV, 16U) / 16U;
                vTailPadParams.blockNum = MXFP8_V_TAIL_BOUNDARY - runInfo.kvTailLen;
                vTailPadParams.dstGap = runInfo.kvTailLen;
                vTailPadParams.initValue = 0;
                InitConstValue(mm2BTail.ReinterpretCast<bfloat16_t>().ReinterpretCast<half>()[runInfo.kvTailLen * 16U],
                               vTailPadParams);
            }
            // MTE1 must wait for both the GM copy and the padding clear.
            SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
            MatmulFull<bfloat16_t, bfloat16_t, T, 128, dVBaseSize, MXFP_DIVISOR_SIZE, ABLayout::MK, ABLayout::KN,
                       L0AType, L0BType>(mm2A.ReinterpretCast<bfloat16_t>()[pTailOffset],
                                         mm2BTail.ReinterpretCast<bfloat16_t>(), mmL0ABuffers_, mmL0BBuffers_,
                                         mm2ResL0C, tailParam);
            SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId); // 释放KV slot
        }

        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mmResUbTensor = ubBmm2ResBuf_.template ReinterpretCast<T>();
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE + AIV0_AIV1_OFFSET);
        FixpipeMm2(mmResUbTensor, mm2ResL0C, runInfo);
        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T& dstTensor, const LocalTensor<T>& l0C, RunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (constInfo_.dSizeV + 7) >> 3 << 3;

        if constexpr (!USE_DN) {
            fixpipeParams.mSize = (runInfo.actMSize + 31) >> 5 << 5;
            fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
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

    __aicore__ inline void IterateBmm2Dn(uint32_t pL1BufId, RunInfoX& runInfo)
    {
        LocalTensor<uint8_t> mm2A = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        uint32_t pSyncIdx = CC_L1P_0 + pL1BufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(pSyncIdx + AIV0_AIV1_OFFSET);

        constexpr uint32_t pScaleOffset = mBaseSize * s2BaseSize; // pScale在L1的偏移量
        constexpr uint32_t sizeOfInt16 = 2;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            LocalTensor<int16_t> mm2AScaleTensor =
                mm2A.ReinterpretCast<int16_t>()[pScaleOffset / sizeOfInt16 +
                                                mBaseSize * s2SplitSize / MXFP_GROUP_SIZE / sizeOfInt16];
            InitConstValue(mm2AScaleTensor,
                           {1, mBaseSize * s2SplitSize / MXFP_GROUP_SIZE * sizeof(SCALE_T) / blockBytes, 0, 0x7f7f});
        } else if (runInfo.actSingleLoopS2Size <= s2SplitSize) {
            LocalTensor<int16_t> mm2AScaleTensor = mm2A.ReinterpretCast<int16_t>()[pScaleOffset / sizeOfInt16];
            InitConstValue(mm2AScaleTensor,
                           {1, mBaseSize * s2SplitSize / MXFP_GROUP_SIZE * sizeof(SCALE_T) / blockBytes, 0, 0x7f7f});
        }
        LocalTensor<SCALE_T> mm2AScaleFakeTensor = mm2A.ReinterpretCast<SCALE_T>()[pScaleOffset];

        uint32_t l0cBufId = l0cBufId_;
        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        WaitFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);
        LocalTensor<T> mm2ResL0C = l0CBuffers_[l0cBufId * L0C_BUF_BYTES].template ReinterpretCast<T>();

        constexpr uint32_t baseK = s2SplitSize;
        uint64_t l1BaseKOffset = baseK * mBaseSize;
        uint64_t l1ScaleOffset = l1BaseKOffset / MXFP_GROUP_SIZE;
        // Exclude the tail from the main PV task.
        uint32_t tailLen = 0;
        if constexpr (HAS_V_TAIL) {
            if (runInfo.s2Idx + runInfo.actSingleLoopS2Size >= runInfo.actS2Size) {
                tailLen = runInfo.kvTailLen;
            }
        }
        uint32_t kMainSize = runInfo.actSingleLoopS2Size - tailLen;
        uint32_t kLoops = (kMainSize + baseK - 1) / baseK;
        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            uint32_t kvL1BufId = kvL1BufId_;
            kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
            LocalTensor<uint8_t> mm2B = l1KVBuffers_[kvL1BufId * MM_KV_BUF_BYTES];
            if (k == kLoops - 1) {
                realK = kMainSize - k * baseK;
            }
            WaitFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
            LocalTensor<KV_T> mm2BTensor = mm2B.ReinterpretCast<KV_T>();
            if (unlikely(runInfo.isLastS2Loop)) {
                InitValueL1BufferNoTrans(mm2BTensor, realK, (uint32_t)constInfo_.dSizeV);
            }
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(mm2BTensor, baseK + baseK / MXFP_GROUP_SIZE, dVBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyValueSlice(mm2BTensor, runInfo.s2Idx + k * baseK, realK, 0, constInfo_.dSizeV, runInfo);
            uint32_t vScaleOffset = (((realK) + 63) >> 6 << 6) * dVBaseSize;
            LocalTensor<SCALE_T> mm2BScaleTensor = mm2B.ReinterpretCast<SCALE_T>()[vScaleOffset];
            CopyValueScaleSlice(mm2BScaleTensor, (runInfo.s2Idx + k * baseK) / MXFP_DIVISOR_SIZE,
                                (realK + 63) / MXFP_DIVISOR_SIZE, 0, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);

            WaitFlag<HardEvent::MTE2_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId);
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, dVBaseSize, (realK + 63) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE,
                            USE_DN, false, k == 0, k == 0);
            MatmulFullMX<INPUT_T, KV_T, T, 128, dVBaseSize, baseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                mm2A.ReinterpretCast<INPUT_T>()[k * l1BaseKOffset], mm2BTensor, mmL0ABuffers_, mmL0BBuffers_, mm2ResL0C,
                param, mm2AScaleFakeTensor[k * l1ScaleOffset], mm2BScaleTensor);
            SetFlag<HardEvent::MTE1_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId);
        }

        SetFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_BUFFER_ID0 + l0cBufId);

        LocalTensor<T> mmResUbTensor = ubBmm2ResBuf_.template ReinterpretCast<T>();
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_FREE + AIV0_AIV1_OFFSET);
        FixpipeMm2(mmResUbTensor, mm2ResL0C, runInfo);
        SetFlag<HardEvent::FIX_M>(L0C_BUFFER_ID0 + l0cBufId);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM2_READY + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void InitValueL1BufferNAxis(const LocalTensor<KV_T>& valueL1, const uint32_t k, const uint32_t n)
    {
        InitConstValueParams<half> initConstValueParams;
        initConstValueParams.repeatTimes = n / 32U;
        initConstValueParams.blockNum = k;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>(), initConstValueParams);
    }

    __aicore__ inline void InitValueL1BufferNoTrans(const LocalTensor<KV_T>& valueL1, const uint32_t realK,
                                                    const uint32_t curN)
    {
        InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        uint32_t curPadK = (realK + 63) >> 6 << 6;
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

    __aicore__ inline void InitValueL1BufferTrans(const LocalTensor<KV_T>& valueL1, const uint32_t realK,
                                                  const uint32_t curN)
    {
        InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        uint32_t curPadK = ((realK) + 63) >> 6 << 6;
        // nd2nz pading to 64 align, (n,k)->(k1,n1,n0,k0)
        if (curPadK - realK < 32U) {
            return;
        }
        // pad n1, n0, 16 for half
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = AttentionCommon::Align(curN, 16U);
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        uint64_t kAlign = AttentionCommon::Align(realK, 16U) / 32U * 16U;
        offset = curN * kAlign;
        InitConstValue(valueL1.template ReinterpretCast<half>()[offset], initConstValueParams);
    }
}; // QuantFlashAttnBlockCubeMxfp8

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool hasVTail = false>
class QuantFlashAttnBlockCubeMxfp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_V_TAIL = hasVTail;
    // D=192 / D_v=128(config5) 场景 L1 超预算, 定向回退为 3 槽;
    // P 槽深度与 D 无关, producer(AIV) 领先 consumer(AIC) 恒为 2, 3 槽为理论下限
    static constexpr bool USE_3_P_SLOTS = (dBaseSize == 192U && dVBaseSize == 128U);

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using ConstInfoX = std::conditional_t<HAS_V_TAIL, ConstInfoVTail_t, ConstInfo_t>;
    using RunInfoX = std::conditional_t<HAS_V_TAIL, RunInfoVTailX, ::RunInfoX>;
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8Dummy(ConstInfoX& constInfo){};
    __aicore__ inline void InitCrossCoreSync() {}
    __aicore__ inline void UnInitCrossCoreSync() {}
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_

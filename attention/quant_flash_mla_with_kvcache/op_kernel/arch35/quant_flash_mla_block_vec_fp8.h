/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_block_vec_fp8.h
 * \brief QuantFlashMlaWithKvcache FA阶段vec block（自FIA MLA block_vec裁剪适配）
 *        静态张量编程: 所有buffer地址为constexpr偏移, 使用Mutex核内同步 + CrossCore核间同步
 */

#ifndef QUANT_FLASH_MLA_BLOCK_VEC_FP8_H_
#define QUANT_FLASH_MLA_BLOCK_VEC_FP8_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"
#include "quant_flash_mla_with_kvcache_public_def.h"

/* ============静态张量编程所需的宏与常量============= */
#ifndef OFFSET_OF_MEMBER
#define OFFSET_OF_MEMBER(TYPE, MEMBER) ((uint64_t) & ((TYPE*)0)->MEMBER)
#endif
#ifndef SIZE_OF_MEMBER
#define SIZE_OF_MEMBER(TYPE, MEMBER) sizeof(((TYPE*)0)->MEMBER)
#endif
#ifndef BUFFER_SIZE_BYTE_32K
#define BUFFER_SIZE_BYTE_32K 32768U
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <bool first, bool last>
__simd_vf__ inline void QmlaOutputUpdate(__ubuf__ float* dst, __ubuf__ float* cur, __ubuf__ float* exp,
                                         __ubuf__ float* sum, uint16_t rows, float outScale)
{
    using namespace AscendC;
    using namespace AscendC::Reg;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> e0, s0, x00, x01, x02, x03, o00, o01, o02, o03;
    for (uint16_t i = 0; i < rows; ++i) {
        if constexpr (!first) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e0, exp + i);
        }
        if constexpr (last) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sum + i);
        }
        for (uint16_t n = 0; n < 2U; ++n) {
            LoadAlign(x00, cur + (i) * 512U + n * 256U + 0U);
            if constexpr (!first) {
                LoadAlign(o00, dst + (i) * 512U + n * 256U + 0U);
            }
            LoadAlign(x01, cur + (i) * 512U + n * 256U + 64U);
            if constexpr (!first) {
                LoadAlign(o01, dst + (i) * 512U + n * 256U + 64U);
            }
            LoadAlign(x02, cur + (i) * 512U + n * 256U + 128U);
            if constexpr (!first) {
                LoadAlign(o02, dst + (i) * 512U + n * 256U + 128U);
            }
            LoadAlign(x03, cur + (i) * 512U + n * 256U + 192U);
            if constexpr (!first) {
                LoadAlign(o03, dst + (i) * 512U + n * 256U + 192U);
            }
            if constexpr (!first) {
                Mul(o00, o00, e0, mask);
            }
            if constexpr (!first) {
                Mul(o01, o01, e0, mask);
            }
            if constexpr (!first) {
                Mul(o02, o02, e0, mask);
            }
            if constexpr (!first) {
                Mul(o03, o03, e0, mask);
            }
            if constexpr (!first) {
                Add(x00, o00, x00, mask);
            }
            if constexpr (!first) {
                Add(x01, o01, x01, mask);
            }
            if constexpr (!first) {
                Add(x02, o02, x02, mask);
            }
            if constexpr (!first) {
                Add(x03, o03, x03, mask);
            }
            if constexpr (last) {
                Div(x00, x00, s0, mask);
                Muls(x00, x00, outScale, mask);
            }
            if constexpr (last) {
                Div(x01, x01, s0, mask);
                Muls(x01, x01, outScale, mask);
            }
            if constexpr (last) {
                Div(x02, x02, s0, mask);
                Muls(x02, x02, outScale, mask);
            }
            if constexpr (last) {
                Div(x03, x03, s0, mask);
                Muls(x03, x03, outScale, mask);
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + n * 256U + 0U, x00, mask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + n * 256U + 64U, x01, mask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + n * 256U + 128U, x02, mask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + n * 256U + 192U, x03, mask);
        }
    }
}

template <bool FIRST, bool MASK>
__simd_vf__ inline void QmlaSoftmax256(__ubuf__ fp8_e4m3fn_t* out, __ubuf__ float* src, __ubuf__ float* qscale,
                                       __ubuf__ float* oldMax, __ubuf__ float* newMax, __ubuf__ float* newSum,
                                       __ubuf__ uint8_t* maskData, __ubuf__ uint8_t* indices, uint16_t rows,
                                       uint32_t cols, float scale, float kscale)
{
    using namespace AscendC::Reg;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg bytes = CreateMask<uint8_t, MaskPattern::ALL>();
    uint32_t outCount = 128U;
    MaskReg outMask = UpdateMask<fp8_e4m3fn_t>(outCount);
    RegTensor<float> x0, x1, x2, x3, factor, maximum, tmp, minusInf;
    RegTensor<float> e0, e1, e2, e3, sum;
    RegTensor<fp8_e4m3fn_t> c0, c1, packed, ordered;
    RegTensor<uint8_t> idx;
    LoadAlign(idx, indices);
    Duplicate(minusInf, -3.402823466e38f);
    auto maxWrite = newMax;
    UnalignReg maxStore;
    for (uint16_t i = 0; i < rows; ++i) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(factor, qscale + i);
        Muls(factor, factor, scale, all);
        Muls(factor, factor, kscale, all);
        LoadAlign(x0, src + i * 256U + 0U);
        Mul(x0, x0, factor, all);
        uint32_t count0 = cols > 0U ? cols - 0U : 0U;
        MaskReg valid0 = UpdateMask<float>(count0);
        Select(x0, x0, minusInf, valid0);
        if constexpr (MASK) {
            MaskReg masked0;
            LoadAlign<uint32_t, MaskDist::DIST_DS>(masked0, (__ubuf__ uint32_t*)(maskData + i * 256U + 0U));
            Select(x0, minusInf, x0, masked0);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(src + i * 256U + 0U, x0, all);
        LoadAlign(x1, src + i * 256U + 64U);
        Mul(x1, x1, factor, all);
        uint32_t count1 = cols > 64U ? cols - 64U : 0U;
        MaskReg valid1 = UpdateMask<float>(count1);
        Select(x1, x1, minusInf, valid1);
        if constexpr (MASK) {
            MaskReg masked1;
            LoadAlign<uint32_t, MaskDist::DIST_DS>(masked1, (__ubuf__ uint32_t*)(maskData + i * 256U + 64U));
            Select(x1, minusInf, x1, masked1);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(src + i * 256U + 64U, x1, all);
        LoadAlign(x2, src + i * 256U + 128U);
        Mul(x2, x2, factor, all);
        uint32_t count2 = cols > 128U ? cols - 128U : 0U;
        MaskReg valid2 = UpdateMask<float>(count2);
        Select(x2, x2, minusInf, valid2);
        if constexpr (MASK) {
            MaskReg masked2;
            LoadAlign<uint32_t, MaskDist::DIST_DS>(masked2, (__ubuf__ uint32_t*)(maskData + i * 256U + 128U));
            Select(x2, minusInf, x2, masked2);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(src + i * 256U + 128U, x2, all);
        LoadAlign(x3, src + i * 256U + 192U);
        Mul(x3, x3, factor, all);
        uint32_t count3 = cols > 192U ? cols - 192U : 0U;
        MaskReg valid3 = UpdateMask<float>(count3);
        Select(x3, x3, minusInf, valid3);
        if constexpr (MASK) {
            MaskReg masked3;
            LoadAlign<uint32_t, MaskDist::DIST_DS>(masked3, (__ubuf__ uint32_t*)(maskData + i * 256U + 192U));
            Select(x3, minusInf, x3, masked3);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(src + i * 256U + 192U, x3, all);
        Max(maximum, x0, x1, all);
        Max(tmp, x2, x3, all);
        Max(maximum, maximum, tmp, all);
        Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(maximum, maximum, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maxWrite, maximum, maxStore, 1);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(maxWrite, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    if constexpr (!FIRST) {
        LoadAlign(maximum, newMax);
        LoadAlign(tmp, oldMax);
        Max(maximum, maximum, tmp, all);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(newMax, maximum, all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto sumWrite = newSum;
    UnalignReg sumStore;
    for (uint16_t i = 0; i < rows; ++i) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(maximum, newMax + i);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(x0, x1, src + i * 256U);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(x2, x3, src + i * 256U + 128U);
        ExpSub(e0, x0, maximum, all);
        ExpSub(e1, x1, maximum, all);
        ExpSub(e2, x2, maximum, all);
        ExpSub(e3, x3, maximum, all);
        Add(sum, e0, e1, all);
        Add(tmp, e2, e3, all);
        Add(sum, sum, tmp, all);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(sum, sum, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumWrite, sum, sumStore, 1);
        Muls(e0, e0, 448.0f, all);
        Muls(e1, e1, 448.0f, all);
        Cast<fp8_e4m3fn_t, float, castTraitRintZero>(c0, e0, all);
        Cast<fp8_e4m3fn_t, float, castTraitRintTwo>(c1, e1, all);
        Or((RegTensor<uint8_t>&)packed, (RegTensor<uint8_t>&)c0, (RegTensor<uint8_t>&)c1, bytes);
        Gather(ordered, packed, idx);
        auto out0 = out + i * 32U + 0U;
        StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(out0, ordered, 33U, 1U,
                                                                                               outMask);
        Muls(e2, e2, 448.0f, all);
        Muls(e3, e3, 448.0f, all);
        Cast<fp8_e4m3fn_t, float, castTraitRintZero>(c0, e2, all);
        Cast<fp8_e4m3fn_t, float, castTraitRintTwo>(c1, e3, all);
        Or((RegTensor<uint8_t>&)packed, (RegTensor<uint8_t>&)c0, (RegTensor<uint8_t>&)c1, bytes);
        Gather(ordered, packed, idx);
        auto out1 = out + i * 32U + 4224U;
        StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(out1, ordered, 33U, 1U,
                                                                                               outMask);
    }
    StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumWrite, sumStore, 0);
}

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
    LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned576,
    DTemplateType dVTemplateType = DTemplateType::Aligned512, bool hasAtten = false, uint8_t KvLayoutType = 0,
    bool isFd = false, bool bmm2Write2Ub = true>
class QuantFlashMlaBlockVecFp8 {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2

    static constexpr uint32_t initOutputEventId = 10U;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND ||
         layout == LayOutTypeEnum::LAYOUT_SBH) ?
            MaskFormat::SG :
            MaskFormat::GS;

    static constexpr bool USE_DN = false;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool IS_PER_TOKEN_HEAD = true;
    // MLA全量化: Q per-token-head descale, TND下[T, N1]排布(TNG)
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN, IS_PER_TOKEN_HEAD, true>();

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    uint32_t negativeIntScalar = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using ConstInfoX = QmlaConstInfo;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // Q: TND, cu_seq(含前导0)+seq_used, int32
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;

    /* =====================核间同步ID（与cube block一致）==================== */
    // Cross-core sync (CROSS_CORE_SYNC_MODE = 4)
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CC_MM_0 = 0U;  // bmm1 result slot 0
    static constexpr uint32_t CC_MM_1 = 1U;  // bmm1 result slot 1
    static constexpr uint32_t CC_MM_2 = 2U;  // bmm2 result slot 0
    static constexpr uint32_t CC_MM_3 = 3U;  // bmm2 result slot 1
    static constexpr uint32_t CC_L1P_0 = 4U; // L1 P slot 0 (AIV→AIC)
    static constexpr uint32_t CC_L1P_1 = 5U;
    static constexpr uint32_t CC_L1P_2 = 6U;

    // Vec internal Mutex/Event IDs
    // Mutex ID 和 SetFlag/WaitFlag Event ID 共享同一硬件池(0-27)
    // Mutex 用 0-7 (vec1/vec2/lse/mask), SetFlag/WaitFlag 用 8-9 (Q scale)
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0; // vec2 result V↔MTE3 (Mutex)
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2; // vec1 result slot 0 V↔MTE3 (Mutex)
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3; // vec1 result slot 1 (Mutex)
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;  // LSE/broadcast output (Mutex)
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;  // (Mutex)
    static constexpr uint32_t UB_IN_MASK_EVENT0 = 6;      // mask input MTE2↔V (Mutex)
    static constexpr uint32_t UB_IN_MASK_EVENT1 = 7;      // (Mutex)
    static constexpr uint32_t UB_IN_QSCALE_EVENT0 = 8;    // Q scale input MTE2↔V (SetFlag)
    static constexpr uint32_t UB_IN_QSCALE_EVENT1 = 9;    // (SetFlag)

    /* =====================Buffer尺寸常量==================== */
    // UB cross-core region (same as cube block)
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);

    // L1 (shared with cube, same offsets)
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);

    // Vec-specific UB buffer sizes
    static constexpr uint32_t UB_STAGE2_OUT_BUF_BYTES = (mBaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t UB_STAGE1_OUT_BUFCNT = 1U;
    static constexpr uint32_t UB_STAGE1_OUT_BUF_BYTES = (mBaseSize / CV_RATIO + 1U) * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t UB_MASK_BUFCNT = 2U;
    static constexpr uint32_t UB_MASK_BUF_BYTES = (mBaseSize / CV_RATIO) * s2BaseSize;
    static constexpr uint32_t UB_SOFTMAX_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U;
    static constexpr uint32_t UB_SOFTMAX_BUF_ELEMS = UB_SOFTMAX_BUF_BYTES / sizeof(T);
    static constexpr uint32_t UB_QSCALE_BUFCNT = 2U;
    static constexpr uint32_t UB_QSCALE_BUF_BYTES = (mBaseSize / CV_RATIO) * sizeof(float);
    static constexpr uint32_t UB_PSCALE_BUFCNT = 3U;
    static constexpr uint32_t UB_PSCALE_BUF_BYTES = 256U;
    static constexpr uint32_t UB_COMMON_TMP_BUF_BYTES = 512U;
    static constexpr uint32_t UB_VSELR_INDEXES_BUFCNT = 4U;
    static constexpr uint32_t UB_VSELR_INDEXES_BUF_BYTES = 128U;
    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 1U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8U;
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = (mBaseSize / CV_RATIO) * 8U * sizeof(float);

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    QSeqParserType* qSeqParserPtr_ = nullptr;
    GlobalTensor<uint8_t> attenMaskGmInt_;
    GlobalTensor<float> deScaleQGm_;
    GlobalTensor<float> deScaleKGm_;
    GlobalTensor<float> deScaleVGm_;
    FaGmTensor<float, Q_SCALE_FORMAT, int32_t, true> queryScaleGm_;
    CopyQueryScaleGmToUb<float, Q_SCALE_FORMAT> copyQueryScaleGmToUb_;
    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    /* =====================静态LocalTensor变量==================== */
    // UB (shared with cube, same offsets for cross-core region)
    LocalTensor<uint8_t> ubMmResBuffers_;
    // L1 (shared with cube, same offsets)
    LocalTensor<uint8_t> l1PBuffers_;
    // Vec-specific UB buffers
    LocalTensor<T> stage2OutBuf_;
    LocalTensor<uint8_t> stage1OutBufs_;
    LocalTensor<uint8_t> maskInBufs_;
    LocalTensor<T> softmaxSumBuf_;
    LocalTensor<T> softmaxMaxBuf_;
    LocalTensor<T> softmaxExpBuf_;
    LocalTensor<T> preLoopMaxBuf_;
    LocalTensor<T> preLoopSumBuf_;
    LocalTensor<T> firstLoopSumBuf_;
    LocalTensor<float> qScaleInputBufs_;
    LocalTensor<T> pScaleBufs_;
    LocalTensor<uint8_t> commonTBuf_;
    LocalTensor<float> lseOutBuf_;
    LocalTensor<float> maxBrdcstBuf_;
    LocalTensor<float> sumBrdcstBuf_;

    // TBuf for vselrIndexes (ProcessVec1Vf API requires TBuf<>*)
    TBuf<> vselrIndexesBuf_[UB_VSELR_INDEXES_BUFCNT];

    // Buffer ID trackers
    uint32_t mmResUbBufId_ = 0U;
    uint32_t vec1ResUbBufId_ = 0U;
    uint32_t lseOutUbBufId_ = 0U;

    const ConstInfoX& constInfo_;
    T negativeFloatScalar_;
    float deScaleKValue_{1.0f};
    float deScaleVValue_{1.0f};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32};

    __aicore__ inline QuantFlashMlaBlockVecFp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                        __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                        QSeqParserType& qParser, __gm__ uint8_t* attenMask)
    {
        qSeqParserPtr_ = &qParser;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T*)&tmp1);

        InitVecInput(dequantScaleQuery, dequantScaleKey, dequantScaleValue, softmaxLse, attentionOut, workspace,
                     attenMask);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                        __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                        __gm__ uint8_t* attenMask)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }

        if constexpr (HAS_MASK) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }

        // MLA全量化dequantScale: Q per-token-head, KV per-tensor
        if (dequantScaleQuery != nullptr) {
            deScaleQGm_.SetGlobalBuffer((__gm__ float*)dequantScaleQuery);
            InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, 1,
                             queryScaleGm_, dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            deScaleKGm_.SetGlobalBuffer((__gm__ float*)dequantScaleKey);
            deScaleKValue_ = this->deScaleKGm_.GetValue(0);
        }
        if (dequantScaleValue != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float*)dequantScaleValue);
            deScaleVValue_ = this->deScaleVGm_.GetValue(0);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float*)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim,
                                            FaGmTensor<float, Q_SCALE_FORMAT, int32_t, true>& qScaleGmTensor,
                                            __gm__ uint8_t* gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float*)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_TN) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, *qSeqParserPtr_);
        }
    }

    // MLA q_descale: 复用common层 CopyQueryScaleGmToUb
    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float>& dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX& runInfo)
    {
        FaUbTensor<float> ubTensor{
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoordGs1Merge gmCoord{
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb_(ubTensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float>& dstTensor, RunInfoX& runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // L1 P 三缓冲（与cube block相同偏移）
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, mBaseSize * 576U, 3U * 576U * s2BaseSize);

        /*--------------------------------------------UB--------------------------------------------*/
        struct UbLayout {
            /* Cross-core region (same as cube block) */
            uint8_t bmm1Res[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES]; // 2 * 16384 = 32768
            uint8_t bmm2Res[UB_MM2_RES_BUFCNT][UB_MM2_RES_BUF_BYTES]; // 2 * 65536 = 131072
            /* Vec-specific region */
            uint8_t stage2Out[UB_STAGE2_OUT_BUF_BYTES];                       // 65536
            uint8_t stage1Out[UB_STAGE1_OUT_BUFCNT][UB_STAGE1_OUT_BUF_BYTES]; // 2 * 4224 = 8448
            uint8_t maskIn[UB_MASK_BUFCNT][UB_MASK_BUF_BYTES];                // 2 * 4096 = 8192
            uint8_t softmaxSum[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES];      // 3 * 256 = 768
            uint8_t softmaxMax[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES];      // 3 * 256 = 768
            uint8_t softmaxExp[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES];      // 3 * 256 = 768
            uint8_t preLoopMax[UB_SOFTMAX_BUF_BYTES];                         // 256
            uint8_t preLoopSum[UB_SOFTMAX_BUF_BYTES];                         // 256
            uint8_t firstLoopSum[UB_SOFTMAX_BUF_BYTES];                       // 256
            uint8_t qScaleIn[UB_QSCALE_BUFCNT][UB_QSCALE_BUF_BYTES];          // 2 * 128 = 256
            uint8_t pScale[UB_PSCALE_BUFCNT][UB_PSCALE_BUF_BYTES];            // 3 * 256 = 768
            uint8_t commonTmp[UB_COMMON_TMP_BUF_BYTES];                       // 512
            uint8_t lseOut[UB_LSE_OUT_BUFCNT][UB_LSE_OUT_BUF_BYTES];          // 2 * 1024 = 2048
            uint8_t maxBrdcst[UB_BRDCST_BUF_BYTES];                           // 2048
            uint8_t sumBrdcst[UB_BRDCST_BUF_BYTES];                           // 2048
        };
        static_assert(sizeof(UbLayout) <= 256 * 1024, "UB buffer too large");

        // Cross-core region
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, 0,
                                               SIZE_OF_MEMBER(UbLayout, bmm1Res) + SIZE_OF_MEMBER(UbLayout, bmm2Res));

        // Vec-specific buffers
        stage2OutBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, stage2Out),
                                             SIZE_OF_MEMBER(UbLayout, stage2Out))
                            .template ReinterpretCast<T>();
        stage1OutBufs_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, stage1Out),
                                              SIZE_OF_MEMBER(UbLayout, stage1Out));
        maskInBufs_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, maskIn),
                                           SIZE_OF_MEMBER(UbLayout, maskIn));
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxSum),
                                              SIZE_OF_MEMBER(UbLayout, softmaxSum))
                             .template ReinterpretCast<T>();
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxMax),
                                              SIZE_OF_MEMBER(UbLayout, softmaxMax))
                             .template ReinterpretCast<T>();
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxExp),
                                              SIZE_OF_MEMBER(UbLayout, softmaxExp))
                             .template ReinterpretCast<T>();
        preLoopMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, preLoopMax),
                                              SIZE_OF_MEMBER(UbLayout, preLoopMax))
                             .template ReinterpretCast<T>();
        preLoopSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, preLoopSum),
                                              SIZE_OF_MEMBER(UbLayout, preLoopSum))
                             .template ReinterpretCast<T>();
        firstLoopSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, firstLoopSum),
                                                SIZE_OF_MEMBER(UbLayout, firstLoopSum))
                               .template ReinterpretCast<T>();
        qScaleInputBufs_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, qScaleIn),
                                                SIZE_OF_MEMBER(UbLayout, qScaleIn))
                               .template ReinterpretCast<float>();
        pScaleBufs_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, pScale), SIZE_OF_MEMBER(UbLayout, pScale))
                .template ReinterpretCast<T>();
        commonTBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, commonTmp),
                                           SIZE_OF_MEMBER(UbLayout, commonTmp));
        lseOutBuf_ =
            LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, lseOut), SIZE_OF_MEMBER(UbLayout, lseOut))
                .template ReinterpretCast<float>();
        maxBrdcstBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, maxBrdcst),
                                             SIZE_OF_MEMBER(UbLayout, maxBrdcst))
                            .template ReinterpretCast<float>();
        sumBrdcstBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, sumBrdcst),
                                             SIZE_OF_MEMBER(UbLayout, sumBrdcst))
                            .template ReinterpretCast<float>();

        // vselrIndexesBuf: ProcessVec1Vf API requires TBuf<>*
        // 先用一个dummy TBuf推进TPipe内部UB指针到静态布局之后，避免与跨核bmm区域冲突
        TBuf<> ubLayoutPadBuf;
        GetTPipePtr()->InitBuffer(ubLayoutPadBuf, sizeof(UbLayout));
        GetTPipePtr()->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)],
                                  UB_VSELR_INDEXES_BUF_BYTES);
        GetTPipePtr()->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);

        LocalTensor<uint8_t> vselrIndexesTensor =
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        vselrIndexesTensor.SetValue(0, 0x7f);
        for (int i = 0; i < 128; i++) {
            vselrIndexesTensor.SetValue(i, i << 1);
        }
        vselrIndexesTensor =
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 64; i++) {
            vselrIndexesTensor.SetValue(i, i << 2);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // Pre-arm: V→MTE2 for Q scale (initial state: V "done" so MTE2 can proceed)
        SetFlag<HardEvent::V_MTE2>(UB_IN_QSCALE_EVENT0);
        SetFlag<HardEvent::V_MTE2>(UB_IN_QSCALE_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::V_MTE2>(UB_IN_QSCALE_EVENT0);
        WaitFlag<HardEvent::V_MTE2>(UB_IN_QSCALE_EVENT1);
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        // CROSS_CORE_SYNC: 每个slot set一个flag
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_1);
        if constexpr (BMM2_TOUB) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_2);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_3);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX& runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize;

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.realGSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = 0; // 新接口无batch维度mask张量, causal在UB内动态构造
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool isSkipMask = IsSkipAttentionmask(maskInfo);
        if (unlikely(!isSkipMask)) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUb, attenMaskGmInt_, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
        }
    }

    __aicore__ inline void ProcessVec1(RunInfoX runInfo)
    {
        uint32_t mmResUbBufId = mmResUbBufId_;
        mmResUbBufId_ = (mmResUbBufId_ + 1) % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        uint32_t l1CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;

        LocalTensor<INPUT_T> pL1Tensor =
            l1PBuffers_[pL1BufId * 576U * s2BaseSize + 512U * s2BaseSize].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMmResBuffers_[mmResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1CrossCoreSyncIdx);
            return;
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        ProcessVec1Nd(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1CrossCoreSyncIdx);
        Vec1PostProcess(runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qSeqParserPtr_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum);
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(attentionOutGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qSeqParserPtr_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum);
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 3e+99);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T>& pL1Tensor, LocalTensor<T>& mm1ResUbTensor,
                                         RunInfoX runInfo)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;

        LocalTensor<float> sumUb =
            softmaxSumBuf_[(runInfo.mloop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS].template ReinterpretCast<float>();
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS].template ReinterpretCast<float>();
        LocalTensor<float> expUb =
            softmaxExpBuf_[(runInfo.loop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS].template ReinterpretCast<float>();
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf_;

        int64_t stage1Offset = runInfo.loop % UB_STAGE1_OUT_BUFCNT;

        float descaleQK = 1.0f;

        LocalTensor<float> queryScaleUb =
            qScaleInputBufs_[(runInfo.mloop % DB) * (UB_QSCALE_BUF_BYTES / sizeof(float))];
        LocalTensor<T> pScaleUb;

        if constexpr (HAS_MASK) {
            const uint32_t maskBufId = runInfo.loop & (DB - 1);
            attenMaskUb = maskInBufs_[maskBufId * UB_MASK_BUF_BYTES];
            Mutex::Lock<PIPE_MTE2>(UB_IN_MASK_EVENT0 + maskBufId);
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
            Mutex::Unlock<PIPE_MTE2>(UB_IN_MASK_EVENT0 + maskBufId);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        // MLA全量化: 首个S2 loop加载per-token Q scale, 取pScale buffer
        if (unlikely(runInfo.isFirstS2Loop)) {
            queryScaleUb = qScaleInputBufs_[(runInfo.mloop % DB) * (UB_QSCALE_BUF_BYTES / sizeof(float))];
            WaitFlag<HardEvent::V_MTE2>(UB_IN_QSCALE_EVENT0 + runInfo.mloop % DB);
            CopyQueryScaleTile(queryScaleUb, runInfo);
            SetFlag<HardEvent::MTE2_V>(UB_IN_QSCALE_EVENT0 + runInfo.mloop % DB);
            WaitFlag<HardEvent::MTE2_V>(UB_IN_QSCALE_EVENT0 + runInfo.mloop % DB);
        }
        pScaleUb = pScaleBufs_[(runInfo.loop % 3) * (UB_PSCALE_BUF_BYTES / sizeof(T))];

        LocalTensor<T> mmRes = mm1ResUbTensor;
        auto stage1CastTensor =
            stage1OutBufs_[stage1Offset * UB_STAGE1_OUT_BUF_BYTES].template ReinterpretCast<INPUT_T>();

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + stage1Offset);

        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        auto idx = vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        auto tmp = apiTmpBuffer.template ReinterpretCast<float>();
        if (runInfo.isFirstS2Loop) {
            QmlaSoftmax256<true, HAS_MASK>(
                (__ubuf__ fp8_e4m3fn_t*)stage1CastTensor.GetPhyAddr(), (__ubuf__ float*)mmRes.GetPhyAddr(),
                (__ubuf__ float*)queryScaleUb.GetPhyAddr(), (__ubuf__ float*)maxUb.GetPhyAddr(),
                (__ubuf__ float*)maxUb.GetPhyAddr(), (__ubuf__ float*)sumUb.GetPhyAddr(),
                (__ubuf__ uint8_t*)attenMaskUb.GetPhyAddr(), (__ubuf__ uint8_t*)idx.GetPhyAddr(), runInfo.actVecMSize,
                s2CalcSize, constInfo_.scaleValue, deScaleKValue_);
        } else {
            QmlaSoftmax256<false, HAS_MASK>(
                (__ubuf__ fp8_e4m3fn_t*)stage1CastTensor.GetPhyAddr(), (__ubuf__ float*)mmRes.GetPhyAddr(),
                (__ubuf__ float*)queryScaleUb.GetPhyAddr(), (__ubuf__ float*)maxUb.GetPhyAddr(),
                (__ubuf__ float*)tmp[64].GetPhyAddr(), (__ubuf__ float*)tmp.GetPhyAddr(),
                (__ubuf__ uint8_t*)attenMaskUb.GetPhyAddr(), (__ubuf__ uint8_t*)idx.GetPhyAddr(), runInfo.actVecMSize,
                s2CalcSize, constInfo_.scaleValue, deScaleKValue_);
        }

        if constexpr (HAS_MASK) {
            const uint32_t maskBufId = runInfo.loop & (DB - 1);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        // ===================DataCopy to L1 ====================
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + stage1Offset);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + stage1Offset);

        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo_.subBlockIdx * (mBaseSize * 16);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2BaseSize / 32, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + stage1Offset);
        vec1ResUbBufId_ = (vec1ResUbBufId_ + 1) % UB_STAGE1_OUT_BUFCNT;
    }

    __aicore__ inline void Vec1PostProcess(RunInfoX runInfo)
    {
        LocalTensor<float> sumUb =
            softmaxSumBuf_[(runInfo.mloop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS].template ReinterpretCast<float>();
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS].template ReinterpretCast<float>();
        LocalTensor<float> expUb =
            softmaxExpBuf_[(runInfo.loop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS].template ReinterpretCast<float>();
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf_;

        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::V_MTE2>(UB_IN_QSCALE_EVENT0 + runInfo.mloop % DB);
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float>& sumUb, LocalTensor<float>& maxUb)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
            if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             RunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = runInfo.actVecMSize;
        uint32_t gmDealRowCount = runInfo.actVecMSize;

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        LocalTensor<float> lseUb = lseOutBuf_[lseOutUbBufId_ * (UB_LSE_OUT_BUF_BYTES / sizeof(float))];
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);

        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        // 新接口LSE输出统一为NT格式[N2*G, T]
        uint32_t prefixBS1 = qSeqParserPtr_->GetTBase(runInfo.bIdx);
        uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
        DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                 gmDealRowCount, constInfo_);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                               LocalTensor<float>& maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(sumBrdcstBuf_, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDSumGm_[gmOffset], sumBrdcstBuf_, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;

        // Copy max to gm
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(maxBrdcstBuf_, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxBrdcstBuf_, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                                       LocalTensor<float>& maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        // bmm2 result UB双buffer: 消费slot n时cube并行写slot n^1
        uint32_t mm2ResUbBufId = runInfo.loop % UB_MM2_RES_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_2 + mm2ResUbBufId;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        if constexpr (BMM2_TOUB) {
            ProcessVec2OnUb(runInfo, mmSyncIdx, mm2ResUbBufId);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(RunInfoX runInfo, uint32_t mmSyncIdx, uint32_t mm2ResUbBufId)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            return;
        }
        uint32_t vecMSize = runInfo.actVecMSize;
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = stage2OutBuf_;
        LocalTensor<T> mmRes =
            ubMmResBuffers_[UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES + mm2ResUbBufId * UB_MM2_RES_BUF_BYTES]
                .template ReinterpretCast<T>();

        auto exp = softmaxExpBuf_[(runInfo.loop % 3U) * UB_SOFTMAX_BUF_ELEMS];
        auto sum = softmaxSumBuf_[(runInfo.mloop % 3U) * UB_SOFTMAX_BUF_ELEMS];
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        if (runInfo.isFirstS2Loop) {
            if (runInfo.isLastS2Loop) {
                QmlaOutputUpdate<true, true>((__ubuf__ float*)vec2ResUb.GetPhyAddr(),
                                             (__ubuf__ float*)mmRes.GetPhyAddr(), (__ubuf__ float*)exp.GetPhyAddr(),
                                             (__ubuf__ float*)sum.GetPhyAddr(), vecMSize, deScaleVValue_ / 448.0f);
            } else {
                QmlaOutputUpdate<true, false>((__ubuf__ float*)vec2ResUb.GetPhyAddr(),
                                              (__ubuf__ float*)mmRes.GetPhyAddr(), (__ubuf__ float*)exp.GetPhyAddr(),
                                              (__ubuf__ float*)sum.GetPhyAddr(), vecMSize, deScaleVValue_ / 448.0f);
            }
        } else {
            if (runInfo.isLastS2Loop) {
                QmlaOutputUpdate<false, true>((__ubuf__ float*)vec2ResUb.GetPhyAddr(),
                                              (__ubuf__ float*)mmRes.GetPhyAddr(), (__ubuf__ float*)exp.GetPhyAddr(),
                                              (__ubuf__ float*)sum.GetPhyAddr(), vecMSize, deScaleVValue_ / 448.0f);
            } else {
                QmlaOutputUpdate<false, false>((__ubuf__ float*)vec2ResUb.GetPhyAddr(),
                                               (__ubuf__ float*)mmRes.GetPhyAddr(), (__ubuf__ float*)exp.GetPhyAddr(),
                                               (__ubuf__ float*)sum.GetPhyAddr(), vecMSize, deScaleVValue_ / 448.0f);
            }
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        if (runInfo.isLastS2Loop) {
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize);
        }
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);

        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX& runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = (constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow;
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor = softmaxMaxBuf_[(runInfo.mloop % (PRELOAD_N + 1)) * UB_SOFTMAX_BUF_ELEMS]
                                                   .template ReinterpretCast<float>()[mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX& runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = (uint32_t)dTemplateAlign64};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.realN2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T>& ubTensor, GmCoordGs1Merge& gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qSeqParserPtr_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qSeqParserPtr_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else { // BNSD
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
};

// AIC编译单元使用的Vec空壳: kernel在AIV分支外不会实例化Vec接口, 仅需类型/常量成员
template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
    LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned576,
    DTemplateType dVTemplateType = DTemplateType::Aligned512, bool hasAtten = false, uint8_t KvLayoutType = 0,
    bool isFd = false, bool bmm2Write2Ub = true>
class QuantFlashMlaBlockVecFp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = QmlaConstInfo;
    __aicore__ inline QuantFlashMlaBlockVecFp8Dummy(ConstInfoX& constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_BLOCK_VEC_FP8_H_

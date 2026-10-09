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
 * \file mixed_quant_flash_attn_utils.h
 * \brief
 */

#ifndef MIXED_QUANT_FLASH_ATTN_UTILS_H_
#define MIXED_QUANT_FLASH_ATTN_UTILS_H_

#include "util.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../../../common/op_kernel/memcopy/offset_calculator_v2.h"
#include "mixed_quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"

using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;
using namespace regbaseutil;

template <
    typename Q_T, typename OUTPUT_T, typename KV_T, typename KV_SCALE_T, bool PAGE_ATTENTION, LayOutTypeEnum LAYOUT_Q,
    uint8_t LAYOUT_KV, LayOutTypeEnum LAYOUT_OUT, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    bool HAS_MASK = false, bool USE_DN = false, uint8_t QUANT_COMPUTE_MODE = 0, typename... Args>
struct MQFAType {
    using qType = Q_T;
    using outType = OUTPUT_T;
    using kvType = KV_T;
    using kvScaleType = KV_SCALE_T;

    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LayOutTypeEnum layoutQ = LAYOUT_Q;
    static constexpr uint8_t layoutKV = LAYOUT_KV;
    static constexpr LayOutTypeEnum layoutOut = LAYOUT_OUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr bool useDN = USE_DN;
    static constexpr uint8_t quantComputeMode = QUANT_COMPUTE_MODE;
};

template <FA_LAYOUT LAYOUT_T, typename SEQLEN_T>
class SeqLensTool {
public:
    ActualSeqLensParser<ActualSeqLensMode::ACCUM, SEQLEN_T, true> cuSeqLensParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, SEQLEN_T, false> seqUsedParser;

    __aicore__ inline void Init(__gm__ uint8_t* cuSeqLensGmAddr, uint32_t cuSeqLensDims, __gm__ uint8_t* seqUsedGmAddr,
                                uint32_t seqUsedDims, uint64_t defaultSeqUsedVal)
    {
        cuSeqLensParser.Init(cuSeqLensGmAddr, cuSeqLensDims, seqUsedGmAddr, seqUsedDims);
        seqUsedParser.Init(seqUsedGmAddr, seqUsedDims, defaultSeqUsedVal);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx)
    {
        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            return cuSeqLensParser.GetActualSeqLength(bIdx);
        } else {
            return seqUsedParser.GetActualSeqLength(bIdx);
        }
    }
};

#define MOD2(x) ((x) & 1)

__aicore__ constexpr uint16_t Align64Func(uint16_t data)
{
    return (data + 63) >> 6 << 6;
}

template <>
struct GmLayoutParams<GmFormat::PA_BnNDBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnNDBs;
};

template <>
struct GmLayoutParams<GmFormat::BNDS> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BNDS;
};

template <>
struct GmLayoutParams<GmFormat::PA_NZ_V_SCALE> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_V_SCALE_PA_NZ;
};

template <>
struct GmLayout<GmFormat::BNDS> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t d, uint32_t s)
    {
        shape = AscendC::MakeShape(b, n, d, s);
        uint64_t sStride = 1;
        uint64_t dStride = sStride * s;
        uint64_t nStride = dStride * d;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, dStride, sStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_NZ_V_SCALE> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d1, uint32_t d0)
    {
        shape = AscendC::MakeShape(n, d1, blockSize, d0);
        uint64_t d0Stride = 1;
        uint64_t bsStride = d0Stride * d0;
        uint64_t d1Stride = bsStride * blockSize;
        uint64_t nStride = d1Stride * d1;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, d1Stride, bsStride, d0Stride);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnNDBs, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, uint32_t blockSize, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, d, blockSize, 0ULL, 0ULL);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize();
        uint64_t bsIdx = s2Idx % GetBlockSize();
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + dIdx * GetStrideD() + bsIdx * GetStrideBlockSize();
        return offset;
    }

    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BNDS, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t d, uint32_t s2)
    {
        gmLayout.MakeLayout(b, n2, d, s2);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + dIdx * GetStrideD() + s2Idx * GetStrideS2();
        return offset;
    }

    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<3>(gmLayout.stride);
    }

    __aicore__ inline uint32_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimD()
    {
        return AscendC::Std::get<2>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimS2()
    {
        return AscendC::Std::get<3>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_V_SCALE_PA_NZ, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;
    static constexpr uint32_t d0 = 16;
    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d1, uint32_t d0s0,
                                GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d1, d0s0);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize();
        uint64_t bsIdx = s2Idx % GetBlockSize();
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t bs1Idx = bsIdx / (GetD0BS0() / d0);
        uint32_t d1Idx = dIdx / d0;
        uint32_t d0Idx = dIdx % d0;
        uint64_t offset = blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + d1Idx * GetStrideD1() +
                          bs1Idx * GetStrideBS1() + d0Idx * GetStrideD0BS0() * (GetD0BS0() / d0);
        return offset;
    }

    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD1()
    {
        return AscendC::Std::get<2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBS1()
    {
        return AscendC::Std::get<3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD0BS0()
    {
        return AscendC::Std::get<4>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD1()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBS1()
    {
        return AscendC::Std::get<2>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD0BS0()
    {
        return AscendC::Std::get<3>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return GetBS1() * GetD0BS0() / d0;
    }
};

template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
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
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0>
__aicore__ inline constexpr GmFormat GetKDescaleGmFormat()
{
    if constexpr (kvLayoutType == 0) {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        }
    } else if constexpr (kvLayoutType == 1) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) {
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ_K_SCALE;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0>
__aicore__ inline constexpr GmFormat GetVDescaleGmFormat()
{
    if constexpr (kvLayoutType == 0) {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNDS;
        }
    } else if constexpr (kvLayoutType == 1) {
        return GmFormat::PA_BnNDBs;
    } else if constexpr (kvLayoutType == 2) {
        return GmFormat::PA_BnNDBs;
    } else {
        return GmFormat::PA_NZ_V_SCALE;
    }
}

uint32_t ceil(double x)
{
    uint32_t intPart = static_cast<uint32_t>(x);
    double epsilon = 1e-9;
    double diff = std::abs(x - static_cast<double>(intPart));
    return diff < epsilon ? intPart : intPart + 1;
}

template <typename qType, uint8_t quantComputeMode>
struct TypeLookup {
    using kv_dtype = void;
    using kvscale_dtype = void;
};

template <typename qType>
struct TypeLookup<qType, A16C4_KV_MXFP4_SOFTMAX_FP32> {
    using kv_dtype = fp4x2_e2m1_t;
    using kvscale_dtype = fp8_e8m0_t;
};

template <typename qType>
struct TypeLookup<qType, A16C4_KV_HIF4_SOFTMAX_FP32> {
    using kv_dtype = hifloat4x2_t;
    using kvscale_dtype = hif4_scale;
};

template <typename qType>
struct TypeLookup<qType, AntiquantMode_FP8_PC> {
    using kv_dtype = float8_e4m3_t;
    using kvscale_dtype = std::conditional_t<std::is_same_v<qType, bfloat16_t>, bfloat16_t, half>;
};

#endif // MIXED_QUANT_FLASH_ATTN_UTILS_H_

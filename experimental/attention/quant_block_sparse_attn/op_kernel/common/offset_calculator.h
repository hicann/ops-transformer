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
 * \file offset_calculator.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_H
#define OFFSET_CALCULATOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::GlobalTensor;

// ----------------------------------------------GmLayout--------------------------------
enum class GmFormat {
    TND = 0,
    TNGD = 1,
    PA_BnBsND = 2,
    PA_BnNBsD = 3,
    PA_BnNBs = 4,
    BNGSD = 5,
    NGTD = 6,
};

template <GmFormat FORMAT>
struct GmLayout {};

template <>
struct GmLayout<GmFormat::TNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::TND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::NGTD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t tStride = dStride * d;
        uint64_t gStride = tStride * t;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnBsND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d)
    {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t bsStride = nStride * n;
        uint64_t bnStride = bsStride * blockSize;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnNBsD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d)
    {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t bsStride = dStride * d;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnNBs> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize)
    {
        shape = AscendC::MakeShape(n, blockSize);
        uint64_t bsStride = 1;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride);
    }
};

// ----------------------------------------------ActualSeqLensParser--------------------------------
enum class ActualSeqLensMode {
    BY_BATCH = 0,
    ACCUM = 1,
};

template <ActualSeqLensMode MODE, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
class ActualSeqLensParser {};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, false> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims,
                                uint64_t defaultVal = 0)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    }

    __aicore__ inline uint64_t GetMxVscaleTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        uint64_t vScaleTBaseOffset = 0;
        for (uint32_t idx = 0; idx < bIdx; idx++) {
            vScaleTBaseOffset += ((GetActualSeqLength(idx) + 63) >> 6);
        }
        return vScaleTBaseOffset;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return (actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    }

    __aicore__ inline uint64_t GetFullActualSeqLength(uint32_t bIdx) const
    {
        return GetActualSeqLength(bIdx);
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return actualSeqLengthsGm.GetValue(actualLenDims - 1);
    }

private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <typename ACTLEN_T>
class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T, false> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline uint32_t GetActualLenDims() const
    {
        return actualLenDims;
    }

private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims = 0;
    uint64_t defaultVal = 0;
};

// ----------------------------------------------BlockTableParser--------------------------------
class BlockTableParser {
public:
    __aicore__ inline BlockTableParser() = default;

    __aicore__ inline void Init(GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        this->blockTableGm = blockTableGm;
        this->maxblockNumPerBatch = maxblockNumPerBatch;
    }

    __aicore__ inline int32_t GetBlockIdx(uint32_t bIdx, uint64_t blockIdxInBatch) const
    {
        uint64_t blockTableOffset = static_cast<uint64_t>(bIdx) * maxblockNumPerBatch + blockIdxInBatch;
        return blockTableGm.GetValue(blockTableOffset);
    }

private:
    GlobalTensor<int32_t> blockTableGm;
    uint32_t maxblockNumPerBatch;
};

// ----------------------------------------------WITH_ZERO_HEAD=true--------------------------------
template <ActualSeqLensMode MODE, typename ACTLEN_T>
class ActualSeqLensParser<MODE, ACTLEN_T, true> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> cuSeqLensGm, GlobalTensor<ACTLEN_T> seqUsedGM,
                                uint32_t cuSeqLensSize, uint32_t seqUsedSize, uint64_t defaultVal = 0)
    {
        this->cuSeqLensGm = cuSeqLensGm;
        this->seqUsedGM = seqUsedGM;
        this->cuSeqLensSize = cuSeqLensSize;
        this->seqUsedSize = seqUsedSize;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        return cuSeqLensGm.GetValue(bIdx);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (seqUsedSize == 0) {
            return cuSeqLensGm.GetValue(bIdx + 1) - cuSeqLensGm.GetValue(bIdx);
        }
        return seqUsedGM.GetValue(bIdx);
    }

    __aicore__ inline uint64_t GetFullActualSeqLength(uint32_t bIdx) const
    {
        return cuSeqLensGm.GetValue(bIdx + 1) - cuSeqLensGm.GetValue(bIdx);
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return cuSeqLensGm.GetValue(cuSeqLensSize);
    }

private:
    GlobalTensor<ACTLEN_T> cuSeqLensGm;
    GlobalTensor<ACTLEN_T> seqUsedGM;
    uint32_t cuSeqLensSize;
    uint32_t seqUsedSize;
};

// ----------------------------------------------GmLayoutParams--------------------------------
enum class FormatCategory {
    GM_Q_OUT_TND = 0,
    GM_KV_TND = 1,
    GM_KV_PA_BNBD = 2,
    GM_ANTIQ_BnNBs = 3,
};

template <GmFormat FORMAT>
struct GmLayoutParams {};

template <>
struct GmLayoutParams<GmFormat::TNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};
template <>
struct GmLayoutParams<GmFormat::TND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_TND;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnBsND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnNBsD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnNBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnNBs;
};
template <>
struct GmLayoutParams<GmFormat::NGTD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};

// ----------------------------------------------OffsetCalculator--------------------------------
template <GmFormat FORMAT, FormatCategory CATEGORY, typename ACTLEN_T, bool WITH_ZERO_HEAD = false>
struct OffsetCalculatorImpl {};

template <GmFormat FORMAT, typename ACTLEN_T, bool WITH_ZERO_HEAD>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_TND, ACTLEN_T, WITH_ZERO_HEAD> {
    GmLayout<FORMAT> gmLayout;
    using SeqLensQParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, WITH_ZERO_HEAD>;
    SeqLensQParserType actualSeqLensQParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ,
                                uint32_t actualLenQDims)
    {
        actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims);
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
    }

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, const SeqLensQParserType& parser)
    {
        actualSeqLensQParser = parser;
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensQParser.GetTBase(bIdx) + s1Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + dIdx * GetStrideD();
        return offset;
    }

    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return GetStrideT();
    }

    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T, bool WITH_ZERO_HEAD>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_TND, ACTLEN_T, WITH_ZERO_HEAD> {
    GmLayout<FORMAT> gmLayout;
    using SeqLensKVParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, WITH_ZERO_HEAD>;
    SeqLensKVParserType actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmKV,
                                uint32_t actualLenKVDims)
    {
        actualSeqLensKVParser.Init(actualSeqLengthsGmKV, actualLenKVDims);
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline void Init(uint32_t n2, uint32_t d, const SeqLensKVParserType& parser)
    {
        actualSeqLensKVParser = parser;
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensKVParser.GetTBase(bIdx) + s2Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + dIdx * GetStrideD();
        return offset;
    }

    __aicore__ inline uint64_t GetStrideT()
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
        return GetStrideT();
    }

    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<2>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize();
        uint64_t bsIdx = s2Idx % GetBlockSize();
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);

        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx * GetStrideD();
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

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<2>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnNBs, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n, uint32_t blockSize, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n, blockSize);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t nIdx, uint32_t sIdx)
    {
        uint64_t blockIdxInBatch = sIdx / GetStrideBlockSize();
        uint64_t bsIdx = sIdx % GetStrideBlockSize();
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset = blockIdx * GetStrideBlockNum() + nIdx * GetStrideN() + bsIdx * GetStrideBlockSize();

        return offset;
    }

    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride);
    }

    __aicore__ inline uint32_t GetDimN()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
struct OffsetCalculator
    : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY, ACTLEN_T, WITH_ZERO_HEAD> {};

template <typename Q_T, GmFormat FORMAT, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    OffsetCalculator<FORMAT, ACTLEN_T, WITH_ZERO_HEAD> offsetCalculator;
};

#endif

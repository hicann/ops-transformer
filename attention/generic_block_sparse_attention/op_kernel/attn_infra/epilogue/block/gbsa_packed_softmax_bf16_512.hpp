/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * This file is derived from GBSA's existing BF16 online-softmax epilogue and
 * distributed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GBSA_PACKED_SOFTMAX_BF16_512_HPP
#define GBSA_PACKED_SOFTMAX_BF16_512_HPP

#include "gbsa_block_epilogue_online_softmax_arch35_reg_low_prec_bf16.hpp"
#include "gbsa_block_epilogue_online_softmax_arch35_reg_low_prec.hpp"

namespace NpuArch::Epilogue::Block {

// Packed FP8-P with FP16/BF16 scores supports up to two groups of16 rows
// and KV tile<=512. FP16 retains its 256-column probability-rounding units
// inside the grouped tile, then rebases them to the tile maximum for one PV.
// Online statistics and inter-tile corrections remain FP32.
template <class OutputType_, class LayoutS_, class ScoreElement_ = bfloat16_t>
class GbsaPackedOnlineSoftmax512
    : public BlockEpilogue<EpilogueOnlineSoftmaxBsa, OutputType_, Gemm::GemmType<ScoreElement_, LayoutS_>> {
public:
    using Base = BlockEpilogue<EpilogueOnlineSoftmaxBsa, OutputType_, Gemm::GemmType<ScoreElement_, LayoutS_>>;
    using ArchTag = typename Base::ArchTag;
    using ElementInput = ScoreElement_;
    static constexpr bool IS_BF16 = AscendC::IsSameType<ElementInput, bfloat16_t>::value;
    using ElementOutput = typename OutputType_::Element;
    using LayoutOutput = typename OutputType_::Layout;
    static_assert(AscendC::IsSameType<ElementOutput, fp8_e4m3fn_t>::value,
                  "Packed512 implementation requires FP8 probability output");
    static constexpr uint32_t SLOTS = 2U;
    static constexpr uint32_t P_SLOTS = 3U;
    static constexpr uint32_t CHUNK = 128U;

public:
    __aicore__ inline GbsaPackedOnlineSoftmax512(Arch::Resource<ArchTag>& resource, float scale, UBufTileHelper& helper)
        : Base(resource, scale, helper)
    {
        if constexpr (IS_BF16) {
            scaleValue_ = AscendC::ToBfloat16(scale);
            minValue_ = AscendC::ToBfloat16(-3.389531390315715675e+38);
        } else {
            scaleValue_ = static_cast<ElementInput>(scale);
            minValue_ = static_cast<ElementInput>(-65504.0f);
        }
        subBlock_ = AscendC::GetSubBlockIdx();
        for (uint32_t i = 0; i < SLOTS; ++i) {
            s_[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                helper.sStartOffset +
                helper.qBaseTilePerSubCore * helper.kvBaseTilePerSubCore * sizeof(ElementInput) * i);
            p_[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                helper.pStartOffset +
                helper.qBaseTilePerSubCore * helper.kvBaseTilePerSubCore * sizeof(ElementOutput) * i);
        }
        for (uint32_t i = 0; i < P_SLOTS; ++i) {
            delta_[i] = resource.ubBuf.template GetBufferByByte<float>(
                helper.dmStartOffset + UBufTileHelper::STATS_ROW_NUM * sizeof(float) * i);
        }
        globalMax_ = resource.ubBuf.template GetBufferByByte<float>(helper.gmStartOffset);
        globalSum_ = resource.ubBuf.template GetBufferByByte<float>(helper.glStartOffset);
        localMax_ = resource.ubBuf.template GetBufferByByte<float>(helper.lmStartOffset);
        localSum_ = resource.ubBuf.template GetBufferByByte<float>(helper.llStartOffset);
    }

    __aicore__ inline void SetPQuantScale(float value)
    {
        Base::SetPQuantScale(value);
        pScale_ = value;
    }

    template <class TensorP>
    __aicore__ inline void operator()(TensorP& l1P, GemmCoord shape, uint32_t first, uint32_t sSlot, uint32_t pSlot,
                                      Arch::CrossCoreFlag qkReady, Arch::CrossCoreFlag pReady,
                                      uint32_t paddedPColumns = 0U)
    {
        if (IS_BF16 && shape.n() <= 256U && !pairMaskActive_) {
            Base::operator()(l1P, shape, first, sSlot, pSlot, qkReady, pReady, paddedPColumns);
            return;
        }
        // The private path has n<=512 and at most32 query/head rows. Thus
        // each subcore owns at most16 rows, fitting one FP32 statistics vector.
        const uint32_t mOffset = RoundUp(shape.m(), 2U) / 2U;
        uint32_t m = shape.m() < mOffset ? shape.m() : mOffset;
        m = subBlock_ == 0U ? m : shape.m() - m;
        if (m == 0U) {
            this->template WaitCrossCoreSync<4, PIPE_V>(qkReady);
            this->template SetCrossCoreSync<4, PIPE_V>(qkReady);
            this->template WaitCrossCoreSync<4, PIPE_MTE3>(pReady);
            this->template SetCrossCoreSync<4, PIPE_MTE3>(pReady);
            return;
        }
        const uint32_t n = shape.n();
        const uint16_t mRound = RoundUp(m, 16U);
        const uint16_t nRound = RoundUp(n, 16U);
        auto* scores = reinterpret_cast<__ubuf__ ElementInput*>(s_[sSlot].GetPhyAddr());
        auto* probabilities = reinterpret_cast<__ubuf__ ElementOutput*>(p_[sSlot].GetPhyAddr());
        auto* oldMax = reinterpret_cast<__ubuf__ float*>(globalMax_.GetPhyAddr());
        auto* oldSum = reinterpret_cast<__ubuf__ float*>(globalSum_.GetPhyAddr());
        auto* nextMax = first ? oldMax : reinterpret_cast<__ubuf__ float*>(localMax_.GetPhyAddr());
        auto* nextSum = first ? oldSum : reinterpret_cast<__ubuf__ float*>(localSum_.GetPhyAddr());
        auto* delta = reinterpret_cast<__ubuf__ float*>(delta_[pSlot].GetPhyAddr());

        this->template WaitCrossCoreSync<4, PIPE_V>(qkReady);
        if (pairMaskActive_) {
            ApplyPairMask(scores, reinterpret_cast<__ubuf__ uint16_t*>(pairIndices_.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint16_t*>(pairBitmap_.GetPhyAddr()), pairBegin_, pairVisible_,
                          static_cast<uint16_t>(m), n, nRound, pairShift_);
        }
        if constexpr (!IS_BF16) {
            ScaleAndMax(scores, nextMax, static_cast<uint16_t>(m), n, nRound);
        } else if (n <= 256U) {
            const uint32_t tail = (n - 1U) % 128U + 1U;
            if (n <= 128U) {
                Base::template ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::ONE>(
                    scores, nextMax, m, tail, RoundUp(tail, 32U), scaleValue_, nRound);
            } else {
                Base::template ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::TWO>(
                    scores, nextMax, m, tail, RoundUp(tail, 32U), scaleValue_, nRound);
            }
            ClampEmptyMaximum(nextMax, m);
        } else {
            ScaleAndMax(scores, nextMax, static_cast<uint16_t>(m), n, nRound);
        }
        if (!first) {
            MergeMaximum(nextMax, oldMax, m);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(sSlot + 2U);
        if constexpr (!IS_BF16) {
            ExpSumAndProbabilityFp16(probabilities, scores, nextMax, oldMax, nextSum, static_cast<uint16_t>(m), n,
                                     mRound, nRound, pScale_, first != 0U);
        } else if (n <= 256U) {
            const uint32_t tail = (n - 1U) % 128U + 1U;
            if (n <= 128U) {
                Base::template ComputeExpSubSum16FP8<KvBaseTileRegSplitStagesBf16::ONE>(
                    probabilities, scores, nextMax, nextSum, m, tail, mRound, nRound, tail / 2U, (tail + 1U) / 2U,
                    pScale_);
            } else {
                Base::template ComputeExpSubSum16FP8<KvBaseTileRegSplitStagesBf16::TWO>(
                    probabilities, scores, nextMax, nextSum, m, tail, mRound, nRound, tail / 2U, (tail + 1U) / 2U,
                    pScale_);
            }
        } else {
            ExpSumAndProbability(probabilities, scores, nextMax, nextSum, static_cast<uint16_t>(m), n, mRound, nRound,
                                 pScale_);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(sSlot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(sSlot);
        this->template SetCrossCoreSync<4, PIPE_V>(qkReady);

        const uint32_t copyN = paddedPColumns == 0U ? n : paddedPColumns;
        auto layout = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, RoundUp(copyN, 32U));
        auto ub = tla::MakeTensor(p_[sSlot], layout, Arch::PositionUB{});
        auto ubTile = GetTile(ub, tla::MakeCoord(0, 0), tla::MakeShape(m, copyN));
        auto l1Tile = GetTile(l1P, tla::MakeCoord(subBlock_ * mOffset, 0), tla::MakeShape(m, copyN));
        this->template WaitCrossCoreSync<4, PIPE_MTE3>(pReady);
        this->CopyPUbToPL1(l1Tile, ubTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(sSlot + 2U);
        this->template SetCrossCoreSync<4, PIPE_MTE3>(pReady);
        if (!first) {
            MergeSumAndDelta(oldSum, delta, oldMax, nextSum, nextMax, m);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SetPairMask(AscendC::LocalTensor<uint16_t> indices, AscendC::LocalTensor<uint32_t> bitmap,
                                       uint32_t begin, uint32_t visible, uint16_t shift, bool active)
    {
        pairIndices_ = indices;
        pairBitmap_ = bitmap;
        pairBegin_ = begin;
        pairVisible_ = visible;
        pairShift_ = shift;
        pairMaskActive_ = active;
    }

protected:
    AscendC::LocalTensor<uint16_t> pairIndices_;
    AscendC::LocalTensor<uint32_t> pairBitmap_;
    uint32_t pairBegin_ = 0U, pairVisible_ = 0U;
    uint16_t pairShift_ = 0U;
    bool pairMaskActive_ = false;
    float pScale_ = 448.0f;
    ElementInput scaleValue_;
    ElementInput minValue_;
    uint32_t subBlock_;
    AscendC::LocalTensor<ElementInput> s_[SLOTS];
    AscendC::LocalTensor<ElementOutput> p_[SLOTS];
    AscendC::LocalTensor<float> delta_[P_SLOTS];
    AscendC::LocalTensor<float> globalMax_, globalSum_, localMax_, localSum_;

    // Each AIV owns one query. Form its column mask once per128 columns,
    // then reuse it for all GQA rows. Negative infinity gives exact zero P.
    static __simd_vf__ inline void ApplyPairMask(__ubuf__ ElementInput* scores, __ubuf__ uint16_t* ids,
                                                 __ubuf__ uint16_t* bitmap, uint16_t begin, uint16_t visible,
                                                 uint16_t rows, uint16_t columns, uint16_t pitch, uint16_t shift)
    {
        using namespace AscendC::Reg;
        RegTensor<int16_t> lanes;
        RegTensor<uint16_t> logical, offset, block, token, membership, innerMask, inner;
        RegTensor<ElementInput> values, minimum;
        Arange(lanes, int16_t(0));
        Duplicate(innerMask, uint16_t((1U << shift) - 1U));
        if constexpr (IS_BF16) {
            Duplicate(minimum, AscendC::ToBfloat16(-__builtin_inff()));
        } else {
            Duplicate(minimum, static_cast<ElementInput>(-__builtin_inff()));
        }
        uint32_t left = columns;
        for (uint16_t chunk = 0U; chunk < (columns + 127U) / 128U; ++chunk) {
            MaskReg active = UpdateMask<uint16_t>(left);
            Adds(logical, (RegTensor<uint16_t>&)lanes, uint16_t(begin + chunk * 128U), active);
            ShiftRights(offset, logical, int16_t(shift), active);
            ShiftLefts(offset, offset, int16_t(1), active);
            Gather(block, ids, offset, active);
            ShiftLefts(offset, block, int16_t(1), active);
            Gather(membership, bitmap, offset, active);
            And(inner, logical, innerMask, active);
            ShiftLefts(token, block, int16_t(shift), active);
            Add(token, token, inner, active);
            MaskReg causal, selected;
            Compares<uint16_t, AscendC::CMPMODE::LT>(causal, token, visible, active);
            Compares<uint16_t, AscendC::CMPMODE::EQ>(selected, membership, uint16_t(1), causal);
            for (uint16_t row = 0U; row < rows; ++row) {
                LoadAlign(values, scores + row * pitch + chunk * 128U);
                Select(values, values, minimum, selected);
                StoreAlign(scores + row * pitch + chunk * 128U, values, active);
            }
        }
    }

    static __simd_vf__ inline void ClampEmptyMaximum(__ubuf__ float* maxima, uint32_t rows)
    {
        using namespace AscendC::Reg;
        RegTensor<float> value;
        MaskReg active = UpdateMask<float>(rows);
        LoadAlign(value, maxima);
        Maxs(value, value, -3.389531390315715675e+38f, active);
        StoreAlign(maxima, value, active);
    }

    __simd_vf__ inline void ScaleAndMax(__ubuf__ ElementInput* src, __ubuf__ float* maxima, uint16_t m, uint32_t n,
                                        uint16_t pitch)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait zero = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                           AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait one = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                          AscendC::RoundMode::UNKNOWN};
        RegTensor<ElementInput> values, maximum, scale;
        RegTensor<float> even, odd, reducedEven, reducedOdd, reduced;
        UnalignReg tail;
        MaskReg full = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg floatFull = CreateMask<float, MaskPattern::ALL>();
        const uint16_t chunks = (n + CHUNK - 1U) / CHUNK;
        Duplicate(scale, scaleValue_);
        for (uint16_t row = 0U; row < m; ++row) {
            Duplicate(maximum, minValue_);
            for (uint16_t chunk = 0U; chunk < chunks; ++chunk) {
                uint32_t active = n - chunk * CHUNK;
                active = active < CHUNK ? active : CHUNK;
                MaskReg valid = UpdateMask<ElementInput>(active);
                LoadAlign(values, src + row * pitch + chunk * CHUNK);
                Mul(values, values, scale, full);
                StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(src + row * pitch + chunk * CHUNK, values, valid);
                Max<ElementInput, MaskMergeMode::MERGING>(maximum, maximum, values, valid);
            }
            Cast<float, ElementInput, zero>(even, maximum, full);
            Cast<float, ElementInput, one>(odd, maximum, full);
            ReduceMax(reducedEven, even, floatFull);
            ReduceMax(reducedOdd, odd, floatFull);
            Max(reduced, reducedEven, reducedOdd, floatFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maxima, reduced, tail, 1);
        }
        vstas(tail, maxima, 0, POST_UPDATE);
    }

    __simd_vf__ inline void ExpSumAndProbability(__ubuf__ ElementOutput* dst, __ubuf__ ElementInput* src,
                                                 __ubuf__ float* maxima, __ubuf__ float* sums, uint16_t m, uint32_t n,
                                                 uint16_t blockStride, uint16_t pitch, float pScale)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait upZero = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait upOne = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                            AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait downZero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait downTwo = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::CAST_RINT};
        RegTensor<ElementInput> values;
        RegTensor<float> even, odd, expEven, expOdd, maximum, sum, zero;
        RegTensor<ElementOutput> pEven, pOdd, output, unused;
        MaskReg bfFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg floatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg byteFull = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg byte128 = CreateMask<uint8_t, MaskPattern::VL128>();
        UnalignReg tail;
        const uint16_t chunks = (n + CHUNK - 1U) / CHUNK;
        Duplicate(zero, 0.0f);
        for (uint16_t row = 0U; row < m; ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maximum, maxima + row);
            Duplicate(sum, 0.0f);
            for (uint16_t chunk = 0U; chunk < chunks; ++chunk) {
                uint32_t active = n - chunk * CHUNK;
                active = active < CHUNK ? active : CHUNK;
                uint32_t activeEven = (active + 1U) / 2U;
                uint32_t activeOdd = active / 2U;
                MaskReg validEven = UpdateMask<float>(activeEven);
                MaskReg validOdd = UpdateMask<float>(activeOdd);
                LoadAlign(values, src + row * pitch + chunk * CHUNK);
                Cast<float, ElementInput, upZero>(even, values, bfFull);
                Cast<float, ElementInput, upOne>(odd, values, bfFull);
                FusedExpSub(expEven, even, maximum, validEven);
                FusedExpSub(expOdd, odd, maximum, validOdd);
                Add<float, MaskMergeMode::MERGING>(sum, sum, expEven, validEven);
                Add<float, MaskMergeMode::MERGING>(sum, sum, expOdd, validOdd);
                Muls(expEven, expEven, pScale, validEven);
                Muls(expOdd, expOdd, pScale, validOdd);
                // Copying a full128-column NZ chunk initializes the complete
                // RoundUp(n,64) PV tail. Statistics use valid lanes only.
                Select(expEven, expEven, zero, validEven);
                Select(expOdd, expOdd, zero, validOdd);
                Cast<ElementOutput, float, downZero>(pEven, expEven, floatFull);
                Cast<ElementOutput, float, downTwo>(pOdd, expOdd, floatFull);
                Or((RegTensor<uint8_t>&)pEven, (RegTensor<uint8_t>&)pEven, (RegTensor<uint8_t>&)pOdd, byteFull);
                DeInterleave(output, unused, pEven, pEven);
                StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(dst + row * 32U + chunk * blockStride * CHUNK,
                                                                         output, blockStride, byte128);
            }
            ReduceSum(sum, sum, floatFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sums, sum, tail, 1);
        }
        vstas(tail, sums, 0, POST_UPDATE);
    }

    // Preserve half Sub/Exp/Mul and half local sums from the FP16 epilogue.
    // Quantize each original 256-column unit before rebasing it to the common
    // 512-column maximum. This keeps the existing FP16 probability rounding
    // while using the grouped QK, a single P tile and the existing PV pipeline.
    __simd_vf__ inline void ExpSumAndProbabilityFp16(__ubuf__ ElementOutput* dst, __ubuf__ ElementInput* src,
                                                     __ubuf__ float* maxima, __ubuf__ float* previousMaxima,
                                                     __ubuf__ float* sums, uint16_t m, uint32_t n, uint16_t blockStride,
                                                     uint16_t pitch, float pScale, bool first)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait upZero = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait upOne = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                            AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait upTwo = {RegLayout::TWO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                            AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait downZero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait downTwo = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::CAST_RINT};
        RegTensor<ElementInput> values0, values1, localMaximum, halfMaximum, halfZero, exponent, localSum;
        RegTensor<float> maximum, prefixMaximum, groupMaximum, correction, sum, groupSum, even, odd;
        RegTensor<ElementOutput> pEven, pOdd, output, unused;
        MaskReg halfFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg floatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg byteFull = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg byte128 = CreateMask<uint8_t, MaskPattern::VL128>();
        UnalignReg tail;
        const uint16_t segments = (n + 255U) / 256U;
        Duplicate(halfZero, static_cast<ElementInput>(0));
        for (uint16_t row = 0U; row < m; ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maximum, maxima + row);
            if (first) {
                Duplicate(prefixMaximum, -65504.0f);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(prefixMaximum, previousMaxima + row);
            }
            Duplicate(sum, 0.0f);
            for (uint16_t segment = 0U; segment < segments; ++segment) {
                uint32_t firstCount = min(128U, n - segment * 256U);
                uint32_t secondCount = n > segment * 256U + 128U ? min(128U, n - segment * 256U - 128U) : 0U;
                MaskReg valid0 = UpdateMask<ElementInput>(firstCount);
                MaskReg valid1 = UpdateMask<ElementInput>(secondCount);
                LoadAlign(values0, src + row * pitch + segment * 256U);
                Duplicate(localMaximum, static_cast<ElementInput>(-65504.0f));
                Max<ElementInput, MaskMergeMode::MERGING>(localMaximum, localMaximum, values0, valid0);
                if (n > segment * 256U + 128U) {
                    LoadAlign(values1, src + row * pitch + segment * 256U + 128U);
                    Max<ElementInput, MaskMergeMode::MERGING>(localMaximum, localMaximum, values1, valid1);
                }
                ReduceMax(localMaximum, localMaximum, halfFull);
                Duplicate(localMaximum, localMaximum, halfFull);
                Cast<float, ElementInput, upZero>(groupMaximum, localMaximum, halfFull);
                Max(prefixMaximum, prefixMaximum, groupMaximum, floatFull);
                FusedExpSub(correction, prefixMaximum, maximum, floatFull);
                Cast<ElementInput, float, downZero>(halfMaximum, prefixMaximum, floatFull);
                Duplicate(halfMaximum, halfMaximum, halfFull);
                Duplicate(localSum, static_cast<ElementInput>(0));
                const uint16_t parts = (min(256U, n - segment * 256U) + 127U) / 128U;
                for (uint16_t part = 0U; part < parts; ++part) {
                    uint32_t active = min(128U, n - segment * 256U - part * 128U);
                    MaskReg valid = UpdateMask<ElementInput>(active);
                    LoadAlign(exponent, src + row * pitch + segment * 256U + part * 128U);
                    Sub(exponent, exponent, halfMaximum, valid);
                    Exp(exponent, exponent, valid);
                    Add<ElementInput, MaskMergeMode::MERGING>(localSum, localSum, exponent, valid);
                    Muls(exponent, exponent, static_cast<ElementInput>(pScale), valid);
                    Select(exponent, exponent, halfZero, valid);
                    Cast<float, ElementInput, upZero>(even, exponent, halfFull);
                    Cast<float, ElementInput, upOne>(odd, exponent, halfFull);
                    Cast<ElementOutput, float, downZero>(pEven, even, floatFull);
                    Cast<ElementOutput, float, downTwo>(pOdd, odd, floatFull);
                    // Rebase already rounded probabilities, leaving the P
                    // scale and the Cube's 512-column PV contract unchanged.
                    Cast<float, ElementOutput, upZero>(even, pEven, byteFull);
                    Cast<float, ElementOutput, upTwo>(odd, pOdd, byteFull);
                    Mul(even, even, correction, floatFull);
                    Mul(odd, odd, correction, floatFull);
                    Cast<ElementOutput, float, downZero>(pEven, even, floatFull);
                    Cast<ElementOutput, float, downTwo>(pOdd, odd, floatFull);
                    Or((RegTensor<uint8_t>&)pEven, (RegTensor<uint8_t>&)pEven, (RegTensor<uint8_t>&)pOdd, byteFull);
                    DeInterleave(output, unused, pEven, pEven);
                    StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                        dst + row * 32U + (segment * 2U + part) * blockStride * CHUNK, output, blockStride, byte128);
                }
                ReduceSum(localSum, localSum, halfFull);
                Cast<float, ElementInput, upZero>(groupSum, localSum, halfFull);
                Mul(groupSum, groupSum, correction, floatFull);
                Add(sum, sum, groupSum, floatFull);
            }
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sums, sum, tail, 1);
        }
        vstas(tail, sums, 0, POST_UPDATE);
    }

    __simd_vf__ inline void MergeMaximum(__ubuf__ float* current, __ubuf__ float* previous, uint32_t rows)
    {
        using namespace AscendC::Reg;
        RegTensor<float> now, old, result;
        MaskReg active = UpdateMask<float>(rows);
        LoadAlign(now, current);
        LoadAlign(old, previous);
        Max(result, now, old, active);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(current, result, active);
    }

    __simd_vf__ inline void MergeSumAndDelta(__ubuf__ float* globalSum, __ubuf__ float* correction,
                                             __ubuf__ float* globalMax, __ubuf__ float* localSum,
                                             __ubuf__ float* localMax, uint32_t rows)
    {
        using namespace AscendC::Reg;
        RegTensor<float> nowMax, oldMax, alpha, oldSum, addSum, result;
        MaskReg active = UpdateMask<float>(rows);
        LoadAlign(nowMax, localMax);
        LoadAlign(oldMax, globalMax);
        FusedExpSub(alpha, oldMax, nowMax, active);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(correction, alpha, active);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(globalMax, nowMax, active);
        LoadAlign(oldSum, globalSum);
        LoadAlign(addSum, localSum);
        Mul(result, alpha, oldSum, active);
        Add(result, result, addSum, active);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(globalSum, result, active);
    }
};

template <class OutputType, class ScoreType>
struct GbsaPackedSoftmax512Selector {
    using Type = BlockEpilogue<EpilogueOnlineSoftmaxBsa, OutputType, ScoreType>;
};

template <class OutputType, class LayoutS>
struct GbsaPackedSoftmax512Selector<OutputType, Gemm::GemmType<bfloat16_t, LayoutS>> {
    using Type = GbsaPackedOnlineSoftmax512<OutputType, LayoutS>;
};

template <class OutputType, class LayoutS>
struct GbsaPackedSoftmax512Selector<OutputType, Gemm::GemmType<half, LayoutS>> {
    using Type = GbsaPackedOnlineSoftmax512<OutputType, LayoutS, half>;
};

} // namespace NpuArch::Epilogue::Block
#endif

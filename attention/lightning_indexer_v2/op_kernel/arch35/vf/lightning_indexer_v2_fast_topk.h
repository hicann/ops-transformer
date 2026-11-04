/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_FAST_TOPK_VF_H
#define LIGHTNING_INDEXER_V2_FAST_TOPK_VF_H

#include "lightning_indexer_v2_fast_vector.h"

namespace LIV2Fast::vf {
namespace Reg = AscendC::Reg;

template <int32_t BYTE>
__simd_callee__ inline void MaskHistogramPrefix(Reg::RegTensor<uint8_t> (&bytes)[kRadixKeyBytes], uint32_t prefix,
                                                Reg::MaskReg& active)
{
    if constexpr (BYTE < kRadixMostSignificantByte) {
        Reg::MaskReg prefixMatchMask;
        Reg::Compares<uint8_t, AscendC::CMPMODE::EQ>(
            prefixMatchMask, bytes[kRadixMostSignificantByte],
            static_cast<uint8_t>(prefix >> (kRadixMostSignificantByte * kBitsPerByte)), active);
        active = prefixMatchMask;
    }
    if constexpr (BYTE < kRadixSecondByte) {
        Reg::MaskReg prefixMatchMask;
        Reg::Compares<uint8_t, AscendC::CMPMODE::EQ>(prefixMatchMask, bytes[kRadixSecondByte],
                                                     static_cast<uint8_t>(prefix >> (kRadixSecondByte * kBitsPerByte)),
                                                     active);
        active = prefixMatchMask;
    }
    if constexpr (BYTE < 1) {
        Reg::MaskReg prefixMatchMask;
        Reg::Compares<uint8_t, AscendC::CMPMODE::EQ>(prefixMatchMask, bytes[1],
                                                     static_cast<uint8_t>(prefix >> kBitsPerByte), active);
        active = prefixMatchMask;
    }
}

template <int32_t BYTE>
__simd_vf__ inline void HistogramByte(__ubuf__ uint32_t* keys, __ubuf__ uint32_t* histogram, uint32_t length,
                                      uint32_t prefix)
{
    static_assert(BYTE >= 0 && BYTE <= kRadixMostSignificantByte);
    static constexpr Reg::CastTrait castEven = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castOdd = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::UNKNOWN};
    Reg::RegTensor<uint16_t> firstLowHalfwords, firstHighHalfwords, secondLowHalfwords, secondHighHalfwords;
    Reg::RegTensor<uint8_t> bytes[kRadixKeyBytes];
    Reg::RegTensor<uint16_t> lowerBinCounts, upperBinCounts;
    Reg::RegTensor<uint32_t> lowerEvenBinCounts, lowerOddBinCounts, upperEvenBinCounts, upperOddBinCounts;
    Reg::MaskReg allHalfwordLanes = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg allWordLanes = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(lowerBinCounts, static_cast<uint16_t>(0));
    Reg::Duplicate(upperBinCounts, static_cast<uint16_t>(0));
    uint32_t remaining = length;
    auto* source16 = reinterpret_cast<__ubuf__ uint16_t*>(keys);
    // Each iteration handles 256 uint32_t keys, addressed as 512 uint16_t elements.
    const uint16_t repeats = static_cast<uint16_t>((length + kRadixLastBin) / kRadixKeysPerIteration);
    for (uint16_t block = static_cast<uint16_t>(0); block < repeats; ++block) {
        const uint64_t offset = static_cast<uint64_t>(block) * kRadixHalfwordsPerIteration;
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_DINTLV_B16>(firstLowHalfwords, firstHighHalfwords,
                                                                 source16 + offset);
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_DINTLV_B16>(secondLowHalfwords, secondHighHalfwords,
                                                                 source16 + offset + kRadixKeysPerIteration);
        Reg::DeInterleave(bytes[0], bytes[1], reinterpret_cast<Reg::RegTensor<uint8_t>&>(firstLowHalfwords),
                          reinterpret_cast<Reg::RegTensor<uint8_t>&>(secondLowHalfwords));
        Reg::DeInterleave(bytes[kRadixSecondByte], bytes[kRadixMostSignificantByte],
                          reinterpret_cast<Reg::RegTensor<uint8_t>&>(firstHighHalfwords),
                          reinterpret_cast<Reg::RegTensor<uint8_t>&>(secondHighHalfwords));
        Reg::MaskReg active = Reg::UpdateMask<uint8_t>(remaining);
        MaskHistogramPrefix<BYTE>(bytes, prefix, active);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            lowerBinCounts, bytes[BYTE], active);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            upperBinCounts, bytes[BYTE], active);
    }
    Reg::Cast<uint32_t, uint16_t, castEven>(lowerEvenBinCounts, lowerBinCounts, allHalfwordLanes);
    Reg::Cast<uint32_t, uint16_t, castOdd>(lowerOddBinCounts, lowerBinCounts, allHalfwordLanes);
    Reg::Cast<uint32_t, uint16_t, castEven>(upperEvenBinCounts, upperBinCounts, allHalfwordLanes);
    Reg::Cast<uint32_t, uint16_t, castOdd>(upperOddBinCounts, upperBinCounts, allHalfwordLanes);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogram, lowerEvenBinCounts, lowerOddBinCounts,
                                                              allWordLanes);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogram + kRadixHalfBins, upperEvenBinCounts,
                                                              upperOddBinCounts, allWordLanes);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void SelectRadixByte(FastVectorBuffers& buffers, __ubuf__ uint32_t* keys, uint32_t length,
                                       int32_t byte, uint32_t& prefix, uint32_t& rank);

__aicore__ inline uint32_t KthSortableKey(FastVectorBuffers& buffers, __ubuf__ uint32_t* keys, uint32_t length,
                                          uint32_t selectedCount = kTopK)
{
    uint32_t prefix = 0U;
    uint32_t rank = selectedCount;
    // Refine the elementBytes-byte sortable key from the most significant byte to the least.
    for (int32_t byte = kRadixMostSignificantByte; byte >= 0; --byte) {
        SelectRadixByte(buffers, keys, length, byte, prefix, rank);
    }
    return prefix;
}

__simd_vf__ inline void FindTargetBin(__ubuf__ uint32_t* histogram, uint32_t descendingRank,
                                      __ubuf__ uint32_t* selectedBin, __ubuf__ uint32_t* nextRank)
{
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg targetBinMask;
    Reg::RegTensor<uint32_t> total, rank, ascending, counts, packed;
    Reg::RegTensor<uint32_t> target, selectedBinPrefixCount, greater, remaining;
    Reg::RegTensor<int32_t> index;
    Reg::UnalignRegForStore cache;
    // The last of 256 prefix-sum bins contains the total active key count.
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(total, histogram + kRadixLastBin);
    Reg::Duplicate(rank, descendingRank);
    Reg::Sub(ascending, total, rank, full);
    Reg::Adds(ascending, ascending, static_cast<uint32_t>(1), full);
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    for (uint16_t group = static_cast<uint16_t>(0); group < static_cast<uint16_t>(kRadixVectorGroups); ++group) {
        const uint32_t begin = static_cast<uint32_t>(group) * kFloatRegLanes;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(counts, histogram + begin);
        Reg::Compare<uint32_t, AscendC::CMPMODE::GE>(targetBinMask, counts, ascending, full);
        Reg::Arange(index, static_cast<int32_t>(begin));
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(
            packed, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), targetBinMask);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(selectedBin, packed, cache);
    }
    Reg::StoreUnAlignPost(selectedBin, cache);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(target, selectedBin);
    Reg::Gather(selectedBinPrefixCount, histogram, target, full);
    Reg::Sub(greater, total, selectedBinPrefixCount, full);
    Reg::Sub(remaining, rank, greater, full);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nextRank, remaining, full);
}

__aicore__ inline void SelectRadixByte(FastVectorBuffers& buffers, __ubuf__ uint32_t* keys, uint32_t length,
                                       int32_t byte, uint32_t& prefix, uint32_t& rank)
{
    auto* histogram = reinterpret_cast<__ubuf__ uint32_t*>(buffers.GetHistogram().GetPhyAddr());
    if (byte == kRadixMostSignificantByte) {
        HistogramByte<kRadixMostSignificantByte>(keys, histogram, length, prefix);
    } else if (byte == kRadixSecondByte) {
        HistogramByte<kRadixSecondByte>(keys, histogram, length, prefix);
    } else if (byte == 1) {
        HistogramByte<1>(keys, histogram, length, prefix);
    } else {
        HistogramByte<0>(keys, histogram, length, prefix);
    }
    auto* target = reinterpret_cast<__ubuf__ uint32_t*>(buffers.GetPositions().GetPhyAddr());
    auto* nextRank = reinterpret_cast<__ubuf__ uint32_t*>(buffers.GetNextRank().GetPhyAddr());
    FindTargetBin(histogram, rank, target, nextRank);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(FastEvent::SCALAR_READ);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(FastEvent::SCALAR_READ);
    const uint32_t selected = buffers.GetPositions().GetValue(0);
    rank = buffers.GetNextRank().GetValue(0);
    prefix |= selected << (static_cast<uint32_t>(byte) * kBitsPerByte);
}

template <AscendC::CMPMODE COMPARE_MODE>
__simd_vf__ inline void CompactPositionsVF(__ubuf__ uint32_t* keys, uint32_t begin, uint32_t length, uint16_t repeats,
                                           uint32_t kth, __ubuf__ uint32_t* positions)
{
    Reg::RegTensor<uint32_t> key, packed;
    Reg::RegTensor<int32_t> index;
    Reg::MaskReg selected;
    Reg::UnalignRegForStore cache;
    uint32_t remaining = length;
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    for (uint16_t block = 0; block < repeats; ++block) {
        const uint64_t offset = begin + static_cast<uint64_t>(block) * kFloatRegLanes;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(key, keys + offset);
        Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(remaining);
        Reg::Compares<uint32_t, COMPARE_MODE>(selected, key, kth, valid);
        // Output key indices use 32-bit lanes; address arithmetic remains 64-bit.
        Reg::Arange(index, static_cast<int32_t>(offset));
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(
            packed, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), selected);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(positions, packed, cache);
    }
    Reg::StoreUnAlignPost(positions, cache);
}

__aicore__ inline uint32_t ReadCompactedCount()
{
    // AR counts bytes. Keep the original V->S handoff before the scalar read.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(FastEvent::SCALAR_READ);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(FastEvent::SCALAR_READ);
    return static_cast<uint32_t>(Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(uint32_t);
}

__aicore__ inline uint32_t CompactStrictPositions(__ubuf__ uint32_t* keys, uint32_t length, uint32_t kth,
                                                  __ubuf__ uint32_t* positions)
{
    CompactPositionsVF<AscendC::CMPMODE::GT>(
        keys, 0U, length, static_cast<uint16_t>((length + (kFloatRegLanes - 1U)) / kFloatRegLanes), kth, positions);
    return ReadCompactedCount();
}

__aicore__ inline uint32_t CompactEqualPositionsChunk(__ubuf__ uint32_t* keys, uint32_t begin, uint32_t length,
                                                      uint32_t kth, __ubuf__ uint32_t* positions)
{
    CompactPositionsVF<AscendC::CMPMODE::EQ>(
        keys, begin, length, static_cast<uint16_t>(kSelectionChunkElements / kFloatRegLanes), kth, positions);
    return ReadCompactedCount();
}

__aicore__ inline void SelectStableTopkPositions(FastVectorBuffers& buffers, __ubuf__ uint32_t* keys, uint32_t length,
                                                 uint32_t kth, uint32_t selectedCount = kTopK)
{
    auto* positions = reinterpret_cast<__ubuf__ uint32_t*>(buffers.GetPositions().GetPhyAddr());
    const uint32_t strict = CompactStrictPositions(keys, length, kth, positions);
    uint32_t equal = 0U;
    for (uint32_t begin = 0U; begin < length && strict + equal < selectedCount; begin += kSelectionChunkElements) {
        const uint32_t chunk = length - begin < kSelectionChunkElements ? length - begin : kSelectionChunkElements;
        equal += CompactEqualPositionsChunk(keys, begin, chunk, kth, positions + strict + equal);
    }
}

__simd_vf__ inline void GatherSelectedTopkVF(__ubuf__ uint32_t* positions, __ubuf__ uint32_t* source,
                                             __ubuf__ uint32_t* destination)
{
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> index, value;
    for (uint16_t chunk = static_cast<uint16_t>(0); chunk < static_cast<uint16_t>(kTopK / kFloatRegLanes); ++chunk) {
        const uint32_t begin = static_cast<uint64_t>(chunk) * kFloatRegLanes;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(index, positions + begin);
        Reg::Gather(value, source, index, full);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(destination + begin, value, full);
    }
}

__aicore__ inline void GatherSelectedTopk(FastVectorBuffers& buffers, __ubuf__ uint32_t* source,
                                          __ubuf__ uint32_t* destination)
{
    auto* positions = reinterpret_cast<__ubuf__ uint32_t*>(buffers.GetPositions().GetPhyAddr());
    GatherSelectedTopkVF(positions, source, destination);
}

// Gather only the requested output count; UB tail lanes remain masked.
__simd_vf__ inline void GatherSelectedCountVF(__ubuf__ uint32_t* positions, __ubuf__ uint32_t* source,
                                              __ubuf__ uint32_t* destination, uint32_t selectedCount)
{
    Reg::RegTensor<uint32_t> index, value;
    uint32_t remaining = selectedCount;
    const uint16_t repeats = (selectedCount + (kFloatRegLanes - 1U)) / kFloatRegLanes;
    for (uint16_t chunk = 0; chunk < repeats; ++chunk) {
        auto mask = Reg::UpdateMask<uint32_t>(remaining);
        const uint32_t begin = static_cast<uint64_t>(chunk) * kFloatRegLanes;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(index, positions + begin);
        Reg::Gather(value, source, index, mask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(destination + begin, value, mask);
    }
}

__aicore__ inline void GatherSelectedCount(FastVectorBuffers& buffers, __ubuf__ uint32_t* source,
                                           __ubuf__ uint32_t* destination, uint32_t selectedCount)
{
    auto* positions = reinterpret_cast<__ubuf__ uint32_t*>(buffers.GetPositions().GetPhyAddr());
    GatherSelectedCountVF(positions, source, destination, selectedCount);
}

// Invert the sortable-key mapping without changing the FP32 score bits.
__simd_vf__ inline void SortableKeysToFloatBits(__ubuf__ uint32_t* keys, uint32_t count)
{
    Reg::RegTensor<uint32_t> value, sign, zero, allOne, highBit, xorMask;
    Reg::MaskReg full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero, static_cast<uint32_t>(0), full);
    Reg::Duplicate(allOne, static_cast<uint32_t>(kUint32AllBits), full);
    Reg::Duplicate(highBit, static_cast<uint32_t>(kUint32SignBit), full);
    uint32_t remaining = count;
    const uint16_t repeats = (count + (kFloatRegLanes - 1U)) / kFloatRegLanes;
    for (uint16_t chunk = 0; chunk < repeats; ++chunk) {
        const uint64_t offset = static_cast<uint64_t>(chunk) * kFloatRegLanes;
        auto mask = Reg::UpdateMask<uint32_t>(remaining);
        Reg::MaskReg nonnegativeMask;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(value, keys + offset);
        Reg::And(sign, value, highBit, mask);
        Reg::Compare<uint32_t, AscendC::CMPMODE::GT>(nonnegativeMask, sign, zero, mask);
        Reg::Select(xorMask, highBit, allOne, nonnegativeMask);
        Reg::Xor(value, value, xorMask, mask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(keys + offset, value, mask);
    }
}

template <bool APPLY_CAUSAL_MASK>
__simd_callee__ inline void SelectWrsFragment(Reg::MaskReg& selected, Reg::RegTensor<uint32_t>& keys,
                                              uint32_t threshold, uint64_t offset, uint32_t validKeys,
                                              Reg::MaskReg& full)
{
    Reg::Compares<uint32_t, AscendC::CMPMODE::GT>(selected, keys, threshold, full);
    if constexpr (APPLY_CAUSAL_MASK) {
        Reg::RegTensor<int32_t> index;
        Reg::MaskReg visibleKeyMask;
        // Output key indices use 32-bit lanes; address arithmetic remains 64-bit.
        Reg::Arange(index, static_cast<int32_t>(offset));
        Reg::Compares<uint32_t, AscendC::CMPMODE::LT>(
            visibleKeyMask, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), validKeys, full);
        Reg::And(selected, selected, visibleKeyMask, full);
    }
}

__simd_callee__ inline void AppendWrsKey(__ubuf__ uint32_t*& destination, Reg::RegTensor<uint32_t>& key,
                                         Reg::MaskReg& selected, Reg::UnalignRegForStore& cache)
{
    Reg::RegTensor<uint32_t> packed;
    Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(packed, key, selected);
    Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(destination, packed, cache);
}

__simd_callee__ inline void AppendWrsIndex(__ubuf__ uint32_t*& destination, uint64_t offset, Reg::MaskReg& selected,
                                           Reg::UnalignRegForStore& cache)
{
    Reg::RegTensor<int32_t> index;
    // Output key indices use 32-bit lanes; address arithmetic remains 64-bit.
    Reg::Arange(index, static_cast<int32_t>(offset));
    AppendWrsKey(destination, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), selected, cache);
}

template <bool APPLY_CAUSAL_MASK>
__simd_callee__ inline void CompactWrsRow(Reg::RegTensor<uint32_t> (&sortableKeys)[kWrsFragmentCount],
                                          __ubuf__ uint32_t* keys, __ubuf__ uint32_t* indices,
                                          __ubuf__ uint32_t* counts, uint32_t threshold, uint64_t globalKeyOffset,
                                          uint32_t validKeys, Reg::MaskReg& full)
{
    Reg::RegTensor<uint32_t> zero, elementBytes, selectedBytes, byteCount;
    Reg::MaskReg fragment0CandidateMask, fragment1CandidateMask, fragment2CandidateMask, fragment3CandidateMask;
    Reg::UnalignRegForStore cache;
    Reg::Duplicate(zero, static_cast<uint32_t>(0), full);
    Reg::Duplicate(elementBytes, static_cast<uint32_t>(sizeof(uint32_t)), full);
    SelectWrsFragment<APPLY_CAUSAL_MASK>(fragment0CandidateMask, sortableKeys[0], threshold, globalKeyOffset + 0U,
                                         validKeys, full);
    SelectWrsFragment<APPLY_CAUSAL_MASK>(fragment1CandidateMask, sortableKeys[1], threshold,
                                         globalKeyOffset + kFloatRegLanes, validKeys, full);
    SelectWrsFragment<APPLY_CAUSAL_MASK>(fragment2CandidateMask, sortableKeys[kWrsFragment2], threshold,
                                         globalKeyOffset + kWrsFragment2Offset, validKeys, full);
    SelectWrsFragment<APPLY_CAUSAL_MASK>(fragment3CandidateMask, sortableKeys[kWrsFragment3], threshold,
                                         globalKeyOffset + kWrsFragment3Offset, validKeys, full);
    Reg::Select(byteCount, elementBytes, zero, fragment0CandidateMask);
    Reg::Select(selectedBytes, elementBytes, zero, fragment1CandidateMask);
    Reg::Add(byteCount, byteCount, selectedBytes, full);
    Reg::Select(selectedBytes, elementBytes, zero, fragment2CandidateMask);
    Reg::Add(byteCount, byteCount, selectedBytes, full);
    Reg::Select(selectedBytes, elementBytes, zero, fragment3CandidateMask);
    Reg::Add(byteCount, byteCount, selectedBytes, full);
    Reg::Reduce<Reg::ReduceType::SUM>(byteCount, byteCount, full);
    Reg::Store(counts, byteCount, static_cast<uint32_t>(1));
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    AppendWrsKey(keys, sortableKeys[0], fragment0CandidateMask, cache);
    AppendWrsKey(keys, sortableKeys[1], fragment1CandidateMask, cache);
    AppendWrsKey(keys, sortableKeys[kWrsFragment2], fragment2CandidateMask, cache);
    AppendWrsKey(keys, sortableKeys[kWrsFragment3], fragment3CandidateMask, cache);
    Reg::StoreUnAlignPost(keys, cache);
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    AppendWrsIndex(indices, globalKeyOffset + 0U, fragment0CandidateMask, cache);
    AppendWrsIndex(indices, globalKeyOffset + kFloatRegLanes, fragment1CandidateMask, cache);
    AppendWrsIndex(indices, globalKeyOffset + kWrsFragment2Offset, fragment2CandidateMask, cache);
    AppendWrsIndex(indices, globalKeyOffset + kWrsFragment3Offset, fragment3CandidateMask, cache);
    Reg::StoreUnAlignPost(indices, cache);
}

__simd_callee__ inline void ConvertWrsRowToKeys(Reg::RegTensor<uint32_t> (&keys)[kWrsFragmentCount],
                                                Reg::RegTensor<float> (&scores)[kWrsFragmentCount],
                                                liV2Vector1::FloatSortConstCtx<float>& context, Reg::MaskReg& full)
{
    liV2Vector1::FloatToSortableKey<float>(keys[0], scores[0], context, full);
    liV2Vector1::FloatToSortableKey<float>(keys[1], scores[1], context, full);
    liV2Vector1::FloatToSortableKey<float>(keys[kWrsFragment2], scores[kWrsFragment2], context, full);
    liV2Vector1::FloatToSortableKey<float>(keys[kWrsFragment3], scores[kWrsFragment3], context, full);
}

// Production VF: QK bank-plane read, ReLU-weighted FP32 accumulation,
// sortable-key conversion, and matching key/index compact writes share one
// scope. The QK handoff is retired only after this scope launches.
template <bool APPLY_CAUSAL_MASK = false, bool SPLIT_FIRST_ROW = false, bool SPLIT_SECOND_ROW = false,
          uint32_t HEADS = 0U>
__simd_vf__ inline void WrsSortCompactPair(__ubuf__ float* qk, __ubuf__ float* weights, __ubuf__ uint32_t* key0,
                                           __ubuf__ uint32_t* key1, __ubuf__ uint32_t* index0,
                                           __ubuf__ uint32_t* index1, __ubuf__ uint32_t* counts, uint32_t threshold0,
                                           uint32_t threshold1, uint64_t globalKeyOffset, uint32_t validKeys0 = 0,
                                           uint32_t validKeys1 = 0, uint32_t runtimeHeads = kHeads,
                                           uint32_t queryHeadStride = kHeads)
{
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> firstScores[kWrsFragmentCount], secondScores[kWrsFragmentCount];
    AccumulateWrsPair<SPLIT_FIRST_ROW, SPLIT_SECOND_ROW, HEADS>(qk, weights, firstScores, secondScores, full,
                                                                runtimeHeads, queryHeadStride);
    Reg::RegTensor<uint32_t> firstKeys[kWrsFragmentCount], secondKeys[kWrsFragmentCount];
    liV2Vector1::FloatSortConstCtx<float> context;
    liV2Vector1::InitFloatSortConstCtx(context, full);
    ConvertWrsRowToKeys(firstKeys, firstScores, context, full);
    ConvertWrsRowToKeys(secondKeys, secondScores, context, full);
    // Keep compaction and counts in the prefixMatchMask VF as WRS; no scalar handoff.
    CompactWrsRow<APPLY_CAUSAL_MASK>(firstKeys, key0, index0, counts, threshold0, globalKeyOffset, validKeys0, full);
    CompactWrsRow<APPLY_CAUSAL_MASK>(secondKeys, key1, index1, counts + 1, threshold1, globalKeyOffset, validKeys1,
                                     full);
}

__simd_vf__ inline void GenerateInitialTopkIndices(__ubuf__ uint32_t* index_scratch)
{
    auto full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<int32_t> index;
    for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(kTopK / kFloatRegLanes); ++chunk) {
        const uint32_t begin = static_cast<uint64_t>(chunk) * kFloatRegLanes;
        Reg::Arange(index, static_cast<int32_t>(begin));
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(index_scratch + begin,
                                                             reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), full);
    }
}

} // namespace LIV2Fast::vf
#endif

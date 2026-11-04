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
 * \file vf_fast_topk.h
 * \brief 直方图阈值筛选、候选压缩及分数与索引恢复VF
 */
#ifndef VF_FAST_TOPK_H
#define VF_FAST_TOPK_H

#include "../quant_lightning_indexer_v2_fast_common_arch35.h"
#include "vf_topk_16_gather_quant_v2.h"

#if defined(__CCE__) && !defined(__ASC_NPU_HOST__)
namespace QLIV2FastTopK {
using namespace AscendC;
using QLIV2Fast::TOPK_STATE_CAPACITY;
constexpr uint32_t MERGE_INPUT_CAPACITY = TOPK_STATE_CAPACITY + QLIV2Fast::TOPK_CANDIDATE_CAPACITY;
constexpr uint32_t UINT16_MODULUS = 1U << 16; // uint16索引减去状态区长度时的回绕基数
inline constexpr Reg::CastTrait kCastEven{Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                          RoundMode::UNKNOWN};
inline constexpr Reg::CastTrait kCastOdd{Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                         RoundMode::UNKNOWN};

// 这两个寄存器仅供当前调用使用，需要跨调用保留的累积分布存放在UB
template <bool BroadcastRank>
__simd_vf__ inline void SelectHighVF(__ubuf__ uint16_t* cdf, __ubuf__ uint32_t* high, __ubuf__ uint32_t* nextRank,
                                     uint32_t rank, uint32_t padding);
// 256个直方图桶分为两个128元素uint16寄存器，偏移128为后半区起点
struct HighCdf {
    Reg::RegTensor<uint16_t> lower, upper;
    __simd_callee__ inline static HighCdf Zero()
    {
        HighCdf r;
        Reg::Duplicate(r.lower, static_cast<uint16_t>(0));
        Reg::Duplicate(r.upper, static_cast<uint16_t>(0));
        return r;
    }
    __simd_callee__ inline static HighCdf Load(__ubuf__ uint16_t* src)
    {
        HighCdf r;
        Reg::LoadAlign<typename std::remove_reference_t<decltype(r.lower)>::ActualT, Reg::LoadDist::DIST_NORM>(
            r.lower, src + (0));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(r.upper)>::ActualT, Reg::LoadDist::DIST_NORM>(
            r.upper, src + (128));
        return r;
    }
    __simd_callee__ inline void Add(Reg::RegTensor<uint16_t> a, Reg::RegTensor<uint16_t> b, Reg::MaskReg active)
    {
        Reg::RegTensor<uint8_t> low, high;
        Reg::DeInterleave(low, high, reinterpret_cast<Reg::RegTensor<uint8_t>&>(a),
                          reinterpret_cast<Reg::RegTensor<uint8_t>&>(b));
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(lower, high,
                                                                                                          active);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(upper, high,
                                                                                                          active);
    }
    __simd_callee__ inline void Store(__ubuf__ uint16_t* dst)
    {
        Reg::MaskReg all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::StoreAlign<typename std::remove_reference_t<decltype(lower)>::ActualT, Reg::StoreDist::DIST_NORM>(
            dst + (0), lower, all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(upper)>::ActualT, Reg::StoreDist::DIST_NORM>(
            dst + (128), upper, all);
    }
    template <bool BroadcastRank>
    __aicore__ inline static void Select(LocalTensor<uint16_t> cdf, LocalTensor<uint32_t> high,
                                         LocalTensor<uint32_t> nextRank, uint32_t rank, uint32_t padding)
    {
        SelectHighVF<BroadcastRank>(reinterpret_cast<__ubuf__ uint16_t*>(cdf.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint32_t*>(high.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint32_t*>(nextRank.GetPhyAddr()), rank, padding);
    }
};

template <bool BroadcastRank>
__simd_vf__ inline void SelectHighVF(__ubuf__ uint16_t* cdf, __ubuf__ uint32_t* high, __ubuf__ uint32_t* nextRank,
                                     uint32_t rank, uint32_t padding)
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::MaskReg all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>(),
                 all32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    HighCdf values = HighCdf::Load(cdf);
    Reg::RegTensor<uint16_t> lower, upper, previous0, previous1, previous, one;
    Reg::RegTensor<uint32_t> bin0, bin1, bin, rankVector, previous32, remaining;
    Reg::MaskReg below0, below1;
    Reg::Adds(lower, values.lower, static_cast<uint16_t>(padding), all16);
    Reg::Adds(upper, values.upper, static_cast<uint16_t>(padding), all16);
    Reg::Compares<typename std::remove_reference_t<decltype(lower)>::ActualT, CMPMODE::LT>(
        below0, lower, static_cast<uint16_t>(rank), all16);
    Reg::Compares<typename std::remove_reference_t<decltype(upper)>::ActualT, CMPMODE::LT>(
        below1, upper, static_cast<uint16_t>(rank), all16);
    Reg::Reduce<Reg::ReduceType::MAX>(previous0, lower, below0);
    Reg::Reduce<Reg::ReduceType::MAX>(previous1, upper, below1);
    Reg::Max(previous, previous0, previous1, all16);
    Reg::Duplicate(one, static_cast<uint16_t>(1));
    Reg::Reduce<Reg::ReduceType::SUM>(bin0, one, below0);
    Reg::Reduce<Reg::ReduceType::SUM>(bin1, one, below1);
    Reg::Add(bin, bin0, bin1, all32);
    Reg::Duplicate(rankVector, rank);
    Reg::Cast<typename std::remove_reference_t<decltype(previous32)>::ActualT,
              typename std::remove_reference_t<decltype(previous)>::ActualT, kCastEven>(previous32, previous, all16);
    Reg::Sub(remaining, rankVector, previous32, all32);
    Reg::StoreAlign<typename std::remove_reference_t<decltype(bin)>::ActualT, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        high + (0), bin, all32);
    if constexpr (BroadcastRank) {
        // 原FindKthVFImpl按普通向量读取rank，须将首lane结果广播到整个256字节通信区
        Reg::RegTensor<uint32_t> firstLane, rankBroadcast;
        Reg::Duplicate(firstLane, static_cast<uint32_t>(0));
        Reg::Gather(rankBroadcast, remaining, firstLane);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nextRank, rankBroadcast, all32);
    } else {
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(nextRank, remaining, all32);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// 在双行处理的VF内调用，第二行无需额外的入口屏障
// 每次处理128个uint16分数，打包为高16位分数、低16位相对索引的4字节记录
template <bool FenceInput = true>
__simd_callee__ inline void PackFilteredRowBody(__ubuf__ uint16_t* input, __ubuf__ uint32_t* output,
                                                __ubuf__ uint32_t* count, uint32_t threshold, uint32_t valid)
{
    Reg::RegTensor<uint16_t> key, relative, zipped0, zipped1;
    Reg::RegTensor<int16_t> relativeSigned;
    Reg::RegTensor<uint32_t> kth, compact0, compact1, byteCount, nextByteCount;
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>(), selected0, selected1;
    if constexpr (FenceInput) {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    Reg::Duplicate(kth, threshold);
    Reg::Duplicate(byteCount, static_cast<uint32_t>(0));
    Reg::ClearSpr<SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore cache;
    for (uint16_t begin = 0; begin < static_cast<uint16_t>(valid); begin += 128) {
        Reg::LoadAlign<typename std::remove_reference_t<decltype(key)>::ActualT, Reg::LoadDist::DIST_NORM>(
            key, input + (static_cast<int64_t>(begin)));
        Reg::Arange(relativeSigned, static_cast<int16_t>(begin));
        relative = reinterpret_cast<Reg::RegTensor<uint16_t>&>(relativeSigned);
        Reg::Interleave(zipped0, zipped1, relative, key);
        Reg::Compare<
            typename std::remove_reference_t<decltype(reinterpret_cast<Reg::RegTensor<uint32_t>&>(zipped0))>::ActualT,
            CMPMODE::GT>(selected0, reinterpret_cast<Reg::RegTensor<uint32_t>&>(zipped0), kth, all);
        Reg::Adds(nextByteCount, byteCount, static_cast<uint32_t>(4), all);
        Reg::Select(byteCount, nextByteCount, byteCount, selected0);
        Reg::Compare<
            typename std::remove_reference_t<decltype(reinterpret_cast<Reg::RegTensor<uint32_t>&>(zipped1))>::ActualT,
            CMPMODE::GT>(selected1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(zipped1), kth, all);
        Reg::Adds(nextByteCount, byteCount, static_cast<uint32_t>(4), all);
        Reg::Select(byteCount, nextByteCount, byteCount, selected1);
        Reg::Squeeze<typename std::remove_reference_t<decltype(compact0)>::ActualT, Reg::GatherMaskMode::STORE_REG>(
            compact0, reinterpret_cast<Reg::RegTensor<uint32_t>&>(zipped0), selected0);
        Reg::StoreUnAlign<typename std::remove_reference_t<decltype(compact0)>::ActualT,
                          Reg::PostLiteral::POST_MODE_UPDATE>(output, compact0, cache);
        Reg::Squeeze<typename std::remove_reference_t<decltype(compact1)>::ActualT, Reg::GatherMaskMode::STORE_REG>(
            compact1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(zipped1), selected1);
        Reg::StoreUnAlign<typename std::remove_reference_t<decltype(compact1)>::ActualT,
                          Reg::PostLiteral::POST_MODE_UPDATE>(output, compact1, cache);
    }
    Reg::StoreUnAlignPost(output, cache);
    Reg::Reduce<Reg::ReduceType::SUM>(byteCount, byteCount, all);
    Reg::StoreAlign<typename std::remove_reference_t<decltype(byteCount)>::ActualT,
                    Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(count + (0), byteCount, all);
}

// 32位记录拆为两组128元素uint16；每轮256条，255为向上取整偏置
// 全局索引每64K一段（右移16位），512K窗口最多8段，cuts记录段间边界
__simd_vf__ inline void UnpackCandidates(__ubuf__ uint32_t* records, __ubuf__ uint16_t* outputKeys,
                                         __ubuf__ uint16_t* cuts, uint32_t keyTileStart, uint32_t count,
                                         uint32_t dmaCount, uint32_t candidateOffset, __ubuf__ uint16_t* highCdf)
{
    Reg::MaskReg all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>(), reached;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::RegTensor<uint16_t> boundaries, segments, next, zero;
    Reg::RegTensor<int16_t> seq;
    Reg::Duplicate(boundaries, static_cast<uint16_t>(MERGE_INPUT_CAPACITY));
    if (candidateOffset != 0U) {
        Reg::LoadAlign<typename std::remove_reference_t<decltype(boundaries)>::ActualT, Reg::LoadDist::DIST_NORM>(
            boundaries, cuts + (0));
    }
    Reg::Arange(seq, static_cast<int16_t>(1));
    segments = reinterpret_cast<Reg::RegTensor<uint16_t>&>(seq);
    Reg::Compares<typename std::remove_reference_t<decltype(segments)>::ActualT, CMPMODE::LE>(
        reached, segments, static_cast<uint16_t>(keyTileStart >> 16), all);
    Reg::Mins(next, boundaries, static_cast<uint16_t>(TOPK_STATE_CAPACITY + candidateOffset), all);
    Reg::Select(boundaries, next, boundaries, reached);
    Reg::MaskReg cutMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL8>();
    Reg::StoreAlign<typename std::remove_reference_t<decltype(boundaries)>::ActualT, Reg::StoreDist::DIST_NORM>(
        cuts + (0), boundaries, cutMask);
    auto* lowIndices = reinterpret_cast<__ubuf__ uint16_t*>(records);
    HighCdf cdf = HighCdf::Load(highCdf);
    Reg::Duplicate(zero, static_cast<uint16_t>(0));
    uint32_t validRemaining = count, dmaRemaining = dmaCount;
    const uint16_t repeats = static_cast<uint16_t>((dmaCount + 255U) / 256U);
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        uint32_t begin = static_cast<uint32_t>(static_cast<int64_t>(repeat) * 256);
        Reg::RegTensor<uint16_t> low0, high0, low1, high1, keys0, keys1;
        Reg::LoadAlign<typename std::remove_reference_t<decltype(low0)>::ActualT, Reg::LoadDist::DIST_DINTLV_B16>(
            low0, high0, lowIndices + (static_cast<int64_t>(2U * begin)));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(low1)>::ActualT, Reg::LoadDist::DIST_DINTLV_B16>(
            low1, high1, lowIndices + (static_cast<int64_t>(2U * begin + 256U)));
        Reg::MaskReg valid0 = Reg::UpdateMask<uint16_t>(validRemaining);
        Reg::MaskReg valid1 = Reg::UpdateMask<uint16_t>(validRemaining);
        Reg::Select(keys0, high0, zero, valid0);
        Reg::Select(keys1, high1, zero, valid1);
        cdf.Add(keys0, keys1, Reg::UpdateMask<uint8_t>(dmaRemaining));
        Reg::Adds(low0, low0, static_cast<uint16_t>(keyTileStart), all);
        Reg::Adds(low1, low1, static_cast<uint16_t>(keyTileStart), all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(keys0)>::ActualT, Reg::StoreDist::DIST_NORM>(
            outputKeys + (static_cast<int64_t>(begin)), keys0, all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(keys1)>::ActualT, Reg::StoreDist::DIST_NORM>(
            outputKeys + (static_cast<int64_t>(begin + 128U)), keys1, all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(low0)>::ActualT, Reg::StoreDist::DIST_NORM>(
            lowIndices + (static_cast<int64_t>(begin)), low0, all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(low1)>::ActualT, Reg::StoreDist::DIST_NORM>(
            lowIndices + (static_cast<int64_t>(begin + 128U)), low1, all);
    }
    cdf.Store(highCdf);
}

// 索引按高低16位恢复：8个64K分段对应7个边界；每轮128个候选输出两个64元素int32向量
template <bool REBASE>
__simd_vf__ inline void GatherIndices(__ubuf__ int32_t* state, __ubuf__ uint16_t* candidates, __ubuf__ uint16_t* cuts,
                                      __ubuf__ int16_t* positions, __ubuf__ int32_t* destination, uint32_t indexBase)
{
    Reg::MaskReg all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>(),
                 all32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::RegTensor<uint16_t> cut1, cut2, cut3, cut4, cut5, cut6, cut7;
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut1)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut1,
                                                                                                           cuts + (0));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut2)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut2,
                                                                                                           cuts + (1));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut3)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut3,
                                                                                                           cuts + (2));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut4)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut4,
                                                                                                           cuts + (3));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut5)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut5,
                                                                                                           cuts + (4));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut6)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut6,
                                                                                                           cuts + (5));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(cut7)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(cut7,
                                                                                                           cuts + (6));
    for (uint16_t begin = 0; begin < TOPK_STATE_CAPACITY; begin += 128) {
        Reg::RegTensor<uint16_t> selected, offset, low, high, absolute0, absolute1;
        Reg::RegTensor<uint32_t> p0, p1, previous0, previous1, result0, result1;
        Reg::MaskReg candidate, segment, state0, state1;
        Reg::LoadAlign<typename std::remove_reference_t<decltype(selected)>::ActualT, Reg::LoadDist::DIST_NORM>(
            selected, reinterpret_cast<__ubuf__ uint16_t*>(positions) + (static_cast<int64_t>(begin)));
        Reg::Compares<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(
            candidate, selected, static_cast<uint16_t>(TOPK_STATE_CAPACITY), all16);
        Reg::Adds(offset, selected, static_cast<uint16_t>(UINT16_MODULUS - TOPK_STATE_CAPACITY), all16);
        Reg::Gather(low, candidates, offset, candidate);
        Reg::Duplicate(high, static_cast<uint16_t>(0));
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut1, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(1), segment);
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut2, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(2), segment);
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut3, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(3), segment);
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut4, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(4), segment);
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut5, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(5), segment);
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut6, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(6), segment);
        Reg::Compare<typename std::remove_reference_t<decltype(selected)>::ActualT, CMPMODE::GE>(segment, selected,
                                                                                                 cut7, all16);
        Reg::Duplicate<typename std::remove_reference_t<decltype(high)>::ActualT, Reg::MaskMergeMode::MERGING>(
            high, static_cast<uint16_t>(7), segment);
        Reg::Interleave(absolute0, absolute1, low, high);
        if constexpr (REBASE) {
            Reg::Adds(reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute0),
                      reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute0), indexBase, all32);
            Reg::Adds(reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute1),
                      reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute1), indexBase, all32);
        }
        Reg::UnPack<typename std::remove_reference_t<decltype(p0)>::ActualT,
                    typename std::remove_reference_t<decltype(selected)>::ActualT, Reg::HighLowPart::LOWEST>(p0,
                                                                                                             selected);
        Reg::UnPack<typename std::remove_reference_t<decltype(p1)>::ActualT,
                    typename std::remove_reference_t<decltype(selected)>::ActualT, Reg::HighLowPart::HIGHEST>(p1,
                                                                                                              selected);
        Reg::Compares<typename std::remove_reference_t<decltype(p0)>::ActualT, CMPMODE::LT>(
            state0, p0, static_cast<uint32_t>(TOPK_STATE_CAPACITY), all32);
        Reg::Compares<typename std::remove_reference_t<decltype(p1)>::ActualT, CMPMODE::LT>(
            state1, p1, static_cast<uint32_t>(TOPK_STATE_CAPACITY), all32);
        Reg::Gather(previous0, reinterpret_cast<__ubuf__ uint32_t*>(state), p0, state0);
        Reg::Gather(previous1, reinterpret_cast<__ubuf__ uint32_t*>(state), p1, state1);
        Reg::Select(result0, previous0, reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute0), state0);
        Reg::Select(result1, previous1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute1), state1);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(result0)>::ActualT, Reg::StoreDist::DIST_NORM>(
            reinterpret_cast<__ubuf__ uint32_t*>(destination) + (static_cast<int64_t>(begin)), result0, all32);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(result1)>::ActualT, Reg::StoreDist::DIST_NORM>(
            reinterpret_cast<__ubuf__ uint32_t*>(destination) + (static_cast<int64_t>(begin + 64)), result1, all32);
    }
}

// MX按选定高字节统计低字节累计直方图，直接向定秩步骤提供寄存器结果
// 每轮读取256个uint16（512字节），按高低8位分别进行直方图筛选
__simd_callee__ inline void HistogramLowBins(Reg::RegTensor<uint16_t>& bins0, Reg::RegTensor<uint16_t>& bins1,
                                             __ubuf__ uint16_t* keys, __ubuf__ uint32_t* targetHigh, uint16_t repeats)
{
    Reg::RegTensor<uint8_t> selectedHigh, low, high;
    Reg::MaskReg all8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>(), selected;
    Reg::Duplicate(bins0, static_cast<uint16_t>(0));
    Reg::Duplicate(bins1, static_cast<uint16_t>(0));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(selectedHigh)>::ActualT, Reg::LoadDist::DIST_BRC_B8>(
        selectedHigh, reinterpret_cast<__ubuf__ uint8_t*>(targetHigh) + (0));
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        Reg::LoadAlign<typename std::remove_reference_t<decltype(low)>::ActualT, Reg::LoadDist::DIST_DINTLV_B8>(
            low, high, reinterpret_cast<__ubuf__ uint8_t*>(keys) + (static_cast<int64_t>(repeat) * 512));
        Reg::Compare<typename std::remove_reference_t<decltype(high)>::ActualT, CMPMODE::EQ>(selected, high,
                                                                                             selectedHigh, all8);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(bins0, low,
                                                                                                          selected);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(bins1, low,
                                                                                                          selected);
    }
}

// 高字节左移8位再合并低字节，恢复16位分数阈值
// nextK复用同一通信区：入口保存所选桶内排名，出口保存完整阈值，与调度层暂存区对应
__simd_callee__ inline void SelectLowByteKth(__ubuf__ uint16_t* keys, __ubuf__ uint32_t* targetHigh,
                                             __ubuf__ uint32_t* nextK, uint16_t repeats)
{
    Reg::RegTensor<uint16_t> bins0, bins1, rank, one;
    Reg::RegTensor<uint32_t> count0, count1, lowByte, highByte, kth;
    Reg::MaskReg all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg all32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>(), below0, below1;
    HistogramLowBins(bins0, bins1, keys, targetHigh, repeats);
    Reg::LoadAlign<typename std::remove_reference_t<decltype(rank)>::ActualT, Reg::LoadDist::DIST_BRC_B16>(
        rank, reinterpret_cast<__ubuf__ uint16_t*>(nextK) + (0));
    Reg::Compare<typename std::remove_reference_t<decltype(bins0)>::ActualT, CMPMODE::LT>(below0, bins0, rank, all16);
    Reg::Compare<typename std::remove_reference_t<decltype(bins1)>::ActualT, CMPMODE::LT>(below1, bins1, rank, all16);
    Reg::Duplicate(one, static_cast<uint16_t>(1));
    Reg::Reduce<Reg::ReduceType::SUM>(count0, one, below0);
    Reg::Reduce<Reg::ReduceType::SUM>(count1, one, below1);
    Reg::Add(lowByte, count0, count1, all32);
    Reg::LoadAlign<typename std::remove_reference_t<decltype(highByte)>::ActualT, Reg::LoadDist::DIST_BRC_B32>(
        highByte, targetHigh + (0));
    Reg::ShiftLefts(highByte, highByte, static_cast<int16_t>(8), all32);
    Reg::Add(kth, highByte, lowByte, all32);
    Reg::StoreAlign<typename std::remove_reference_t<decltype(kth)>::ActualT, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        nextK + (0), kth, all32);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// 候选总数补齐到256元素；以128个uint16为一个寄存器，255为256元素向上取整的偏置
__simd_vf__ inline void RepairPadding(__ubuf__ uint16_t* keys, uint32_t count)
{
    // 已对齐时无需补零，提前结束可避免额外VF读写；其余分支分别保护部分向量及整向量尾块
    if (count % 256U == 0U) {
        return;
    }

    Reg::RegTensor<uint16_t> value, zero;
    Reg::MaskReg all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    uint32_t valid = count % 128U, begin = TOPK_STATE_CAPACITY + count / 128U * 128U;
    Reg::Duplicate(zero, static_cast<uint16_t>(0));
    if (valid != 0U) {
        Reg::MaskReg active = Reg::UpdateMask<uint16_t>(valid);
        Reg::LoadAlign<typename std::remove_reference_t<decltype(value)>::ActualT, Reg::LoadDist::DIST_NORM>(
            value, keys + (static_cast<int64_t>(begin)));
        Reg::Select(value, value, zero, active);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(value)>::ActualT, Reg::StoreDist::DIST_NORM>(
            keys + (static_cast<int64_t>(begin)), value, all);
        begin += 128U;
    }
    const uint32_t end = TOPK_STATE_CAPACITY + (count + 255U) / 256U * 256U;
    if (begin < end) {
        Reg::StoreAlign<typename std::remove_reference_t<decltype(zero)>::ActualT, Reg::StoreDist::DIST_NORM>(
            keys + (static_cast<int64_t>(begin)), zero, all);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}
} // namespace QLIV2FastTopK
#endif

namespace QLIV2FastTopK {
// 单个寄存器容纳64个int32或128个uint16，边界表分两段初始化
__simd_vf__ inline void PrepareIdentityVF(__ubuf__ int32_t* indexScratch, __ubuf__ uint16_t* candidateCuts)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    for (uint16_t begin = 0; begin < TOPK_STATE_CAPACITY; begin += 64) {
        Reg::RegTensor<int32_t> index;
        Reg::Arange(index, static_cast<int32_t>(begin));
        Reg::StoreAlign<typename std::remove_reference_t<decltype(index)>::ActualT, Reg::StoreDist::DIST_NORM>(
            indexScratch + (static_cast<int64_t>(begin)), index, all);
    }
    Reg::RegTensor<uint16_t> empty;
    Reg::Duplicate(empty, static_cast<uint16_t>(MERGE_INPUT_CAPACITY));
    Reg::MaskReg all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::StoreAlign<typename std::remove_reference_t<decltype(empty)>::ActualT, Reg::StoreDist::DIST_NORM>(
        candidateCuts + (0), empty, all16);
    Reg::StoreAlign<typename std::remove_reference_t<decltype(empty)>::ActualT, Reg::StoreDist::DIST_NORM>(
        candidateCuts + (128), empty, all16);
}

// 每轮512个分数拆成4个128元素寄存器，起点依次为0、128、256、384
__simd_vf__ inline void InitializeHistogramVF(__ubuf__ uint16_t* input, __ubuf__ uint16_t* highCdfOutput,
                                              __ubuf__ uint32_t* nextK)
{
    Reg::MaskReg all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>(),
                 byteMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> a, b, c, d, minimum, ab, cd;
    Reg::LoadAlign<typename std::remove_reference_t<decltype(a)>::ActualT, Reg::LoadDist::DIST_NORM>(a, input + (0));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(b)>::ActualT, Reg::LoadDist::DIST_NORM>(b, input + (128));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(c)>::ActualT, Reg::LoadDist::DIST_NORM>(c, input + (256));
    Reg::LoadAlign<typename std::remove_reference_t<decltype(d)>::ActualT, Reg::LoadDist::DIST_NORM>(d, input + (384));
    HighCdf cdf = HighCdf::Zero();
    cdf.Add(a, b, byteMask);
    cdf.Add(c, d, byteMask);
    for (uint16_t begin = 512; begin < TOPK_STATE_CAPACITY; begin += 512) {
        Reg::RegTensor<uint16_t> n0, n1, n2, n3;
        Reg::LoadAlign<typename std::remove_reference_t<decltype(n0)>::ActualT, Reg::LoadDist::DIST_NORM>(
            n0, input + (static_cast<int64_t>(begin)));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(n1)>::ActualT, Reg::LoadDist::DIST_NORM>(
            n1, input + (static_cast<int64_t>(begin + 128)));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(n2)>::ActualT, Reg::LoadDist::DIST_NORM>(
            n2, input + (static_cast<int64_t>(begin + 256)));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(n3)>::ActualT, Reg::LoadDist::DIST_NORM>(
            n3, input + (static_cast<int64_t>(begin + 384)));
        cdf.Add(n0, n1, byteMask);
        cdf.Add(n2, n3, byteMask);
        Reg::Min(a, a, n0, all);
        Reg::Min(b, b, n1, all);
        Reg::Min(c, c, n2, all);
        Reg::Min(d, d, n3, all);
    }
    cdf.Store(highCdfOutput);
    Reg::Min(ab, a, b, all);
    Reg::Min(cd, c, d, all);
    Reg::Min(ab, ab, cd, all);
    Reg::Reduce<Reg::ReduceType::MIN>(minimum, ab, all);
    Reg::StoreAlign<typename std::remove_reference_t<decltype(minimum)>::ActualT, Reg::StoreDist::DIST_NORM>(
        reinterpret_cast<__ubuf__ uint16_t*>(nextK) + (0), minimum, all);
}

__simd_vf__ inline void PackRowVF(__ubuf__ uint16_t* input, __ubuf__ uint32_t* packedRecords, __ubuf__ uint32_t* counts,
                                  uint32_t kth, uint32_t valid)
{
    PackFilteredRowBody(input, packedRecords, counts, kth, static_cast<uint32_t>(valid));
}

__simd_vf__ inline void PackPairVF(__ubuf__ uint16_t* input0, __ubuf__ uint16_t* input1,
                                   __ubuf__ uint32_t* packedRecords, __ubuf__ uint32_t* counts, uint32_t kth0,
                                   uint32_t kth1, uint32_t valid)
{
    PackFilteredRowBody(input0, packedRecords, counts, kth0, static_cast<uint32_t>(valid));
    PackFilteredRowBody<false>(input1, packedRecords + TOPK_STATE_CAPACITY, counts + 1, kth1,
                               static_cast<uint32_t>(valid));
}
// MX在寄存器内完成低字节定秩，避免写回并重读256个uint32直方图元素
__simd_vf__ inline void FindThresholdMxVF(__ubuf__ uint16_t* mergeKeys, __ubuf__ uint32_t* targetHigh,
                                          __ubuf__ uint32_t* nextK, uint32_t length)
{
    SelectLowByteKth(mergeKeys, targetHigh, nextK, static_cast<uint16_t>(length / 256U));
}

// 每轮恢复256个分数，即两个128元素uint16寄存器
__simd_vf__ inline void GatherKeysVF(__ubuf__ int16_t* compactIndices, __ubuf__ uint16_t* mergeKeys,
                                     __ubuf__ uint16_t* intermediateKeyResults, __ubuf__ uint16_t* highCdfOutput)
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::MaskReg all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>(),
                 byteMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    HighCdf cdf = HighCdf::Zero();
    auto* positions = reinterpret_cast<__ubuf__ uint16_t*>(compactIndices);
    for (uint16_t begin = 0; begin < TOPK_STATE_CAPACITY; begin += 256) {
        Reg::RegTensor<uint16_t> p0, p1, k0, k1;
        Reg::LoadAlign<typename std::remove_reference_t<decltype(p0)>::ActualT, Reg::LoadDist::DIST_NORM>(
            p0, positions + (static_cast<int64_t>(begin)));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(p1)>::ActualT, Reg::LoadDist::DIST_NORM>(
            p1, positions + (static_cast<int64_t>(begin + 128)));
        Reg::Gather(k0, mergeKeys, p0, all);
        Reg::Gather(k1, mergeKeys, p1, all);
        cdf.Add(k0, k1, byteMask);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(k0)>::ActualT, Reg::StoreDist::DIST_NORM>(
            intermediateKeyResults + (static_cast<int64_t>(begin)), k0, all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(k1)>::ActualT, Reg::StoreDist::DIST_NORM>(
            intermediateKeyResults + (static_cast<int64_t>(begin + 128)), k1, all);
    }
    cdf.Store(highCdfOutput);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

} // namespace QLIV2FastTopK

#endif // VF_FAST_TOPK_H

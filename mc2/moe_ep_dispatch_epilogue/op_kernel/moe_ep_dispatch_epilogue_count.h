/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_COUNT_H
#define MOE_EP_DISPATCH_EPILOGUE_COUNT_H

namespace MoeEpDispatchEpilogueCount {
using namespace AscendC;
using IntReg = Reg::RegTensor<int32_t>;
constexpr uint32_t INT32_LANE_COUNT = 64;     // 一个RegTensor<int32_t>的通道数。
constexpr uint32_t HISTOGRAM_PAGE_BINS = 256; // 一页可统计的专家数，对应uint8_t的取值范围。
constexpr uint32_t HISTOGRAM_REG_BINS = 128;  // 一个直方图结果寄存器容纳的桶数。
constexpr uint32_t SCAN_INITIAL_STEP = 1;     // 前缀和首先合并相邻元素。
constexpr uint32_t SCAN_STEP_FACTOR = 2;      // 每轮将前缀和覆盖范围扩大一倍。

// 将直方图的uint16计数零扩展为int32，再累加到UB计数表。
__simd_callee__ inline void AddHistogramCounts(__ubuf__ int32_t* counts, Reg::RegTensor<uint16_t>& hist, uint32_t size)
{
    auto all = Reg::CreateMask<int32_t>();
    Reg::RegTensor<uint16_t> zero, low, high;
    Reg::Duplicate(zero, static_cast<uint16_t>(0));
    Reg::Interleave(low, high, hist, zero);
    IntReg lane, index, previous, sum;
    Reg::Arange(lane, 0);
    uint32_t remaining = size;
    auto valid = Reg::UpdateMask<int32_t>(remaining);
    Reg::Gather(previous, counts, reinterpret_cast<Reg::RegTensor<uint32_t>&>(lane), valid);
    Reg::Add(sum, previous, reinterpret_cast<IntReg&>(low), valid);
    Reg::Scatter(counts, sum, reinterpret_cast<Reg::RegTensor<uint32_t>&>(lane), valid);
    if (size > INT32_LANE_COUNT) {
        valid = Reg::UpdateMask<int32_t>(remaining);
        Reg::Adds(index, lane, static_cast<int32_t>(INT32_LANE_COUNT), all);
        Reg::Gather(previous, counts, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), valid);
        Reg::Add(sum, previous, reinterpret_cast<IntReg&>(high), valid);
        Reg::Scatter(counts, sum, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), valid);
    }
}

// 每页256个专家；padding为-1，先筛选再统计，避免低字节截断串桶。
__simd_vf__ __aicore__ inline void CountTopkHits(__ubuf__ int32_t* topk, __ubuf__ int32_t* counts, uint32_t elements,
                                                 uint32_t experts, uint32_t expertBase)
{
    auto all = Reg::CreateMask<int32_t>();
    for (uint32_t page = 0; page < experts; page += HISTOGRAM_PAGE_BINS) {
        Reg::RegTensor<uint16_t> low, high;
        Reg::Duplicate(low, static_cast<uint16_t>(0));
        Reg::Duplicate(high, static_cast<uint16_t>(0));
        uint32_t remaining = elements;
        uint32_t pageSize = experts - page < HISTOGRAM_PAGE_BINS ? experts - page : HISTOGRAM_PAGE_BINS;
        for (uint32_t offset = 0; offset < elements; offset += INT32_LANE_COUNT) {
            auto valid = Reg::UpdateMask<int32_t>(remaining);
            IntReg ids;
            Reg::LoadAlign(ids, topk + offset);
            Reg::Adds(ids, ids, -static_cast<int32_t>(expertBase + page), valid);
            Reg::MaskReg lower, upper, selected;
            Reg::Compares<int32_t, CMPMODE::GE>(lower, ids, 0, valid);
            Reg::Compares<int32_t, CMPMODE::LT>(upper, ids, static_cast<int32_t>(pageSize), valid);
            Reg::And(selected, lower, upper, all);
            // B32掩码只选择每个int32的最低字节；每tile至多128*32项，不溢出uint16。
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(
                low, reinterpret_cast<Reg::RegTensor<uint8_t>&>(ids), selected);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(
                high, reinterpret_cast<Reg::RegTensor<uint8_t>&>(ids), selected);
        }
        AddHistogramCounts(counts + page, low, pageSize < HISTOGRAM_REG_BINS ? pageSize : HISTOGRAM_REG_BINS);
        if (pageSize > HISTOGRAM_REG_BINS) {
            AddHistogramCounts(counts + page + HISTOGRAM_REG_BINS, high, pageSize - HISTOGRAM_REG_BINS);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// 64个整数做六轮倍增前缀和，避免逐专家读取UB再由Scalar累加。
__simd_callee__ inline void InclusiveScan(IntReg& value)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg lane, index, previous, zero;
    Reg::Arange(lane, 0);
    Reg::Duplicate(zero, 0);
    for (uint32_t step = SCAN_INITIAL_STEP; step < INT32_LANE_COUNT; step *= SCAN_STEP_FACTOR) {
        Reg::MaskReg valid;
        Reg::Adds(index, lane, -static_cast<int32_t>(step), all);
        Reg::Compares<int32_t, CMPMODE::GE>(valid, index, 0, all);
        Reg::Maxs(index, index, 0, all);
        Reg::Gather(previous, value, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index));
        Reg::Select(previous, previous, zero, valid);
        Reg::Add(value, value, previous, all);
    }
}

// 专家起点计算完成后，将每rank此前核的计数原地归约到行首，不增加UB。
__simd_vf__ __aicore__ inline void ReduceRankCorePrefixes(__ubuf__ int32_t* counts, uint32_t ranks, uint32_t stride)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg lane, index, value, sum, total;
    Reg::Arange(lane, 0);
    for (uint32_t rank = 0; rank < ranks; ++rank) {
        Reg::Duplicate(total, 0);
        uint32_t remaining = stride;
        for (uint32_t offset = 0; offset < stride; offset += INT32_LANE_COUNT) {
            auto valid = Reg::UpdateMask<int32_t>(remaining);
            Reg::Adds(index, lane, static_cast<int32_t>(rank * stride + offset), all);
            Reg::Gather(value, counts, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), valid);
            Reg::Reduce<Reg::ReduceType::SUM>(sum, value, valid);
            Reg::Duplicate(sum, sum, all);
            Reg::Add(total, total, sum, all);
        }
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(counts + rank * stride, total, all);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// 原地把总计数变为metadata起点；padding计数为0，不改变跨rank前缀和。
__simd_vf__ __aicore__ inline void BuildMetadataStarts(__ubuf__ int32_t* counts, __ubuf__ int32_t* corePrefixes,
                                                       __ubuf__ int32_t* rankOffsets, uint32_t ranks, uint32_t stride)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg lane, index, value, prefix, carry, sum, zero, core, rank, divisor, firstIndex;
    Reg::Arange(lane, 0);
    Reg::Duplicate(zero, 0);
    Reg::Duplicate(carry, 0);
    Reg::Duplicate(divisor, static_cast<int32_t>(stride));
    uint32_t remaining = ranks * stride;
    for (uint32_t base = 0; base < ranks * stride; base += INT32_LANE_COUNT) {
        auto valid = Reg::UpdateMask<int32_t>(remaining);
        Reg::Adds(index, lane, static_cast<int32_t>(base), all);
        Reg::Gather(value, counts, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), valid);
        Reg::Select(value, value, zero, valid);
        Reg::Move(prefix, value);
        InclusiveScan(prefix);
        Reg::Sub(prefix, prefix, value, all);
        Reg::Add(prefix, prefix, carry, all);
        Reg::Div(rank, index, divisor, all);
        Reg::Muls(firstIndex, rank, static_cast<int32_t>(stride), all);
        Reg::MaskReg firstExpert;
        Reg::Compare<int32_t, CMPMODE::EQ>(firstExpert, index, firstIndex, valid);
        Reg::Scatter(rankOffsets, prefix, reinterpret_cast<Reg::RegTensor<uint32_t>&>(rank), firstExpert);
        Reg::Gather(core, corePrefixes, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), valid);
        Reg::Add(core, core, prefix, valid);
        Reg::Scatter(counts, core, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index), valid);
        Reg::Reduce<Reg::ReduceType::SUM>(sum, value, valid);
        Reg::Duplicate(sum, sum, all);
        Reg::Add(carry, carry, sum, all);
    }
    Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rankOffsets + ranks, carry, all);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

} // namespace MoeEpDispatchEpilogueCount
#endif

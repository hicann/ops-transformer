/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_EP_DISPATCH_EPILOGUE_ROUTE_H
#define MOE_EP_DISPATCH_EPILOGUE_ROUTE_H

#include "moe_ep_dispatch_epilogue_tiling.h"

namespace MoeEpDispatchEpilogueRoute {
using namespace AscendC;
using MoeEpDispatchEpilogueCount::IntReg;
using MoeEpDispatchEpilogueCount::INT32_LANE_COUNT;

// Kernel route-frame fields and UB element units.
constexpr uint32_t MOE_EP_UB_BLOCK_BYTES = 32U;
constexpr uint32_t MOE_EP_INT32_PER_BLOCK = MOE_EP_UB_BLOCK_BYTES / sizeof(int32_t);
constexpr uint32_t MOE_EP_INT32_PER_INT64 = sizeof(int64_t) / sizeof(int32_t);
constexpr uint32_t MOE_EP_SLOT_META_SECTIONS = 2U;
constexpr uint32_t MOE_EP_SLOT_SRC_RANK_OFFSET = 0U;
constexpr uint32_t MOE_EP_SLOT_TOKEN_OFFSET = 1U;
constexpr uint32_t MOE_EP_RECV_META_FIELDS = 4U;
constexpr uint32_t MOE_EP_RECV_META_RANK_OFFSET = 0U;
constexpr uint32_t MOE_EP_RECV_META_TOKEN_OFFSET = 1U;
constexpr uint32_t MOE_EP_RECV_META_TOPK_OFFSET = 2U;
constexpr uint32_t MOE_EP_RECV_META_X_OFFSET = 3U;
constexpr uint32_t MOE_EP_ROUTE_HIT_COUNT_OFFSET = 0U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_COUNT_OFFSET = 1U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_FIELDS = MOE_EP_INT32_PER_BLOCK;
constexpr uint32_t MOE_EP_ROUTE_GROUP_HITS_OFFSET = 0U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_META_OFFSET = 1U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_WEIGHT_OFFSET = 2U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_META_ROW_OFFSET = 3U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_DST_OFFSET = 4U;
constexpr uint32_t MOE_EP_ROUTE_GROUP_EXPERT_OFFSET = 6U;

// Gather/Scatter require aligned UB roots. Express every field offset in the index.
__simd_callee__ inline void GatherAt(IntReg& value, __ubuf__ int32_t* root, IntReg& index, uint32_t field,
                                     Reg::MaskReg& mask)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg position;
    Reg::Adds(position, index, static_cast<int32_t>(field), all);
    Reg::Gather(value, root, reinterpret_cast<Reg::RegTensor<uint32_t>&>(position), mask);
}

__simd_callee__ inline void ScatterAt(__ubuf__ int32_t* root, IntReg& value, IntReg& index, uint32_t field,
                                      Reg::MaskReg& mask)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg position;
    Reg::Adds(position, index, static_cast<int32_t>(field), all);
    Reg::Scatter(root, value, reinterpret_cast<Reg::RegTensor<uint32_t>&>(position), mask);
}

__simd_callee__ inline void ScanCarry(IntReg& value, IntReg& carry)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg total;
    Reg::Reduce<Reg::ReduceType::SUM>(total, value, all);
    Reg::Duplicate(total, total, all);
    MoeEpDispatchEpilogueCount::InclusiveScan(value);
    Reg::Add(value, value, carry, all);
    Reg::Add(carry, carry, total, all);
}

__simd_callee__ inline void DecodeRouteIndex(IntReg& slot, IntReg& topk, IntReg& flatRouteIndex, IntReg& divisor,
                                             uint32_t k, Reg::MaskReg& all)
{
    Reg::Div(slot, flatRouteIndex, divisor, all);
    Reg::Muls(topk, slot, static_cast<int32_t>(k), all);
    Reg::Sub(topk, flatRouteIndex, topk, all);
}

__simd_callee__ inline void CountBlockExpertHits(IntReg& hitCount, IntReg& earlierHits, IntReg& ids,
                                                 IntReg& flatRouteIndex, uint32_t base, uint32_t routeCount,
                                                 IntReg& one, IntReg& zero, Reg::MaskReg& valid, Reg::MaskReg& all)
{
    IntReg index, previous, increment;
    Reg::Duplicate(hitCount, 0);
    Reg::Duplicate(earlierHits, 0);
    // Match only this vector block. Frame counts carry stable ordinals across blocks.
    const uint32_t blockEnd = base + INT32_LANE_COUNT < routeCount ? base + INT32_LANE_COUNT : routeCount;
    for (uint32_t prev = base; prev < blockEnd; ++prev) {
        // Reuse the loaded IDs: no repeated UB broadcast or token/topk division here.
        Reg::Duplicate(index, static_cast<int32_t>(prev - base));
        Reg::Gather(previous, ids, reinterpret_cast<Reg::RegTensor<uint32_t>&>(index));
        Reg::MaskReg same, earlier, selected;
        Reg::Compare<int32_t, CMPMODE::EQ>(same, ids, previous, valid);
        Reg::Select(increment, one, zero, same);
        Reg::Add(hitCount, hitCount, increment, all);
        Reg::Compares<int32_t, CMPMODE::GT>(earlier, flatRouteIndex, static_cast<int32_t>(prev), valid);
        Reg::And(selected, same, earlier, all);
        Reg::Select(increment, one, zero, selected);
        Reg::Add(earlierHits, earlierHits, increment, all);
    }
}

__simd_callee__ inline void StoreBlockHits(__ubuf__ int32_t* output, __ubuf__ int32_t* offsets, __ubuf__ int32_t* xBase,
                                           __ubuf__ int32_t* cursor, __ubuf__ int32_t* metadataCursor,
                                           const MoeEpDispatchEpilogueRouteLayout layout, IntReg& j, IntReg& expert,
                                           IntReg& flatRouteIndex, IntReg& topk, IntReg& hitCount, IntReg& earlierHits,
                                           IntReg& current, IntReg& groupCarry, IntReg& one, IntReg& zero,
                                           Reg::MaskReg& valid, Reg::MaskReg& all)
{
    IntReg mbase, mrow, xindex, lo, hi, off, carryBit, lastOrdinal;
    GatherAt(current, cursor, expert, 0U, valid);
    GatherAt(mbase, metadataCursor, expert, 0U, valid);
    Reg::Add(mrow, mbase, earlierHits, all);
    Reg::Muls(xindex, expert, static_cast<int32_t>(MOE_EP_INT32_PER_INT64), all);
    GatherAt(lo, xBase, xindex, 0U, valid);
    GatherAt(hi, xBase, xindex, 1, valid);
    Reg::Add(off, current, earlierHits, all);
    Reg::Add(off, lo, off, all);
    Reg::MaskReg carryMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(carryMask, reinterpret_cast<Reg::RegTensor<uint32_t>&>(off),
                                        reinterpret_cast<Reg::RegTensor<uint32_t>&>(lo), valid);
    Reg::Select(carryBit, one, zero, carryMask);
    Reg::Add(hi, hi, carryBit, all);
    Reg::Muls(xindex, j, static_cast<int32_t>(MOE_EP_INT32_PER_INT64), all);
    ScatterAt(output, off, xindex, layout.dst, valid);
    ScatterAt(output, hi, xindex, layout.dst + 1U, valid);
    ScatterAt(output, off, j, layout.recvIdx, valid);
    ScatterAt(output, topk, j, layout.topk, valid);
    ScatterAt(output, flatRouteIndex, j, layout.flatRouteIndex, valid);
    IntReg frameBefore, frameOrdinal;
    GatherAt(frameBefore, offsets, expert, 0U, valid);
    Reg::Add(frameOrdinal, frameBefore, earlierHits, all);
    ScatterAt(output, frameOrdinal, j, layout.ordinal, valid);

    // Exactly one active writer per expert in this block.
    Reg::MaskReg last, leader, firstBlock;
    Reg::Adds(lastOrdinal, hitCount, -1, all);
    Reg::Compare<int32_t, CMPMODE::EQ>(last, earlierHits, lastOrdinal, valid);
    Reg::Add(current, current, hitCount, all);
    ScatterAt(cursor, current, expert, 0U, last);
    Reg::Add(current, mbase, hitCount, all);
    ScatterAt(metadataCursor, current, expert, 0U, last);
    Reg::Add(current, frameBefore, hitCount, all);
    ScatterAt(offsets, current, expert, 0U, last);
    Reg::Compares<int32_t, CMPMODE::EQ>(leader, earlierHits, 0, valid);
    Reg::Compares<int32_t, CMPMODE::EQ>(firstBlock, frameBefore, 0, valid);
    Reg::And(leader, leader, firstBlock, all);
    IntReg groups, groupIndex;
    Reg::Select(groups, one, zero, leader);
    ScanCarry(groups, groupCarry);
    Reg::Adds(groupIndex, groups, -1, all);
    Reg::Muls(groupIndex, groupIndex, static_cast<int32_t>(MOE_EP_ROUTE_GROUP_FIELDS), all);
    // Store each group's first row now; its full-frame length is known after all blocks.
    ScatterAt(output, mrow, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_META_ROW_OFFSET, leader);
    ScatterAt(output, off, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_DST_OFFSET, leader);
    ScatterAt(output, hi, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_DST_OFFSET + 1U, leader);
    ScatterAt(output, expert, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_EXPERT_OFFSET, leader);
}

__simd_callee__ inline void CollectRouteHits(__ubuf__ int32_t* input, __ubuf__ int32_t* output,
                                             __ubuf__ int32_t* offsets, __ubuf__ int32_t* xBase,
                                             __ubuf__ int32_t* cursor, __ubuf__ int32_t* metadataCursor,
                                             const MoeEpDispatchEpilogueRouteLayout layout, uint32_t routeCount,
                                             uint32_t k, uint32_t kAlign, uint32_t inputStride, uint32_t experts,
                                             uint32_t expertBase, uint32_t planBase, IntReg& lane, IntReg& zero,
                                             IntReg& one, IntReg& divisor, IntReg& hitCarry, IntReg& groupCarry,
                                             Reg::MaskReg& all)
{
    // The first scratch table holds frame counts until group sizing, then metadata offsets.
    uint32_t remaining = experts;
    for (uint32_t base = 0; base < experts; base += INT32_LANE_COUNT) {
        auto active = Reg::UpdateMask<int32_t>(remaining);
        IntReg index;
        Reg::Adds(index, lane, static_cast<int32_t>(base), all);
        ScatterAt(offsets, zero, index, 0U, active);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    remaining = routeCount;
    for (uint32_t base = 0; base < routeCount; base += INT32_LANE_COUNT) {
        auto lanes = Reg::UpdateMask<int32_t>(remaining);
        IntReg flatRouteIndex, slot, topk, index, ids, expert, n, p;
        Reg::Adds(flatRouteIndex, lane, static_cast<int32_t>(base), all);
        DecodeRouteIndex(slot, topk, flatRouteIndex, divisor, k, all);
        Reg::Muls(index, slot, static_cast<int32_t>(inputStride), all);
        Reg::Add(index, index, topk, all);
        GatherAt(ids, input, index, 0U, lanes);
        Reg::Adds(expert, ids, -static_cast<int32_t>(expertBase), lanes);
        Reg::MaskReg lower, upper, valid;
        Reg::Compares<int32_t, CMPMODE::GE>(lower, expert, 0, lanes);
        Reg::Compares<int32_t, CMPMODE::LT>(upper, expert, static_cast<int32_t>(experts), lanes);
        Reg::And(valid, lower, upper, all);
        CountBlockExpertHits(n, p, ids, flatRouteIndex, base, routeCount, one, zero, valid, all);

        IntReg hit, j, end, current;
        Reg::Select(hit, one, zero, valid);
        ScanCarry(hit, hitCarry);
        Reg::Adds(j, hit, -1, all);
        Reg::Adds(end, hit, static_cast<int32_t>(planBase), all);
        Reg::MaskReg lastTopk, firstTopk;
        Reg::Compares<int32_t, CMPMODE::EQ>(lastTopk, topk, static_cast<int32_t>(k - 1U), lanes);
        ScatterAt(output, end, slot, layout.ends, lastTopk);
        Reg::Compares<int32_t, CMPMODE::EQ>(firstTopk, topk, 0, lanes);
        Reg::Muls(index, slot, static_cast<int32_t>(inputStride), all);
        GatherAt(current, input, index, MOE_EP_SLOT_META_SECTIONS * kAlign + MOE_EP_SLOT_TOKEN_OFFSET, lanes);
        ScatterAt(output, current, slot, layout.token, firstTopk);
        GatherAt(current, input, index, MOE_EP_SLOT_META_SECTIONS * kAlign + MOE_EP_SLOT_SRC_RANK_OFFSET, lanes);
        ScatterAt(output, current, slot, layout.rank, firstTopk);

        StoreBlockHits(output, offsets, xBase, cursor, metadataCursor, layout, j, expert, flatRouteIndex, topk, n, p,
                       current, groupCarry, one, zero, valid, all);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <bool HasWeights>
__simd_callee__ inline void AssignGroupOffsets(__ubuf__ int32_t* output, __ubuf__ int32_t* offsets,
                                               const MoeEpDispatchEpilogueRouteLayout layout, uint32_t routeCount,
                                               uint32_t experts, IntReg& lane, IntReg& zero, IntReg& groupCarry,
                                               Reg::MaskReg& all)
{
    IntReg metaCarry, weightCarry;
    Reg::Duplicate(metaCarry, 0);
    Reg::Duplicate(weightCarry, 0);
    // Group IDs are unique. Replace each expert's count only after its group has read it.
    const uint32_t groupCapacity = experts < routeCount ? experts : routeCount;
    uint32_t remaining = groupCapacity;
    for (uint32_t base = 0; base < groupCapacity; base += INT32_LANE_COUNT) {
        auto lanes = Reg::UpdateMask<int32_t>(remaining);
        IntReg group, groupIndex, expert, n, metaSize, metaOffset, weightSize, weightOffset, alignment;
        Reg::Adds(group, lane, static_cast<int32_t>(base), all);
        Reg::MaskReg active;
        Reg::Compare<int32_t, CMPMODE::LT>(active, group, groupCarry, lanes);
        Reg::Muls(groupIndex, group, static_cast<int32_t>(MOE_EP_ROUTE_GROUP_FIELDS), all);
        GatherAt(expert, output, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_EXPERT_OFFSET, active);
        GatherAt(n, offsets, expert, 0U, active);
        Reg::Select(n, n, zero, active);
        Reg::Duplicate(alignment, static_cast<int32_t>(MOE_EP_INT32_PER_BLOCK));
        Reg::Muls(metaSize, n, static_cast<int32_t>(MOE_EP_RECV_META_FIELDS), all);
        Reg::Adds(metaSize, metaSize, static_cast<int32_t>(MOE_EP_INT32_PER_BLOCK - 1U), all);
        Reg::Div(metaSize, metaSize, alignment, all);
        Reg::Muls(metaSize, metaSize, static_cast<int32_t>(MOE_EP_INT32_PER_BLOCK), all);
        Reg::Move(metaOffset, metaSize);
        ScanCarry(metaSize, metaCarry);
        Reg::Sub(metaOffset, metaSize, metaOffset, all);
        ScatterAt(offsets, metaOffset, expert, 0U, active);
        ScatterAt(output, n, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_HITS_OFFSET, active);
        ScatterAt(output, metaOffset, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_META_OFFSET, active);
        if constexpr (HasWeights) {
            Reg::Adds(weightSize, n, static_cast<int32_t>(MOE_EP_INT32_PER_BLOCK - 1U), all);
            Reg::Div(weightSize, weightSize, alignment, all);
            Reg::Muls(weightSize, weightSize, static_cast<int32_t>(MOE_EP_INT32_PER_BLOCK), all);
            Reg::Move(weightOffset, weightSize);
            ScanCarry(weightSize, weightCarry);
            Reg::Sub(weightOffset, weightSize, weightOffset, all);
            ScatterAt(offsets, weightOffset, expert, experts, active);
            ScatterAt(output, weightOffset, groupIndex, layout.groups + MOE_EP_ROUTE_GROUP_WEIGHT_OFFSET, active);
        }
    }
}

template <bool HasWeights>
__simd_callee__ inline void WriteRouteMetadata(__ubuf__ int32_t* input, __ubuf__ int32_t* output,
                                               __ubuf__ int32_t* offsets, const MoeEpDispatchEpilogueRouteLayout layout,
                                               uint32_t routeCount, uint32_t k, uint32_t kAlign, uint32_t inputStride,
                                               uint32_t experts, uint32_t expertBase, IntReg& lane, IntReg& divisor,
                                               IntReg& hitCarry, Reg::MaskReg& all)
{
    uint32_t remaining = routeCount;
    for (uint32_t base = 0; base < routeCount; base += INT32_LANE_COUNT) {
        auto lanes = Reg::UpdateMask<int32_t>(remaining);
        IntReg j, flatRouteIndex, slot, topk, metaIndex, expert, id, ordinal, target, value, position;
        Reg::Adds(j, lane, static_cast<int32_t>(base), all);
        Reg::MaskReg valid;
        Reg::Compare<int32_t, CMPMODE::LT>(valid, j, hitCarry, lanes);
        GatherAt(flatRouteIndex, output, j, layout.flatRouteIndex, valid);
        GatherAt(ordinal, output, j, layout.ordinal, valid);
        DecodeRouteIndex(slot, topk, flatRouteIndex, divisor, k, all);
        Reg::Muls(metaIndex, slot, static_cast<int32_t>(inputStride), all);
        Reg::Add(position, metaIndex, topk, all);
        GatherAt(id, input, position, 0U, valid);
        Reg::Adds(expert, id, -static_cast<int32_t>(expertBase), valid);
        GatherAt(target, offsets, expert, 0U, valid);
        Reg::Muls(position, ordinal, static_cast<int32_t>(MOE_EP_RECV_META_FIELDS), all);
        Reg::Add(target, target, position, all);
        GatherAt(value, input, metaIndex, MOE_EP_SLOT_META_SECTIONS * kAlign + MOE_EP_SLOT_SRC_RANK_OFFSET, valid);
        ScatterAt(output, value, target, layout.metadata + MOE_EP_RECV_META_RANK_OFFSET, valid);
        GatherAt(value, input, metaIndex, MOE_EP_SLOT_META_SECTIONS * kAlign + MOE_EP_SLOT_TOKEN_OFFSET, valid);
        ScatterAt(output, value, target, layout.metadata + MOE_EP_RECV_META_TOKEN_OFFSET, valid);
        ScatterAt(output, topk, target, layout.metadata + MOE_EP_RECV_META_TOPK_OFFSET, valid);
        Reg::Muls(position, j, static_cast<int32_t>(MOE_EP_INT32_PER_INT64), all);
        GatherAt(value, output, position, layout.dst, valid);
        ScatterAt(output, value, target, layout.metadata + MOE_EP_RECV_META_X_OFFSET, valid);
        if constexpr (HasWeights) {
            GatherAt(target, offsets, expert, experts, valid);
            Reg::Add(target, target, ordinal, all);
            Reg::Add(position, metaIndex, topk, all);
            GatherAt(value, input, position, kAlign, valid);
            ScatterAt(output, value, target, layout.weights, valid);
        }
    }
}

template <bool HasWeights>
__simd_vf__ __aicore__ inline void BuildRoutePlan(__ubuf__ int32_t* input, __ubuf__ int32_t* output,
                                                  __ubuf__ int32_t* offsets, __ubuf__ int32_t* xBase,
                                                  __ubuf__ int32_t* cursor, __ubuf__ int32_t* metadataCursor,
                                                  const MoeEpDispatchEpilogueRouteLayout layout, uint32_t tokens,
                                                  uint32_t k, uint32_t kAlign, uint32_t inputStride, uint32_t experts,
                                                  uint32_t expertBase, uint32_t planBase)
{
    auto all = Reg::CreateMask<int32_t>();
    IntReg lane, zero, one, divisor, hitCarry, groupCarry;
    Reg::Arange(lane, 0);
    Reg::Duplicate(zero, 0);
    Reg::Duplicate(one, 1);
    Reg::Duplicate(divisor, static_cast<int32_t>(k));
    Reg::Duplicate(hitCarry, 0);
    Reg::Duplicate(groupCarry, 0);

    const uint32_t routeCount = tokens * k;
    CollectRouteHits(input, output, offsets, xBase, cursor, metadataCursor, layout, routeCount, k, kAlign, inputStride,
                     experts, expertBase, planBase, lane, zero, one, divisor, hitCarry, groupCarry, all);
    AssignGroupOffsets<HasWeights>(output, offsets, layout, routeCount, experts, lane, zero, groupCarry, all);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    WriteRouteMetadata<HasWeights>(input, output, offsets, layout, routeCount, k, kAlign, inputStride, experts,
                                   expertBase, lane, divisor, hitCarry, all);

    Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(output + MOE_EP_ROUTE_HIT_COUNT_OFFSET, hitCarry,
                                                                     all);
    Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(output + MOE_EP_ROUTE_GROUP_COUNT_OFFSET,
                                                                     groupCarry, all);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}
} // namespace MoeEpDispatchEpilogueRoute
#endif

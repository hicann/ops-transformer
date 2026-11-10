/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_QUERY_UNION_HPP
#define GBSA_MXFP8_QUERY_UNION_HPP

#ifdef __DAV_VEC__
namespace GbsaKernelArch35::MxQueryUnion {

static __simd_vf__ inline void CheckOrder(__ubuf__ float* output, __ubuf__ uint32_t* ids, uint16_t count)
{
    using namespace AscendC::Reg;
    RegTensor<int32_t> positions;
    RegTensor<uint32_t> values, previous;
    RegTensor<float> errors, sum, one, zero, reduced;
    uint32_t lanes = 64U;
    MaskReg full = UpdateMask<uint32_t>(lanes);
    Duplicate(sum, 0.0f);
    Duplicate(one, 1.0f);
    Duplicate(zero, 0.0f);
    uint32_t left = count;
    for (uint16_t chunk = 0U; chunk < (count + 63U) / 64U; ++chunk) {
        MaskReg active = UpdateMask<uint32_t>(left);
        LoadAlign(values, ids + chunk * 64U);
        Arange(positions, int32_t(chunk * 64U));
        MaskReg afterFirst, ascending;
        Compares<int32_t, AscendC::CMPMODE::GT>(afterFirst, positions, int32_t(0), active);
        Adds(positions, positions, int32_t(-1), active);
        Maxs(positions, positions, int32_t(0), active);
        Gather(previous, ids, (RegTensor<uint32_t>&)positions, active);
        Compare<uint32_t, AscendC::CMPMODE::GT>(ascending, values, previous, afterFirst);
        Select(errors, zero, one, ascending);
        Select(errors, errors, zero, afterFirst);
        Add(sum, sum, errors, full);
    }
    ReduceSum(reduced, sum, full);
    uint32_t oneCount = 1U;
    MaskReg first = UpdateMask<float>(oneCount);
    StoreAlign(output, reduced, first);
}

// The caller has proved every index is valid and strictly increasing.
static __simd_vf__ inline void BuildBitmap(__ubuf__ uint32_t* bitmap, __ubuf__ uint32_t* ids, uint16_t count,
                                           uint16_t blocks)
{
    using namespace AscendC::Reg;
    RegTensor<uint32_t> zero, one, values;
    Duplicate(zero, uint32_t(0));
    Duplicate(one, uint32_t(1));
    uint32_t left = blocks;
    for (uint16_t chunk = 0U; chunk < (blocks + 63U) / 64U; ++chunk) {
        MaskReg active = UpdateMask<uint32_t>(left);
        StoreAlign(bitmap + chunk * 64U, zero, active);
    }
    left = count;
    for (uint16_t chunk = 0U; chunk < (count + 63U) / 64U; ++chunk) {
        MaskReg active = UpdateMask<uint32_t>(left);
        LoadAlign(values, ids + chunk * 64U);
        Scatter(bitmap, one, values, active);
    }
}

// A register prefix scan assigns one output slot to each selected block.
// Carry stays in vector registers across chunks, avoiding scalar per-run reads.
static __simd_vf__ inline void BuildOrderedUnion(__ubuf__ uint32_t* output, __ubuf__ uint32_t* first,
                                                 __ubuf__ uint32_t* second, __ubuf__ uint32_t* count, uint16_t blocks)
{
    using namespace AscendC::Reg;
    RegTensor<int32_t> lanes, shiftedPositions;
    RegTensor<uint32_t> a, b, prefix, shifted, offsets, positions, carry, zero, lastLane;
    uint32_t allCount = 64U;
    MaskReg full = UpdateMask<uint32_t>(allCount);
    Arange(lanes, int32_t(0));
    Duplicate(carry, uint32_t(0));
    Duplicate(zero, uint32_t(0));
    Duplicate(lastLane, uint32_t(63));
    uint32_t left = blocks;
    for (uint16_t chunk = 0U; chunk < (blocks + 63U) / 64U; ++chunk) {
        MaskReg active = UpdateMask<uint32_t>(left);
        LoadAlign(a, first + chunk * 64U);
        LoadAlign(b, second + chunk * 64U);
        Or(prefix, a, b, full);
        Select(prefix, prefix, zero, active);
        MaskReg emit;
        Compares<uint32_t, AscendC::CMPMODE::NE>(emit, prefix, uint32_t(0), active);
        for (uint16_t step = 1U; step < 64U; step <<= 1U) {
            Adds(shiftedPositions, lanes, -int32_t(step), full);
            Maxs(shiftedPositions, shiftedPositions, int32_t(0), full);
            Gather(shifted, prefix, (RegTensor<uint32_t>&)shiftedPositions);
            MaskReg hasPrevious;
            Compares<int32_t, AscendC::CMPMODE::GE>(hasPrevious, lanes, int32_t(step), full);
            Select(shifted, shifted, zero, hasPrevious);
            Add(prefix, prefix, shifted, full);
        }
        Add(offsets, prefix, carry, full);
        Adds(offsets, offsets, uint32_t(-1), full);
        Adds(positions, (RegTensor<uint32_t>&)lanes, uint32_t(chunk * 64U), full);
        Scatter(output, positions, offsets, emit);
        Gather(shifted, prefix, lastLane);
        Add(carry, carry, shifted, full);
    }
    uint32_t oneCount = 1U;
    MaskReg lane0 = UpdateMask<uint32_t>(oneCount);
    StoreAlign(count, carry, lane0);
}

} // namespace GbsaKernelArch35::MxQueryUnion
#endif
#endif

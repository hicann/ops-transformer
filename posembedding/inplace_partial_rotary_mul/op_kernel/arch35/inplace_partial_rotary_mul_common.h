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
 * \file common.h
 * \brief
 */
#ifndef INPLACE_PARTIAL_ROTARY_MUL_COMMON_H_V35
#define INPLACE_PARTIAL_ROTARY_MUL_COMMON_H_V35

#include "kernel_operator.h"

namespace ops {
using namespace AscendC;
constexpr int32_t MIN_FP32 = 0xFF800000;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 x, T2 y)
{
    if (y != 0 && x != 0) {
        const T1 quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }

    return x;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

constexpr Reg::CastTrait castTraitB162B32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait castTraitB322B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr Reg::CastTrait castTraitB322Int32 = {
    Reg::RegLayout::UNKNOWN,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};

constexpr Reg::CastTrait castTraitB322Int16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};

constexpr Reg::CastTrait castTraitB162Int8 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};

// load 对齐的 bfloat16,float16,bfloat32类型的 input(ub中)数据到 float32类型的dst(寄存器)中
template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__ubuf__ T *input, Reg::RegTensor<float> &dst, Reg::MaskReg &preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ half *)(input) + (offset)));
        Reg::Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, ((__ubuf__ bfloat16_t *)(input) + (offset)));
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst, ((__ubuf__ float *)(input) + (offset)));
    }
}

// load 2个对齐的Tensor 到寄存器中
template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(__ubuf__ T *src1, __ubuf__ T *src2, Reg::RegTensor<float> &dst1,
                                              Reg::RegTensor<float> &dst2, Reg::MaskReg &dst1Preg,
                                              Reg::MaskReg &dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16Q;
        Reg::RegTensor<half> xFp16R;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ half *)(src1) + (src1Offset)));
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ half *)(src2) + (src2Offset)));
        Reg::Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Reg::Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xFp16Q;
        Reg::RegTensor<bfloat16_t> xFp16R;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xFp16Q,
                                                                   ((__ubuf__ bfloat16_t *)(src1) + (src1Offset)));
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xFp16R,
                                                                   ((__ubuf__ bfloat16_t *)(src2) + (src2Offset)));
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst1, ((__ubuf__ float *)(src1) + (src1Offset)));
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst2, ((__ubuf__ float *)(src2) + (src2Offset)));
    }
}

// store
// 对齐的float32类型的src(寄存器)数据到output(ub)中，output数据类型支持bfloat16,float16,bfloat32,int32_t,int16_t,int8_t,uint8_t
template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__ubuf__ T *output, Reg::RegTensor<float> &src, Reg::MaskReg &preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> yFp16;
        Reg::Cast<half, float, castTraitB322B16>(yFp16, src, preg);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(((__ubuf__ half *)output + offset), yFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(((__ubuf__ bfloat16_t *)output + offset), xBf16,
                                                                   preg);
    } else if constexpr (IsSameType<T, int32_t>::value) {
        Reg::RegTensor<int32_t> zInt32;
        Reg::Cast<int32_t, float, castTraitB322Int32>(zInt32, src, preg);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM>(((__ubuf__ int32_t *)output + offset), zInt32, preg);
    } else if constexpr (IsSameType<T, int16_t>::value) {
        Reg::RegTensor<int16_t> zInt16;
        Reg::Cast<int16_t, float, castTraitB322Int16>(zInt16, src, preg);
        Reg::StoreAlign<int16_t, Reg::StoreDist::DIST_PACK_B32>(((__ubuf__ int16_t *)output + offset), zInt16, preg);
    } else if constexpr (IsSameType<T, int8_t>::value) {
        Reg::RegTensor<half> yFp16;
        Reg::RegTensor<int8_t> zInt8;
        Reg::Cast<half, float, castTraitB322Int16>(yFp16, src, preg);
        Reg::Cast<int8_t, half, castTraitB162Int8>(zInt8, yFp16, preg);
        Reg::StoreAlign<int8_t, Reg::StoreDist::DIST_PACK4_B32>(((__ubuf__ int8_t *)output + offset), zInt8, preg);
    } else if constexpr (IsSameType<T, uint8_t>::value) {
        Reg::RegTensor<half> yFp16;
        Reg::RegTensor<uint8_t> zUint8;
        Reg::Cast<half, float, castTraitB322Int16>(yFp16, src, preg);
        Reg::Cast<uint8_t, half, castTraitB162Int8>(zUint8, yFp16, preg);
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(((__ubuf__ uint8_t *)output + offset), zUint8, preg);
    } else {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(((__ubuf__ float *)output + offset), src, preg);
    }
}

} // namespace ops
#endif // INPLACE_PARTIAL_ROTARY_MUL_COMMON_H_V35

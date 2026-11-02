/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH35_VF_H
#define MEGA_MOE_ARCH35_VF_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "../../arch35_92/common/mega_moe_constants.h"

namespace MegaMoeImpl {

using namespace AscendC;

template <typename T>
__simd_vf__ inline void FillUbVF(__ubuf__ T* destination, T value, uint32_t elementCount)
{
    static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>);
    using ValueVector = std::conditional_t<std::is_same_v<T, int16_t>, vector_int16_t, vector_int32_t>;
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(T);
    ValueVector valueReg;
    vector_bool fullMask = sizeof(T) == sizeof(int16_t) ? asc_create_mask_b16(PAT_ALL) : asc_create_mask_b32(PAT_ALL);
    asc_duplicate_scalar(valueReg, value, fullMask);
    const uint16_t repeatCount = static_cast<uint16_t>((elementCount + ELEMENTS_PER_VECTOR - 1U) / ELEMENTS_PER_VECTOR);
    uint32_t remaining = elementCount;
    for (uint16_t i = 0U; i < repeatCount; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * ELEMENTS_PER_VECTOR;
        vector_bool active =
            sizeof(T) == sizeof(int16_t) ? asc_update_mask_b16(remaining) : asc_update_mask_b32(remaining);
        asc_storealign(destination + offset, valueReg, active);
    }
}

template <typename T>
__simd_vf__ inline void CreateRouteIndicesVF(__ubuf__ T* destination, int32_t firstIndex, uint32_t elementCount)
{
    static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>);
    using IndexVector = std::conditional_t<std::is_same_v<T, int16_t>, vector_int16_t, vector_int32_t>;
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(T);
    IndexVector indicesReg;
    for (uint32_t offset = 0U; offset < elementCount; offset += ELEMENTS_PER_VECTOR) {
        uint32_t remaining = elementCount - offset;
        vector_bool active =
            sizeof(T) == sizeof(int16_t) ? asc_update_mask_b16(remaining) : asc_update_mask_b32(remaining);
        asc_arange(indicesReg, static_cast<T>(firstIndex + static_cast<int32_t>(offset)));
        asc_storealign(destination + offset, indicesReg, active);
    }
}

// 源和目的均32B对齐且不重叠；尾段只读取有效BF16，支持只有32B/64B的TopK权重槽。
__simd_callee__ inline void ConvertBf16ToFp32(__ubuf__ float* destination, __ubuf__ bfloat16_t* source,
                                              uint32_t elementCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(float);
    vector_bfloat16_t inputReg;
    vector_float outputReg;
    vector_bool fullMask = asc_create_mask_b32(PAT_ALL);
    const uint16_t fullRepeatCount = static_cast<uint16_t>(elementCount / ELEMENTS_PER_VECTOR);
    for (uint16_t i = 0U; i < fullRepeatCount; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * ELEMENTS_PER_VECTOR;
        asc_loadalign_unpack(inputReg, source + offset);
        asc_bfloat162float(outputReg, inputReg, fullMask);
        asc_storealign(destination + offset, outputReg, fullMask);
    }
    const uint32_t tailOffset = static_cast<uint32_t>(fullRepeatCount) * ELEMENTS_PER_VECTOR;
    const uint32_t tailCount = elementCount - tailOffset;
    // 无尾段时使用有效基址，避免传入UB空间外的尾后指针。
    const uint32_t safeOffset = tailOffset * static_cast<uint32_t>(tailCount != 0U);
    uint32_t maskCount = tailCount;
    vector_bool active = asc_update_mask_b32(maskCount);
    vector_int32_t indexReg;
    asc_arange(indexReg, 0);
    // b32索引将BF16放入偶数b16 lane，未激活项不访问UB。
    asc_gather(inputReg, source + safeOffset, reinterpret_cast<vector_uint32_t&>(indexReg), active);
    asc_bfloat162float(outputReg, inputReg, active);
    asc_storealign(destination + safeOffset, outputReg, active);
}

__simd_vf__ inline void CastBf16ToFp32VF(__ubuf__ float* destination, __ubuf__ bfloat16_t* source,
                                         uint32_t elementCount)
{
    ConvertBf16ToFp32(destination, source, elementCount);
}

// 调用方保证长度按64项对齐；Unpermute整行和去重成员分段均覆盖完整向量加载范围。
template <bool InitAccum, bool ApplyWeight>
__simd_vf__ inline void AccumulateBf16ExpertSlotVF(__ubuf__ float* accumUb, __ubuf__ bfloat16_t* inputUb,
                                                   float expertScale, uint32_t elementCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(float);
    const uint16_t repeatCount = static_cast<uint16_t>((elementCount + ELEMENTS_PER_VECTOR - 1U) / ELEMENTS_PER_VECTOR);
    vector_bool active = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t inputReg;
    vector_float valueReg;
    for (uint16_t i = 0U; i < repeatCount; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * ELEMENTS_PER_VECTOR;
        // 64 个连续 BF16 放在偶数 b16 lane，转换后与 64 个 FP32 累加值逐项对应。
        asc_loadalign_unpack(inputReg, inputUb + offset);
        asc_bfloat162float(valueReg, inputReg, active);
        if constexpr (ApplyWeight) {
            // 保持独立 FP32 乘法和加法，不使用 FMA 或乘加后转半精度的接口。
            asc_mul_scalar(valueReg, valueReg, expertScale, active);
        }
        if constexpr (!InitAccum) {
            vector_float accumReg;
            asc_loadalign(accumReg, accumUb + offset);
            asc_add(valueReg, accumReg, valueReg, active);
        }
        asc_storealign(accumUb + offset, valueReg, active);
    }
}

// 与原CAST_RINT保持相同的RINT、SAT和ZEROING模式；调用方保证长度按64项对齐。
__simd_vf__ inline void CastFp32ToBf16VF(__ubuf__ bfloat16_t* destination, __ubuf__ float* source,
                                         uint32_t elementCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(float);
    vector_bool fullMask = asc_create_mask_b32(PAT_ALL);
    vector_float inputReg;
    vector_bfloat16_t outputReg;
    for (uint32_t offset = 0U; offset < elementCount; offset += ELEMENTS_PER_VECTOR) {
        asc_loadalign(inputReg, source + offset);
        asc_float2bfloat16_rn_sat(outputReg, inputReg, fullMask);
        // 每个32bit槽取低16bit连续存储，避免误用BF16重载的PK_B16。
        asc_storealign_pack(reinterpret_cast<__ubuf__ uint32_t*>(destination + offset),
                            reinterpret_cast<vector_uint32_t&>(outputReg), fullMask);
    }
}

template <int32_t InputMask = -1>
__simd_callee__ inline void InclusivePrefixSumInt32(__ubuf__ int32_t* values, uint32_t elementCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(int32_t);
    vector_int32_t prefixReg;
    vector_int32_t shiftedReg;
    vector_int32_t carryReg;
    vector_int32_t laneIndexReg;
    vector_int32_t shiftedIndexReg;
    vector_int32_t inputMaskReg;
    vector_bool fullMask = asc_create_mask_b32(PAT_ALL);

    asc_arange(laneIndexReg, 0);
    asc_duplicate_scalar(carryReg, 0, fullMask);
    if constexpr (InputMask != -1) {
        asc_duplicate_scalar(inputMaskReg, InputMask, fullMask);
    }
    for (uint32_t vectorOffset = 0U; vectorOffset < elementCount; vectorOffset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = elementCount - vectorOffset;
        uint32_t maskCount = validCount;
        vector_bool activeMask = asc_update_mask_b32(maskCount);
        validCount -= maskCount;
        asc_loadalign(prefixReg, values + vectorOffset);
        if constexpr (InputMask != -1) {
            asc_and(prefixReg, prefixReg, inputMaskReg, activeMask);
        }

        for (uint32_t shift = 1U; shift < validCount; shift <<= 1U) {
            asc_add_scalar(shiftedIndexReg, laneIndexReg, -static_cast<int32_t>(shift), fullMask);
            asc_max_scalar(shiftedIndexReg, shiftedIndexReg, 0, fullMask);
            asc_gather(shiftedReg, prefixReg, reinterpret_cast<vector_uint32_t&>(shiftedIndexReg));
            vector_bool shiftedLaneMask;
            asc_ge_scalar(shiftedLaneMask, laneIndexReg, static_cast<int32_t>(shift), activeMask);
            // asc_add 将未激活 lane 清零；复用 shiftedReg 暂存，再选择以保留原 MERGING 语义。
            asc_add(shiftedReg, prefixReg, shiftedReg, shiftedLaneMask);
            asc_select(prefixReg, shiftedReg, prefixReg, shiftedLaneMask);
        }

        asc_add(prefixReg, prefixReg, carryReg, activeMask);
        asc_storealign(values + vectorOffset, prefixReg, activeMask);

        asc_duplicate_scalar(shiftedIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
        asc_gather(carryReg, prefixReg, reinterpret_cast<vector_uint32_t&>(shiftedIndexReg));
    }
}

namespace Mxfp8 {

__simd_callee__ inline void ConvertMxScaleToBf16(__ubuf__ fp8_e8m0_t* srcPtr, __ubuf__ bfloat16_t* dyScaleBf16Ptr,
                                                 uint32_t scaleLen, uint32_t bf16RepeatSize, uint16_t repeatTimes)
{
    auto* scalePtr = reinterpret_cast<__ubuf__ uint8_t*>(srcPtr);
    auto* dstPtr = reinterpret_cast<__ubuf__ uint16_t*>(dyScaleBf16Ptr);
    vector_uint8_t scaleReg;
    vector_uint16_t bf16Bits;
    vector_bool nanMask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_bool mask = asc_update_mask_b16(scaleLen);
        asc_loadalign_unpack_postupdate(scaleReg, scalePtr, bf16RepeatSize);
        // UNPACK_B8 already zero-extends each scale byte to a uint16_t lane.
        asc_shiftleft_scalar(bf16Bits, reinterpret_cast<vector_uint16_t&>(scaleReg), int16_t{7}, mask);
        asc_eq_scalar(nanMask, bf16Bits, uint16_t{0x7f80}, mask);
        vdup(bf16Bits, uint16_t{0x7fc0}, nanMask, MODE_MERGING);
        asc_storealign_intlv(dstPtr, bf16Bits, bf16Bits);
        dstPtr += bf16RepeatSize * 2U;
    }
    asc_mem_bar(VST_VLD);
}

__simd_callee__ inline void ConvertMxScaleToFp32(__ubuf__ bfloat16_t* dyScaleBf16Ptr, __ubuf__ float* dyScaleFp32Ptr,
                                                 uint32_t quantCount, uint32_t fp32RepeatSize, uint16_t repeatTimes)
{
    vector_bfloat16_t bf16Reg;
    vector_float fp32Reg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_bool mask = asc_update_mask_b32(quantCount);
        asc_loadalign_unpack_postupdate(bf16Reg, dyScaleBf16Ptr, fp32RepeatSize);
        asc_bfloat162float(fp32Reg, bf16Reg, mask);
        asc_storealign_intlv(reinterpret_cast<__ubuf__ uint32_t*>(dyScaleFp32Ptr),
                             reinterpret_cast<vector_uint32_t&>(fp32Reg), reinterpret_cast<vector_uint32_t&>(fp32Reg));
        dyScaleFp32Ptr += fp32RepeatSize * 2U;
    }
    asc_mem_bar(VST_VLD);
}

template <typename T, bool InitAccum = true, bool ApplyWeight = false>
__simd_callee__ inline void DequantizeMxFp8Token(__ubuf__ T* tokenPtr, __ubuf__ float* dyScaleFp32Ptr,
                                                 __ubuf__ float* sumDstPtr, uint32_t tokenLen, uint32_t fp32RepeatSize,
                                                 uint16_t repeatTimes, float weight = 0.0f)
{
    static_assert(IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value);
    using TokenReg =
        typename std::conditional<IsSameType<T, fp8_e4m3fn_t>::value, vector_fp8_e4m3fn_t, vector_fp8_e5m2_t>::type;
    TokenReg tokenReg;
    vector_float valueReg;
    vector_float scaleReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_bool mask = asc_update_mask_b32(tokenLen);
        asc_loadalign_brc_postupdate_v3(scaleReg, dyScaleFp32Ptr, 8);
        asc_loadalign_unpack_postupdate_v2(tokenReg, tokenPtr, fp32RepeatSize);
        if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
            asc_e4m32float(valueReg, tokenReg, mask);
        } else {
            asc_e5m22float(valueReg, tokenReg, mask);
        }
        asc_mul(valueReg, scaleReg, valueReg, mask);
        if constexpr (ApplyWeight) {
            asc_mul_scalar(valueReg, valueReg, weight, mask);
        }
        if constexpr (!InitAccum) {
            vector_float accumReg;
            asc_loadalign(accumReg, sumDstPtr);
            asc_add(valueReg, accumReg, valueReg, mask);
        }
        asc_storealign_postupdate(sumDstPtr, valueReg, fp32RepeatSize, mask);
    }
}

template <typename T, typename XType>
__simd_vf__ inline void DeQuantMxFp8Vf(__ubuf__ XType* inAddr, __ubuf__ float* sumDstPtr,
                                       __ubuf__ bfloat16_t* dyScaleBf16Ptr, __ubuf__ float* dyScaleFp32Ptr,
                                       uint32_t scaleLen, uint32_t tokenLen)
{
    auto* tokenPtr = reinterpret_cast<__ubuf__ T*>(inAddr);
    uint32_t tokenStorageElementCount = ((tokenLen + ALIGN_256 - 1U) / ALIGN_256 * ALIGN_256) / 2U;
    auto* scalePtr = reinterpret_cast<__ubuf__ fp8_e8m0_t*>(inAddr + tokenStorageElementCount);
    constexpr uint32_t bf16RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint32_t fp32RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t bf16RepeatTimes = (scaleLen + bf16RepeatSize - 1U) / bf16RepeatSize;
    uint32_t quantCount = scaleLen * 2U;
    uint16_t scaleFp32RepeatTimes = (quantCount + fp32RepeatSize - 1U) / fp32RepeatSize;
    uint16_t tokenFp32RepeatTimes = (tokenLen + fp32RepeatSize - 1U) / fp32RepeatSize;
    ConvertMxScaleToBf16(scalePtr, dyScaleBf16Ptr, scaleLen, bf16RepeatSize, bf16RepeatTimes);
    ConvertMxScaleToFp32(dyScaleBf16Ptr, dyScaleFp32Ptr, quantCount, fp32RepeatSize, scaleFp32RepeatTimes);
    DequantizeMxFp8Token(tokenPtr, dyScaleFp32Ptr, sumDstPtr, tokenLen, fp32RepeatSize, tokenFp32RepeatTimes);
}

// MTE 输入槽覆盖尾批64B的 scale 读取；独立反量化仍使用原暂存流程。
// 调用方保证 tokenLen 按64项对齐且 scaleLen == tokenLen / 32。
template <typename T, bool InitAccum, bool ApplyWeight>
__simd_vf__ inline void AccumulateMxFp8VF(__ubuf__ T* tokenPtr, __ubuf__ fp8_e8m0_t* scalePtr, __ubuf__ float* accumPtr,
                                          uint32_t scaleLen, uint32_t tokenLen, float weight)
{
    static_assert(IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value);
    using TokenReg =
        typename std::conditional<IsSameType<T, fp8_e4m3fn_t>::value, vector_fp8_e4m3fn_t, vector_fp8_e5m2_t>::type;
    constexpr uint32_t fp32RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t SCALE_GROUP_SIZE = 32U;
    constexpr uint32_t SCALES_PER_TOKEN_VECTOR = fp32RepeatSize / SCALE_GROUP_SIZE;
    vector_bool fullMask = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t groupIndexReg;
    asc_arange(reinterpret_cast<vector_int32_t&>(groupIndexReg), 0);
    asc_shiftright_scalar(groupIndexReg, groupIndexReg, int16_t{5}, fullMask);
    const uint16_t scaleRepeatTimes = static_cast<uint16_t>((scaleLen + fp32RepeatSize - 1U) / fp32RepeatSize);
    for (uint16_t batch = 0U; batch < scaleRepeatTimes; ++batch) {
        const uint32_t scaleBase = static_cast<uint32_t>(batch) * fp32RepeatSize;
        uint32_t validScaleCount = scaleLen - scaleBase;
        vector_uint32_t scaleBits;
        {
            vector_uint8_t rawScaleReg;
            uint32_t maskCount = validScaleCount;
            vector_bool scaleMask = asc_update_mask_b32(maskCount);
            validScaleCount -= maskCount;
            vector_bool nanMask;
            asc_loadalign_unpack_v2(rawScaleReg, reinterpret_cast<__ubuf__ uint8_t*>(scalePtr) + scaleBase);
            asc_shiftleft_scalar(scaleBits, reinterpret_cast<vector_uint32_t&>(rawScaleReg), int16_t{23}, scaleMask);
            asc_eq_scalar(nanMask, scaleBits, uint32_t{0x7f800000}, scaleMask);
            vdup(scaleBits, uint32_t{0x7fc00000}, nanMask, MODE_MERGING);
        }
        uint32_t validDataCount = validScaleCount * SCALE_GROUP_SIZE;
        uint16_t repeatTimes = (validDataCount + fp32RepeatSize - 1U) / fp32RepeatSize;
        for (uint16_t i = 0U; i < repeatTimes; ++i) {
            vector_uint32_t scaleIndexReg;
            vector_float scaleReg;
            asc_add_scalar(scaleIndexReg, groupIndexReg, static_cast<uint32_t>(i) * SCALES_PER_TOKEN_VECTOR, fullMask);
            asc_gather(reinterpret_cast<vector_uint32_t&>(scaleReg), scaleBits, scaleIndexReg);
            vector_bool mask = asc_update_mask_b32(tokenLen);
            TokenReg tokenReg;
            vector_float valueReg;
            asc_loadalign_unpack_postupdate_v2(tokenReg, tokenPtr, fp32RepeatSize);
            if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
                asc_e4m32float(valueReg, tokenReg, mask);
            } else {
                asc_e5m22float(valueReg, tokenReg, mask);
            }
            asc_mul(valueReg, scaleReg, valueReg, mask);
            if constexpr (ApplyWeight) {
                asc_mul_scalar(valueReg, valueReg, weight, mask);
            }
            if constexpr (!InitAccum) {
                vector_float accumReg;
                asc_loadalign(accumReg, accumPtr);
                asc_add(valueReg, accumReg, valueReg, mask);
            }
            asc_storealign_postupdate(accumPtr, valueReg, fp32RepeatSize, mask);
        }
    }
}

} // namespace Mxfp8
} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH35_VF_H

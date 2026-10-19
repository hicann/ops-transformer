/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH35_ROWWISE_MX_QUANT_H
#define MEGA_MOE_ARCH35_ROWWISE_MX_QUANT_H

#if defined(__DAV_C310__)
#include "../../../../../common/op_kernel/quantize_functions.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"

namespace MegaMoeImpl {
namespace GeluQuant {

using namespace AscendC;

constexpr uint16_t MX_QUANT_GROUP_ELEMENTS = 32;
constexpr uint16_t FP4_VALUES_PER_BYTE = 2;
// B16 去交织后，每个半区连续 16 个通道归属同一 32 元素 MX 分组。
constexpr int16_t MX_HALF_GROUP_INDEX_SHIFT = 4;

struct TileContext {
    __ubuf__ bfloat16_t* activation;
    __ubuf__ int8_t* quantOutput;
    __ubuf__ uint8_t* scaleOutput;
    uint16_t rowCount;
    uint16_t activationRowStrideElements;
    uint16_t quantRowStrideBytes;
    uint16_t scaleRowStrideBytes;
};

// ReduceDataBlock 的八个结果分别是当前行八个 32 元素 MX 分组的最大值。
template <typename OutputType>
__simd_callee__ inline void ComputeRowScale(Reg::RegTensor<bfloat16_t>& firstHalf,
                                            Reg::RegTensor<bfloat16_t>& secondHalf, Reg::RegTensor<uint8_t>& scaleBytes,
                                            Reg::RegTensor<uint16_t>& inverseScale, Reg::MaskReg all)
{
    constexpr uint16_t maxOutputExponent = Std::IsSame<OutputType, fp8_e4m3fn_t>::value ? Quant::FP8_E4M3_MAX_EXP :
                                           Std::IsSame<OutputType, fp8_e5m2_t>::value   ? Quant::FP8_E5M2_MAX_EXP :
                                           Std::IsSame<OutputType, fp4x2_e2m1_t>::value ? Quant::FP4_E2M1_BF16_MAX_EXP :
                                                                                          Quant::FP4_E1M2_BF16_MAX_EXP;
    Reg::RegTensor<uint16_t> inputExponent0;
    Reg::RegTensor<uint16_t> inputExponent1;
    Reg::RegTensor<uint16_t> maximumExponent;
    Reg::RegTensor<uint16_t> sharedExponent;
    Reg::RegTensor<uint16_t> scaleValue;
    Reg::RegTensor<uint16_t> exponentMask;
    Reg::RegTensor<uint16_t> outputMax;
    Reg::RegTensor<uint16_t> bias;
    Reg::RegTensor<uint16_t> nanScale;
    Reg::RegTensor<uint16_t> zero;
    Reg::RegTensor<uint16_t> nanInverse;
    Reg::RegTensor<uint16_t> specialInverse;
    Reg::MaskReg finiteMask;
    Reg::MaskReg nonzeroMask;
    Reg::MaskReg clampMask;
    Reg::MaskReg specialMask;

    Reg::Duplicate(exponentMask, Quant::MAX_EXP_FOR_BF16);
    Reg::Duplicate(outputMax, maxOutputExponent);
    Reg::Duplicate(bias, Quant::BF16_EXP_BIAS);
    Reg::Duplicate(nanScale, Quant::MAX_EXP_FOR_FP8);
    Reg::Duplicate(zero, 0);
    Reg::Duplicate(nanInverse, Quant::BF16_NAN_CUSTOM);
    Reg::Duplicate(specialInverse, Quant::BF16_SPECIAL_EXP_THRESHOLD);

    Reg::And(inputExponent0, (Reg::RegTensor<uint16_t>&)firstHalf, exponentMask, all);
    Reg::And(inputExponent1, (Reg::RegTensor<uint16_t>&)secondHalf, exponentMask, all);
    Reg::Max(maximumExponent, inputExponent0, inputExponent1, all);
    Reg::ReduceDataBlock<Reg::ReduceType::MAX>(maximumExponent, maximumExponent, all);
    Reg::Compare<uint16_t, CMPMODE::NE>(finiteMask, maximumExponent, exponentMask, all);
    Reg::Compare<uint16_t, CMPMODE::NE>(nonzeroMask, maximumExponent, zero, all);
    Reg::Compare<uint16_t, CMPMODE::LE>(clampMask, maximumExponent, outputMax, all);
    Reg::Select<uint16_t>(maximumExponent, outputMax, maximumExponent, clampMask);
    Reg::Sub(sharedExponent, maximumExponent, outputMax, all);
    Reg::ShiftRights(scaleValue, sharedExponent, Quant::SHR_NUM_FOR_BF16, all);
    Reg::Select<uint16_t>(scaleValue, scaleValue, nanScale, finiteMask);
    Reg::Select<uint16_t>(scaleValue, scaleValue, zero, nonzeroMask);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scaleBytes, scaleValue);

    Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExponent, bias, all);
    Reg::Sub(inverseScale, bias, sharedExponent, all);
    Reg::Select<uint16_t>(inverseScale, inverseScale, nanInverse, finiteMask);
    Reg::Select<uint16_t>(inverseScale, inverseScale, zero, nonzeroMask);
    Reg::Select<uint16_t>(inverseScale, specialInverse, inverseScale, specialMask);
}

template <typename OutputType, uint32_t TileN>
__simd_vf__ inline void RunGeluMxQuant(TileContext context)
{
    // 当前每行由两个 BF16 寄存器完整装载，其他行宽需要配套调整寄存器算法。
    static_assert(TileN == AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t) * 2U,
                  "逐行 MX 量化需要两个完整 BF16 寄存器的列宽");
    constexpr uint16_t rowElementCount = TileN;
    constexpr uint16_t scaleBytesPerRow = TileN / MX_QUANT_GROUP_ELEMENTS;
    constexpr uint16_t fp4SourceWordsPerRow = TileN * sizeof(bfloat16_t) / sizeof(uint32_t);
    constexpr uint16_t fp4PackedBytesPerRow = TileN / FP4_VALUES_PER_BYTE;
    Reg::RegTensor<bfloat16_t> firstHalf;
    Reg::RegTensor<bfloat16_t> secondHalf;
    Reg::RegTensor<uint8_t> scaleBytes;
    Reg::RegTensor<uint16_t> inverseScale;
    Reg::RegTensor<uint16_t> inverseScaleExpanded;
    Reg::RegTensor<uint16_t> groupIndex;
    Reg::MaskReg all = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int16_t>&>(groupIndex), static_cast<int16_t>(0));
    Reg::ShiftRights(groupIndex, groupIndex, MX_HALF_GROUP_INDEX_SHIFT, all);

    for (uint16_t row = 0; row < context.rowCount; ++row) {
        __ubuf__ bfloat16_t* source = context.activation + row * context.activationRowStrideElements;
        __ubuf__ int8_t* quantOutput = context.quantOutput + row * context.quantRowStrideBytes;
        __ubuf__ uint8_t* scaleOutput = context.scaleOutput + row * context.scaleRowStrideBytes;

        // 在任一输出覆盖当前行之前，先读完全部 BF16 元素。
        Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            firstHalf, secondHalf, source, rowElementCount);
        ComputeRowScale<OutputType>(firstHalf, secondHalf, scaleBytes, inverseScale, all);
        Reg::Gather(inverseScaleExpanded, inverseScale, groupIndex);

        if constexpr (Std::IsSame<OutputType, fp4x2_e2m1_t>::value || Std::IsSame<OutputType, fp4x2_e1m2_t>::value) {
            // FP4 打包使用 ComputeFp4Data 的 B32 去交织顺序，分组最大值使用 B16 顺序；
            // 因此在原位写入 scale 前重新加载数据。
            __ubuf__ uint32_t* fp4Source =
                (__ubuf__ uint32_t*)(context.activation + row * context.activationRowStrideElements);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                (Reg::RegTensor<uint32_t>&)firstHalf, (Reg::RegTensor<uint32_t>&)secondHalf, fp4Source,
                fp4SourceWordsPerRow);
        }

        // StoreUnAlign 同时支持预取路径的紧凑布局和 UB 中按行留步长的布局。
        Reg::UnalignRegForStore scaleStoreState;
        Reg::StoreUnAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scaleOutput, scaleBytes, scaleStoreState,
                                                                       scaleBytesPerRow);
        Reg::StoreUnAlignPost<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scaleOutput, scaleStoreState, 0);

        Reg::Mul(firstHalf, firstHalf, (Reg::RegTensor<bfloat16_t>&)inverseScaleExpanded, all);
        Reg::Mul(secondHalf, secondHalf, (Reg::RegTensor<bfloat16_t>&)inverseScaleExpanded, all);
        if constexpr (Std::IsSame<OutputType, fp8_e4m3fn_t>::value || Std::IsSame<OutputType, fp8_e5m2_t>::value) {
            Reg::RegTensor<float> firstEven;
            Reg::RegTensor<float> firstOdd;
            Reg::RegTensor<float> secondEven;
            Reg::RegTensor<float> secondOdd;
            Reg::RegTensor<OutputType> fp8FirstEven;
            Reg::RegTensor<OutputType> fp8FirstOdd;
            Reg::RegTensor<OutputType> fp8SecondEven;
            Reg::RegTensor<OutputType> fp8SecondOdd;
            static constexpr Reg::CastTrait toFloatEven = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr Reg::CastTrait toFloatOdd = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr Reg::CastTrait toFp8Even0 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait toFp8Odd0 = {Reg::RegLayout::TWO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait toFp8Even1 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait toFp8Odd1 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::Cast<float, bfloat16_t, toFloatEven>(firstEven, firstHalf, all);
            Reg::Cast<float, bfloat16_t, toFloatOdd>(firstOdd, firstHalf, all);
            Reg::Cast<float, bfloat16_t, toFloatEven>(secondEven, secondHalf, all);
            Reg::Cast<float, bfloat16_t, toFloatOdd>(secondOdd, secondHalf, all);
            Reg::Cast<OutputType, float, toFp8Even0>(fp8FirstEven, firstEven, all);
            Reg::Cast<OutputType, float, toFp8Odd0>(fp8FirstOdd, firstOdd, all);
            Reg::Cast<OutputType, float, toFp8Even1>(fp8SecondEven, secondEven, all);
            Reg::Cast<OutputType, float, toFp8Odd1>(fp8SecondOdd, secondOdd, all);
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8FirstEven, (Reg::RegTensor<uint8_t>&)fp8FirstEven,
                     (Reg::RegTensor<uint8_t>&)fp8FirstOdd, all);
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8SecondEven, (Reg::RegTensor<uint8_t>&)fp8SecondEven,
                     (Reg::RegTensor<uint8_t>&)fp8SecondOdd, all);
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8FirstEven, (Reg::RegTensor<uint8_t>&)fp8FirstEven,
                     (Reg::RegTensor<uint8_t>&)fp8SecondEven, all);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                quantOutput, (Reg::RegTensor<int8_t>&)fp8FirstEven, rowElementCount, all);
        } else {
            Reg::RegTensor<OutputType> fp4First;
            Reg::RegTensor<OutputType> fp4Second;
            static constexpr Reg::CastTrait toFp4First = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait toFp4Second = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Reg::Cast<OutputType, bfloat16_t, toFp4First>(fp4First, firstHalf, all);
            Reg::Cast<OutputType, bfloat16_t, toFp4Second>(fp4Second, secondHalf, all);
            Reg::Add((Reg::RegTensor<uint8_t>&)fp4First, (Reg::RegTensor<uint8_t>&)fp4First,
                     (Reg::RegTensor<uint8_t>&)fp4Second, all);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B32>(
                quantOutput, (Reg::RegTensor<int8_t>&)fp4First, fp4PackedBytesPerRow, all);
        }
    }
}

} // namespace GeluQuant
} // namespace MegaMoeImpl

#endif // __DAV_C310__ 条件编译结束
#endif // MEGA_MOE_ARCH35_ROWWISE_MX_QUANT_H 头文件保护结束

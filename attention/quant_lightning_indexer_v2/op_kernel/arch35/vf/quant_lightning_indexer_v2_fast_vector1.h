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
 * \file quant_lightning_indexer_v2_fast_vector1.h
 * \brief 权重处理、加权归约、分数编码及scale有效性检查VF
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_VECTOR1_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_VECTOR1_H

#include "../../../../lightning_indexer_v2/op_kernel/arch35/common/vf/lightning_indexer_v2_vector1_base.h"

namespace QLIV2FastVector1 {
using namespace AscendC;
// BF16符号位为bit15；0x7fff为正数编码上界，0x7fc0为规范NaN，0xffff用于逐位翻转
constexpr uint16_t BF16_SIGN_MASK = 0x8000U;
constexpr float FLOAT32_MAX_VALUE = 3.402823466e38f;

// 共用四个分数片段的编码；标量比较替代AND取符号位，保留热路径原有的四条计算指令
__simd_callee__ inline void EncodeScoreKey(Reg::RegTensor<uint16_t>& out, Reg::RegTensor<bfloat16_t>& score,
                                           Reg::RegTensor<uint16_t>& allOne, Reg::RegTensor<uint16_t>& signMask,
                                           Reg::MaskReg& all)
{
    Reg::RegTensor<uint16_t> flip;
    Reg::MaskReg negative, valid;
    Reg::Compares<uint16_t, CMPMODE::GT>(negative, reinterpret_cast<Reg::RegTensor<uint16_t>&>(score),
                                         static_cast<uint16_t>(0x7fff), all);
    Reg::Select(flip, allOne, signMask, negative);
    Reg::Compares<uint16_t, CMPMODE::NE>(valid, reinterpret_cast<Reg::RegTensor<uint16_t>&>(score),
                                         static_cast<uint16_t>(0x7fc0), all);
    Reg::Xor(out, reinterpret_cast<Reg::RegTensor<uint16_t>&>(score), flip, valid);
}

// Fixpipe交接数据采用分存储体、分平面的布局，每个Vector核接收两行Query
// 同一Query行内相邻头相隔512字节，四个64列片段之间相隔localM * 512字节
// APPLY_KEY_SCALE仅控制Key scale；FP8的Query scale已在有效权重中合入，MX的scale由Cube处理
template <bool APPLY_KEY_SCALE>
__simd_vf__ inline void WeightedReduceVF(__ubuf__ uint16_t* out, __ubuf__ float* qk, __ubuf__ float* weight,
                                         __ubuf__ float* keyScale, uint32_t outStride, uint32_t alignedGSize,
                                         uint32_t gSize)
{
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();
    // 256列分为4个64元素FP32寄存器；输出按128元素BF16寄存器分为两段
    Reg::RegTensor<float> sum0[4];
    Reg::RegTensor<float> sum1[4];
    liV2Vector1::DuplicateZero(sum0, maskB32);
    liV2Vector1::DuplicateZero(sum1, maskB32);

    // 逐行处理Query并延后加载Key scale，缩短寄存器生命周期，同时保持原有头遍历和FMA计算顺序
    // UB步长由固定tile约束，先用64位计算，再按VF地址接口位宽保存
    const uint32_t rowStride = static_cast<uint32_t>(static_cast<int64_t>(alignedGSize) * 128);
    const uint32_t planeStride = static_cast<uint32_t>(static_cast<int64_t>(rowStride) * 2);
    // 每个g归约两行Query的四个片段，共8组ReLU/FMA；此处仅预载首个g的5组结果
    // row0的4组及row1First跨迭代保留，第二行其余3组在循环内按需加载
    {
        Reg::RegTensor<float> row0[4], row1First, w0, w1;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w0, weight);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w1, weight + alignedGSize);
        Reg::LoadAlign<float>(row0[0], qk + 0U * planeStride);
        Reg::Relu(row0[0], row0[0], maskB32);
        Reg::LoadAlign<float>(row0[1], qk + 1U * planeStride);
        Reg::Relu(row0[1], row0[1], maskB32);
        Reg::LoadAlign<float>(row0[2], qk + 2U * planeStride);
        Reg::Relu(row0[2], row0[2], maskB32);
        Reg::LoadAlign<float>(row0[3], qk + 3U * planeStride);
        Reg::Relu(row0[3], row0[3], maskB32);
        Reg::LoadAlign<float>(row1First, qk + rowStride);
        Reg::Relu(row1First, row1First, maskB32);
        // 当前g的8组FMA与下一g的5组预加载交错，保持累加顺序并缩短加载到使用的依赖链
        for (uint16_t gIdx = 0U; gIdx < gSize - 1U; ++gIdx) {
            Reg::RegTensor<float> tail[3], incoming;
            auto qkAddr = Reg::CreateAddrReg<float>(gIdx, 128);
            auto weightAddr = Reg::CreateAddrReg<float>(gIdx, 1);
            Reg::LoadAlign<float>(incoming, qk + rowStride + 1U * planeStride, qkAddr);
            Reg::Relu(tail[0], incoming, maskB32);
            Reg::MulAddDst(sum0[0], row0[0], w0, maskB32);
            Reg::LoadAlign<float>(incoming, qk + rowStride + 2U * planeStride, qkAddr);
            Reg::Relu(tail[1], incoming, maskB32);
            Reg::MulAddDst(sum0[1], row0[1], w0, maskB32);
            Reg::LoadAlign<float>(incoming, qk + rowStride + 3U * planeStride, qkAddr);
            Reg::Relu(tail[2], incoming, maskB32);
            Reg::MulAddDst(sum0[2], row0[2], w0, maskB32);
            Reg::LoadAlign<float>(incoming, qk + 128U, qkAddr);
            Reg::Relu(row0[0], incoming, maskB32);
            Reg::MulAddDst(sum0[3], row0[3], w0, maskB32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w0, weight + 1U, weightAddr);
            Reg::LoadAlign<float>(incoming, qk + 128U + 1U * planeStride, qkAddr);
            Reg::Relu(row0[1], incoming, maskB32);
            Reg::MulAddDst(sum1[0], row1First, w1, maskB32);
            Reg::LoadAlign<float>(incoming, qk + 128U + 2U * planeStride, qkAddr);
            Reg::Relu(row0[2], incoming, maskB32);
            Reg::MulAddDst(sum1[1], tail[0], w1, maskB32);
            Reg::LoadAlign<float>(incoming, qk + 128U + 3U * planeStride, qkAddr);
            Reg::Relu(row0[3], incoming, maskB32);
            Reg::MulAddDst(sum1[2], tail[1], w1, maskB32);
            Reg::LoadAlign<float>(incoming, qk + 128U + rowStride, qkAddr);
            Reg::Relu(row1First, incoming, maskB32);
            Reg::MulAddDst(sum1[3], tail[2], w1, maskB32);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w1, weight + alignedGSize + 1U, weightAddr);
        }
        // 排空最后一个g：不再预取下一g，避免越界；gSize=1时直接由此完成全部8组归约
        Reg::MulAddDst(sum0[0], row0[0], w0, maskB32);
        Reg::MulAddDst(sum0[1], row0[1], w0, maskB32);
        Reg::MulAddDst(sum0[2], row0[2], w0, maskB32);
        Reg::MulAddDst(sum0[3], row0[3], w0, maskB32);
        Reg::MulAddDst(sum1[0], row1First, w1, maskB32);
        Reg::RegTensor<float> last;
        Reg::LoadAlign<float>(last, qk + (gSize - 1U) * 128U + rowStride + 1U * planeStride);
        Reg::Relu(last, last, maskB32);
        Reg::MulAddDst(sum1[1], last, w1, maskB32);
        Reg::LoadAlign<float>(last, qk + (gSize - 1U) * 128U + rowStride + 2U * planeStride);
        Reg::Relu(last, last, maskB32);
        Reg::MulAddDst(sum1[2], last, w1, maskB32);
        Reg::LoadAlign<float>(last, qk + (gSize - 1U) * 128U + rowStride + 3U * planeStride);
        Reg::Relu(last, last, maskB32);
        Reg::MulAddDst(sum1[3], last, w1, maskB32);
    }

    Reg::RegTensor<float> scale[4];
    if constexpr (APPLY_KEY_SCALE) {
        Reg::LoadAlign<float>(scale[0], keyScale);
        Reg::LoadAlign<float>(scale[1], keyScale + 64U);
        Reg::LoadAlign<float>(scale[2], keyScale + 128U);
        Reg::LoadAlign<float>(scale[3], keyScale + 192U);
    }

    if constexpr (APPLY_KEY_SCALE) {
        Reg::Mul(sum0[0], sum0[0], scale[0], maskB32);
        Reg::Mul(sum0[1], sum0[1], scale[1], maskB32);
        Reg::Mul(sum0[2], sum0[2], scale[2], maskB32);
        Reg::Mul(sum0[3], sum0[3], scale[3], maskB32);
        Reg::Mul(sum1[0], sum1[0], scale[0], maskB32);
        Reg::Mul(sum1[1], sum1[1], scale[1], maskB32);
        Reg::Mul(sum1[2], sum1[2], scale[2], maskB32);
        Reg::Mul(sum1[3], sum1[3], scale[3], maskB32);
    }

    constexpr static Reg::CastTrait evenCast = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING,
                                                RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait oddCast = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                               RoundMode::CAST_ROUND};
    Reg::RegTensor<bfloat16_t> bf16Row0[2];
    Reg::RegTensor<bfloat16_t> bf16Row1[2];
    Reg::DeInterleave(sum0[0], sum0[1], sum0[0], sum0[1]);
    Reg::DeInterleave(sum0[2], sum0[3], sum0[2], sum0[3]);
    Reg::DeInterleave(sum1[0], sum1[1], sum1[0], sum1[1]);
    Reg::DeInterleave(sum1[2], sum1[3], sum1[2], sum1[3]);
    Reg::Cast<bfloat16_t, float, oddCast>(bf16Row0[0], sum0[1], maskB32);
    Reg::Cast<bfloat16_t, float, evenCast>(bf16Row0[0], sum0[0], maskB32);
    Reg::Cast<bfloat16_t, float, oddCast>(bf16Row0[1], sum0[3], maskB32);
    Reg::Cast<bfloat16_t, float, evenCast>(bf16Row0[1], sum0[2], maskB32);
    Reg::Cast<bfloat16_t, float, oddCast>(bf16Row1[0], sum1[1], maskB32);
    Reg::Cast<bfloat16_t, float, evenCast>(bf16Row1[0], sum1[0], maskB32);
    Reg::Cast<bfloat16_t, float, oddCast>(bf16Row1[1], sum1[3], maskB32);
    Reg::Cast<bfloat16_t, float, evenCast>(bf16Row1[1], sum1[2], maskB32);

    // 标准NaN映射为0；有符号有限值及Inf采用相同的可排序键转换，通过标量比较减少一次额外的按位与操作
    Reg::RegTensor<uint16_t> key0[2], key1[2], allOne, signMask;
    Reg::Duplicate(allOne, static_cast<uint16_t>(0xffff));
    Reg::Duplicate(signMask, BF16_SIGN_MASK);
    EncodeScoreKey(key0[0], bf16Row0[0], allOne, signMask, maskB16);
    EncodeScoreKey(key0[1], bf16Row0[1], allOne, signMask, maskB16);
    EncodeScoreKey(key1[0], bf16Row1[0], allOne, signMask, maskB16);
    EncodeScoreKey(key1[1], bf16Row1[1], allOne, signMask, maskB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(out, key0[0], maskB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(out + 128U, key0[1], maskB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(out + outStride, key1[0], maskB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(out + outStride + 128U, key1[1], maskB16);
}

// Query scale已融合到有效权重；alignedGSize为补齐后的物理g步长，gSize为实际归约的g数
template <bool APPLY_KEY_SCALE>
__aicore__ inline void WeightedReduce(const LocalTensor<uint16_t>& out, const LocalTensor<float>& qk,
                                      const LocalTensor<float>& weight, const LocalTensor<float>& keyScale,
                                      uint32_t outStride, uint32_t alignedGSize, uint32_t gSize)
{
    auto* outPtr = reinterpret_cast<__ubuf__ uint16_t*>(out.GetPhyAddr());
    auto* qkPtr = reinterpret_cast<__ubuf__ float*>(qk.GetPhyAddr());
    auto* weightPtr = reinterpret_cast<__ubuf__ float*>(weight.GetPhyAddr());
    __ubuf__ float* scalePtr = nullptr;
    if constexpr (APPLY_KEY_SCALE) {
        scalePtr = reinterpret_cast<__ubuf__ float*>(keyScale.GetPhyAddr());
    }
    WeightedReduceVF<APPLY_KEY_SCALE>(outPtr, qkPtr, weightPtr, scalePtr, outStride, alignedGSize, gSize);
}

__aicore__ inline void WeightedReduceFp8(const LocalTensor<uint16_t>& out, const LocalTensor<float>& qk,
                                         const LocalTensor<float>& weight, const LocalTensor<float>& keyScale,
                                         uint32_t outStride, uint32_t alignedGSize, uint32_t gSize)
{
    WeightedReduce<true>(out, qk, weight, keyScale, outStride, alignedGSize, gSize);
}

__aicore__ inline void WeightedReduceMx(const LocalTensor<uint16_t>& out, const LocalTensor<float>& qk,
                                        const LocalTensor<float>& weight, uint32_t outStride, uint32_t alignedGSize,
                                        uint32_t gSize)
{
    // MXFP8反量化已由Cube的Mmad完成，此处不使用scale参数
    WeightedReduce<false>(out, qk, weight, weight, outStride, alignedGSize, gSize);
}

} // namespace QLIV2FastVector1

namespace QLIV2FastVector1 {
template <typename T>
__simd_vf__ inline void FillAlignedRangeVF(__ubuf__ T* dst, T value, uint32_t begin, uint32_t end)
{
    // 单个向量寄存器256字节；2字节元素使用16位位置索引，其余使用32位
    constexpr uint32_t lanes = QLIV2Fast::VECTOR_REGISTER_BYTES / sizeof(T);
    using PositionT = std::conditional_t<sizeof(T) == 2U, int16_t, int32_t>;
    Reg::MaskReg all = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::RegTensor<T> fill;
    Reg::RegTensor<PositionT> positions;
    Reg::Duplicate(fill, value, all);
    for (uint16_t block = begin / lanes; block < (end + lanes - 1U) / lanes; ++block) {
        Reg::MaskReg afterBegin;
        Reg::MaskReg beforeEnd;
        Reg::MaskReg padding;
        Reg::Arange(positions, static_cast<PositionT>(block * lanes));
        Reg::Compares<PositionT, CMPMODE::GE>(afterBegin, positions, static_cast<PositionT>(begin), all);
        Reg::Compares<PositionT, CMPMODE::LT>(beforeEnd, positions, static_cast<PositionT>(end), all);
        Reg::And(padding, afterBegin, beforeEnd, all);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(dst + block * lanes, fill, padding);
    }
}

__simd_vf__ inline void PrepareWeightsVF(__ubuf__ float* effective, __ubuf__ float* scratch, uint32_t paddedHeadNum)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    // 每个AIV共8行权重，单次寄存器处理64个float（256字节）
    for (uint16_t i = 0; i < 8 * paddedHeadNum; i += 64) {
        Reg::RegTensor<float> w, scale, result;
        Reg::LoadAlign<typename std::remove_reference_t<decltype(w)>::ActualT, Reg::LoadDist::DIST_NORM>(
            w, effective + (static_cast<int64_t>(i)));
        Reg::LoadAlign<typename std::remove_reference_t<decltype(scale)>::ActualT, Reg::LoadDist::DIST_NORM>(
            scale, scratch + (static_cast<int64_t>(i)));
        Reg::Mul(result, w, scale, all);
        Reg::StoreAlign<typename std::remove_reference_t<decltype(result)>::ActualT, Reg::StoreDist::DIST_NORM>(
            effective + (static_cast<int64_t>(i)), result, all);
    }
}

__simd_vf__ void InvalidScaleVF(__ubuf__ uint32_t* out, __ubuf__ float* in, uint32_t count)
{
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg nonnegative, finite;
    Reg::RegTensor<float> value;
    Reg::RegTensor<uint32_t> invalid, current, zero, one;
    Reg::Duplicate(invalid, static_cast<uint32_t>(0));
    Reg::Duplicate(zero, static_cast<uint32_t>(0));
    Reg::Duplicate(one, static_cast<uint32_t>(1));
    // 每次检查64个float，63为向上取整偏置；末轮用有效元素掩码排除补齐区
    for (uint16_t i = 0; i < (count + 63U) / 64U; ++i) {
        Reg::LoadAlign(value, in + i * 64U);
        Reg::Compares<float, CMPMODE::GE>(nonnegative, value, 0.0f, all);
        Reg::Compares<float, CMPMODE::LE>(finite, value, FLOAT32_MAX_VALUE, all);
        Reg::And(nonnegative, nonnegative, finite, all);
        Reg::Select(current, zero, one, nonnegative);
        uint32_t remaining = count - static_cast<uint32_t>(i) * 64U;
        auto active = Reg::UpdateMask<float>(remaining);
        Reg::Select(current, current, zero, active);
        Reg::Or(invalid, invalid, current, all);
    }
    Reg::Reduce<Reg::ReduceType::MAX>(invalid, invalid, all);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(out, invalid, all);
}
} // namespace QLIV2FastVector1

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_VECTOR1_H

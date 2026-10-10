/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VF_COMMON_DEF_MXFP8_H_
#define VF_COMMON_DEF_MXFP8_H_
#include "kernel_tensor.h"

namespace NpuArch::Epilogue::Block::Mxfp8VF {
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

constexpr static AscendC::MicroAPI::CastTrait h2iZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait h2iOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr half NUM_127 = static_cast<half>(127.0f);
constexpr half NUM_NEG_127 = static_cast<half>(-127.0f);
// OCP P-scale e8m0: shared_exp + 127 = K_block - K_global - emax + 127
// e4m3 emax=8 → 127-8=119。不可照搬 mxfp4 e2m1 emax=2 的 NUM_NEG_125。
constexpr half NUM_NEG_119 = static_cast<half>(-119.0f);
constexpr half ZERO_VALUE = static_cast<half>(0.0f);
constexpr int16_t SHIFT_VALUE = 23;

constexpr uint8_t NUM_128 = static_cast<uint8_t>(128);
constexpr int16_t NUM_1 = static_cast<int16_t>(1);
constexpr int16_t NUM_2 = static_cast<int16_t>(2);
constexpr int8_t indexSubLength = static_cast<int8_t>(32);

constexpr half LN2 = static_cast<half>(0.6931471806f);
constexpr half INV_LN2 = static_cast<half>(1.4426950409f);
// OCP P snap: m_snap = (K + NEG_EIGHT_VALE)*ln2。e8m0 仍按未偏置 K 编（Adds 在 Store 之后）。
// 目标 P∈[256,512)×(448/512)=[224,448)。fp16 在 4–8 的 ulp=1/256，
// -log2(224)≈-7.807355 四舍五入会落到 -7.80859375 → 2^(emax+1)≈448.38>448，仍会 NaN。
// 取更保守的精确 fp16：-7.8046875 → P∈[2^7.8046875, 2^8.8046875)≈[223.59,447.17)。
constexpr half NEG_EIGHT_VALE = static_cast<half>(-7.8046875f);
constexpr half TWO_VALE = static_cast<half>(2.0f);
constexpr half MIN_VALUE = static_cast<half>(-65504.0f);

// P 量化：fp16 Exp 后 Cast 到 float，再以 CAST_RINT 量化到 fp8_e4m3。
// Ascend 950 没有 bfloat16 到 fp8_e4m3 的单条 Cast。

// LoadAlign(+64)：128 half 的 even/odd 再 Interleave 成连续 qs[0:64) / qs[64:128)
#define BSA_MXFP8_FP16_SPLIT_CONSEC64(fLo, fHi, srcHalf, preg16) \
    do { \
        Cast<float, T, castTraitZero>((fLo), (srcHalf), (preg16)); \
        Cast<float, T, castTraitOne>((fHi), (srcHalf), (preg16)); \
        Interleave((fLo), (fHi), (fLo), (fHi)); \
    } while (0)

// FusedExpSub 的目的寄存器只能是 float。源为 half 时，RegLayout::ZERO / ONE 分别取偶数、奇数元素。
// Interleave 之后得到连续的 qs[0:64) 与 qs[64:128)。
#define BSA_MXFP8_FUSED_EXPSUB_SPLIT_CONSEC64(fLo, fHi, srcHalf, maxHalf, preg16) \
    do { \
        FusedExpSub<float, half, MicroAPI::RegLayout::ZERO, MicroAPI::MaskMergeMode::ZEROING>((fLo), (srcHalf), \
                                                                                              (maxHalf), (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>((fHi), (srcHalf), \
                                                                                             (maxHalf), (preg16)); \
        Interleave((fLo), (fHi), (fLo), (fHi)); \
    } while (0)

// qs64 只取低 64 个元素，用 FusedExpSub 的 RegLayout::ZERO。
#define BSA_MXFP8_FUSED_EXPSUB_EVEN64(fLo, srcHalf, maxHalf, preg16) \
    do { \
        FusedExpSub<float, half, MicroAPI::RegLayout::ZERO, MicroAPI::MaskMergeMode::ZEROING>((fLo), (srcHalf), \
                                                                                              (maxHalf), (preg16)); \
    } while (0)

#define BSA_MXFP8_EXPSUB_EVEN64_4WAY_MINS(f0, f1, f2, f3, c0, c1, c2, c3, maxHalf, preg16, pregFp32) \
    do { \
        BSA_MXFP8_FUSED_EXPSUB_EVEN64((f0), (c0), (maxHalf), (preg16)); \
        BSA_MXFP8_FUSED_EXPSUB_EVEN64((f1), (c1), (maxHalf), (preg16)); \
        BSA_MXFP8_FUSED_EXPSUB_EVEN64((f2), (c2), (maxHalf), (preg16)); \
        BSA_MXFP8_FUSED_EXPSUB_EVEN64((f3), (c3), (maxHalf), (preg16)); \
    } while (0)

#define BSA_MXFP8_EXPSUB_SPLIT_4WAY_MINS(f0, f1, f2, f3, f4, f5, f6, f7, c0, c1, c2, c3, maxHalf, preg16, pregFp32) \
    do { \
        FusedExpSub<float, half, MicroAPI::RegLayout::ZERO, MicroAPI::MaskMergeMode::ZEROING>((f0), (c0), (maxHalf), \
                                                                                              (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>((f4), (c0), (maxHalf), \
                                                                                             (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ZERO, MicroAPI::MaskMergeMode::ZEROING>((f1), (c1), (maxHalf), \
                                                                                              (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>((f5), (c1), (maxHalf), \
                                                                                             (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ZERO, MicroAPI::MaskMergeMode::ZEROING>((f2), (c2), (maxHalf), \
                                                                                              (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>((f6), (c2), (maxHalf), \
                                                                                             (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ZERO, MicroAPI::MaskMergeMode::ZEROING>((f3), (c3), (maxHalf), \
                                                                                              (preg16)); \
        FusedExpSub<float, half, MicroAPI::RegLayout::ONE, MicroAPI::MaskMergeMode::ZEROING>((f7), (c3), (maxHalf), \
                                                                                             (preg16)); \
        Interleave((f0), (f4), (f0), (f4)); \
        Interleave((f1), (f5), (f1), (f5)); \
        Interleave((f2), (f6), (f2), (f6)); \
        Interleave((f3), (f7), (f3), (f7)); \
    } while (0)

// ProcessVec1DnMxfp8：4 路 float→e4m3 RINT + Gather + DATA_BLOCK_COPY（指针 POST_MODE_UPDATE）
#define BSA_MXFP8_PACK_STORE_E4M3_DBC(pPtr, x0, x1, x2, x3, idx, pregFp32, pregAll8, blkStride, repStride) \
    do { \
        RegTensor<fp8_e4m3fn_t> _q0; \
        Cast<fp8_e4m3fn_t, float, castTraitRintZero>(_q0, (x0), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintOne>((RegTensor<fp8_e4m3fn_t>&)(x0), (x1), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintTwo>((RegTensor<fp8_e4m3fn_t>&)(x1), (x2), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintThree>((RegTensor<fp8_e4m3fn_t>&)(x2), (x3), (pregFp32)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x0), (pregAll8)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x1), (pregAll8)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x2), (pregAll8)); \
        Gather((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (idx)); \
        StoreAlign<fp8_e4m3fn_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>( \
            ((__ubuf__ fp8_e4m3fn_t*&)(pPtr)), (RegTensor<fp8_e4m3fn_t>&)_q0, (blkStride), (repStride), (pregAll8)); \
    } while (0)

// 连续 256B store（qs64 / 旧路径）
#define BSA_MXFP8_PACK_STORE_E4M3(pDest, offset, x0, x1, x2, x3, idx, pregFp32, pregAll8) \
    do { \
        RegTensor<fp8_e4m3fn_t> _q0; \
        Cast<fp8_e4m3fn_t, float, castTraitRintZero>(_q0, (x0), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintOne>((RegTensor<fp8_e4m3fn_t>&)(x0), (x1), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintTwo>((RegTensor<fp8_e4m3fn_t>&)(x1), (x2), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintThree>((RegTensor<fp8_e4m3fn_t>&)(x2), (x3), (pregFp32)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x0), (pregAll8)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x1), (pregAll8)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x2), (pregAll8)); \
        Gather((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (idx)); \
        StoreAlign(((__ubuf__ uint8_t*&)(pDest)) + (offset), (RegTensor<uint8_t>&)_q0, (pregAll8)); \
    } while (0)

// 4 行×64 e4m3 打成 256B 后按 VL128 拆 M（2 行×64 = 128B）
#define BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, offLo, offHi, x0, x1, x2, x3, idx, pregFp32, pregAll8, pregLo, pregHi) \
    do { \
        RegTensor<fp8_e4m3fn_t> _q0; \
        Cast<fp8_e4m3fn_t, float, castTraitRintZero>(_q0, (x0), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintOne>((RegTensor<fp8_e4m3fn_t>&)(x0), (x1), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintTwo>((RegTensor<fp8_e4m3fn_t>&)(x1), (x2), (pregFp32)); \
        Cast<fp8_e4m3fn_t, float, castTraitRintThree>((RegTensor<fp8_e4m3fn_t>&)(x2), (x3), (pregFp32)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x0), (pregAll8)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x1), (pregAll8)); \
        Or((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)(x2), (pregAll8)); \
        Gather((RegTensor<uint8_t>&)_q0, (RegTensor<uint8_t>&)_q0, (idx)); \
        StoreAlign(((__ubuf__ uint8_t*&)(pDest)) + (offLo), (RegTensor<uint8_t>&)_q0, (pregLo)); \
        StoreAlign(((__ubuf__ uint8_t*&)(pDest)) + (offHi), (RegTensor<uint8_t>&)_q0, (pregHi)); \
    } while (0)

} // namespace NpuArch::Epilogue::Block::Mxfp8VF
#endif // VF_COMMON_DEF_MXFP8_H_

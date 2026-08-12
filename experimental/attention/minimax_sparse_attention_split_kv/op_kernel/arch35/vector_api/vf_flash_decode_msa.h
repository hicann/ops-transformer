/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Local copy of the FaVectorApi ReduceFinalRes_VF family for
 * minimax_sparse_attention_split_kv Phase2 combine (FP32 reduce).
 *
 * Rationale: this operator's Phase1 does not divide accumOut by the softmax
 * sum, so Phase2 must perform an extra per-row normalization by the original
 * Phase1 rowSum before the cross-split scale-and-reduce. Instead of a separate
 * UB-level AscendC::Divs pre-pass, the per-row normalize is a Mul here: CopyLseIn
 * pre-inverts the compact per-row rowSum in place (AscendC::Reciprocal on
 * lseSumTmpBuf), so this kernel Mul-multiplies vregAccumOut by the broadcast
 * 1/rowSum (equivalent to the old Div, cheaper) before the existing
 * Mul(..., vregLse, ...) / Add.
 *
 * FP32-only path (accumOut is float). The bf16 O_partial path (innerPrecise==1)
 * does NOT add a bf16 branch here — instead the epilogue does a vector
 * AscendC::Cast (bf16 accumOut -> fp32 staging UB) BEFORE calling this reduce,
 * so the reduce always sees fp32 accumOut. Keeps the verified fp32 binary
 * byte-identical (the bf16 path's only bf16-specific code is that one vector
 * Cast in the epilogue's per-split loop).
 *
 * This header is a near-verbatim copy of the ReduceFinalRes_* family in
 * attention/common/op_kernel/arch35/vf/vf_flash_decode.h, with:
 *   - an extra rowSumUb / rowSumLocal parameter (per-row original rowSum,
 *     pre-inverted to 1/rowSum by CopyLseIn);
 *   - one MicroAPI::Mul (was Div) per z-block, multiplying vregAccumOut by the
 *     broadcast 1/rowSum before the existing Mul(..., vregLse, ...) / Add.
 *
 * Include ordering: must be included AFTER
 *   ../../../../../common/op_kernel/arch35/vf/vf_flash_decode.h
 * which provides FLT_ZERO / FLT_MAX_NEW and the MicroAPI types.
 */
#ifndef VF_FLASH_DECODE_MSA_H
#define VF_FLASH_DECODE_MSA_H

namespace FaVectorApiSplitKv {

// 处理循环splitKVIndex=0的场景，vregDst需要置0
template <typename T>
__simd_vf__ void ReduceFinalRes_0_VF(__ubuf__ T * dstUb, __ubuf__ T * lseUb, __ubuf__ T * accumOutUb,
                                      __ubuf__ T * rowSumUb, uint16_t k, uint16_t z,
                                      uint32_t dealNum1Reg, uint32_t repStride, const uint16_t floatRepSize,
                                      const uint16_t dLoops, uint32_t dealRowCount, uint32_t splitKVIndex)
{
    MicroAPI::RegTensor<T> vregDst;
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::RegTensor<T> vregAccumOut;
    MicroAPI::RegTensor<T> vregRowSum;
    uint32_t n = dealNum1Reg;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {  // repeat g

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BLK>(
            vregLse,
            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        // rowSum is per-row (constant across the D-dim z-blocks): broadcast-load once per row.
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregRowSum, (__ubuf__ float*&)rowSumUb + k);
        for (z = 0; z < dLoops; z++) {
            // splitKVIndex=0的场景，vregDst不需要load，直接置0
            MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregDst, FLT_ZERO, pregTailN);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregAccumOut, (__ubuf__ float*&)accumOutUb + k * repStride * 8 + z * floatRepSize);
            // Embedded regbase Mul: normalize Phase1 accumOut by its original per-row rowSum.
            // rowSum was pre-inverted to 1/rowSum by CopyLseIn's Reciprocal on the compact
            // lseSumTmpBuf, so this is a Mul (cheaper than Div). Broadcast-loaded via
            // DIST_BRC_B32 (one reciprocal per row).
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregAccumOut, vregRowSum, pregTailN);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_0(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
                                           LocalTensor<T>& accumOutLocal, LocalTensor<T>& rowSumLocal,
                                           uint32_t dealRowCount,
                                           uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ T * dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T * lseUb = (__ubuf__ T *)lseLocal.GetPhyAddr();
    __ubuf__ T * accumOutUb = (__ubuf__ T *)accumOutLocal.GetPhyAddr();
    __ubuf__ T * rowSumUb = (__ubuf__ T *)rowSumLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    ReduceFinalRes_0_VF<T>(dstUb, lseUb, accumOutUb, rowSumUb, k, z, dealNum1Reg, repStride, floatRepSize,
                        dLoops, dealRowCount, splitKVIndex);
}

// 处理循环splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
template <typename T>
__simd_vf__ void ReduceFinalRes_Rest_VF(
    __ubuf__ T * dstUb, __ubuf__ T * lseUb, __ubuf__ T * accumOutUb,
    __ubuf__ T * rowSumUb, uint16_t k, uint16_t z,
    uint32_t dealNum1Reg, uint32_t repStride,
    const uint16_t floatRepSize, const uint16_t dLoops,
    uint32_t dealRowCount, uint32_t splitKVIndex)
{
    MicroAPI::RegTensor<T> vregDst;
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::RegTensor<T> vregAccumOut;
    MicroAPI::RegTensor<T> vregRowSum;
    uint32_t n = dealNum1Reg;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);
    uint32_t stride = (0x1 << 16) | 0x8;

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {  // repeat g
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BLK>(
            vregLse,
            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        // rowSum is per-row (constant across the D-dim z-blocks): broadcast-load once per row.
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregRowSum, (__ubuf__ float*&)rowSumUb + k);
        for (z = 0; z < dLoops; z++) {
            // splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregDst, (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregAccumOut, (__ubuf__ float*&)accumOutUb + k * repStride * 8 + z * floatRepSize);
            // Embedded regbase Mul: normalize Phase1 accumOut by its original per-row rowSum.
            // rowSum was pre-inverted to 1/rowSum by CopyLseIn's Reciprocal on the compact
            // lseSumTmpBuf, so this is a Mul (cheaper than Div). Broadcast-loaded via
            // DIST_BRC_B32 (one reciprocal per row).
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregAccumOut, vregRowSum, pregTailN);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_Rest(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
                                                 LocalTensor<T>& accumOutLocal, LocalTensor<T>& rowSumLocal,
                                                 uint32_t dealRowCount,
                                                 uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ T * dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T * lseUb = (__ubuf__ T *)lseLocal.GetPhyAddr();
    __ubuf__ T * accumOutUb = (__ubuf__ T *)accumOutLocal.GetPhyAddr();
    __ubuf__ T * rowSumUb = (__ubuf__ T *)rowSumLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    ReduceFinalRes_Rest_VF<T>(
        dstUb, lseUb, accumOutUb, rowSumUb, k, z, dealNum1Reg, repStride,
        floatRepSize, dLoops, dealRowCount, splitKVIndex);
}

template <typename T>
__aicore__ inline void ReduceFinalRes_VF(
    LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
    LocalTensor<T>& accumOutLocal, LocalTensor<T>& rowSumLocal,
    uint32_t dealRowCount, uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_0(
            dstLocal, lseLocal, accumOutLocal, rowSumLocal,
            dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_Rest(
            dstLocal, lseLocal, accumOutLocal, rowSumLocal,
            dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}

// ============================================================================
// bf16 O_partial path (innerPrecise==1): regbase-fused bf16->fp32 cast.
//
// The fp32 path above keeps accumOut fp32 and reduces in fp32. The bf16 path
// here reads bf16 accumOut straight into a register and fuses the bf16->fp32
// Cast into the reduce (no separate vector Cast staging pass / staging UB):
//   LoadAlign<bf16>(128) -> Cast<float,bf16,ZERO/ONE>  (2x64 fp32, even/odd)
//   fp32 Mul(rowSum) Mul(lse) Add(dst)  on both halves
//   Cast<bf16,float,ZERO/ONE>{SAT,CAST_ROUND}  (2x64 bf16 in even/odd lanes)
//   Or  (bitwise merge of even-filled | odd-filled -> 128 sequential bf16)
//   StoreAlign<bf16, DATA_BLOCK_COPY>  (vsstb; dataBlockStride=1 => contiguous)
// This mirrors the shipped built-in
// block_epilogue_online_softmax_arch35_reg_low_prec_bf16.hpp::ComputeExpSubSum16
// (Or+vsstb is the proven store for an Or-merged bf16 reg; plain vst cannot
// split it, which was the scheme-B 471 "Do not know how to split" failure).
//
// dst (O) is bf16 here, so the cross-split accumulation is bf16-rounded per
// split (innerPrecise==1 is the INNER_LOW path -> accepted). lse/rowSum/dst
// scale compute stay fp32. Only D=128 (headDimAlignFp32==128, no tail) is
// handled; D!=128 needs tail-mask care (production D=128).
// ============================================================================

template <typename Tdst>  // Tdst = bfloat16_t (accumOut + dst dtype)
__simd_vf__ void ReduceFinalRes_0_VF_BF16(__ubuf__ Tdst * dstUb, __ubuf__ float * lseUb,
                                          __ubuf__ Tdst * accumOutUb, __ubuf__ float * rowSumUb,
                                          uint16_t k, uint32_t dealRowCount,
                                          uint32_t splitKVIndex, uint32_t repStride)
{
    using namespace AscendC::MicroAPI;
    constexpr static CastTrait castUpZero = {
        RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static CastTrait castUpOne = {
        RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static CastTrait castDownZero = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static CastTrait castDownOne = {
        RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};

    RegTensor<Tdst> vregAccumRaw;
    RegTensor<Tdst> vregDstRaw;
    RegTensor<float> vregAccumEven, vregAccumOdd;
    RegTensor<float> vregDstEven, vregDstOdd;
    RegTensor<float> vregLse, vregRowSum;
    RegTensor<Tdst> vregDstBf16Even, vregDstBf16Odd, vregDstBf16;
    MaskReg pregFull = CreateMask<Tdst, MaskPattern::ALL>();
    MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {
        LoadAlign<float, LoadDist::DIST_BLK>(vregLse,
            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vregRowSum, (__ubuf__ float*&)rowSumUb + k);
        LoadAlign<Tdst, LoadDist::DIST_NORM>(vregAccumRaw, accumOutUb + k * repStride * 8);
        Cast<float, Tdst, castUpZero>(vregAccumEven, vregAccumRaw, pregFull);
        Cast<float, Tdst, castUpOne >(vregAccumOdd,  vregAccumRaw, pregFull);
        // splitKVIndex==0: dst starts at zero
        Duplicate<float, MaskMergeMode::ZEROING, float>(vregDstEven, FLT_ZERO, pregFloatFull);
        Duplicate<float, MaskMergeMode::ZEROING, float>(vregDstOdd,  FLT_ZERO, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumEven, vregAccumEven, vregRowSum, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumEven, vregLse, vregAccumEven, pregFloatFull);
        Add<float, MaskMergeMode::ZEROING>(vregDstEven, vregDstEven, vregAccumEven, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumOdd, vregAccumOdd, vregRowSum, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumOdd, vregLse, vregAccumOdd, pregFloatFull);
        Add<float, MaskMergeMode::ZEROING>(vregDstOdd, vregDstOdd, vregAccumOdd, pregFloatFull);
        Cast<Tdst, float, castDownZero>(vregDstBf16Even, vregDstEven, pregFloatFull);
        Cast<Tdst, float, castDownOne >(vregDstBf16Odd,  vregDstOdd,  pregFloatFull);
        Or((RegTensor<uint16_t>&)vregDstBf16,
           (RegTensor<uint16_t>&)vregDstBf16Even, (RegTensor<uint16_t>&)vregDstBf16Odd, pregFull);
        // dataBlockStride=1 => 8x32B datablocks laid out contiguously (128 bf16).
        StoreAlign<Tdst, DataCopyMode::DATA_BLOCK_COPY>(
            dstUb + k * repStride * 8, vregDstBf16, 1U, pregFull);
    }
}

template <typename Tdst>
__simd_vf__ void ReduceFinalRes_Rest_VF_BF16(__ubuf__ Tdst * dstUb, __ubuf__ float * lseUb,
                                             __ubuf__ Tdst * accumOutUb, __ubuf__ float * rowSumUb,
                                             uint16_t k, uint32_t dealRowCount,
                                             uint32_t splitKVIndex, uint32_t repStride)
{
    using namespace AscendC::MicroAPI;
    constexpr static CastTrait castUpZero = {
        RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static CastTrait castUpOne = {
        RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static CastTrait castDownZero = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static CastTrait castDownOne = {
        RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};

    RegTensor<Tdst> vregAccumRaw;
    RegTensor<Tdst> vregDstRaw;
    RegTensor<float> vregAccumEven, vregAccumOdd;
    RegTensor<float> vregDstEven, vregDstOdd;
    RegTensor<float> vregLse, vregRowSum;
    RegTensor<Tdst> vregDstBf16Even, vregDstBf16Odd, vregDstBf16;
    MaskReg pregFull = CreateMask<Tdst, MaskPattern::ALL>();
    MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {
        LoadAlign<float, LoadDist::DIST_BLK>(vregLse,
            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(vregRowSum, (__ubuf__ float*&)rowSumUb + k);
        // splitKVIndex>0: dst = load prior bf16 result, cast to fp32 even/odd
        LoadAlign<Tdst, LoadDist::DIST_NORM>(vregDstRaw, dstUb + k * repStride * 8);
        Cast<float, Tdst, castUpZero>(vregDstEven, vregDstRaw, pregFull);
        Cast<float, Tdst, castUpOne >(vregDstOdd,  vregDstRaw, pregFull);
        LoadAlign<Tdst, LoadDist::DIST_NORM>(vregAccumRaw, accumOutUb + k * repStride * 8);
        Cast<float, Tdst, castUpZero>(vregAccumEven, vregAccumRaw, pregFull);
        Cast<float, Tdst, castUpOne >(vregAccumOdd,  vregAccumRaw, pregFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumEven, vregAccumEven, vregRowSum, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumEven, vregLse, vregAccumEven, pregFloatFull);
        Add<float, MaskMergeMode::ZEROING>(vregDstEven, vregDstEven, vregAccumEven, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumOdd, vregAccumOdd, vregRowSum, pregFloatFull);
        Mul<float, MaskMergeMode::ZEROING>(vregAccumOdd, vregLse, vregAccumOdd, pregFloatFull);
        Add<float, MaskMergeMode::ZEROING>(vregDstOdd, vregDstOdd, vregAccumOdd, pregFloatFull);
        Cast<Tdst, float, castDownZero>(vregDstBf16Even, vregDstEven, pregFloatFull);
        Cast<Tdst, float, castDownOne >(vregDstBf16Odd,  vregDstOdd,  pregFloatFull);
        Or((RegTensor<uint16_t>&)vregDstBf16,
           (RegTensor<uint16_t>&)vregDstBf16Even, (RegTensor<uint16_t>&)vregDstBf16Odd, pregFull);
        StoreAlign<Tdst, DataCopyMode::DATA_BLOCK_COPY>(
            dstUb + k * repStride * 8, vregDstBf16, 1U, pregFull);
    }
}

template <typename Tdst>
__aicore__ inline void ReduceFinalRes_0_BF16(LocalTensor<Tdst>& dstLocal, LocalTensor<float>& lseLocal,
                                             LocalTensor<Tdst>& accumOutLocal, LocalTensor<float>& rowSumLocal,
                                             uint32_t dealRowCount,
                                             uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ Tdst * dstUb = (__ubuf__ Tdst *)dstLocal.GetPhyAddr();
    __ubuf__ float * lseUb = (__ubuf__ float *)lseLocal.GetPhyAddr();
    __ubuf__ Tdst * accumOutUb = (__ubuf__ Tdst *)accumOutLocal.GetPhyAddr();
    __ubuf__ float * rowSumUb = (__ubuf__ float *)rowSumLocal.GetPhyAddr();
    uint16_t k = 0;
    uint32_t repStride = headDimAlignFp32 / 8;
    ReduceFinalRes_0_VF_BF16<Tdst>(dstUb, lseUb, accumOutUb, rowSumUb, k, dealRowCount, splitKVIndex, repStride);
}

template <typename Tdst>
__aicore__ inline void ReduceFinalRes_Rest_BF16(LocalTensor<Tdst>& dstLocal, LocalTensor<float>& lseLocal,
                                                LocalTensor<Tdst>& accumOutLocal, LocalTensor<float>& rowSumLocal,
                                                uint32_t dealRowCount,
                                                uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ Tdst * dstUb = (__ubuf__ Tdst *)dstLocal.GetPhyAddr();
    __ubuf__ float * lseUb = (__ubuf__ float *)lseLocal.GetPhyAddr();
    __ubuf__ Tdst * accumOutUb = (__ubuf__ Tdst *)accumOutLocal.GetPhyAddr();
    __ubuf__ float * rowSumUb = (__ubuf__ float *)rowSumLocal.GetPhyAddr();
    uint16_t k = 0;
    uint32_t repStride = headDimAlignFp32 / 8;
    ReduceFinalRes_Rest_VF_BF16<Tdst>(
        dstUb, lseUb, accumOutUb, rowSumUb, k, dealRowCount, splitKVIndex, repStride);
}

template <typename Tdst>
__aicore__ inline void ReduceFinalRes_VF_BF16(
    LocalTensor<Tdst>& dstLocal, LocalTensor<float>& lseLocal,
    LocalTensor<Tdst>& accumOutLocal, LocalTensor<float>& rowSumLocal,
    uint32_t dealRowCount, uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_0_BF16<Tdst>(
            dstLocal, lseLocal, accumOutLocal, rowSumLocal,
            dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_Rest_BF16<Tdst>(
            dstLocal, lseLocal, accumOutLocal, rowSumLocal,
            dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}

} // namespace FaVectorApiSplitKv

#endif // VF_FLASH_DECODE_MSA_H

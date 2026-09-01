/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VF_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VF_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_vf.h"

namespace QkvRmsNormRopeCacheWithKScale {

/*
 * 本文件把一个 D128 head 表示为两个 FP32 D64 寄存器：low=[0, 64)，high=[64, 128)。
 * Dynamic MX 量化再把每行切成四个连续 D32 block；每个 block 生成一个 E8M0 scale，
 * 因而一行的 scale 物理布局始终是 [D0..31, D32..63, D64..95, D96..127]。
 *
 * 主数据流有两种：
 *
 *   逐行通路（小尾块和通用回退）
 *     BF16 x -> RMSNorm -> M-RoPE y -> D32 amax/scale -> E4M3
 *
 *   RowBatch16 通路（Q/K 主路径）
 *     Stage  : 16 rows x -> rms^2，并生成未除 rms 的 M-RoPE 结果 z 和 max(abs(z))
 *        | VEC_STORE -> VEC_LOAD：Stage 写完 scratch，Scale 才能读取
 *     Scale  : max(abs(z))/rms -> E8M0 scale，同时算 1/(rms*scale)
 *        | VEC_STORE -> VEC_LOAD：合并倒数写完，Quant 才能读取
 *     Quant  : z * 1/(rms*scale) -> E4M3，并按两行连续 256 B 写出
 *        | VEC_LOAD -> VEC_STORE：本批读取结束后，下一批才可复用 scratch
 *
 * RowBatch16 延迟归一化依据：令 z=M-RoPE(x*gamma)，y=z/rms，则
 *   max(abs(y)) = max(abs(z))/rms
 *   y/scale = z/(rms*scale)
 * 这样每行的 Sqrt/Div 可以在 16 行 Scale 阶段并行执行，逐元素只保留一次 Mul。
 *
 * 一个 RowBatch16 scratch record 的连续区域为：
 *   data       : 16 rows x 128 FP32 z
 *   max        : 16 rows x 4 uint32 amax bits
 *   倒数 : 16 rows x 4 FP32 合并倒数
 *   rms^2      : 8 head-pair x 32 B sparse records
 * 各 helper 只通过上述固定布局交接数据；输出 scale 仍是每行连续四个 E8M0 byte。
 *
 * LocalMemBar 只约束同一 UB scratch 的 Vector load/store 生命周期，不是 CPU 线程同步。
 *
 * 上层入口路由如下：
 *
 *   V
 *     在共享 vf.h 中调用 VScaleFp8D128ToNtdVfImpl，只做 vScale 缩放、
 *     E4M3 Cast 和 staging 布局，不需要 M-RoPE/MX scratch。
 *
 *   Q
 *     统一进入 QRmsNormMropeMxD128GlobalTileWaveVfImpl<HAS_Q_TAIL>。每个 token
 *     的完整 16-head batch 各占一个 scratch record；若 Nq%16==8，每两个
 *     token 的 tail 共用一个 record，奇数末 token 只使用低八行。整个 tile
 *     按 Stage-all -> Scale-all -> Quant-all 排布，以常数次屏障换取更大的
 *     Vector 指令窗口。
 *
 *   K
 *     Nk==8 时复用 GlobalTileWave<true>，每两个 token 填满 16 行；
 *     Nk 为 2/4 时进入 KRmsNormMropeMxD128RowBatch16EvenVfImpl，循环复用一个
 *     RowBatch16 record；不足一批或其他 Nk 由 KRmsNormMropeMxD128VfImpl 逐行处理。
 *     这些分支只改变 SIMD lane 利用率和 scratch 生命周期，不改变每行四个
 *     D32 scale、E8M0/E4M3 公式和输出顺序。
 */

constexpr uint32_t MROPE_MX_FP32_ABS_MASK = 0x7fffffffU;
constexpr uint32_t MROPE_MX_FP32_INF_BITS = 0x7f800000U;
constexpr uint32_t MROPE_MX_FP32_CUBLAS_FOLDED_ROUND_OFFSET = 0xfc1fffffU;
constexpr uint32_t MROPE_MX_FP32_CUBLAS_SUBNORMAL_EDGE = 0x04600001U;
constexpr uint32_t MROPE_MX_FP32_EXP_BIAS_CUBLAS = 0x7f000000U;
constexpr uint32_t MROPE_MX_FP32_NEG_EXP_UNIT = 0xff800000U;
constexpr uint32_t MROPE_MX_FP32_NAN_BITS = 0x7f810000U;
constexpr uint32_t MROPE_MX_E8M0_NAN = 0x000000ffU;
constexpr int16_t MROPE_MX_FP32_EXP_SHIFT = 23;
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8_ONE = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8_TWO = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8_THREE = {
    MicroAPI::RegLayout::THREE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__simd_callee__ inline void RmsNormMropeFp32D128(
    MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh, MicroAPI::RegTensor<bfloat16_t> &inLowBf16,
    MicroAPI::RegTensor<bfloat16_t> &inHighBf16, MicroAPI::RegTensor<float> &gammaCosLow,
    MicroAPI::RegTensor<float> &gammaCosHigh, MicroAPI::RegTensor<float> &gammaSinLow,
    MicroAPI::RegTensor<float> &gammaSinHigh, float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    // 逐行参考通路：先得到 norm=x/rms，再完成 gamma 和旋转；输出仍为两个连续 D64。
    MicroAPI::RegTensor<float> normLow;
    MicroAPI::RegTensor<float> normHigh;
    MicroAPI::RegTensor<float> rotateTmp;
    RmsNormBf16ToFp32D128<false>(normLow, normHigh, inLowBf16, inHighBf16, gammaCosLow, gammaCosHigh, epsilon, mask64,
                                 maskFirst);
    MicroAPI::Mul(outLow, normLow, gammaCosLow, mask64);
    MicroAPI::Mul(rotateTmp, normHigh, gammaSinHigh, mask64);
    MicroAPI::Sub(outLow, outLow, rotateTmp, mask64);
    MicroAPI::Mul(outHigh, normHigh, gammaCosHigh, mask64);
    MicroAPI::Mul(rotateTmp, normLow, gammaSinLow, mask64);
    MicroAPI::Add(outHigh, outHigh, rotateTmp, mask64);
}

__simd_callee__ inline void MxReduceMaxD32x2(MicroAPI::RegTensor<uint32_t> &maxValue, MicroAPI::RegTensor<float> &input,
                                             MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskLow32,
                                             MicroAPI::MaskReg maskHigh32)
{
    // 一个 D64 寄存器包含两个 D32 block。本函数输出这两个 block 的绝对值最大值。
    MicroAPI::RegTensor<uint32_t> absMask;
    MicroAPI::RegTensor<uint32_t> absValue;
    MicroAPI::RegTensor<uint32_t> lowMax;
    MicroAPI::RegTensor<uint32_t> highMax;
    MicroAPI::RegTensor<uint32_t> interleaveScratch;

    // Step 1: 清除 FP32 符号位得到 abs(input)，整数比较保持非负 FP32 的大小顺序。
    MicroAPI::Duplicate(absMask, MROPE_MX_FP32_ABS_MASK);
    MicroAPI::And(absValue, (MicroAPI::RegTensor<uint32_t> &)input, absMask, mask64);
    // Step 2: 分别归约低/高 32 lanes；Interleave 把两个标量结果放入后续可拼接布局。
    MicroAPI::ReduceMax(lowMax, absValue, maskLow32);
    MicroAPI::ReduceMax(highMax, absValue, maskHigh32);
    MicroAPI::Interleave(maxValue, interleaveScratch, lowMax, highMax);
}

__simd_callee__ inline void MxQuantCublasScaleD32x4(MicroAPI::RegTensor<float> &reciprocal,
                                                    MicroAPI::RegTensor<uint32_t> &scaleExp,
                                                    MicroAPI::RegTensor<uint32_t> &maxValue, MicroAPI::MaskReg mask4)
{
    // 输入是四个 D32 amax 的原始 FP32 bit；输出是四个 E8M0 byte（暂存为 uint32）
    // 和对应的 FP32 倒数=1/scale。整个转换使用整数指数域，保持既定 MX 合同。
    MicroAPI::MaskReg normalPredicate;
    MicroAPI::MaskReg invalidPredicate;
    MicroAPI::RegTensor<uint32_t> adjustedBits;

    // For max > 0x04600001:
    //   ((max + 0x001fffff) >> 23) - 8
    // == ((max + 0xfc1fffff) mod 2^32) >> 23.
    // ZEROING keeps every lower value, including the locked edge, at scale 0.
    MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(normalPredicate, maxValue,
                                                            MROPE_MX_FP32_CUBLAS_SUBNORMAL_EDGE, mask4);
    MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GE>(invalidPredicate, maxValue, MROPE_MX_FP32_INF_BITS, mask4);
    MicroAPI::Adds(adjustedBits, maxValue, MROPE_MX_FP32_CUBLAS_FOLDED_ROUND_OFFSET, mask4);
    MicroAPI::ShiftRights<uint32_t, int16_t, MicroAPI::MaskMergeMode::ZEROING>(
        scaleExp, adjustedBits, MROPE_MX_FP32_EXP_SHIFT, normalPredicate);

    // E8M0 指数 e 直接映射为 FP32 2^(-e) 的指数位，因此无需逐 lane 浮点除法。
    MicroAPI::Muls<uint32_t>((MicroAPI::RegTensor<uint32_t> &)reciprocal, scaleExp, MROPE_MX_FP32_NEG_EXP_UNIT, mask4);
    MicroAPI::Adds<uint32_t>((MicroAPI::RegTensor<uint32_t> &)reciprocal, (MicroAPI::RegTensor<uint32_t> &)reciprocal,
                             MROPE_MX_FP32_EXP_BIAS_CUBLAS, mask4);
    // Inf/NaN 走固定编码；MERGING 只覆盖 invalid lanes，不改变正常结果。
    MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(scaleExp, MROPE_MX_E8M0_NAN, invalidPredicate);
    MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>((MicroAPI::RegTensor<uint32_t> &)reciprocal,
                                                                    MROPE_MX_FP32_NAN_BITS, invalidPredicate);
}

__simd_callee__ inline void MxQuantCublasD128(__ubuf__ uint8_t *outBytes, MicroAPI::AddrReg outAddr,
                                              __ubuf__ uint8_t *scaleBytes, MicroAPI::RegTensor<float> &inLow,
                                              MicroAPI::RegTensor<float> &inHigh,
                                              MicroAPI::RegTensor<uint32_t> &scaleGatherLowIndex,
                                              MicroAPI::RegTensor<uint32_t> &scaleGatherHighIndex,
                                              MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskLow32,
                                              MicroAPI::MaskReg maskHigh32, MicroAPI::MaskReg mask4)
{
    // 单行直接量化通路：D128 -> 4×D32 amax -> 4 scales -> 128 E4M3 bytes。
    // 它用于不能组成完整 RowBatch16 的尾行，也定义了批量通路必须保持的输出顺序。
    MicroAPI::RegTensor<uint32_t> maxLow;
    MicroAPI::RegTensor<uint32_t> maxHigh;
    MicroAPI::RegTensor<uint32_t> maxCombined;
    MicroAPI::RegTensor<uint32_t> interleaveScratch;
    // Step 1: low/high 各产生两个 D32 amax，再拼成行内四个 amax。
    MxReduceMaxD32x2(maxLow, inLow, mask64, maskLow32, maskHigh32);
    MxReduceMaxD32x2(maxHigh, inHigh, mask64, maskLow32, maskHigh32);
    MicroAPI::Interleave(maxCombined, interleaveScratch, maxLow, maxHigh);

    MicroAPI::RegTensor<float> reciprocalCombined;
    MicroAPI::RegTensor<uint32_t> scaleExpCombined;
    // Step 2: 一次生成四个 E8M0 scale 和四个 FP32 倒数。
    MxQuantCublasScaleD32x4(reciprocalCombined, scaleExpCombined, maxCombined, mask4);

    MicroAPI::RegTensor<uint32_t> scaleExpLow;
    MicroAPI::RegTensor<uint32_t> scaleExpHigh;
    MicroAPI::DeInterleave(scaleExpLow, scaleExpHigh, scaleExpCombined, scaleExpCombined);

    MicroAPI::RegTensor<fp8_e4m3fn_t> output;
    MicroAPI::RegTensor<uint16_t> scaleB16;
    MicroAPI::RegTensor<uint8_t> scale;
    MicroAPI::RegTensor<float> expandedScale;
    MicroAPI::UnalignReg scaleStore;

    // Step 3: Gather 将每个 D32 倒数广播到对应 32 lanes；乘、Cast、Store 完成量化。
    // scale 分两次各写 2 B，最终仍形成连续的四字节 [s0,s1,s2,s3]。
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExpLow);
    MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale, scaleB16);
    MicroAPI::Gather(expandedScale, reciprocalCombined, scaleGatherLowIndex);
    MicroAPI::Mul(inLow, inLow, expandedScale, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8>(output, inLow, mask64);
    MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outBytes, (MicroAPI::RegTensor<uint8_t> &)output,
                                                                       outAddr, mask64);
    MicroAPI::StoreUnAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleBytes, scale, scaleStore, 2U);

    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExpHigh);
    MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale, scaleB16);
    MicroAPI::Gather(expandedScale, reciprocalCombined, scaleGatherHighIndex);
    MicroAPI::Mul(inHigh, inHigh, expandedScale, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8>(output, inHigh, mask64);
    MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
        outBytes + QKV_K_SCALE_D128_HALF_SIZE, (MicroAPI::RegTensor<uint8_t> &)output, outAddr, mask64);
    MicroAPI::StoreUnAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleBytes, scale, scaleStore, 2U);
    MicroAPI::StoreUnAlignPost(scaleBytes, scaleStore, 0U);
}

__simd_callee__ inline void PrepareMropeGammaD128(
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh,
    MicroAPI::RegTensor<uint32_t> &mropeIndex, __ubuf__ float *rawToken, MicroAPI::MaskReg mask64)
{
    // 同一 token 的所有 heads 共用 cos/sin。先按 M-RoPE index Gather，再预乘 gamma，
    // 后续每行旋转只需四次 Mul 和一次 Add/Sub，不重复计算 gamma*cos/sin。
    MicroAPI::RegTensor<float> cosValue;
    MicroAPI::RegTensor<float> sinValue;
    MicroAPI::Gather(cosValue, rawToken, mropeIndex, mask64);
    MicroAPI::Gather(sinValue, rawToken + QKV_K_SCALE_D128_HALF_SIZE, mropeIndex, mask64);
    MicroAPI::Mul(gammaSinLow, gammaLow, sinValue, mask64);
    MicroAPI::Mul(gammaSinHigh, gammaHigh, sinValue, mask64);
    MicroAPI::Mul(gammaCosLow, gammaLow, cosValue, mask64);
    MicroAPI::Mul(gammaCosHigh, gammaHigh, cosValue, mask64);
}

__simd_callee__ inline void LoadRmsNormMropeFp32D128(
    MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh, __ubuf__ bfloat16_t *inputBf16,
    MicroAPI::AddrReg inputAddr, MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh, float epsilon,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    // 从自然 D128 BF16 行加载两个 D64，并进入“先归一化、后旋转”的逐行通路。
    MicroAPI::RegTensor<bfloat16_t> inLowBf16;
    MicroAPI::RegTensor<bfloat16_t> inHighBf16;
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(inLowBf16, inputBf16, inputAddr);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        inHighBf16, inputBf16 + QKV_K_SCALE_D128_HALF_SIZE, inputAddr);
    RmsNormMropeFp32D128(outLow, outHigh, inLowBf16, inHighBf16, gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh,
                         epsilon, mask64, maskFirst);
}

// RowBatch16 的 Stage 不生成 y=z/rms，而只生成 rms^2 和未归一化的 z。
// MX scale 只依赖 max(abs(y))，因此 Scale 阶段可以用 max(abs(z))/rms 得到相同输入；
// Quant 阶段再用 z/(rms*scale) 一次完成原来的归一化与量化缩放。
__simd_callee__ inline void LoadRmsSquaredMropeUnnormalizedFp32D128(
    MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh, MicroAPI::RegTensor<float> &rmsSquared,
    __ubuf__ bfloat16_t *inputBf16, MicroAPI::AddrReg inputAddr, MicroAPI::RegTensor<float> &gammaCosLow,
    MicroAPI::RegTensor<float> &gammaCosHigh, MicroAPI::RegTensor<float> &gammaSinLow,
    MicroAPI::RegTensor<float> &gammaSinHigh, float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    MicroAPI::RegTensor<bfloat16_t> inLowBf16;
    MicroAPI::RegTensor<bfloat16_t> inHighBf16;
    MicroAPI::RegTensor<float> inLow;
    MicroAPI::RegTensor<float> inHigh;
    MicroAPI::RegTensor<float> squareLow;
    MicroAPI::RegTensor<float> squareHigh;
    MicroAPI::RegTensor<float> squareSum;
    MicroAPI::RegTensor<float> rotateTmp;

    // Step 1: BF16 D128 -> 两个 FP32 D64；BF16 值可精确扩展到 FP32。
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(inLowBf16, inputBf16, inputAddr);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        inHighBf16, inputBf16 + QKV_K_SCALE_D128_HALF_SIZE, inputAddr);
    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(inLow, inLowBf16, mask64);
    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(inHigh, inHighBf16, mask64);
    // Step 2: sum(low^2 + high^2)/128 + epsilon。Add 后 Reduce 覆盖全部 128 个元素。
    MicroAPI::Mul(squareLow, inLow, inLow, mask64);
    MicroAPI::Mul(squareHigh, inHigh, inHigh, mask64);
    MicroAPI::Add(squareSum, squareLow, squareHigh, mask64);
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(rmsSquared, squareSum,
                                                                                                mask64);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(rmsSquared, rmsSquared, QKV_K_SCALE_D128_RECIP,
                                                                   maskFirst);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(rmsSquared, rmsSquared, epsilon, maskFirst);

    // Step 3: 不除 rms，直接生成 z。Scale 阶段稍后把 rms 合入倒数。
    MicroAPI::Mul(outLow, inLow, gammaCosLow, mask64);
    MicroAPI::Mul(rotateTmp, inHigh, gammaSinHigh, mask64);
    MicroAPI::Sub(outLow, outLow, rotateTmp, mask64);
    MicroAPI::Mul(outHigh, inHigh, gammaCosHigh, mask64);
    MicroAPI::Mul(rotateTmp, inLow, gammaSinLow, mask64);
    MicroAPI::Add(outHigh, outHigh, rotateTmp, mask64);
}

template <bool REDUCE_MAX = true>
__simd_callee__ inline void StoreAndReduceMropeMxD128x2(
    __ubuf__ float *row0LowScratch, __ubuf__ float *row0HighScratch, __ubuf__ float *row1LowScratch,
    __ubuf__ float *row1HighScratch, __ubuf__ uint32_t *maxScratch, MicroAPI::AddrReg dataAddr,
    MicroAPI::AddrReg maxAddr, MicroAPI::RegTensor<float> &row0Low, MicroAPI::RegTensor<float> &row0High,
    MicroAPI::RegTensor<float> &row1Low, MicroAPI::RegTensor<float> &row1High, MicroAPI::RegTensor<uint32_t> &absMask,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg mask8)
{
    // 两行是 RowBatch16 的最小 producer 单元。
    // Step 1: 先把 row0/row1 的 z 按自然行序写入 data scratch，供 Quant 阶段重读。
    MicroAPI::StoreAlign<float>(row0LowScratch, row0Low, dataAddr, mask64);
    MicroAPI::StoreAlign<float>(row0HighScratch, row0High, dataAddr, mask64);
    MicroAPI::StoreAlign<float>(row1LowScratch, row1Low, dataAddr, mask64);
    MicroAPI::StoreAlign<float>(row1HighScratch, row1High, dataAddr, mask64);
    if constexpr (REDUCE_MAX) {
        // Step 2: 将两行、两个 D64 重排成 D32 流；清符号位后逐级 Max。
        // 最终一次 ReduceMaxWithDataBlock 同时得到 2 rows × 4 D32 的八个 amax。
        MicroAPI::DeInterleave(row0Low, row0High, row0Low, row0High);
        MicroAPI::DeInterleave(row1Low, row1High, row1Low, row1High);
        MicroAPI::DeInterleave(row0Low, row1Low, row0Low, row1Low);
        MicroAPI::DeInterleave(row0High, row1High, row0High, row1High);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row0Low, (MicroAPI::RegTensor<uint32_t> &)row0Low, absMask,
                      mask64);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row0High, (MicroAPI::RegTensor<uint32_t> &)row0High, absMask,
                      mask64);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row1Low, (MicroAPI::RegTensor<uint32_t> &)row1Low, absMask,
                      mask64);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row1High, (MicroAPI::RegTensor<uint32_t> &)row1High, absMask,
                      mask64);
        MicroAPI::Max((MicroAPI::RegTensor<uint32_t> &)row0Low, (MicroAPI::RegTensor<uint32_t> &)row0Low,
                      (MicroAPI::RegTensor<uint32_t> &)row1Low, mask64);
        MicroAPI::Max((MicroAPI::RegTensor<uint32_t> &)row0High, (MicroAPI::RegTensor<uint32_t> &)row0High,
                      (MicroAPI::RegTensor<uint32_t> &)row1High, mask64);
        MicroAPI::Max((MicroAPI::RegTensor<uint32_t> &)row0Low, (MicroAPI::RegTensor<uint32_t> &)row0Low,
                      (MicroAPI::RegTensor<uint32_t> &)row0High, mask64);
        MicroAPI::ReduceMaxWithDataBlock((MicroAPI::RegTensor<uint32_t> &)row0Low,
                                         (MicroAPI::RegTensor<uint32_t> &)row0Low, mask64);
        MicroAPI::StoreAlign<uint32_t>(maxScratch, (MicroAPI::RegTensor<uint32_t> &)row0Low, maxAddr, mask8);
    }
}

template <bool REDUCE_MAX = true>
__simd_callee__ inline void StageMropeMxHeadPairD128(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *row0LowScratch, __ubuf__ float *row0HighScratch,
    __ubuf__ float *row1LowScratch, __ubuf__ float *row1HighScratch, __ubuf__ uint32_t *maxScratch,
    MicroAPI::AddrReg inputAddr, MicroAPI::AddrReg dataAddr, MicroAPI::AddrReg maxAddr, uint32_t inputHeadStride,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<uint32_t> &absMask, float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst,
    MicroAPI::MaskReg mask8)
{
    // 已归一化 producer：两行各自完整执行 RMSNorm+M-RoPE，再共同写 data/max scratch。
    // REDUCE_MAX=false 只用于最终奇 token 的 8-row tail；它稍后走单行直接量化并自行算 amax。
    MicroAPI::RegTensor<float> row0Low;
    MicroAPI::RegTensor<float> row0High;
    MicroAPI::RegTensor<float> row1Low;
    MicroAPI::RegTensor<float> row1High;
    LoadRmsNormMropeFp32D128(row0Low, row0High, inputBf16, inputAddr, gammaCosLow, gammaCosHigh, gammaSinLow,
                             gammaSinHigh, epsilon, mask64, maskFirst);
    LoadRmsNormMropeFp32D128(row1Low, row1High, inputBf16 + inputHeadStride, inputAddr, gammaCosLow, gammaCosHigh,
                             gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
    StoreAndReduceMropeMxD128x2<REDUCE_MAX>(row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch,
                                            maxScratch, dataAddr, maxAddr, row0Low, row0High, row1Low, row1High,
                                            absMask, mask64, mask8);
}

__simd_callee__ inline void StageMropeMxHeadPairDeferredRmsD128(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *row0LowScratch, __ubuf__ float *row0HighScratch,
    __ubuf__ float *row1LowScratch, __ubuf__ float *row1HighScratch, __ubuf__ uint32_t *maxScratch,
    __ubuf__ float *rmsSquaredScratch, MicroAPI::AddrReg inputAddr, MicroAPI::AddrReg dataAddr,
    MicroAPI::AddrReg maxAddr, MicroAPI::AddrReg rmsAddr, uint32_t inputHeadStride,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<uint32_t> &absMask, float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst,
    MicroAPI::MaskReg mask8)
{
    // 延迟归一化 producer：每对 head 同时产生
    //   dataScratch : 两行未归一化 z
    //   maxScratch  : 两行各四个 max(abs(z))
    //   rmsScratch  : 两行 rms^2，按 [row0,row1,0,0,0,0,0,0] 稀疏存放
    // 稀疏布局让每个 pair 只需一次对齐 Store；Scale 阶段用 rmsGatherIndex 恢复并广播。
    MicroAPI::RegTensor<float> row0Low;
    MicroAPI::RegTensor<float> row0High;
    MicroAPI::RegTensor<float> row1Low;
    MicroAPI::RegTensor<float> row1High;
    MicroAPI::RegTensor<float> row0RmsSquared;
    MicroAPI::RegTensor<float> row1RmsSquared;
    MicroAPI::RegTensor<float> rmsPair;
    MicroAPI::RegTensor<float> rmsPairScratch;
    // Step 1: 两行独立计算 rms^2 和 z。
    LoadRmsSquaredMropeUnnormalizedFp32D128(row0Low, row0High, row0RmsSquared, inputBf16, inputAddr, gammaCosLow,
                                            gammaCosHigh, gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
    LoadRmsSquaredMropeUnnormalizedFp32D128(row1Low, row1High, row1RmsSquared, inputBf16 + inputHeadStride, inputAddr,
                                            gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, epsilon, mask64,
                                            maskFirst);
    // Step 2: 写 z 并归约两行的八个 D32 amax。
    StoreAndReduceMropeMxD128x2(row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch, maxScratch, dataAddr,
                                maxAddr, row0Low, row0High, row1Low, row1High, absMask, mask64, mask8);
    // Step 3: 合成稀疏 rms^2 记录：[rms²_row0, rms²_row1, 0,0,0,0,0,0]。
    MicroAPI::Interleave(rmsPair, rmsPairScratch, row0RmsSquared, row1RmsSquared);
    MicroAPI::StoreAlign<float>(rmsSquaredScratch, rmsPair, rmsAddr, mask8);
}

__simd_callee__ inline void MxQuantCublasScaleRowBatch16DeferredRms(
    __ubuf__ uint32_t *maxScratch, __ubuf__ float *rmsSquaredScratch, __ubuf__ float *reciprocalScratch,
    __ubuf__ uint8_t *scaleBytes, MicroAPI::RegTensor<uint32_t> &rmsGatherIndex, MicroAPI::MaskReg maskScale,
    MicroAPI::MaskReg maskScaleB16)
{
    // 一次处理 16 rows × 4 D32 = 64 个 scale lane。
    // 输入 max(abs(z)) 与 rms^2，输出 E8M0 scale 和 FP32 合并倒数。
    MicroAPI::RegTensor<uint32_t> maxValue;
    MicroAPI::RegTensor<float> rmsSquaredSparse;
    MicroAPI::RegTensor<float> rmsSquared;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::RegTensor<float> normalizedMax;
    MicroAPI::RegTensor<float> reciprocal;
    MicroAPI::RegTensor<uint32_t> scaleExp;
    MicroAPI::RegTensor<uint16_t> scaleB16;

    // Step 1: 读取 64 个 D32 amax；把 16 个稀疏 rms^2 广播成每行四份。
    MicroAPI::LoadAlign<uint32_t>(maxValue, maxScratch);
    // rmsSquaredSparse: [row0,row1,空×6, row2,row3,空×6, ...]
    MicroAPI::LoadAlign<float>(rmsSquaredSparse, rmsSquaredScratch);
    MicroAPI::Gather(rmsSquared, rmsSquaredSparse, rmsGatherIndex);
    // Step 2: rms=sqrt(rms^2)，normalizedMax=max(abs(z))/rms。
    MicroAPI::Sqrt(rms, rmsSquared, maskScale);
    MicroAPI::Div(normalizedMax, (MicroAPI::RegTensor<float> &)maxValue, rms, maskScale);
    // Step 3: 由 normalizedMax 生成 E8M0 scale；再把 1/scale 合并成 1/(rms*scale)。
    MxQuantCublasScaleD32x4(reciprocal, scaleExp, (MicroAPI::RegTensor<uint32_t> &)normalizedMax, maskScale);
    MicroAPI::Div(reciprocal, reciprocal, rms, maskScale);

    // Step 4: 64×uint32 exponent 压成连续 64×E8M0 byte；倒数保留 FP32 写入 scratch。
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExp);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
        reinterpret_cast<__ubuf__ uint16_t *>(scaleBytes), scaleB16, maskScaleB16);
    MicroAPI::StoreAlign<float>(reciprocalScratch, reciprocal, maskScale);
}

// Q 的 8-head tail 把两个 token 拼成一个 RowBatch16。两半数据已经完成逐行 RMSNorm，
// 因此直接对 16 行统一算 scale，再将低/高 32 个 scale byte 分别写回 token0/token1。
__simd_callee__ inline void MxQuantCublasScaleRowBatch16Split(
    __ubuf__ uint32_t *maxScratch, __ubuf__ float *reciprocalScratch, __ubuf__ uint8_t *scaleBytes0,
    __ubuf__ uint8_t *scaleBytes1, MicroAPI::RegTensor<uint32_t> &scaleHighGatherIndex, MicroAPI::MaskReg maskScale,
    MicroAPI::MaskReg maskScaleB16, MicroAPI::MaskReg maskScaleHalfB16)
{
    // Step 1: 对两 token 的 16 行统一生成 64 个 scale/倒数。
    MicroAPI::RegTensor<uint32_t> maxValue;
    MicroAPI::RegTensor<float> reciprocal;
    MicroAPI::RegTensor<uint32_t> scaleExp;
    MicroAPI::RegTensor<uint32_t> scaleExpHigh;
    MicroAPI::RegTensor<uint16_t> scaleB16;
    MicroAPI::RegTensor<uint16_t> scaleHighB16;

    MicroAPI::LoadAlign<uint32_t>(maxValue, maxScratch);
    MxQuantCublasScaleD32x4(reciprocal, scaleExp, maxValue, maskScale);

    // Step 2: 低 8 行直接写 token0；Gather 高 8 行后写 token1。
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExp);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
        reinterpret_cast<__ubuf__ uint16_t *>(scaleBytes0), scaleB16, maskScaleHalfB16);
    MicroAPI::Gather(scaleExpHigh, scaleExp, scaleHighGatherIndex);
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleHighB16, scaleExpHigh);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
        reinterpret_cast<__ubuf__ uint16_t *>(scaleBytes1), scaleHighB16, maskScaleHalfB16);

    MicroAPI::StoreAlign<float>(reciprocalScratch, reciprocal, maskScale);
}

template <uint32_t TOKEN_SCALE_COUNT>
__simd_callee__ inline void StoreMropeMxScaleRowBatch16ToPadded(__ubuf__ uint32_t *scaleScratch,
                                                                __ubuf__ uint8_t *paddedScaleBytes, uint16_t tokenCount)
{
    // K 的一个 RowBatch16 可能由多个 token 组成。scratch 中 scale 按 16 行紧凑排列，
    // 公共输出却按 token 使用固定 32 B pitch；本函数逐 token 重排到该 padded 布局。
    MicroAPI::RegTensor<uint32_t> tokenScaleExp;
    MicroAPI::RegTensor<uint16_t> tokenScaleB16;
    uint32_t tokenScaleCount = TOKEN_SCALE_COUNT;
    MicroAPI::MaskReg tokenScaleMaskB16 = MicroAPI::UpdateMask<uint16_t>(tokenScaleCount);
    for (uint16_t localToken = 0U; localToken < tokenCount; ++localToken) {
        MicroAPI::AddrReg scaleLoadAddr = MicroAPI::CreateAddrReg<uint32_t>(localToken, TOKEN_SCALE_COUNT);
        MicroAPI::LoadAlign<uint32_t>(tokenScaleExp, scaleScratch, scaleLoadAddr);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(tokenScaleB16, tokenScaleExp);
        MicroAPI::AddrReg scaleStoreAddr = MicroAPI::CreateAddrReg<uint16_t>(localToken, 16U);
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
            reinterpret_cast<__ubuf__ uint16_t *>(paddedScaleBytes), tokenScaleB16, scaleStoreAddr, tokenScaleMaskB16);
    }
}

__simd_callee__ inline void MxQuantCublasScaleRowBatch16DeferredRmsToPadded(
    __ubuf__ uint32_t *maxScratch, __ubuf__ float *rmsSquaredScratch, __ubuf__ float *reciprocalScratch,
    __ubuf__ uint8_t *paddedScaleBytes, uint16_t tokenCount, uint16_t headSize,
    MicroAPI::RegTensor<uint32_t> &rmsGatherIndex, MicroAPI::MaskReg maskScale, MicroAPI::MaskReg maskScaleB16)
{
    // Nk=2/4：先按完整 RowBatch16 算 scale，再借用 maxScratch 暂存 uint32 exponent，
    // 将紧凑的 token×head×4 结果散到每 token 固定 pitch。倒数仍保持连续 64×FP32。
    MicroAPI::RegTensor<uint32_t> maxValue;
    MicroAPI::RegTensor<float> rmsSquaredSparse;
    MicroAPI::RegTensor<float> rmsSquared;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::RegTensor<float> normalizedMax;
    MicroAPI::RegTensor<float> reciprocal;
    MicroAPI::RegTensor<uint32_t> scaleExp;

    MicroAPI::LoadAlign<uint32_t>(maxValue, maxScratch);
    MicroAPI::LoadAlign<float>(rmsSquaredSparse, rmsSquaredScratch);
    MicroAPI::Gather(rmsSquared, rmsSquaredSparse, rmsGatherIndex);
    MicroAPI::Sqrt(rms, rmsSquared, maskScale);
    MicroAPI::Div(normalizedMax, (MicroAPI::RegTensor<float> &)maxValue, rms, maskScale);
    MxQuantCublasScaleD32x4(reciprocal, scaleExp, (MicroAPI::RegTensor<uint32_t> &)normalizedMax, maskScale);
    MicroAPI::Div(reciprocal, reciprocal, rms, maskScale);

    // exponent 的 producer Store 必须完成，后面的逐 token Load 才能安全重排。
    MicroAPI::StoreAlign<uint32_t>(maxScratch, scaleExp, maskScale);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    if (headSize == 2U) {
        StoreMropeMxScaleRowBatch16ToPadded<8U>(maxScratch, paddedScaleBytes, tokenCount);
    } else {
        StoreMropeMxScaleRowBatch16ToPadded<16U>(maxScratch, paddedScaleBytes, tokenCount);
    }

    MicroAPI::StoreAlign<float>(reciprocalScratch, reciprocal, maskScale);
}

__simd_callee__ inline void MxQuantCublasDataRowBatch16(__ubuf__ float *dataScratch, __ubuf__ float *reciprocalScratch,
                                                        __ubuf__ uint8_t *outBytes, uint16_t pairCount,
                                                        MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFp8)
{
    // Quant consumer 每次处理两行。dataScratch 是自然 row-major FP32 z；
    // 倒数 scratch 是每行四个 FP32 1/(rms*scale)。最终一次写出连续 2×128 B。
    for (uint16_t pairIdx = 0U; pairIdx < pairCount; ++pairIdx) {
        MicroAPI::RegTensor<float> reciprocalFp32;
        MicroAPI::RegTensor<float> x0Zero;
        MicroAPI::RegTensor<float> x1Zero;
        MicroAPI::RegTensor<float> x0One;
        MicroAPI::RegTensor<float> x1One;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x0ZeroFp8;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x1ZeroFp8;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x0OneFp8;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x1OneFp8;
        // Step 1: E2B_B32 将两行的 8 个 D32 倒数广播到对应数据 lanes。
        MicroAPI::AddrReg reciprocalAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_E2B_B32>(reciprocalFp32, reciprocalScratch, reciprocalAddr);
        MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        // Step 2: 两次双路 Load 取回两行 D128；DeInterleave 得到四个独立 lane stream。
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(x0Zero, x1Zero, dataScratch, dataAddr);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            x0One, x1One, dataScratch + QKV_K_SCALE_D128_FULL_SIZE, dataAddr);
        MicroAPI::DeInterleave(x0Zero, x0One, x0Zero, x0One);
        MicroAPI::DeInterleave(x1Zero, x1One, x1Zero, x1One);
        // Step 3: z/(rms*scale)，随后四种 Cast layout 把四个 stream 放到互不重叠的 byte lanes。
        MicroAPI::Mul(x0Zero, x0Zero, reciprocalFp32, mask64);
        MicroAPI::Mul(x1Zero, x1Zero, reciprocalFp32, mask64);
        MicroAPI::Mul(x0One, x0One, reciprocalFp32, mask64);
        MicroAPI::Mul(x1One, x1One, reciprocalFp32, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8>(x0ZeroFp8, x0Zero, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8_ONE>(x1ZeroFp8, x1Zero, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8_TWO>(x0OneFp8, x0One, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8_THREE>(x1OneFp8, x1One, mask64);
        // Step 4: 这里的 byte Add 是合并互斥 lanes，不是对 E4M3 数值求和；三次合并后恢复
        // row0[128 B] + row1[128 B] 的自然顺序，再由一次 Store 写出 256 B。
        MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8, (MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8,
                      (MicroAPI::RegTensor<uint8_t> &)x0OneFp8, maskFp8);
        MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)x1ZeroFp8, (MicroAPI::RegTensor<uint8_t> &)x1ZeroFp8,
                      (MicroAPI::RegTensor<uint8_t> &)x1OneFp8, maskFp8);
        MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8, (MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8,
                      (MicroAPI::RegTensor<uint8_t> &)x1ZeroFp8, maskFp8);
        MicroAPI::AddrReg outAddr = MicroAPI::CreateAddrReg<uint8_t>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
            outBytes, (MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8, outAddr, maskFp8);
    }
}

__simd_callee__ inline void QuantizeMropeMxRowBatch8AfterReady(__ubuf__ float *dataScratch, __ubuf__ uint8_t *outBytes,
                                                               __ubuf__ uint8_t *scaleBytes, MicroAPI::MaskReg mask64)
{
    // 奇数个 token 留下的最后 8 行无法与下一 token 拼成 RowBatch16。
    // Stage 已经生成归一化后的 y，因此逐行复用 MxQuantCublasD128，避免为半批构造额外布局。
    MicroAPI::RegTensor<uint32_t> scaleGatherLowIndex;
    MicroAPI::RegTensor<uint32_t> scaleGatherHighIndex;
    MicroAPI::MaskReg maskLow32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg maskHigh32;
    MicroAPI::Not(maskHigh32, maskLow32, mask64);
    uint32_t scaleCount = MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg mask4 = MicroAPI::UpdateMask<float>(scaleCount);
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)scaleGatherLowIndex, 0);
    MicroAPI::ShiftRights(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(5), mask64);
    MicroAPI::ShiftLefts(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(1), mask64);
    MicroAPI::Adds(scaleGatherHighIndex, scaleGatherLowIndex, 1U, mask64);
    for (uint16_t row = 0U; row < 8U; ++row) {
        MicroAPI::RegTensor<float> rowLow;
        MicroAPI::RegTensor<float> rowHigh;
        MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(row, QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::AddrReg outAddr = MicroAPI::CreateAddrReg<uint8_t>(row, QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::LoadAlign<float>(rowLow, dataScratch, dataAddr);
        MicroAPI::LoadAlign<float>(rowHigh, dataScratch + QKV_K_SCALE_D128_HALF_SIZE, dataAddr);
        MxQuantCublasD128(outBytes, outAddr, scaleBytes + static_cast<uint32_t>(row) * MROPE_MX_SCALE_COUNT_D128,
                          rowLow, rowHigh, scaleGatherLowIndex, scaleGatherHighIndex, mask64, maskLow32, maskHigh32,
                          mask4);
    }
}

template <bool REDUCE_MAX>
__simd_callee__ inline void StageQTailToken8D128(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ uint8_t *waveScratch, uint16_t rowOffset, uint32_t inputHeadStride,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<uint32_t> &absMask, float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst,
    MicroAPI::MaskReg mask8)
{
    // 把一个 token 的 8 个 Q tail heads 写入 RowBatch16 scratch 的低半或高半。
    // rowOffset=0/8 决定所属半区；四个 pair 恰好覆盖八行。
    __ubuf__ float *dataScratch =
        reinterpret_cast<__ubuf__ float *>(waveScratch) + static_cast<uint32_t>(rowOffset) * QKV_K_SCALE_D128_FULL_SIZE;
    __ubuf__ float *row0LowScratch = dataScratch;
    __ubuf__ float *row0HighScratch = dataScratch + QKV_K_SCALE_D128_HALF_SIZE;
    __ubuf__ float *row1LowScratch = dataScratch + QKV_K_SCALE_D128_FULL_SIZE;
    __ubuf__ float *row1HighScratch = dataScratch + QKV_K_SCALE_D128_FULL_SIZE + QKV_K_SCALE_D128_HALF_SIZE;
    __ubuf__ uint32_t *maxScratch = reinterpret_cast<__ubuf__ uint32_t *>(waveScratch + MROPE_MX_ROW_BATCH_DATA_BYTES) +
                                    static_cast<uint32_t>(rowOffset) * MROPE_MX_SCALE_COUNT_D128;
    for (uint16_t pairIdx = 0U; pairIdx < 4U; ++pairIdx) {
        MicroAPI::AddrReg inputAddr = MicroAPI::CreateAddrReg<bfloat16_t>(pairIdx, 2U * inputHeadStride);
        MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::AddrReg maxAddr = MicroAPI::CreateAddrReg<uint32_t>(pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
        StageMropeMxHeadPairD128<REDUCE_MAX>(inputBf16, row0LowScratch, row0HighScratch, row1LowScratch,
                                             row1HighScratch, maxScratch, inputAddr, dataAddr, maxAddr, inputHeadStride,
                                             gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, absMask, epsilon,
                                             mask64, maskFirst, mask8);
    }
}

/*
 * Q GlobalTileWave 的理论依据
 * ===========================
 *
 * 1. 为什么选择 16 行作为批量单位
 *
 *    一个 D128 行包含四个 D32 scale。16 rows x 4 scales = 64 values，正好填满
 *    FP32 Vector 的 64 lanes。逐 head 通路一次只有四个有效 scale lanes，而 RowBatch16
 *    可以用一次 Load/Sqrt/Div/指数转换同时处理 64 个 scale，摊薄向量指令的固定开销。
 *
 * 2. 为什么可以把 RMSNorm 推迟到 Scale 阶段
 *
 *    Stage 先算 z=M-RoPE(x*gamma) 和 rms^2，不生成 y=z/rms。对正常有限行，rms 是
 *    一整行共享的非负标量，所以 max(abs(y_D32))=max(abs(z_D32))/rms；Quant 又可以计算
 *    z/(rms*scale)。这把每行重复的 Sqrt/Div 改成 16 行并行的 Vector 运算，同时把
 *    “乘 1/rms”和“乘 1/scale”合成一次逐元素 Mul。数学公式和四个 D32 scale 的顺序
 *    不变；FP32 的具体求值顺序以本实现为准。
 *
 * 3. 为什么要按整 tile 展开 Stage -> Scale -> Quant
 *
 *    每个 RowBatch16 使用互不重叠的 scratch record，record 之间没有读写依赖。因此无需
 *    每处理一个 token/batch 就等待并关闭流水窗口，可以先生产整 tile，再统一消费：
 *
 *      record-local: (Stage -> Bar -> Scale -> Bar -> Quant -> Bar) x R
 *      tile-wave   :  Stage x R -> Bar -> Scale x R -> Bar -> Quant x R -> Bar
 *
 *    对 R 个独立 record，生命周期屏障由随 R 线性增长变为每 tile 常数个。更长的同类
 *    指令区间也给硬件更多独立 load、归约和算术链，便于双发和乱序调度；这能保持
 *    Vector 流水，避免短循环在每个 token 末尾反复排空。代价是同时保留 R 份
 *    UB scratch；Host tiling 通过 tokenSize 限制该工作集，保证所有 record 都在 UB 内。
 *
 * 4. 为什么 8-head tail 要跨 token 拼接
 *
 *    单个 tail 只有 8 rows x 4 scales = 32 个有效 lanes。token 2k 写 record 的低八行，
 *    token 2k+1 写高八行，两者合成完整 64-lane Scale。token 数为奇数时，最后一个 tail
 *    没有配对对象，只处理已初始化的低八行并回退到逐行 scale，因而优化不依赖
 *    token 为偶数。
 *
 * 5. 这项优化没有改变什么
 *
 *    输入、E8M0/E4M3 数值公式、每行四个 D32 scale、输出顺序和 GM 搬运量都不变；收益来自
 *    SIMD lane 利用率、固定指令摊销、屏障合并和更大的可调度指令窗口，而不是
 *    减少有效元素。
 */
template <bool HAS_Q_TAIL>
__simd_vf__ inline void QRmsNormMropeMxD128GlobalTileWaveVfImpl(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma, __ubuf__ float *rawCosSin, __ubuf__ uint32_t *gatherIndex,
    __ubuf__ fp8_e4m3fn_t *out, __ubuf__ fp8_e8m0_t *outScale, __ubuf__ uint8_t *scratch, uint16_t tokenSize,
    uint32_t inputTokenStride, uint16_t headSize, uint32_t inputHeadStride, uint32_t scaleTokenStrideWords,
    float epsilon)
{
    // 固定寄存器准备：gamma、M-RoPE Gather index、abs mask，以及两个批量 Gather index。
    MicroAPI::RegTensor<uint32_t> mropeIndex;
    MicroAPI::RegTensor<uint32_t> absMask;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t mask8Count = 8U;
    MicroAPI::MaskReg mask8 = MicroAPI::UpdateMask<uint32_t>(mask8Count);
    MicroAPI::MaskReg maskFp8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskScale = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    uint32_t scaleB16Count = MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg maskScaleB16 = MicroAPI::UpdateMask<uint16_t>(scaleB16Count);
    uint32_t scaleHalfB16Count = 8U * MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg maskScaleHalfB16 = MicroAPI::UpdateMask<uint16_t>(scaleHalfB16Count);
    MicroAPI::LoadAlign<uint32_t>(mropeIndex, gatherIndex);
    MicroAPI::LoadAlign<float>(gammaLow, gamma);
    MicroAPI::LoadAlign<float>(gammaHigh, gamma + QKV_K_SCALE_D128_HALF_SIZE);
    MicroAPI::Duplicate(absMask, MROPE_MX_FP32_ABS_MASK);

    // rmsGatherIndex 将稀疏 rms^2 布局
    // [r0,r1,0×6,r2,r3,0×6,...] 展开为 [r0×4,r1×4,...,r15×4]。
    MicroAPI::RegTensor<uint32_t> rmsGatherIndex;
    MicroAPI::RegTensor<uint32_t> rmsGatherBase;
    MicroAPI::RegTensor<uint32_t> rmsGatherOne;
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)rmsGatherIndex, 0);
    MicroAPI::ShiftRights(rmsGatherBase, rmsGatherIndex, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftLefts(rmsGatherBase, rmsGatherBase, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftRights(rmsGatherIndex, rmsGatherIndex, static_cast<int16_t>(2), maskScale);
    MicroAPI::Duplicate(rmsGatherOne, 1U);
    MicroAPI::And(rmsGatherIndex, rmsGatherIndex, rmsGatherOne, maskScale);
    MicroAPI::Add(rmsGatherIndex, rmsGatherBase, rmsGatherIndex, maskScale);

    // scaleHighGatherIndex 选出 RowBatch16 的高 8 行，供跨 token tail 分别写 scale。
    MicroAPI::RegTensor<uint32_t> scaleHighGatherIndex;
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)scaleHighGatherIndex, 0);
    MicroAPI::ShiftLefts(scaleHighGatherIndex, scaleHighGatherIndex, static_cast<int16_t>(27), maskScale);
    MicroAPI::ShiftRights(scaleHighGatherIndex, scaleHighGatherIndex, static_cast<int16_t>(27), maskScale);
    MicroAPI::Adds(scaleHighGatherIndex, scaleHighGatherIndex, 32U, maskScale);

    // scratch 索引：先排 token-major 的完整 batch，再排每两个 token 共用的 tail record。
    const uint16_t fullBatchCount = static_cast<uint16_t>(headSize / MROPE_MX_ROW_BATCH_ROWS);
    const uint16_t fullHeadCount = static_cast<uint16_t>(fullBatchCount * MROPE_MX_ROW_BATCH_ROWS);
    const uint16_t prefixScratchBatchCount = static_cast<uint16_t>(tokenSize * fullBatchCount);
    const uint16_t tailPairCount = HAS_Q_TAIL ? static_cast<uint16_t>(tokenSize / 2U) : 0U;
    const bool hasOddTailToken = HAS_Q_TAIL && (tokenSize & 1U) != 0U;
    constexpr uint32_t scaleBatchStride = MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128;

    // Phase 1 - Stage：先生产整 tile 的 data/max/rms^2 scratch，不在 token 内关闭流水窗口。
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        MicroAPI::RegTensor<float> gammaCosLow;
        MicroAPI::RegTensor<float> gammaCosHigh;
        MicroAPI::RegTensor<float> gammaSinLow;
        MicroAPI::RegTensor<float> gammaSinHigh;
        PrepareMropeGammaD128(gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, gammaLow, gammaHigh, mropeIndex,
                              rawCosSin + static_cast<uint32_t>(tokenIdx) * 3U * QKV_K_SCALE_D128_FULL_SIZE, mask64);
        __ubuf__ bfloat16_t *tokenInput = inputBf16 + static_cast<uint32_t>(tokenIdx) * inputTokenStride;

        // 完整前缀使用延迟归一化：每个 batch 正好 16 heads。
        for (uint16_t batchIdx = 0U; batchIdx < fullBatchCount; ++batchIdx) {
            const uint16_t scratchIdx = static_cast<uint16_t>(tokenIdx * fullBatchCount + batchIdx);
            __ubuf__ uint8_t *batchScratch =
                scratch + static_cast<uint32_t>(scratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(batchScratch);
            __ubuf__ float *row0LowScratch = dataScratch;
            __ubuf__ float *row0HighScratch = dataScratch + QKV_K_SCALE_D128_HALF_SIZE;
            __ubuf__ float *row1LowScratch = dataScratch + QKV_K_SCALE_D128_FULL_SIZE;
            __ubuf__ float *row1HighScratch = dataScratch + QKV_K_SCALE_D128_FULL_SIZE + QKV_K_SCALE_D128_HALF_SIZE;
            __ubuf__ uint32_t *maxScratch =
                reinterpret_cast<__ubuf__ uint32_t *>(batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES);
            __ubuf__ float *rmsSquaredScratch =
                reinterpret_cast<__ubuf__ float *>(batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES +
                                                   MROPE_MX_ROW_BATCH_MAX_BYTES + MROPE_MX_ROW_BATCH_RECIP_BYTES);
            __ubuf__ bfloat16_t *batchInput =
                tokenInput + static_cast<uint32_t>(batchIdx) * MROPE_MX_ROW_BATCH_ROWS * inputHeadStride;
            for (uint16_t pairIdx = 0U; pairIdx < 8U; ++pairIdx) {
                MicroAPI::AddrReg inputAddr = MicroAPI::CreateAddrReg<bfloat16_t>(pairIdx, 2U * inputHeadStride);
                MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
                MicroAPI::AddrReg maxAddr = MicroAPI::CreateAddrReg<uint32_t>(pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
                MicroAPI::AddrReg rmsAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 8U);
                StageMropeMxHeadPairDeferredRmsD128(
                    batchInput, row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch, maxScratch,
                    rmsSquaredScratch, inputAddr, dataAddr, maxAddr, rmsAddr, inputHeadStride, gammaCosLow,
                    gammaCosHigh, gammaSinLow, gammaSinHigh, absMask, epsilon, mask64, maskFirst, mask8);
            }
        }

        if constexpr (HAS_Q_TAIL) {
            // Nq%16==8 时，token 2k 写低八行，token 2k+1 写高八行，共用一个 record。
            const uint16_t tailScratchIdx = static_cast<uint16_t>(prefixScratchBatchCount + tokenIdx / 2U);
            __ubuf__ uint8_t *tailScratch =
                scratch + static_cast<uint32_t>(tailScratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            const uint16_t tailRowOffset = static_cast<uint16_t>((tokenIdx & 1U) * 8U);
            __ubuf__ bfloat16_t *tailInput = tokenInput + static_cast<uint32_t>(fullHeadCount) * inputHeadStride;
            const bool isFinalOddToken = hasOddTailToken && tokenIdx + 1U == tokenSize;
            if (isFinalOddToken) {
                // 没有配对 token：不生成批量 max，Quant 时逐行重新归约，避免读取空高半。
                StageQTailToken8D128<false>(tailInput, tailScratch, tailRowOffset, inputHeadStride, gammaCosLow,
                                            gammaCosHigh, gammaSinLow, gammaSinHigh, absMask, epsilon, mask64,
                                            maskFirst, mask8);
            } else {
                StageQTailToken8D128<true>(tailInput, tailScratch, tailRowOffset, inputHeadStride, gammaCosLow,
                                           gammaCosHigh, gammaSinLow, gammaSinHigh, absMask, epsilon, mask64, maskFirst,
                                           mask8);
            }
        }
    }

    // Stage 的 scratch Store 全部完成后，Scale 才能开始 Load。
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    // Phase 2a - Scale 完整前缀：64 lanes 同时完成 16 rows 的 rms/scale/倒数。
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        __ubuf__ uint8_t *tokenScaleBytes = reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
                                            static_cast<uint32_t>(tokenIdx) * scaleTokenStrideWords * sizeof(uint32_t);
        for (uint16_t batchIdx = 0U; batchIdx < fullBatchCount; ++batchIdx) {
            const uint16_t scratchIdx = static_cast<uint16_t>(tokenIdx * fullBatchCount + batchIdx);
            __ubuf__ uint8_t *batchScratch =
                scratch + static_cast<uint32_t>(scratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            __ubuf__ uint32_t *maxScratch =
                reinterpret_cast<__ubuf__ uint32_t *>(batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES);
            __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
            __ubuf__ float *rmsSquaredScratch =
                reinterpret_cast<__ubuf__ float *>(batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES +
                                                   MROPE_MX_ROW_BATCH_MAX_BYTES + MROPE_MX_ROW_BATCH_RECIP_BYTES);
            MxQuantCublasScaleRowBatch16DeferredRms(
                maxScratch, rmsSquaredScratch, reciprocalScratch,
                tokenScaleBytes + static_cast<uint32_t>(batchIdx) * scaleBatchStride, rmsGatherIndex, maskScale,
                maskScaleB16);
        }
    }
    // Phase 2b - Scale 跨 token tail：16 rows 一次计算，scale 分回两个 token。
    for (uint16_t pairIdx = 0U; pairIdx < tailPairCount; ++pairIdx) {
        const uint16_t token0 = static_cast<uint16_t>(pairIdx * 2U);
        const uint16_t token1 = static_cast<uint16_t>(token0 + 1U);
        __ubuf__ uint8_t *tailScratch =
            scratch + static_cast<uint32_t>(prefixScratchBatchCount + pairIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
        __ubuf__ uint32_t *maxScratch =
            reinterpret_cast<__ubuf__ uint32_t *>(tailScratch + MROPE_MX_ROW_BATCH_DATA_BYTES);
        __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
            tailScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
        __ubuf__ uint8_t *scaleBytes0 =
            reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            (static_cast<uint32_t>(token0) * scaleTokenStrideWords + fullHeadCount) * sizeof(uint32_t);
        __ubuf__ uint8_t *scaleBytes1 =
            reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            (static_cast<uint32_t>(token1) * scaleTokenStrideWords + fullHeadCount) * sizeof(uint32_t);
        MxQuantCublasScaleRowBatch16Split(maxScratch, reciprocalScratch, scaleBytes0, scaleBytes1, scaleHighGatherIndex,
                                          maskScale, maskScaleB16, maskScaleHalfB16);
    }

    // 只有批量 Scale 写过倒数时才需要这条 producer -> consumer 屏障。
    if (prefixScratchBatchCount != 0U || tailPairCount != 0U) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
    // Phase 3a - Quant 完整前缀：每次处理一对 head，按自然 row-major 写 E4M3。
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        __ubuf__ uint8_t *tokenOutBytes = reinterpret_cast<__ubuf__ uint8_t *>(out) +
                                          static_cast<uint32_t>(tokenIdx) * headSize * QKV_K_SCALE_D128_FULL_SIZE;
        for (uint16_t batchIdx = 0U; batchIdx < fullBatchCount; ++batchIdx) {
            const uint16_t scratchIdx = static_cast<uint16_t>(tokenIdx * fullBatchCount + batchIdx);
            __ubuf__ uint8_t *batchScratch =
                scratch + static_cast<uint32_t>(scratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(batchScratch);
            __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
            MxQuantCublasDataRowBatch16(
                dataScratch, reciprocalScratch,
                tokenOutBytes + static_cast<uint32_t>(batchIdx) * MROPE_MX_ROW_BATCH_ROWS * QKV_K_SCALE_D128_FULL_SIZE,
                8U, mask64, maskFp8);
        }
    }
    // Phase 3b - Quant tail：同一 RowBatch16 的低/高八行分别写回 token0/token1。
    for (uint16_t pairIdx = 0U; pairIdx < tailPairCount; ++pairIdx) {
        const uint16_t token0 = static_cast<uint16_t>(pairIdx * 2U);
        const uint16_t token1 = static_cast<uint16_t>(token0 + 1U);
        __ubuf__ uint8_t *tailScratch =
            scratch + static_cast<uint32_t>(prefixScratchBatchCount + pairIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
        __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(tailScratch);
        __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
            tailScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
        __ubuf__ uint8_t *outBytes0 =
            reinterpret_cast<__ubuf__ uint8_t *>(out) +
            (static_cast<uint32_t>(token0) * headSize + fullHeadCount) * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *outBytes1 =
            reinterpret_cast<__ubuf__ uint8_t *>(out) +
            (static_cast<uint32_t>(token1) * headSize + fullHeadCount) * QKV_K_SCALE_D128_FULL_SIZE;
        MxQuantCublasDataRowBatch16(dataScratch, reciprocalScratch, outBytes0, 4U, mask64, maskFp8);
        MxQuantCublasDataRowBatch16(dataScratch + 8U * QKV_K_SCALE_D128_FULL_SIZE,
                                    reciprocalScratch + 8U * MROPE_MX_SCALE_COUNT_D128, outBytes1, 4U, mask64, maskFp8);
    }
    if (hasOddTailToken) {
        // Phase 3c - 最后一个未配对 tail：只处理已初始化的低八行。
        const uint16_t oddToken = static_cast<uint16_t>(tokenSize - 1U);
        __ubuf__ uint8_t *oddScratch =
            scratch + static_cast<uint32_t>(prefixScratchBatchCount + tailPairCount) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
        __ubuf__ uint8_t *oddOutBytes =
            reinterpret_cast<__ubuf__ uint8_t *>(out) +
            (static_cast<uint32_t>(oddToken) * headSize + fullHeadCount) * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *oddScaleBytes =
            reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            (static_cast<uint32_t>(oddToken) * scaleTokenStrideWords + fullHeadCount) * sizeof(uint32_t);
        QuantizeMropeMxRowBatch8AfterReady(reinterpret_cast<__ubuf__ float *>(oddScratch), oddOutBytes, oddScaleBytes,
                                           mask64);
    }
    // Quant 对 scratch 的 Load 完成后，上层才能让下一 tile 覆盖同一 UB 区域。
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

// K RowBatch16 的单-token Stage。headSize 必须为偶数；每次处理两个 heads，
// 将该 token 的数据放到当前 16-row record 中由 tokenRowBegin 指定的连续行区间。
__simd_callee__ inline void StageKMropeMxEvenTokenD128(
    __ubuf__ bfloat16_t *tokenInputBf16, __ubuf__ float *rawToken, __ubuf__ float *tokenDataScratch,
    __ubuf__ uint32_t *tokenMaxScratch, __ubuf__ float *tokenRmsSquaredScratch, uint16_t headSize,
    uint32_t inputHeadStride, float epsilon, MicroAPI::RegTensor<uint32_t> &mropeIndex,
    MicroAPI::RegTensor<uint32_t> &absMask, MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst, MicroAPI::MaskReg mask8)
{
    // cos/sin 只按 token 准备一次，随后被该 token 的所有 K heads 复用。
    MicroAPI::RegTensor<float> gammaCosLow;
    MicroAPI::RegTensor<float> gammaCosHigh;
    MicroAPI::RegTensor<float> gammaSinLow;
    MicroAPI::RegTensor<float> gammaSinHigh;
    PrepareMropeGammaD128(gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, gammaLow, gammaHigh, mropeIndex,
                          rawToken, mask64);

    const uint16_t pairCount = static_cast<uint16_t>(headSize / 2U);
    for (uint16_t pairIdx = 0U; pairIdx < pairCount; ++pairIdx) {
        MicroAPI::AddrReg inputAddr = MicroAPI::CreateAddrReg<bfloat16_t>(pairIdx, 2U * inputHeadStride);
        MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::AddrReg maxAddr = MicroAPI::CreateAddrReg<uint32_t>(pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
        MicroAPI::AddrReg rmsAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 8U);
        StageMropeMxHeadPairDeferredRmsD128(
            tokenInputBf16, tokenDataScratch, tokenDataScratch + QKV_K_SCALE_D128_HALF_SIZE,
            tokenDataScratch + QKV_K_SCALE_D128_FULL_SIZE,
            tokenDataScratch + QKV_K_SCALE_D128_FULL_SIZE + QKV_K_SCALE_D128_HALF_SIZE, tokenMaxScratch,
            tokenRmsSquaredScratch, inputAddr, dataAddr, maxAddr, rmsAddr, inputHeadStride, gammaCosLow, gammaCosHigh,
            gammaSinLow, gammaSinHigh, absMask, epsilon, mask64, maskFirst, mask8);
    }
}

__simd_callee__ inline void FinishKMropeMxD128EvenRows(__ubuf__ float *dataScratch, __ubuf__ uint32_t *maxScratch,
                                                       __ubuf__ float *rmsSquaredScratch,
                                                       __ubuf__ float *reciprocalScratch,
                                                       __ubuf__ uint8_t *batchOutBytes,
                                                       __ubuf__ uint8_t *batchPaddedScaleBytes, uint16_t tokenCount,
                                                       uint16_t headSize, MicroAPI::RegTensor<uint32_t> &rmsGatherIndex,
                                                       MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFp8)
{
    // Stage-ready 后完成一个 K RowBatch16：先生成 scale/倒数，再量化 16 行。
    // Nk=8 时两个 token 恰好形成连续 64B scale；Nk=2/4 需写入每 token 的 padded pitch。
    MicroAPI::MaskReg maskScale = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    uint32_t batchScaleCount = MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg maskScaleB16 = MicroAPI::UpdateMask<uint16_t>(batchScaleCount);
    if (headSize == 8U) {
        MxQuantCublasScaleRowBatch16DeferredRms(maxScratch, rmsSquaredScratch, reciprocalScratch, batchPaddedScaleBytes,
                                                rmsGatherIndex, maskScale, maskScaleB16);
    } else {
        MxQuantCublasScaleRowBatch16DeferredRmsToPadded(maxScratch, rmsSquaredScratch, reciprocalScratch,
                                                        batchPaddedScaleBytes, tokenCount, headSize, rmsGatherIndex,
                                                        maskScale, maskScaleB16);
    }
    // Scale 写倒数完成后，Quant 才能加载；这是 record 内第二个依赖边。
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    MxQuantCublasDataRowBatch16(dataScratch, reciprocalScratch, batchOutBytes, 8U, mask64, maskFp8);
}

__simd_callee__ inline void ProcessKMropeMxD128RowBatch16(
    __ubuf__ bfloat16_t *batchInputBf16, __ubuf__ float *batchRawCosSin, __ubuf__ uint8_t *batchOutBytes,
    __ubuf__ uint8_t *batchPaddedScaleBytes, __ubuf__ float *dataScratch, __ubuf__ uint32_t *maxScratch,
    __ubuf__ float *reciprocalScratch, __ubuf__ float *rmsSquaredScratch, uint16_t batchTokenCount, uint16_t headSize,
    uint32_t inputTokenStride, uint32_t inputHeadStride, uint32_t scaleTokenStrideWords, float epsilon,
    MicroAPI::RegTensor<uint32_t> &mropeIndex, MicroAPI::RegTensor<uint32_t> &absMask,
    MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh,
    MicroAPI::RegTensor<uint32_t> &rmsGatherIndex, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst,
    MicroAPI::MaskReg mask8, MicroAPI::MaskReg maskFp8)
{
    // 一个 K batch 始终填满 16 rows：batchTokenCount * headSize == 16。
    // 例如 Nk=8/4/2 分别合并 2/4/8 个 token。tokenRowBegin 保持最终 row-major 顺序。
    // Phase 1 - Stage 各 token，共享同一 data/max/rms^2 scratch record。
    for (uint16_t localToken = 0U; localToken < batchTokenCount; ++localToken) {
        const uint16_t tokenRowBegin = static_cast<uint16_t>(localToken * headSize);
        __ubuf__ bfloat16_t *tokenInputBf16 = batchInputBf16 + static_cast<uint32_t>(localToken) * inputTokenStride;
        __ubuf__ float *tokenDataScratch =
            dataScratch + static_cast<uint32_t>(tokenRowBegin) * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint32_t *tokenMaxScratch =
            maxScratch + static_cast<uint32_t>(tokenRowBegin) * MROPE_MX_SCALE_COUNT_D128;
        __ubuf__ float *tokenRmsSquaredScratch = rmsSquaredScratch + static_cast<uint32_t>(tokenRowBegin) * 4U;
        __ubuf__ float *rawToken = batchRawCosSin + static_cast<uint32_t>(localToken) * 3U * QKV_K_SCALE_D128_FULL_SIZE;
        StageKMropeMxEvenTokenD128(tokenInputBf16, rawToken, tokenDataScratch, tokenMaxScratch, tokenRmsSquaredScratch,
                                   headSize, inputHeadStride, epsilon, mropeIndex, absMask, gammaLow, gammaHigh, mask64,
                                   maskFirst, mask8);
    }

    // Phase 2/3 - 等待全部 Stage Store，随后一次 Scale + Quant 完成整个 RowBatch16。
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    FinishKMropeMxD128EvenRows(dataScratch, maxScratch, rmsSquaredScratch, reciprocalScratch, batchOutBytes,
                               batchPaddedScaleBytes, batchTokenCount, headSize, rmsGatherIndex, mask64, maskFp8);
    // Quant 已读完 scratch，下一组 token 才能覆盖该 record。
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

__simd_vf__ inline void KRmsNormMropeMxD128RowBatch16EvenVfImpl(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma, __ubuf__ float *rawCosSin, __ubuf__ uint32_t *gatherIndex,
    __ubuf__ fp8_e4m3fn_t *out, __ubuf__ fp8_e8m0_t *outScale, __ubuf__ uint8_t *scratch, uint16_t tokenSize,
    uint16_t headSize, uint32_t inputTokenStride, uint32_t inputHeadStride, uint32_t scaleTokenStrideWords,
    float epsilon)
{
    // 偶数 Nk 且 16%Nk==0 的批量入口。它只保留一个 RowBatch16 scratch，并按 token 组循环复用。
    MicroAPI::RegTensor<uint32_t> mropeIndex;
    MicroAPI::RegTensor<uint32_t> absMask;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t mask8Count = 8U;
    MicroAPI::MaskReg mask8 = MicroAPI::UpdateMask<uint32_t>(mask8Count);
    MicroAPI::MaskReg maskFp8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::LoadAlign<uint32_t>(mropeIndex, gatherIndex);
    MicroAPI::LoadAlign<float>(gammaLow, gamma);
    MicroAPI::LoadAlign<float>(gammaHigh, gamma + QKV_K_SCALE_D128_HALF_SIZE);
    MicroAPI::Duplicate(absMask, MROPE_MX_FP32_ABS_MASK);
    MicroAPI::RegTensor<uint32_t> rmsGatherIndex;
    MicroAPI::RegTensor<uint32_t> rmsGatherBase;
    MicroAPI::RegTensor<uint32_t> rmsGatherOne;
    MicroAPI::MaskReg maskScale = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)rmsGatherIndex, 0);
    MicroAPI::ShiftRights(rmsGatherBase, rmsGatherIndex, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftLefts(rmsGatherBase, rmsGatherBase, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftRights(rmsGatherIndex, rmsGatherIndex, static_cast<int16_t>(2), maskScale);
    MicroAPI::Duplicate(rmsGatherOne, 1U);
    MicroAPI::And(rmsGatherIndex, rmsGatherIndex, rmsGatherOne, maskScale);
    MicroAPI::Add(rmsGatherIndex, rmsGatherBase, rmsGatherIndex, maskScale);

    __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(scratch);
    __ubuf__ uint32_t *maxScratch =
        reinterpret_cast<__ubuf__ uint32_t *>(scratch + static_cast<uint32_t>(MROPE_MX_ROW_BATCH_DATA_BYTES));
    __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
        scratch + static_cast<uint32_t>(MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES));
    __ubuf__ float *rmsSquaredScratch = reinterpret_cast<__ubuf__ float *>(
        scratch + static_cast<uint32_t>(MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES +
                                        MROPE_MX_ROW_BATCH_RECIP_BYTES));
    __ubuf__ uint8_t *outBytes = reinterpret_cast<__ubuf__ uint8_t *>(out);
    __ubuf__ uint8_t *paddedScaleBytes = reinterpret_cast<__ubuf__ uint8_t *>(outScale);
    // 选择能恰好拼成 16 行的 token 数；不会在热循环中处理不完整 batch。
    const uint16_t tokensPerBatch = static_cast<uint16_t>(MROPE_MX_ROW_BATCH_ROWS / headSize);

    uint16_t batchTokenBegin = 0U;
    while (static_cast<uint16_t>(tokenSize - batchTokenBegin) >= tokensPerBatch) {
        const uint16_t batchTokenCount = tokensPerBatch;
        __ubuf__ bfloat16_t *batchInputBf16 = inputBf16 + static_cast<uint32_t>(batchTokenBegin) * inputTokenStride;
        __ubuf__ float *batchRawCosSin =
            rawCosSin + static_cast<uint32_t>(batchTokenBegin) * 3U * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *batchOutBytes =
            outBytes + static_cast<uint32_t>(batchTokenBegin) * headSize * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *batchPaddedScaleBytes =
            paddedScaleBytes + static_cast<uint32_t>(batchTokenBegin) * scaleTokenStrideWords * sizeof(uint32_t);
        ProcessKMropeMxD128RowBatch16(batchInputBf16, batchRawCosSin, batchOutBytes, batchPaddedScaleBytes, dataScratch,
                                      maxScratch, reciprocalScratch, rmsSquaredScratch, batchTokenCount, headSize,
                                      inputTokenStride, inputHeadStride, scaleTokenStrideWords, epsilon, mropeIndex,
                                      absMask, gammaLow, gammaHigh, rmsGatherIndex, mask64, maskFirst, mask8, maskFp8);
        batchTokenBegin = static_cast<uint16_t>(batchTokenBegin + batchTokenCount);
    }
}
__simd_vf__ inline void KRmsNormMropeMxD128VfImpl(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma,
                                                  __ubuf__ float *rawCosSin, __ubuf__ uint32_t *gatherIndex,
                                                  __ubuf__ fp8_e4m3fn_t *out, __ubuf__ fp8_e8m0_t *outScale,
                                                  __ubuf__ uint8_t *scratch, uint16_t tokenSize, uint16_t headSize,
                                                  uint32_t inputTokenStride, uint32_t inputHeadStride,
                                                  uint32_t scaleTokenStrideWords, float epsilon)
{
    // 通用逐 head 回退：不使用 RowBatch16 scratch。每行先完整 RMSNorm+M-RoPE，
    // 再立即算四个 scale 并量化。它覆盖不能安全拼成 16-row batch 的 K shape。
    MicroAPI::RegTensor<uint32_t> mropeIndex;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::LoadAlign<uint32_t>(mropeIndex, gatherIndex);
    MicroAPI::LoadAlign<float>(gammaLow, gamma);
    MicroAPI::LoadAlign<float>(gammaHigh, gamma + QKV_K_SCALE_D128_HALF_SIZE);

    (void)scratch;

    MicroAPI::RegTensor<uint32_t> scaleGatherLowIndex;
    MicroAPI::RegTensor<uint32_t> scaleGatherHighIndex;
    MicroAPI::MaskReg maskLow32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg maskHigh32;
    MicroAPI::Not(maskHigh32, maskLow32, mask64);
    uint32_t scaleCountPerHead = 4U;
    MicroAPI::MaskReg mask4 = MicroAPI::UpdateMask<float>(scaleCountPerHead);
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)scaleGatherLowIndex, 0);
    MicroAPI::ShiftRights(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(5), mask64);
    MicroAPI::ShiftLefts(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(1), mask64);
    MicroAPI::Adds(scaleGatherHighIndex, scaleGatherLowIndex, 1U, mask64);
    // token 外层保证 gamma*cos/sin 只准备一次；head 内层保持输入/输出自然布局。
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        MicroAPI::RegTensor<float> gammaCosLow;
        MicroAPI::RegTensor<float> gammaCosHigh;
        MicroAPI::RegTensor<float> gammaSinLow;
        MicroAPI::RegTensor<float> gammaSinHigh;
        __ubuf__ float *rawToken = rawCosSin + static_cast<uint32_t>(tokenIdx) * 3U * QKV_K_SCALE_D128_FULL_SIZE;
        PrepareMropeGammaD128(gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, gammaLow, gammaHigh, mropeIndex,
                              rawToken, mask64);
        for (uint16_t headIdx = 0U; headIdx < headSize; ++headIdx) {
            MicroAPI::RegTensor<float> outLowFp32;
            MicroAPI::RegTensor<float> outHighFp32;
            MicroAPI::AddrReg inputAddr =
                MicroAPI::CreateAddrReg<bfloat16_t>(tokenIdx, inputTokenStride, headIdx, inputHeadStride);
            LoadRmsNormMropeFp32D128(outLowFp32, outHighFp32, inputBf16, inputAddr, gammaCosLow, gammaCosHigh,
                                     gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
            MicroAPI::AddrReg outAddr =
                MicroAPI::CreateAddrReg<uint8_t>(tokenIdx, static_cast<uint32_t>(headSize) * QKV_K_SCALE_D128_FULL_SIZE,
                                                 headIdx, QKV_K_SCALE_D128_FULL_SIZE);
            __ubuf__ uint8_t *scaleBytes =
                reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
                (static_cast<uint32_t>(tokenIdx) * scaleTokenStrideWords + headIdx) * sizeof(uint32_t);
            MxQuantCublasD128(reinterpret_cast<__ubuf__ uint8_t *>(out), outAddr, scaleBytes, outLowFp32, outHighFp32,
                              scaleGatherLowIndex, scaleGatherHighIndex, mask64, maskLow32, maskHigh32, mask4);
        }
    }
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VF_H_

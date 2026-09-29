/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_SOFTMAX_FP16_TO_FP8_H_
#define VF_SOFTMAX_FP16_TO_FP8_H_

#include "kernel_tensor.h"
#include "vf_common_def_mxfp8_softmax_fp16.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief V1 softmax 主体（__simd_vf__ 内部函数，由 __aicore__ wrapper 调用）
 *
 * @param [out] pDst        NZ 源布局 P（vec1PUb[subLoopIdx]）
 * @param [in,out] accMax   accMax 槽（softmaxMaxUB[stateSlot]，half log2 整数）
 * @param [out] usedMaxOut  本 subLoop k_used（subLoopUsedMaxUB[subLoopIdx]）
 * @param [in] nzIdx        Gather 索引表（pNzIdxUB）
 * @param [in] x            mm1Res 输入（[validRows, colStride] half）
 * @param [in] pScale       外部 headroom 因子（half 标量，默认 1.0）
 * @param [in] validRows    有效 S2 行数（1..128）
 * @param [in] colStride    UB 行跨步 = 128（FixpipeMm1 固定 pitch）
 */
template <typename T = half, typename T2 = fp8_e4m3fn_t>
__simd_vf__ inline void VfSoftmaxFp16ToFp8VF(__ubuf__ T2* pDst, __ubuf__ half* accMax, __ubuf__ half* usedMaxOut,
                                             __ubuf__ uint8_t* nzIdx, __ubuf__ T* x, half pScale, uint32_t validRows,
                                             uint32_t colStride)
{
    // ===== 寄存器声明（一行 128 half = 1 寄存器；命名对齐参考下划线风格）=====
    RegTensor<T> max0, max1, max2, max3;                 // 4 路流式 max 累加器（各收行 i%4 == 路号）
    RegTensor<T> r0, r1, r2, r3;                         // 行加载临时（归约与 P 循环复用）
    RegTensor<T> vreg_cur_max;                           // 树归并结果（[128] 列 max）
    RegTensor<T> vreg_k_cur, vreg_k_used, vreg_norm_max; // 量化指数（half 整数）与 P 归一化基准
    RegTensor<T> vreg_acc_max_load;                      // accMax 槽加载
    RegTensor<T2> vreg_quant_a, vreg_quant_b;            // fp8 打包寄存器（各含 2 行 × 128B）
    RegTensor<float> f_tmp0, f_tmp1; // half→float 中转寄存器（各 64 float，h2iZero/h2iOne 拆偶/奇元素）
    RegTensor<T2> fp8_tmp;           // float→fp8 中转寄存器（TWO/THREE 布局临时）
    RegTensor<uint8_t> vreg_idx;     // 解交织索引表（循环外载入一次，全程复用）
    MaskReg preg_half = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_float = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_u8 = CreateMask<uint8_t, MaskPattern::ALL>();
    // 半掩码对（前半 = 行 i 的 4 单元，后半 = 行 i+1；MxFP4 preg_vl128/_not 同款）
    uint32_t halfLaneCount = 128; // UpdateMask 收左值引用，须用变量
    MaskReg preg_u8_first = UpdateMask<uint8_t>(halfLaneCount);
    MaskReg preg_u8_second;
    MaskNot(preg_u8_second, preg_u8_first, preg_u8);

    // 解交织表载入（循环外一次，全程复用；表由 InitTensorsVec 初始化 idx[i*128+j]=i+2j）
    LoadAlign(vreg_idx, nzIdx);

    Duplicate(max0, MIN_VALUE);
    Duplicate(max1, MIN_VALUE);
    Duplicate(max2, MIN_VALUE);
    Duplicate(max3, MIN_VALUE);
    for (uint32_t i = 0; i + 3 < validRows; i += 4) {
        LoadAlign(r0, x + i * colStride);
        LoadAlign(r1, x + (i + 1) * colStride);
        LoadAlign(r2, x + (i + 2) * colStride);
        LoadAlign(r3, x + (i + 3) * colStride);
        Max(max0, max0, r0, preg_half);
        Max(max1, max1, r1, preg_half);
        Max(max2, max2, r2, preg_half);
        Max(max3, max3, r3, preg_half);
    }

    // 尾行（validRows % 4 余量）逐行并入 max0——max 交换结合律，不破坏归并树
    for (uint32_t i = validRows & ~3U; i < validRows; i++) {
        LoadAlign(r0, x + i * colStride);
        Max(max0, max0, r0, preg_half);
    }
    Max(max0, max0, max2, preg_half);
    Max(max1, max1, max3, preg_half);
    Max(vreg_cur_max, max0, max1, preg_half);

    Muls(vreg_k_cur, vreg_cur_max, INV_LN2, preg_half);                   // ×1/ln2 → log2 域
    Truncate<T, RoundMode::CAST_CEIL>(vreg_k_cur, vreg_k_cur, preg_half); // CEIL → 整数 k_cur
    LoadAlign(vreg_acc_max_load, accMax);                                 // accMax 槽（half 整数）
    Max(vreg_k_used, vreg_k_cur, vreg_acc_max_load, preg_half);           // 合并保持整数
    StoreAlign(accMax, vreg_k_used, preg_half);                           // 写回 accMax（下块用）
    StoreAlign(usedMaxOut, vreg_k_used, preg_half);                       // 暂存 k_i（epilogue pscale 用）
    Muls(vreg_norm_max, vreg_k_used, LN2, preg_half);                     // k·ln2 → P 归一化基准

    for (uint32_t i = 0; i + 3 < validRows; i += 4) {
        LoadAlign(r0, x + i * colStride);
        LoadAlign(r1, x + (i + 1) * colStride);
        LoadAlign(r2, x + (i + 2) * colStride);
        LoadAlign(r3, x + (i + 3) * colStride);
        Sub(r0, r0, vreg_norm_max,
            preg_half); // 减 norm_max 向量：[S2,S1] 行主序，norm_max 恰 [128] 列向量，形状天然对齐
        Sub(r1, r1, vreg_norm_max, preg_half);
        Sub(r2, r2, vreg_norm_max, preg_half);
        Sub(r3, r3, vreg_norm_max, preg_half);
        Exp(r0, r0, preg_half); // 原地 exp（省寄存器，mxfp4 同款）
        Exp(r1, r1, preg_half);
        Exp(r2, r2, preg_half);
        Exp(r3, r3, preg_half);
        Muls(r0, r0, pScale, preg_half); // × 外部 headroom（默认 1.0 时 no-op；分子分母同乘自消）
        Muls(r1, r1, pScale, preg_half);
        Muls(r2, r2, pScale, preg_half);
        Muls(r3, r3, pScale, preg_half);
        // fp8 成对窄化：half → float → fp8（硬件不支持 half→fp8 直接 Cast，经 float 中转）
        // half(16b) --h2iZero/h2iZero--> float(32b, 64元素) --castTraitRintZero/Two/One/Three--> fp8(8b)
        // 行 i 落偶字节位 [0,2,...,254]（ZERO+TWO），行 i+1 落奇字节位 [1,3,...,255]（ONE+THREE）
        // —— 第一对 (行 i = r0, 行 i+1 = r1) ——
        Cast<float, T, h2iZero>(f_tmp0, r0, preg_half);
        Cast<float, T, h2iOne>(f_tmp1, r0, preg_half);
        Cast<T2, float, castTraitRintZero>(vreg_quant_a, f_tmp0, preg_float);
        Cast<T2, float, castTraitRintTwo>(fp8_tmp, f_tmp1, preg_float);
        Or((RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)fp8_tmp, preg_u8);
        Cast<float, T, h2iZero>(f_tmp0, r1, preg_half);
        Cast<float, T, h2iOne>(f_tmp1, r1, preg_half);
        Cast<T2, float, castTraitRintOne>(vreg_quant_b, f_tmp0, preg_float);
        Cast<T2, float, castTraitRintThree>(fp8_tmp, f_tmp1, preg_float);
        Or((RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)fp8_tmp, preg_u8);
        // 异源配对交织 → Gather 解交织 → 半掩码双散布
        Or((RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_b,
           preg_u8);
        // 解交织：[a0,b0,a1,b1,…] → [行i | 行i+1]（各 128B 顺序，前半行 i、后半行 i+1）
        Gather((RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_a, vreg_idx);
        // 存 A：行 i 的 4 个列组单元（寄存器前半）→ [j×4096B + i×32B]，j=0..3
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pDst + i * 32, (RegTensor<T2>&)vreg_quant_a, 128,
                                                           preg_u8_first);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pDst - 4 * 4096 + (i + 1) * 32, (RegTensor<T2>&)vreg_quant_a,
                                                           128, preg_u8_second);
        // —— 第二对 (行 i+2 = r2, 行 i+3 = r3) ——
        Cast<float, T, h2iZero>(f_tmp0, r2, preg_half);
        Cast<float, T, h2iOne>(f_tmp1, r2, preg_half);
        Cast<T2, float, castTraitRintZero>(vreg_quant_b, f_tmp0, preg_float);
        Cast<T2, float, castTraitRintTwo>(fp8_tmp, f_tmp1, preg_float);
        Or((RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)fp8_tmp, preg_u8);
        Cast<float, T, h2iZero>(f_tmp0, r3, preg_half);
        Cast<float, T, h2iOne>(f_tmp1, r3, preg_half);
        Cast<T2, float, castTraitRintOne>(fp8_tmp, f_tmp0, preg_float);
        Cast<T2, float, castTraitRintThree>((RegTensor<T2>&)r0, f_tmp1, preg_float); // r0 已消费，复用为 fp8 目标
        Or((RegTensor<uint8_t>&)fp8_tmp, (RegTensor<uint8_t>&)fp8_tmp, (RegTensor<uint8_t>&)r0, preg_u8);
        // 第二对同构：交织 → 解交织 → 行 i+2 / i+3 半掩码双散布
        Or((RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)fp8_tmp, preg_u8);
        Gather((RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)vreg_quant_b, vreg_idx);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pDst + (i + 2) * 32, (RegTensor<T2>&)vreg_quant_b, 128,
                                                           preg_u8_first);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pDst - 4 * 4096 + (i + 3) * 32, (RegTensor<T2>&)vreg_quant_b,
                                                           128, preg_u8_second);
    }

    for (uint32_t i = validRows & ~3U; i < validRows; i += 2U) {
        LoadAlign(r0, x + i * colStride);
        Sub(r0, r0, vreg_norm_max, preg_half);
        Exp(r0, r0, preg_half);
        Muls(r0, r0, pScale, preg_half);
        Cast<float, T, h2iZero>(f_tmp0, r0, preg_half);
        Cast<float, T, h2iOne>(f_tmp1, r0, preg_half);
        Cast<T2, float, castTraitRintZero>(vreg_quant_a, f_tmp0, preg_float);
        Cast<T2, float, castTraitRintTwo>(fp8_tmp, f_tmp1, preg_float);
        Or((RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)fp8_tmp, preg_u8);
        if (i + 1U < validRows) {
            LoadAlign(r1, x + (i + 1U) * colStride);
            Sub(r1, r1, vreg_norm_max, preg_half);
            Exp(r1, r1, preg_half);
            Muls(r1, r1, pScale, preg_half);
            Cast<float, T, h2iZero>(f_tmp0, r1, preg_half);
            Cast<float, T, h2iOne>(f_tmp1, r1, preg_half);
            Cast<T2, float, castTraitRintOne>(vreg_quant_b, f_tmp0, preg_float);
            Cast<T2, float, castTraitRintThree>(fp8_tmp, f_tmp1, preg_float);
            Or((RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)vreg_quant_b, (RegTensor<uint8_t>&)fp8_tmp,
               preg_u8);
            Or((RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_b,
               preg_u8);
        }
        Gather((RegTensor<uint8_t>&)vreg_quant_a, (RegTensor<uint8_t>&)vreg_quant_a, vreg_idx);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pDst + i * 32, (RegTensor<T2>&)vreg_quant_a, 128,
                                                           preg_u8_first);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pDst - 4 * 4096 + (i + 1) * 32, (RegTensor<T2>&)vreg_quant_a,
                                                           128, preg_u8_second);
    }
}

/*
 * @ingroup VfaVectorApi
 * @brief V1 softmax 主体（__aicore__ wrapper，接收 LocalTensor，内部转指针调 vf）
 *
 * 各参数的槽位偏移由本 wrapper 内部计算，编排层（ComputeVec1）只传 tensor + 索引。
 */
template <typename T = half, typename T2 = fp8_e4m3fn_t>
__aicore__ inline void VfSoftmaxFp16ToFp8(const LocalTensor<T2>& pDst, const LocalTensor<half>& accMax,
                                          const LocalTensor<half>& usedMaxOut, const LocalTensor<uint8_t>& nzIdx,
                                          const LocalTensor<T>& x, half pScale, uint32_t validRows, uint32_t subLoopIdx,
                                          uint32_t mmSlot, uint32_t stateSlot)
{
    constexpr uint32_t s2SubLoopSize = 128;
    constexpr uint32_t s1HalfSize = 128;
    constexpr uint32_t smaxSlot = 128;
    constexpr uint32_t mm1ResSlot = 128 * 128;
    constexpr uint32_t vec1PSlot = s2SubLoopSize * s1HalfSize;

    __ubuf__ T2* pDstPtr = reinterpret_cast<__ubuf__ T2*>(pDst.GetPhyAddr()) + subLoopIdx * vec1PSlot;
    __ubuf__ half* accMaxPtr = reinterpret_cast<__ubuf__ half*>(accMax.GetPhyAddr()) + stateSlot * smaxSlot;
    __ubuf__ half* usedMaxOutPtr = reinterpret_cast<__ubuf__ half*>(usedMaxOut.GetPhyAddr()) + subLoopIdx * smaxSlot;
    __ubuf__ uint8_t* nzIdxPtr = reinterpret_cast<__ubuf__ uint8_t*>(nzIdx.GetPhyAddr());
    __ubuf__ T* xPtr = reinterpret_cast<__ubuf__ T*>(x.GetPhyAddr()) + mmSlot * mm1ResSlot;

    VfSoftmaxFp16ToFp8VF<T, T2>(pDstPtr, accMaxPtr, usedMaxOutPtr, nzIdxPtr, xPtr, pScale, validRows, s1HalfSize);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_SOFTMAX_FP16_TO_FP8_H_

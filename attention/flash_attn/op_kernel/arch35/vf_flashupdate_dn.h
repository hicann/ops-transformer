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
 * \file vf_flashupdate_dn.h
 * \brief 不合轴 DN(template 2) 的 VEC2 向量化实现（累加器为 DN 布局 [dV][m]，每 AIV m=64）。
 *
 *   1) FlashUpdateDnVF          acc = acc*expMax + mm2（除首轮外每一轮，含尾轮；update 只在这里做）
 *   2) FlashUpdateDnLastTransVF 归一 + Cast 配对 + 3 轮 Interleave + DATA_BLOCK_COPY 散写（非 FD 尾轮）
 *   3) FlashOutputDnWorkspaceVF Gather 转置 + 归一，输出 fp32 行主序（FD split-KV 核尾轮）
 *
 * 说明：
 *   - 累加器地址：acc[d*64 + m]（d = dV 列，m = 64 行）；一列 = 256B。
 *   - 尾轮输出布局：[m][dRowND]，dRowND = dTrans + 16（每行多 1 个 32B pad 块），
 *     dTrans = Align16(dSizeV)（D=72 时为 80，组数 5）。
 *   - 行无效：不用模板、不用 if 分支，统一用 0/1 次 for 循环承载（VF 的 for 循环体内不允许运行时 if）。
 */

#ifndef FLASH_ATTN_VF_FLASHUPDATE_DN_H_
#define FLASH_ATTN_VF_FLASHUPDATE_DN_H_

#include "kernel_tensor.h"
#include "../../../common/op_kernel/arch35/vf/vf_basic_block_utils.h"

namespace FlashAttnKernel {
using namespace AscendC;
using namespace Reg;
using namespace FaVectorApi;

/*!
 * \brief acc[d][m] = acc[d][m] * expMax[m] + mm2[d][m]
 * \param acc      DN 累加器（dV * 64 个 float）
 * \param cur      BMM2 本轮结果（DN，dV * 64 个 float）
 * \param expMax   64 个 m 的 rescale 因子
 * \param rows     有效行数（<= 64）
 * \param columns  有效列数（dTrans = 64/80/128，可被 4 整除）
 */
__simd_vf__ inline void FlashUpdateDnVF(__ubuf__ float* acc, __ubuf__ float* cur, __ubuf__ float* expMax, uint32_t rows,
                                        uint16_t columns)
{
    MaskReg mr = UpdateMask<float>(rows);
    RegTensor<float> s;
    LoadAlign(s, expMax);

    uint16_t d = 0;
    for (; d + 4 <= columns; d += 4) {
        RegTensor<float> a0, a1, a2, a3, c0, c1, c2, c3;
        LoadAlign(a0, acc + (d + 0) * 64);
        LoadAlign(c0, cur + (d + 0) * 64);
        LoadAlign(a1, acc + (d + 1) * 64);
        LoadAlign(c1, cur + (d + 1) * 64);
        LoadAlign(a2, acc + (d + 2) * 64);
        LoadAlign(c2, cur + (d + 2) * 64);
        LoadAlign(a3, acc + (d + 3) * 64);
        LoadAlign(c3, cur + (d + 3) * 64);
        MulAddDst(c0, a0, s, mr);
        MulAddDst(c1, a1, s, mr);
        MulAddDst(c2, a2, s, mr);
        MulAddDst(c3, a3, s, mr);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(acc + (d + 0) * 64, c0, mr);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(acc + (d + 1) * 64, c1, mr);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(acc + (d + 2) * 64, c2, mr);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(acc + (d + 3) * 64, c3, mr);
    }
    for (; d < columns; ++d) {
        RegTensor<float> a, c;
        LoadAlign(a, acc + d * 64);
        LoadAlign(c, cur + d * 64);
        MulAddDst(c, a, s, mr);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(acc + d * 64, c, mr);
    }
}

/*!
 * \brief out[m][d] = acc[d][m] / sum[m]，16bit 输出，行距 dRowND = dTrans + 16
 * \param out         输出缓冲（借用的 vec1 槽），64 * dRowND 个 OUT_T
 * \param acc         DN 累加器（已由 FlashUpdateDnVF 更新）
 * \param sum         64 个 m 的归一因子
 * \param maxv        64 个 m 的 running max（用于空行判定）
 * \param rows        有效行数（<= 64）
 * \param needInvalid 0/1：是否需要做空行清零, HAS_MASK=true时生效
 * \param groups      dTrans / 16
 * \param sentinel    空行哨兵（-FLT_MAX）
 */
template <typename OUT_T, bool HAS_MASK>
__simd_vf__ inline void FlashUpdateDnLastTransVF(__ubuf__ OUT_T* out, __ubuf__ float* acc, __ubuf__ float* sum,
                                                 __ubuf__ float* maxv, uint32_t rows, uint16_t needInvalid,
                                                 uint16_t groups, float sentinel)
{
    constexpr uint16_t HALF_PER_BLOCK = 16; // 32B / sizeof(OUT_T)
    MaskReg mr = UpdateMask<float>(rows);
    MaskReg allF32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg full = CreateMask<OUT_T, MaskPattern::ALL>();
    MaskReg inv;

    RegTensor<float> den, zero;
    LoadAlign(den, sum);
    Duplicate(zero, 0.0f);

    if constexpr (HAS_MASK) {
        for (uint16_t once = 0; once < needInvalid; ++once) {
            RegTensor<float> mx, one;
            LoadAlign(mx, maxv);
            Compares<float, CMPMODE::EQ>(inv, mx, sentinel, mr);
            Duplicate(one, 1.0f);
            Select(den, one, den, inv);
        }
    }

    uint16_t dBlocks = groups + 1;
    for (uint16_t g = 0; g < groups; ++g) {
        RegTensor<float> x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
        LoadAlign(x0, acc + (16 * g + 0) * 64);
        LoadAlign(x1, acc + (16 * g + 1) * 64);
        LoadAlign(x2, acc + (16 * g + 2) * 64);
        LoadAlign(x3, acc + (16 * g + 3) * 64);
        LoadAlign(x4, acc + (16 * g + 4) * 64);
        LoadAlign(x5, acc + (16 * g + 5) * 64);
        LoadAlign(x6, acc + (16 * g + 6) * 64);
        LoadAlign(x7, acc + (16 * g + 7) * 64);
        LoadAlign(x8, acc + (16 * g + 8) * 64);
        LoadAlign(x9, acc + (16 * g + 9) * 64);
        LoadAlign(x10, acc + (16 * g + 10) * 64);
        LoadAlign(x11, acc + (16 * g + 11) * 64);
        LoadAlign(x12, acc + (16 * g + 12) * 64);
        LoadAlign(x13, acc + (16 * g + 13) * 64);
        LoadAlign(x14, acc + (16 * g + 14) * 64);
        LoadAlign(x15, acc + (16 * g + 15) * 64);

        Div(x0, x0, den, mr);
        Div(x1, x1, den, mr);
        Div(x2, x2, den, mr);
        Div(x3, x3, den, mr);
        Div(x4, x4, den, mr);
        Div(x5, x5, den, mr);
        Div(x6, x6, den, mr);
        Div(x7, x7, den, mr);
        Div(x8, x8, den, mr);
        Div(x9, x9, den, mr);
        Div(x10, x10, den, mr);
        Div(x11, x11, den, mr);
        Div(x12, x12, den, mr);
        Div(x13, x13, den, mr);
        Div(x14, x14, den, mr);
        Div(x15, x15, den, mr);

        if constexpr (HAS_MASK) {
            // 空行清零：0/1 次循环，Div 之后，16 列一组（不在 j 循环内逐列判断）
            for (uint16_t once = 0; once < needInvalid; ++once) {
                Select(x0, zero, x0, inv);
                Select(x1, zero, x1, inv);
                Select(x2, zero, x2, inv);
                Select(x3, zero, x3, inv);
                Select(x4, zero, x4, inv);
                Select(x5, zero, x5, inv);
                Select(x6, zero, x6, inv);
                Select(x7, zero, x7, inv);
                Select(x8, zero, x8, inv);
                Select(x9, zero, x9, inv);
                Select(x10, zero, x10, inv);
                Select(x11, zero, x11, inv);
                Select(x12, zero, x12, inv);
                Select(x13, zero, x13, inv);
                Select(x14, zero, x14, inv);
                Select(x15, zero, x15, inv);
            }
        }

        // 两列一组打包进 128 lane：列 c 放在 slot rol4(c,1)（即 (c, c+8) 配对），
        // 这样 3 轮 Interleave 之后的 16 列块内顺序是自然的 0..15（实测 rol4(j,1) 的补偿）
        RegTensor<OUT_T> lo, hi, p0, p1, p2, p3, p4, p5, p6, p7;
        Cast<OUT_T, float, castTraitZero>(lo, x0, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x8, allF32);
        Or((RegTensor<uint16_t>&)p0, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x1, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x9, allF32);
        Or((RegTensor<uint16_t>&)p1, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x2, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x10, allF32);
        Or((RegTensor<uint16_t>&)p2, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x3, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x11, allF32);
        Or((RegTensor<uint16_t>&)p3, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x4, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x12, allF32);
        Or((RegTensor<uint16_t>&)p4, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x5, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x13, allF32);
        Or((RegTensor<uint16_t>&)p5, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x6, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x14, allF32);
        Or((RegTensor<uint16_t>&)p6, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);
        Cast<OUT_T, float, castTraitZero>(lo, x7, allF32);
        Cast<OUT_T, float, castTraitOne>(hi, x15, allF32);
        Or((RegTensor<uint16_t>&)p7, (RegTensor<uint16_t>&)lo, (RegTensor<uint16_t>&)hi, full);

        // 3 轮 butterfly：2 列/64 行 -> 4/32 -> 8/16 -> 16 列/8 行
        Interleave(p0, p4, p0, p4);
        Interleave(p1, p5, p1, p5);
        Interleave(p2, p6, p2, p6);
        Interleave(p3, p7, p3, p7);
        Interleave(p0, p2, p0, p2);
        Interleave(p4, p6, p4, p6);
        Interleave(p1, p3, p1, p3);
        Interleave(p5, p7, p5, p7);
        Interleave(p0, p1, p0, p1);
        Interleave(p2, p3, p2, p3);
        Interleave(p4, p5, p4, p5);
        Interleave(p6, p7, p6, p7);

        // 8 条散写：每条 8 行 x 16 列，块步长 dBlocks，行距 dBlocks*8 个块
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 0) * HALF_PER_BLOCK, p0,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 1) * HALF_PER_BLOCK, p1,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 2) * HALF_PER_BLOCK, p2,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 3) * HALF_PER_BLOCK, p3,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 4) * HALF_PER_BLOCK, p4,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 5) * HALF_PER_BLOCK, p5,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 6) * HALF_PER_BLOCK, p6,
                                                              (uint32_t)dBlocks, full);
        StoreAlign<OUT_T, Reg::DataCopyMode::DATA_BLOCK_COPY>(out + (g + dBlocks * 8 * 7) * HALF_PER_BLOCK, p7,
                                                              (uint32_t)dBlocks, full);
    }
}

/*!
 * \brief out[row][d] = acc[d][row] / sum[row]，fp32 行主序（FD workspace 契约）
 * \param out     输出缓冲（借用的 vec1 槽），64 * stride 个 float
 * \param acc     DN 累加器（已由 FlashUpdateDnVF 更新）
 * \param sum     64 个 m 的归一因子
 * \param rows    有效行数（<= 64）
 * \param columns 有效列数（dSizeV）
 * \param stride  行距（dTemplateAlign64）
 */
__simd_vf__ inline void FlashOutputDnWorkspaceVF(__ubuf__ float* out, __ubuf__ float* acc, __ubuf__ float* sum,
                                                 uint16_t rows, uint16_t columns, uint16_t stride)
{
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<uint32_t> lane, base, idx, ridx;
    Arange((RegTensor<int32_t>&)lane, 0);
    Muls(base, lane, 64U, all);

    for (uint16_t r = 0; r < rows; ++r) {
        RegTensor<float> v, den;
        Duplicate(ridx, static_cast<uint32_t>(r));
        Gather(den, sum, ridx, all);
        for (uint16_t c = 0; c < columns; c += 64) {
            uint32_t cnt = static_cast<uint32_t>(columns - c);
            MaskReg m = UpdateMask<float>(cnt);
            Adds(idx, base, static_cast<uint32_t>(c) * 64U + static_cast<uint32_t>(r), all);
            Gather(v, acc, idx, m);
            Div(v, v, den, m);
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(out + r * stride + c, v, m);
        }
    }
}

/* ------------------------------ __aicore__ 包装 ------------------------------ */

__aicore__ inline void FlashUpdateDn(const LocalTensor<float>& accTensor, const LocalTensor<float>& curTensor,
                                     const LocalTensor<float>& expMaxTensor, uint32_t rows, uint16_t columns)
{
    __ubuf__ float* acc = (__ubuf__ float*)accTensor.GetPhyAddr();
    __ubuf__ float* cur = (__ubuf__ float*)curTensor.GetPhyAddr();
    __ubuf__ float* expMax = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    FlashUpdateDnVF(acc, cur, expMax, rows, columns);
}

template <typename OUT_T, bool HAS_MASK>
__aicore__ inline void FlashUpdateDnLastTrans(const LocalTensor<OUT_T>& outTensor, const LocalTensor<float>& accTensor,
                                              const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor,
                                              uint32_t rows, uint16_t needInvalid, uint16_t groups, float sentinel)
{
    __ubuf__ OUT_T* out = (__ubuf__ OUT_T*)outTensor.GetPhyAddr();
    __ubuf__ float* acc = (__ubuf__ float*)accTensor.GetPhyAddr();
    __ubuf__ float* sum = (__ubuf__ float*)sumTensor.GetPhyAddr();
    __ubuf__ float* maxv = (__ubuf__ float*)maxTensor.GetPhyAddr();
    FlashUpdateDnLastTransVF<OUT_T, HAS_MASK>(out, acc, sum, maxv, rows, needInvalid, groups, sentinel);
}

__aicore__ inline void FlashOutputDnWorkspace(const LocalTensor<float>& outTensor, const LocalTensor<float>& accTensor,
                                              const LocalTensor<float>& sumTensor, uint16_t rows, uint16_t columns,
                                              uint16_t stride)
{
    __ubuf__ float* out = (__ubuf__ float*)outTensor.GetPhyAddr();
    __ubuf__ float* acc = (__ubuf__ float*)accTensor.GetPhyAddr();
    __ubuf__ float* sum = (__ubuf__ float*)sumTensor.GetPhyAddr();
    FlashOutputDnWorkspaceVF(out, acc, sum, rows, columns, stride);
}

} // namespace FlashAttnKernel
#endif // FLASH_ATTN_VF_FLASHUPDATE_DN_H_

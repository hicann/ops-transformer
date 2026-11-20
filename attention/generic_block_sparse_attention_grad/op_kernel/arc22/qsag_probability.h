/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATTENTION_QSAG_PROBABILITY_ARCH22_H
#define ATTENTION_QSAG_PROBABILITY_ARCH22_H

#include "kernel_operator.h"

namespace QSAG {
using namespace AscendC;

// chunk 内一次处理的行数（vector 侧分块粒度）
constexpr uint32_t VEC_CHUNK_ROWS = 32;
// Brcb / lse di 广播的 block 元素个数（32B / sizeof(float)）
constexpr uint32_t VEC_BLOCK_ELEMENTS = 8;
// vector 单次指令最大元素个数（256B / sizeof(float)）
constexpr uint32_t VEC_ROW_SEGMENT = 64;

/*
 * 概率梯度计算（arch22 标准 vector 版，替代 A5 的 VF 版）：
 *   score = score * scale
 *   score = exp(score - lse)
 *   grad  = (grad - di) * score
 * 之后把 [count, width) 区间的 score / grad 置 0（被 mask 屏蔽的位置不参与梯度）。
 * score / grad 在 caller 侧已按行连续排布（行步长 = width）。
 * lse / di 每行一个元素，按 8 个 block（32B）对齐排列（由 Brcb 展开）。
 */
template <typename T>
__aicore__ inline void ProbabilityGradRows(LocalTensor<T> score, LocalTensor<T> grad, LocalTensor<T> lse,
                                           LocalTensor<T> di, LocalTensor<float> lseBrcb, LocalTensor<float> diBrcb,
                                           float scale, uint32_t width, uint32_t rows, uint32_t validRows,
                                           uint32_t count)
{
    // 1) score = score * scale
    // 同 PIPE_V 内的向量指令按发射顺序完成且硬件保证依赖，无需 PipeBarrier。
    Muls(score, score, scale, validRows * width);
    // 2) 逐段：score = exp(score - lse); grad = (grad - di) * score
    //
    // lse/di 每行只占 32B 块的首元素（其余 7 个为 padding 0）。向量指令的一个
    // repeat 覆盖 256B（8 个 32B block），BinaryRepeatParams 的 src1BlkStride=0
    // 只会重复读取 src1 的首个 32B block——若该 block 是
    // [lse, 0, 0, 0, 0, 0, 0, 0]，则只有前 8 个元素减到 lse，其余元素减 0。
    // 因此先用 Brcb 把标量展开成 8 个相同值，再让 repeat 复用同一 block。
    // width = ktile ∈ {64,128,256}，恒为 VEC_ROW_SEGMENT 的倍数，统一走 repeat 模式：
    // 每行一条指令，repeat 在行内步进 8 block（64 fp32），lse/di 广播块 rep 步进 0 复用。
    const BinaryRepeatParams broadcastParams{1, 1, 0, 8, 8, 0};
    const BinaryRepeatParams elementwiseParams{1, 1, 1, 8, 8, 8};
    const uint32_t repeats = width / VEC_ROW_SEGMENT;
    for (uint32_t r = 0; r < validRows; ++r) {
        Brcb(lseBrcb, lse[r * VEC_BLOCK_ELEMENTS], 1, BrcbRepeatParams(1, 1));
        Brcb(diBrcb, di[r * VEC_BLOCK_ELEMENTS], 1, BrcbRepeatParams(1, 1));
        const uint32_t offset = r * width;
        Sub(score[offset], score[offset], lseBrcb, VEC_ROW_SEGMENT, repeats, broadcastParams);
        Exp(score[offset], score[offset], VEC_ROW_SEGMENT, repeats, {1, 1, 8, 8});
        Sub(grad[offset], grad[offset], diBrcb, VEC_ROW_SEGMENT, repeats, broadcastParams);
        Mul(grad[offset], grad[offset], score[offset], VEC_ROW_SEGMENT, repeats, elementwiseParams);
    }
    // 3) 把被 mask 屏蔽的区域 [count, width) 置 0
    if (count < width) {
        // 段1：数组版（BIT 模式），一次覆盖 [count对齐到8, 下一个8对齐点)
        uint32_t z0 = count / VEC_BLOCK_ELEMENTS * VEC_BLOCK_ELEMENTS;
        uint32_t rest = count % VEC_BLOCK_ELEMENTS;
        if (rest != 0) {
            uint64_t maskArr[2] = {((1ULL << VEC_BLOCK_ELEMENTS) - 1) & ~((1ULL << rest) - 1), 0ULL};
            Duplicate(score[z0], 0.0f, maskArr, validRows, 1, width / VEC_BLOCK_ELEMENTS);
            Duplicate(grad[z0], 0.0f, maskArr, validRows, 1, width / VEC_BLOCK_ELEMENTS);
        }
        // 段2：标量版（COUNT 模式），逐 64 元素段清零
        for (uint32_t z = (count + VEC_BLOCK_ELEMENTS - 1) / VEC_BLOCK_ELEMENTS * VEC_BLOCK_ELEMENTS; z < width;
             z += VEC_ROW_SEGMENT) {
            uint32_t mask = width - z < VEC_ROW_SEGMENT ? width - z : VEC_ROW_SEGMENT;
            Duplicate(score[z], 0.0f, mask, validRows, 1, width / VEC_BLOCK_ELEMENTS);
            Duplicate(grad[z], 0.0f, mask, validRows, 1, width / VEC_BLOCK_ELEMENTS);
        }
    }
}
} // namespace QSAG
#endif // ATTENTION_QSAG_PROBABILITY_ARCH22_H

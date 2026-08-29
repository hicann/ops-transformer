/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_VF_H

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace MicroAPI;

/*
 * 按传入的 chunkSize 计算 FP32 inclusive cumsum、exp 和 Gamma：
 *   gCum[i] = sum(g[0:i + 1])
 *   gCumExp[i] = exp(gCum[i])
 *   gamma[i, j] = j < i ? exp(gCum[i] - gCum[j]) : 0
 * gCum 计算完后保留在寄存器中直接生成 Gamma，避免再次从 UB 加载以及两次 Broadcast。
 */
__simd_vf__ inline void CumSumExpVF(__ubuf__ float *gAddr, __ubuf__ float *gCumAddr, __ubuf__ float *gCumExpAddr,
                                    __ubuf__ float *gammaAddr, uint32_t chunkSize)
{
    // chunkSize <= 单个 FP32 Vector 寄存器的元素数，因此一个寄存器可保存整个 chunk 的 g/gCum
    RegTensor<float> gCumReg;
    RegTensor<float> shiftedReg;
    RegTensor<float> gCumExpReg;
    RegTensor<int32_t> laneIndexReg;
    RegTensor<int32_t> gatherIndexReg;
    // validMask 只使能当前 chunk 的有效 lane，避免寄存器尾部数据参与计算和写回
    MaskReg allMask = CreateMask<int32_t, MaskPattern::ALL>();
    uint32_t validCount = chunkSize;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg addMask;
    MaskReg gammaMask;

    // gCumReg 初始为原始 g；laneIndexReg 保存 [0, 1, ..., chunkSize - 1]，用于构造移位索引
    LoadAlign<float, LoadDist::DIST_NORM>(gCumReg, gAddr);
    Arange(laneIndexReg, static_cast<int32_t>(0));

    // 使用 Hillis-Steele inclusive scan 计算前缀和。每轮 stride 翻倍，分别合并相邻的
    // 1、2、4、... 个元素，经过 ceil(log2(chunkSize)) 轮后得到：
    //   gCumReg[i] = g[0] + g[1] + ... + g[i]。
    for (int32_t stride = 1; stride < static_cast<int32_t>(chunkSize); stride <<= 1) {
        // shiftedReg[i] 读取上一轮的 gCumReg[i - stride]。索引小于 0 的 lane 保持为0
        // 再通过 addMask 屏蔽，因此这些 lane 保持原值，不会重复累加 gCumReg[0]
        Adds<int32_t>(gatherIndexReg, laneIndexReg, -stride, allMask);
        Maxs<int32_t>(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(0), allMask);
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg));
        Compares<int32_t, CMPMODE::GE>(addMask, laneIndexReg, stride, validMask);
        Add<float, MaskMergeMode::MERGING>(gCumReg, gCumReg, shiftedReg, addMask);
    }

    Exp<float, MaskMergeMode::ZEROING>(gCumExpReg, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumAddr, gCumReg, validMask);
    StoreAlign<float, StoreDist::DIST_NORM>(gCumExpAddr, gCumExpReg, validMask);

    // stageOneMask 是严格下三角矩阵。直接用 laneIndexReg < row 生成行 mask，
    // 在 Gamma 矩阵中写出对角线及上三角为 0，不再从 GM 搬运 mask 或执行两次 Mul。
    Duplicate(gCumExpReg, static_cast<float>(0.0));
    StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaAddr, gCumExpReg, validMask);
    for (uint16_t row = 1; row < static_cast<uint16_t>(chunkSize); ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * chunkSize;
        Duplicate(reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg), static_cast<uint32_t>(row));
        Gather<float, uint32_t>(shiftedReg, gCumReg, reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg));
        Compares<int32_t, CMPMODE::LT>(gammaMask, laneIndexReg, static_cast<int32_t>(row), validMask);
        Sub<float, MaskMergeMode::ZEROING>(gCumExpReg, shiftedReg, gCumReg, gammaMask);
        Exp<float, MaskMergeMode::ZEROING>(gCumExpReg, gCumExpReg, gammaMask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gammaAddr + rowOffset, gCumExpReg, validMask);
    }
}

/*!
 * InverseAIVVF: forward-substitution inverse of the N x N lower-triangular (unit-diagonal) attn block.
 *   inv[0] = e_0; for i=1..N-1: inv[i] = e_i - sum_{j<i} attn[i,j] * inv[j]
 * attnUb/invResUb layout: [halfChunkSize rows, chunkSize cols]; subBlock uses column offset = `offset`,
 *   i.e. element (i, col) is at base + offset + i*chunkSize + col. Only first N columns per row are valid.
 * eiUb layout: identity matrix, row i = e_i, row stride = chunkSize (no column offset).
 *   eiUb must be pre-filled by the caller (unit-diagonal identity in the first N rows).
 */
template <uint32_t N>
__simd_vf__ inline void InverseAIVVFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb, __ubuf__ float *eiUb,
                                         uint32_t offset, uint32_t chunkSize)
{
    // N<=64, 单寄存器容纳一行; UpdateMask 生成前 N 个 lane 有效的掩码
    uint32_t maskLen = N;
    MaskReg maskN = UpdateMask<float>(maskLen);

    // inv[0] = e_0: 加载单位阵首行, 写入结果首行(offset 列偏移)
    RegTensor<float> inv0;
    LoadAlign(inv0, eiUb);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset, inv0, maskN);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 前代法: 逐行求解 inv[i] = e_i - sum_{j<i} attn[i,j] * inv[j]
    RegTensor<uint32_t> idxReg;
    RegTensor<float> acc;
    for (uint16_t i = 1; i < static_cast<uint16_t>(N); ++i) {
        // acc 清零, 准备累加当前行 i 的内积
        Duplicate(acc, static_cast<float>(0.0));
        // 加载 attn 第 i 行(对角块内, 含 offset 列偏移)
        RegTensor<float> li;
        LoadAlign(li, attnUb + offset + i * chunkSize);
        // 累加 sum_{j<i} attn[i,j] * inv[j]: Gather 取出 li[j] 广播, 与已求出的 inv[j] 乘加
        for (uint16_t j = 0; j < i; ++j) {
            Duplicate(idxReg, j);
            RegTensor<float> lijBrc;
            Gather(lijBrc, li, idxReg);
            RegTensor<float> invj;
            LoadAlign(invj, invResUb + offset + j * chunkSize);
            MulAddDst(acc, invj, lijBrc, maskN);
        }
        // inv[i] = e_i - acc, 写回结果第 i 行
        RegTensor<float> ei_i;
        LoadAlign(ei_i, eiUb + i * chunkSize);
        RegTensor<float> invi;
        Sub(invi, ei_i, acc, maskN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + i * chunkSize, invi, maskN);
        // store->load 屏障: 保证本行写入对后续行加载可见(下一轮 i 需读取 inv[i])
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <uint32_t N>
__aicore__ inline void InverseAIVVF(const LocalTensor<float> &attnUb, const LocalTensor<float> &invResUb,
                                    const LocalTensor<float> &eiUb, uint32_t offset, uint32_t chunkSize)
{
    __ubuf__ float *attn = reinterpret_cast<__ubuf__ float *>(attnUb.GetPhyAddr());
    __ubuf__ float *inv = reinterpret_cast<__ubuf__ float *>(invResUb.GetPhyAddr());
    __ubuf__ float *ei = reinterpret_cast<__ubuf__ float *>(eiUb.GetPhyAddr());
    InverseAIVVFImpl<N>(attn, inv, ei, offset, chunkSize);
}
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_VF_H

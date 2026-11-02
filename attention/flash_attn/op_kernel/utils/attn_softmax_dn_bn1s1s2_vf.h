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
 * \file attn_softmax_dn_bn1s1s2_vf.h
 * \brief 不合轴 DN(template 2, BN1_S1 分核) 的 softmax 向量化计算, 只给 mask_mode=3 用。
 *
 * 和 common 目录里那份非量化 DN softmax 实现(ProcessVec1DnUpdateVF)的关系:
 *   - 输出的 P(softmax 的分子)在 UB 里按"两个 32 行半区、中间隔 64 个元素"摆放
 *     (详见下面 ProcessVec1DnLineVF 的注释), 随后搬去 L1 的地址也按这个布局取,
 *     这样外面 vec block 的搬运代码一行都不用改;
 *   - 遮挡关系不再从 GM 读 bool 表, 而是按"第几行、第几列"现算(见下面 maskLine 的说明);
 *   - 只做"已经有 sum/max, 接着往后累加"这一种情况(FA 每次都会先 ResetSoftmaxBuffer),
 *     不做 sink, 也不做量化;
 * 以后改 common 那份实现时, 要回头确认这里是否也要跟着改。
 */

#ifndef FLASH_ATTN_ATTN_SOFTMAX_DN_BN1S1S2_VF_H_
#define FLASH_ATTN_ATTN_SOFTMAX_DN_BN1S1S2_VF_H_

#include "kernel_tensor.h"
#include "../../../common/op_kernel/arch35/vf/vf_basic_block_utils.h"

namespace FlashAttnKernel {
using namespace AscendC;
using namespace Reg;
using namespace FaVectorApi;

/*!
 * \brief 不合轴 DN 的 softmax 向量化计算(接着已有 sum/max 继续累加), 可选 causal 遮挡。
 *
 * 遮挡规则: 第 row 行能看到第 col 列, 当且仅当 maskLine + row >= col。
 *           maskLine 是"本块第 0 行能看到的最后一个块内列号", 每往下一行多看一列;
 *           被挡住的格子写成一个极小值, 后面算 exp 时它自然变成 0。
 * 缩放系数: NEED_MASK=true 时先乘 scale 再做遮挡(被挡的格子就是极小值本身, 不会再乘 scale);
 *           NEED_MASK=false 时和原实现一样, 先拿原始分数求 max, 算 exp 前再乘 scale。
 *
 * P 在 UB 里的摆法(32 列一组, 组内 8 个 544B 槽, 列粒度因此降到 2 列):
 *   组基址 = 组号 g * 4352B; 槽序 = k0.m0 k0.m1 k0.m2 k0.m3 | k1.m0 k1.m1 k1.m2 k1.m3;
 *   每个槽 = 512B 的 16x16 fractal + 32B pad(= 17 个 DataBlock = 544B), 列主序:
 *   槽(k_sub, m) 里第 i 列占 [i*32, i*32+32), 即一个 32B 块 = 某列的 16 行。
 *   于是一列 64 行 = 4 个 32B 块, 落在 m0..m3 四个槽里; 一条 StoreAlign<DATA_BLOCK_COPY>
 *   写 8 个块、块距 17 个 DataBlock, 恰好是 (列 c 的 4 个行块 | 列 c+16 的 4 个行块)。
 *   地址: off(g, c', m) = g*4352 + ((c'/16)*4 + m)*544 + (c'%16)*32, 容量 ceil(s2/32)*4352B。
 *   每 fractal 一个 pad 是必须的: 纯 NZ(512B 步长)8 块同 bank 相位会撞 bank, 组内单个 pad
 *   又让 8 块步长不均匀, 只有 544B(17 个 DataBlock, 奇数)同时满足均匀步长与相位错开。
 *   vec block 搬去 L1 时按 fractal 取数(源 544B / 目的 512B 递增)。
 */
template <typename T, typename T2, bool NEED_MASK, uint16_t tileCols = 128>
__simd_vf__ inline void ProcessVec1DnLineVF(__ubuf__ T2* pOutBuf, __ubuf__ float* scoreBuf,
                                            __ubuf__ float* maxRescaleBuf, __ubuf__ float* runningSumBuf,
                                            __ubuf__ float* runningMaxBuf, const uint32_t rowStride,
                                            const uint32_t validColCount, const T scale, const T minValue,
                                            const int64_t maskLine)
{
    (void)maskLine; // NEED_MASK=false 的那份实例化用不到
    const float inputScale = static_cast<float>(scale);
    // 32 列一组: 组内 8 个槽 = 4352B; 每次迭代 c 推进 1 列 = 32B
    constexpr uint16_t P_COLS_PER_GROUP = 32U;
    constexpr uint32_t P_GROUP_ELEMS = 4352U / sizeof(T2); // 2176 (T2 为 16bit)
    constexpr uint32_t P_COL_ELEMS = 32U / sizeof(T2);     // 16
    constexpr uint32_t P_DATA_BLOCK_STRIDE = 17U;          // 544B

    RegTensor<float> expSum0;
    RegTensor<float> expSum1;
    RegTensor<float> expSum2;
    RegTensor<float> expSum3;
    RegTensor<float> sumTmp0;
    RegTensor<float> sumTmp1;
    RegTensor<half> expEvenHalf0;
    RegTensor<half> expOddHalf0;
    RegTensor<bfloat16_t> expEvenBf16Half0;
    RegTensor<bfloat16_t> expOddBf16Half0;

    RegTensor<float> expValue0;
    RegTensor<float> expValue1;
    RegTensor<float> expValue2;
    RegTensor<float> expValue3;
    RegTensor<half> expEvenHalf1;
    RegTensor<half> expOddHalf1;
    RegTensor<bfloat16_t> expEvenBf16Half1;
    RegTensor<bfloat16_t> expOddBf16Half1;

    RegTensor<float> loadedScore0;
    RegTensor<float> loadedScore1;
    RegTensor<float> loadedScore2;
    RegTensor<float> loadedScore3;
    RegTensor<float> runningMaxValue;
    RegTensor<half> packedHalf0;
    RegTensor<half> packedHalf1;
    RegTensor<half> packedHalf0Alt;
    RegTensor<half> packedHalf1Alt;

    RegTensor<bfloat16_t> packedBf16Half0;
    RegTensor<bfloat16_t> packedBf16Half1;
    RegTensor<bfloat16_t> packedBf16Half0Alt;
    RegTensor<bfloat16_t> packedBf16Half1Alt;

    MaskReg allLanesF32;
    MaskReg allLanesB16;
    MaskReg packLanesB16;
    allLanesF32 = CreateMask<T, MaskPattern::ALL>();
    allLanesB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    uint32_t packLaneCount = (uint32_t)128ULL;
    packLanesB16 = UpdateMask<uint16_t>(packLaneCount);
    RegTensor<float> score0, score1, score2, score3;
    RegTensor<float> tileMax0, tileMax1, tileMax2, tileMax3;
    RegTensor<float> maskedFillValue;

    Duplicate(tileMax0, minValue);
    Duplicate(tileMax1, minValue);
    Duplicate(tileMax2, minValue);
    Duplicate(tileMax3, minValue);
    Duplicate(maskedFillValue, minValue);
    uint32_t validRowCount = rowStride;
    MaskReg validRowMask = UpdateMask<T>(validRowCount);
    // 列粒度: 每次迭代处理 4 列 {c, c+8, c+16, c+24}(c 为本组内的列号), 一组 8 次迭代覆盖 32 列。
    // 末组只需覆盖到有效列: 有效列 <= 8 时迭代数就是列数, 否则 8 次全用(0..31 都会被取列碰到)。
    uint16_t fullGroupCount = static_cast<uint16_t>(validColCount / P_COLS_PER_GROUP);
    uint16_t tailColCount = static_cast<uint16_t>(validColCount - P_COLS_PER_GROUP * fullGroupCount);
    uint16_t tailIterCount = (tailColCount > 8U) ? 8U : tailColCount;
    uint16_t groupCount = static_cast<uint16_t>(fullGroupCount + ((tailColCount != 0U) ? 1U : 0U));
    // 掩码/max 循环仍按"4 个相邻列"推进, 收界到末组里被取列碰到的最远一列((k-1)+24), 再向上取整到 4 列。
    // 补 minValue 的范围也就是这一段(至多 24+3 列), 于是 max/exp/sum 都只看得到有效列与填好的极小值。
    uint16_t fillEndCol =
        static_cast<uint16_t>(P_COLS_PER_GROUP * fullGroupCount + ((tailColCount != 0U) ? (tailIterCount + 24U) : 0U));
    uint16_t maskIterCount = static_cast<uint16_t>((fillEndCol + 3U) >> 2);
    fillEndCol = static_cast<uint16_t>(maskIterCount << 2);
    for (uint16_t colIdx = static_cast<uint16_t>(validColCount); colIdx < fillEndCol; ++colIdx) {
        // 每个 S2 列在 UB 里只占 rowStride 个 float; 这里只写这 rowStride 个, 不要写到下一列去
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)scoreBuf + colIdx * rowStride, maskedFillValue,
                                                     validRowMask);
    }
    mem_bar(VST_VLD);

    if constexpr (NEED_MASK) {
        MaskReg visibleMask0, visibleMask1, visibleMask2, visibleMask3;
        RegTensor<float> rowLimitValue;
        RegTensor<float> colBaseValue;
        RegTensor<float> colThreshold1;
        RegTensor<float> colThreshold2;
        RegTensor<float> colThreshold3;
        // 先乘缩放系数再做遮挡: 被挡的格子直接写成极小值, 不随 scale 变化(scale<=0 也不会错)。
        // Arange 之后第 row 个值就是 maskLine + row, 即这一行能看到的最后一个块内列号;
        // 下面拿它和列号逐列比较, 列号 <= 它的格子保留, 其余换成极小值。
        // 4 个列阈值都从同一个 base 独立算出(而不是链式 +1), 避免 Compare 之间形成
        // 串行依赖, 让 4 列的比较/选择能并行发射。
        Arange(rowLimitValue, (float)maskLine);
        Duplicate(colBaseValue, 0.0f);
        for (uint16_t colGroup = 0; colGroup < maskIterCount; ++colGroup) {
            LoadAlign(score0, scoreBuf + colGroup * rowStride * 4);
            LoadAlign(score1, scoreBuf + colGroup * rowStride * 4 + rowStride);
            LoadAlign(score2, scoreBuf + colGroup * rowStride * 4 + rowStride * 2);
            LoadAlign(score3, scoreBuf + colGroup * rowStride * 4 + rowStride * 3);
            Muls(score0, score0, inputScale, allLanesF32);
            Muls(score1, score1, inputScale, allLanesF32);
            Muls(score2, score2, inputScale, allLanesF32);
            Muls(score3, score3, inputScale, allLanesF32);
            Compare<T, CMPMODE::GE>(visibleMask0, rowLimitValue, colBaseValue, allLanesF32);
            Adds(colThreshold1, colBaseValue, 1.0f, allLanesF32);
            Adds(colThreshold2, colBaseValue, 2.0f, allLanesF32);
            Adds(colThreshold3, colBaseValue, 3.0f, allLanesF32);
            Compare<T, CMPMODE::GE>(visibleMask1, rowLimitValue, colThreshold1, allLanesF32);
            Compare<T, CMPMODE::GE>(visibleMask2, rowLimitValue, colThreshold2, allLanesF32);
            Compare<T, CMPMODE::GE>(visibleMask3, rowLimitValue, colThreshold3, allLanesF32);
            Adds(colBaseValue, colBaseValue, 4.0f, allLanesF32);
            Select(score0, score0, maskedFillValue, visibleMask0);
            Select(score1, score1, maskedFillValue, visibleMask1);
            Select(score2, score2, maskedFillValue, visibleMask2);
            Select(score3, score3, maskedFillValue, visibleMask3);
            // 只能写回本列的 rowStride 个元素; 如果写满 64 个, M 不满 64 行的尾块会把下一列的数据盖掉
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(scoreBuf + colGroup * rowStride * 4, score0, validRowMask);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(scoreBuf + colGroup * rowStride * 4 + rowStride, score1,
                                                         validRowMask);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(scoreBuf + colGroup * rowStride * 4 + rowStride * 2, score2,
                                                         validRowMask);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(scoreBuf + colGroup * rowStride * 4 + rowStride * 3, score3,
                                                         validRowMask);
            Max(tileMax0, tileMax0, score0, allLanesF32);
            Max(tileMax1, tileMax1, score1, allLanesF32);
            Max(tileMax2, tileMax2, score2, allLanesF32);
            Max(tileMax3, tileMax3, score3, allLanesF32);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    } else {
        for (uint16_t colGroup = 0; colGroup < maskIterCount; ++colGroup) {
            LoadAlign(score0, scoreBuf + colGroup * rowStride * 4);
            LoadAlign(score1, scoreBuf + colGroup * rowStride * 4 + rowStride);
            LoadAlign(score2, scoreBuf + colGroup * rowStride * 4 + rowStride * 2);
            LoadAlign(score3, scoreBuf + colGroup * rowStride * 4 + rowStride * 3);
            Max(tileMax0, tileMax0, score0, allLanesF32);
            Max(tileMax1, tileMax1, score1, allLanesF32);
            Max(tileMax2, tileMax2, score2, allLanesF32);
            Max(tileMax3, tileMax3, score3, allLanesF32);
        }
    }

    LoadAlign(runningMaxValue, runningMaxBuf);
    Max(tileMax0, tileMax0, tileMax2, allLanesF32);
    Max(tileMax1, tileMax1, tileMax3, allLanesF32);
    Max(tileMax0, tileMax0, tileMax1, allLanesF32);
    if constexpr (!NEED_MASK) {
        Muls(tileMax0, tileMax0, inputScale, allLanesF32);
    }
    Max(tileMax0, tileMax0, runningMaxValue, allLanesF32);

    FusedExpSub(runningMaxValue, runningMaxValue, tileMax0, allLanesF32);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)runningMaxBuf, tileMax0, allLanesB16);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)maxRescaleBuf, runningMaxValue, allLanesB16);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(expSum0, 0, allLanesF32);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(expSum1, 0, allLanesF32);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(expSum2, 0, allLanesF32);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(expSum3, 0, allLanesF32);

    // 组循环 + 组内 8 次迭代: 两层都是 uint16 从 0 步长 1, 组内层的内层边界先拷成局部量;
    // 组内层每层的指针/列偏移都是"基址 + 迭代号 * 常量", 便于 Compiler 生成 Hardware Loop。
    for (uint16_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
        uint16_t iterCount = ((groupIdx + 1U) == groupCount && tailColCount != 0U) ? tailIterCount : 8U;
        __ubuf__ float* scoreGroup = scoreBuf + groupIdx * P_COLS_PER_GROUP * rowStride;
        __ubuf__ T2* pOutGroupBase = pOutBuf + groupIdx * P_GROUP_ELEMS;
        for (uint16_t colInGroup = 0; colInGroup < iterCount; ++colInGroup) {
            // 4 路取列: 本组内 {c, c+8, c+16, c+24}; 打包时按 (c, c+16) / (c+8, c+24) 配对,
            // 恰好一组落 k0 槽、一组落 k1 槽, 一条 store 的 8 个块正好填满两列
            LoadAlign(loadedScore0, scoreGroup + colInGroup * rowStride);
            LoadAlign(loadedScore1, scoreGroup + (colInGroup + 8U) * rowStride);
            LoadAlign(loadedScore2, scoreGroup + (colInGroup + 16U) * rowStride);
            LoadAlign(loadedScore3, scoreGroup + (colInGroup + 24U) * rowStride);

            if constexpr (!NEED_MASK) {
                // 不遮挡的那份分数还没乘过 scale, 这里补乘; 遮挡那份在求 max 前已经乘过, 不再乘
                Muls(loadedScore0, loadedScore0, inputScale, allLanesF32);
                Muls(loadedScore1, loadedScore1, inputScale, allLanesF32);
                Muls(loadedScore2, loadedScore2, inputScale, allLanesF32);
                Muls(loadedScore3, loadedScore3, inputScale, allLanesF32);
            }

            FusedExpSub(expValue0, loadedScore0, tileMax0, allLanesF32);
            FusedExpSub(expValue1, loadedScore1, tileMax0, allLanesF32);
            FusedExpSub(expValue2, loadedScore2, tileMax0, allLanesF32);
            FusedExpSub(expValue3, loadedScore3, tileMax0, allLanesF32);

            __ubuf__ T2* pOutGroup = pOutGroupBase + colInGroup * P_COL_ELEMS;
            __ubuf__ T2* pOutGroupK1 = pOutGroup + 8U * P_COL_ELEMS; // 组内 k1 的 8 列
            if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(expEvenBf16Half0, expValue0, allLanesF32);
                Cast<T2, T, castTraitZero>(expOddBf16Half0, expValue2, allLanesF32);
                DeInterleave(packedBf16Half0, packedBf16Half0Alt, expEvenBf16Half0, expOddBf16Half0);

                Cast<T2, T, castTraitZero>(expEvenBf16Half1, expValue1, allLanesF32);
                Cast<T2, T, castTraitZero>(expOddBf16Half1, expValue3, allLanesF32);
                DeInterleave(packedBf16Half1, packedBf16Half1Alt, expEvenBf16Half1, expOddBf16Half1);
                /* 8 个 32B 块 = 列 c 的 4 个行块 + 列 c+16 的 4 个行块, 块距 17 个 DataBlock */
                StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pOutGroup, packedBf16Half0, P_DATA_BLOCK_STRIDE,
                                                                   packLanesB16);
                Add(expSum0, expValue0, expSum0, allLanesF32);
                Add(expSum2, expValue2, expSum2, allLanesF32);
                StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pOutGroupK1, packedBf16Half1, P_DATA_BLOCK_STRIDE,
                                                                   packLanesB16);
                Add(expSum1, expValue1, expSum1, allLanesF32);
                Add(expSum3, expValue3, expSum3, allLanesF32);
            } else {
                Cast<T2, T, castTraitZero>(expEvenHalf0, expValue0, allLanesF32);
                Cast<T2, T, castTraitZero>(expOddHalf0, expValue2, allLanesF32);
                DeInterleave(packedHalf0, packedHalf0Alt, expEvenHalf0, expOddHalf0);

                Cast<T2, T, castTraitZero>(expEvenHalf1, expValue1, allLanesF32);
                Cast<T2, T, castTraitZero>(expOddHalf1, expValue3, allLanesF32);

                DeInterleave(packedHalf1, packedHalf1Alt, expEvenHalf1, expOddHalf1);

                /* 8 个 32B 块 = 列 c 的 4 个行块 + 列 c+16 的 4 个行块, 块距 17 个 DataBlock */
                StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pOutGroup, packedHalf0, P_DATA_BLOCK_STRIDE,
                                                                   packLanesB16);
                Add(expSum0, expValue0, expSum0, allLanesF32);
                Add(expSum2, expValue2, expSum2, allLanesF32);

                StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY>(pOutGroupK1, packedHalf1, P_DATA_BLOCK_STRIDE,
                                                                   packLanesB16);
                Add(expSum1, expValue1, expSum1, allLanesF32);
                Add(expSum3, expValue3, expSum3, allLanesF32);
            }
        }
    }

    Add(sumTmp0, expSum2, expSum0, allLanesF32);
    Add(sumTmp1, expSum3, expSum1, allLanesF32);
    Add(sumTmp0, sumTmp0, sumTmp1, allLanesF32);
    RegTensor<float> prevSumValue;
    LoadAlign(prevSumValue, runningSumBuf);
    Mul(prevSumValue, runningMaxValue, prevSumValue, allLanesF32);
    Add(prevSumValue, prevSumValue, sumTmp0, allLanesF32);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)runningSumBuf, prevSumValue, allLanesF32);
}

template <typename T, typename T2, bool NEED_MASK, uint16_t tileCols = 128>
__aicore__ inline void ProcessVec1VfDnLine(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                           const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                           const LocalTensor<T>& expMaxTensor, const uint32_t rowStride,
                                           const uint32_t validColCount, const T scale, const T minValue,
                                           const int64_t maskLine)
{
    __ubuf__ T2* pOutBuf = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* scoreBuf = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* maxRescaleBuf = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* runningSumBuf = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* runningMaxBuf = (__ubuf__ T*)maxTensor.GetPhyAddr();

    ProcessVec1DnLineVF<T, T2, NEED_MASK, tileCols>(pOutBuf, scoreBuf, maxRescaleBuf, runningSumBuf, runningMaxBuf,
                                                    rowStride, validColCount, scale, minValue, maskLine);
}

} // namespace FlashAttnKernel
#endif // FLASH_ATTN_ATTN_SOFTMAX_DN_BN1S1S2_VF_H_

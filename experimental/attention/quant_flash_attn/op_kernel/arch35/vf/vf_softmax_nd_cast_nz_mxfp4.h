/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
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
 * \file vf_softmax_nd_cast_nz_mxfp4.h
 * \brief S 行主序 (lane = s2) 的 softmax + fp4 量化, 输出 [M 行][s2 列] 的紧凑 fp4
 */

#ifndef VF_SOFTMAX_ND_CAST_NZ_MXFP4_H_
#define VF_SOFTMAX_ND_CAST_NZ_MXFP4_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

// ND 索引区 (相对索引基址的字节偏移), 索引按 16bit 元素存放
constexpr uint32_t ND_IDX_XOR_OFFSET = 256;   // 5 * 256B: lane ^ k (k = 16,8,4,2,1), 32 lane 组内全规约
constexpr uint32_t ND_IDX_T_OFFSET = 1536;    // 4 * 256B: [M][4] 组标量转置成 [4][M] 的索引
constexpr uint32_t ND_IDX_SEL_OFFSET = 2560;  // 256B: 组内 4 个组标量抽取索引
constexpr uint32_t ND_IDX_PERM_OFFSET = 2816; // 256B: fp4 打包结果的字节置换索引
constexpr uint32_t ND_IDX_HI_OFFSET = 3072;   // 256B: 取打包结果的高 128B 到低 lane
constexpr uint32_t ND_IDX_TOTAL = 3328;
constexpr uint32_t ND_S2_LANES = 128;   // 单次 VF 处理的 s2 列数 = 寄存器 lane 数
constexpr uint32_t ND_P_ROW_BYTES = 64; // 每 M 行 128 个 fp4 = 64B

template <typename T>
__simd_vf__ inline void init_nd_indexes_vf(__ubuf__ int16_t* idxXor, __ubuf__ int16_t* idxT, __ubuf__ int16_t* idxSel,
                                           __ubuf__ uint8_t* idxPerm)
{
    RegTensor<int16_t> l, t, u, out;
    MaskReg pregAll16 = CreateMask<uint16_t, MaskPattern::ALL>();
    Arange<int16_t, IndexOrder::INCREASE_ORDER>(l, 0);

#define ND_XOR_IDX(K, S, OFF) \
    ShiftRights(t, l, static_cast<int16_t>(S), pregAll16); \
    ShiftRights(u, t, static_cast<int16_t>(1), pregAll16); \
    ShiftLefts(u, u, static_cast<int16_t>(1), pregAll16); \
    Sub(u, t, u, pregAll16); \
    Muls(u, u, static_cast<int16_t>(2 * (K)), pregAll16); \
    Adds(out, l, static_cast<int16_t>(K), pregAll16); \
    Sub(out, out, u, pregAll16); \
    StoreAlign<int16_t>(idxXor + (OFF), out, pregAll16)
    ND_XOR_IDX(16, 4, 0);
    ND_XOR_IDX(8, 3, 128);
    ND_XOR_IDX(4, 2, 256);
    ND_XOR_IDX(2, 1, 384);
    ND_XOR_IDX(1, 0, 512);
#undef ND_XOR_IDX

    // 转置索引: staging 为 [M][4], 取第 g 个组标量给第 j 个 M
    ShiftLefts(t, l, static_cast<int16_t>(2), pregAll16);
    for (int16_t g = 0; g < 4; ++g) {
        Adds(out, t, g, pregAll16);
        StoreAlign<int16_t>(idxT + g * 128, out, pregAll16);
    }

    // 抽取索引: 每个 32 lane 组取一个组标量 (lane 0/32/64/96)
    ShiftRights(u, l, static_cast<int16_t>(2), pregAll16);
    ShiftLefts(u, u, static_cast<int16_t>(2), pregAll16);
    Sub(u, l, u, pregAll16);
    ShiftLefts(u, u, static_cast<int16_t>(5), pregAll16);
    StoreAlign<int16_t>(idxSel, u, pregAll16);

    // fp4 打包置换: 4 个 castTrait 的结果按字节交织在 src 里,
    // 目标字节 j = 64*r + 16*g + m 取自 src 字节 64*g + 4*m + r
    RegTensor<uint8_t> l8, r8, t8, g8, m8, idx8;
    MaskReg pregAll8 = CreateMask<uint8_t, MaskPattern::ALL>();
    Arange((RegTensor<int8_t>&)l8, 0);
    ShiftRights(r8, l8, static_cast<int16_t>(6), pregAll8);
    ShiftRights(t8, l8, static_cast<int16_t>(4), pregAll8);
    ShiftLefts(g8, r8, NUM_2, pregAll8);
    Sub(g8, t8, g8, pregAll8);
    ShiftLefts(t8, t8, static_cast<int16_t>(4), pregAll8);
    Sub(m8, l8, t8, pregAll8);
    ShiftLefts(g8, g8, static_cast<int16_t>(6), pregAll8);
    ShiftLefts(m8, m8, NUM_2, pregAll8);
    Add(idx8, g8, m8, pregAll8);
    Add(idx8, idx8, r8, pregAll8);
    StoreAlign<uint8_t>(idxPerm, idx8, pregAll8);

    // 高 128B 取到低 lane: l -> l + 128 (仅前 128 lane 有效)
    Adds(idx8, l8, static_cast<uint16_t>(ND_S2_LANES), pregAll8);
    StoreAlign<uint8_t>(idxPerm + 256, idx8, pregAll8);
}

template <typename T>
__aicore__ inline void InitNdIndexesCallVF(const LocalTensor<uint8_t>& indexBuf)
{
    __ubuf__ uint8_t* base = (__ubuf__ uint8_t*)indexBuf.GetPhyAddr();
    init_nd_indexes_vf<T>((__ubuf__ int16_t*)(base + ND_IDX_XOR_OFFSET), (__ubuf__ int16_t*)(base + ND_IDX_T_OFFSET),
                          (__ubuf__ int16_t*)(base + ND_IDX_SEL_OFFSET),
                          (__ubuf__ uint8_t*)(base + ND_IDX_PERM_OFFSET));
}

// pDest: [M 行][128 个 s2 的 fp4, 64B]; s: [M 行][128 half]; ulmaxStage: [M][4] half;
// ulmax: [4 组][128 M] half; globalMax: [128 M] half
template <bool clear_gmax, bool hasAtten, typename T, typename T2>
__simd_vf__ inline void softmax_nd_cast_nz_mxfp4_vf(__ubuf__ T2* pDest, __ubuf__ T* s, __ubuf__ T* ulmaxStage,
                                                    __ubuf__ T* ulmax, __ubuf__ T* globalMax, __ubuf__ uint8_t* idxUb,
                                                    uint32_t actM, uint32_t actS2, int32_t maskBase,
                                                    int32_t causalScale, uint32_t s2ChunkOff)
{
    MaskReg pregAll16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg pregAll8 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg pregVl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    // s2 尾片: lane >= actS2 的列无效, 置 -inf 使组内 max 不受影响、exp 结果为 0
    uint32_t actS2Cnt = actS2;
    MaskReg pregS2Keep = UpdateMask<half>(actS2Cnt);
    MaskReg pregS2Invalid;
    MaskNot(pregS2Invalid, pregS2Keep, pregAll16);
    bool s2TailMask = (actS2 < ND_S2_LANES);

    RegTensor<half> negInf;
    Duplicate(negInf, MIN_VALUE);
    Muls(negInf, negInf, TWO_VALE, pregAll16);

    RegTensor<uint16_t> idxX16, idxX8, idxX4, idxX2, idxX1, idxSel;
    LoadAlign<uint16_t>(idxX16, (__ubuf__ uint16_t*)(idxUb + ND_IDX_XOR_OFFSET));
    LoadAlign<uint16_t>(idxX8, (__ubuf__ uint16_t*)(idxUb + ND_IDX_XOR_OFFSET + 256));
    LoadAlign<uint16_t>(idxX4, (__ubuf__ uint16_t*)(idxUb + ND_IDX_XOR_OFFSET + 512));
    LoadAlign<uint16_t>(idxX2, (__ubuf__ uint16_t*)(idxUb + ND_IDX_XOR_OFFSET + 768));
    LoadAlign<uint16_t>(idxX1, (__ubuf__ uint16_t*)(idxUb + ND_IDX_XOR_OFFSET + 1024));
    LoadAlign<uint16_t>(idxSel, (__ubuf__ uint16_t*)(idxUb + ND_IDX_SEL_OFFSET));
    RegTensor<uint8_t> idxPerm, idxHi;
    LoadAlign<uint8_t>(idxPerm, (__ubuf__ uint8_t*)(idxUb + ND_IDX_PERM_OFFSET));
    LoadAlign<uint8_t>(idxHi, (__ubuf__ uint8_t*)(idxUb + ND_IDX_HI_OFFSET));

    // CAUSAL: 行阈值掩码用的 lane 序号(0..ND_S2_LANES-1)与谓词

    RegTensor<half> src0, src1, src2, src3, tmp, grpMax, expSub, grp4;
    RegTensor<bfloat16_t> bf0, bf1, bf2, bf3;
    RegTensor<float4_e2m1x2_t> q0, q1, q2, q3;
    RegTensor<uint8_t> pHi;

    for (uint32_t m = 0; m < actM; m += 4) {
        LoadAlign(src0, s + m * ND_S2_LANES);
        LoadAlign(src1, s + (m + 1) * ND_S2_LANES);
        LoadAlign(src2, s + (m + 2) * ND_S2_LANES);
        LoadAlign(src3, s + (m + 3) * ND_S2_LANES);
        if (s2TailMask) {
            Select(src0, negInf, src0, pregS2Invalid);
            Select(src1, negInf, src1, pregS2Invalid);
            Select(src2, negInf, src2, pregS2Invalid);
            Select(src3, negInf, src3, pregS2Invalid);
        }

        // CAUSAL: 可见 ⇔ (chunkOff+lane) <= row/gSize - maskBase; tl>=ND_S2_LANES-1 时整行全可见, 直接跳过
#define ND_CAUSAL_ROW(reg, row) \
    do { \
        int32_t t_ = static_cast<int32_t>(row) / causalScale - maskBase - static_cast<int32_t>(s2ChunkOff); \
        /* 下界取 -1: 该行本 chunk 无可见 key 时 count=0(全遮), 夹到 0 会误留 1 个 key */ \
        t_ = (t_ < -1) ? \
                 -1 : \
                 ((t_ > static_cast<int32_t>(ND_S2_LANES) - 1) ? (static_cast<int32_t>(ND_S2_LANES) - 1) : t_); \
        uint32_t cnt_ = static_cast<uint32_t>(t_ + 1); \
        MaskReg pregM = UpdateMask<half>(cnt_); \
        Select((reg), (reg), negInf, pregM); \
    } while (0)
        if constexpr (hasAtten) {
            ND_CAUSAL_ROW(src0, m + 0);
            ND_CAUSAL_ROW(src1, m + 1);
            ND_CAUSAL_ROW(src2, m + 2);
            ND_CAUSAL_ROW(src3, m + 3);
        }
#undef ND_CAUSAL_ROW

        // 组内 (32 lane) 全规约 → log2 域组标量, 并就地做 exp(x - (gmax - 2) * ln2)
#define ND_GROUP_MAX(reg, stageOff) \
    Gather(tmp, (reg), idxX16); \
    Max(grpMax, (reg), tmp, pregAll16); \
    Gather(tmp, grpMax, idxX8); \
    Max(grpMax, grpMax, tmp, pregAll16); \
    Gather(tmp, grpMax, idxX4); \
    Max(grpMax, grpMax, tmp, pregAll16); \
    Gather(tmp, grpMax, idxX2); \
    Max(grpMax, grpMax, tmp, pregAll16); \
    Gather(tmp, grpMax, idxX1); \
    Max(grpMax, grpMax, tmp, pregAll16); \
    Maxs(grpMax, grpMax, CAUSAL_CLAMP, pregAll16); \
    Muls(grpMax, grpMax, INV_LN2, pregAll16); \
    Truncate<half, RoundMode::CAST_FLOOR>(grpMax, grpMax, pregAll16); \
    Gather(grp4, grpMax, idxSel); \
    Store(ulmaxStage + (stageOff), grp4, 4); \
    Adds(expSub, grpMax, NEG_TWO_VALE, pregAll16); \
    Muls(expSub, expSub, LN2, pregAll16); \
    Sub((reg), (reg), expSub, pregAll16); \
    Exp((reg), (reg), pregAll16)
        ND_GROUP_MAX(src0, m * 4);
        ND_GROUP_MAX(src1, (m + 1) * 4);
        ND_GROUP_MAX(src2, (m + 2) * 4);
        ND_GROUP_MAX(src3, (m + 3) * 4);
#undef ND_GROUP_MAX

        // 4 行各按列对压成 64B, 依 castTrait 拼接成 256B: [行 0][行 1][行 2][行 3]
        Cast<bfloat16_t, T, castTraitZero>(bf0, src0, pregAll16);
        Cast<bfloat16_t, T, castTraitZero>(bf1, src1, pregAll16);
        Cast<bfloat16_t, T, castTraitZero>(bf2, src2, pregAll16);
        Cast<bfloat16_t, T, castTraitZero>(bf3, src3, pregAll16);
        Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(q0, bf0, pregAll16);
        Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(q1, bf1, pregAll16);
        Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(q2, bf2, pregAll16);
        Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(q3, bf3, pregAll16);
        Or((RegTensor<uint8_t>&)q0, (RegTensor<uint8_t>&)q0, (RegTensor<uint8_t>&)q1, pregAll8);
        Or((RegTensor<uint8_t>&)q2, (RegTensor<uint8_t>&)q2, (RegTensor<uint8_t>&)q3, pregAll8);
        Or((RegTensor<uint8_t>&)q0, (RegTensor<uint8_t>&)q0, (RegTensor<uint8_t>&)q2, pregAll8);
        Gather((RegTensor<uint8_t>&)q0, (RegTensor<uint8_t>&)q0, idxPerm);
        StoreAlign<uint8_t>((__ubuf__ uint8_t*)pDest + m * ND_P_ROW_BYTES, (RegTensor<uint8_t>&)q0, pregVl128);
        Gather(pHi, (RegTensor<uint8_t>&)q0, idxHi);
        StoreAlign<uint8_t>((__ubuf__ uint8_t*)pDest + m * ND_P_ROW_BYTES + ND_S2_LANES, pHi, pregVl128);
    }
}

// [M][4] 组标量转置成 [4][M]: 独立 VF, 与写 stage 的 VF 分开以保证 stage 写完再读
template <bool clear_gmax, typename T>
__simd_vf__ inline void softmax_nd_gmax_transpose_vf(__ubuf__ T* ulmaxStage, __ubuf__ T* ulmax, __ubuf__ T* globalMax,
                                                     __ubuf__ uint8_t* idxUb)
{
    MaskReg pregAll16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<uint16_t> idxT0, idxT1, idxT2, idxT3;
    LoadAlign<uint16_t>(idxT0, (__ubuf__ uint16_t*)(idxUb + ND_IDX_T_OFFSET));
    LoadAlign<uint16_t>(idxT1, (__ubuf__ uint16_t*)(idxUb + ND_IDX_T_OFFSET + 256));
    LoadAlign<uint16_t>(idxT2, (__ubuf__ uint16_t*)(idxUb + ND_IDX_T_OFFSET + 512));
    LoadAlign<uint16_t>(idxT3, (__ubuf__ uint16_t*)(idxUb + ND_IDX_T_OFFSET + 768));
    RegTensor<half> g0, g1, g2, g3, gOld;
    Gather(g0, ulmaxStage, idxT0, pregAll16);
    Gather(g1, ulmaxStage, idxT1, pregAll16);
    Gather(g2, ulmaxStage, idxT2, pregAll16);
    Gather(g3, ulmaxStage, idxT3, pregAll16);
    Max(g0, g0, g1, pregAll16);
    Max(g2, g2, g3, pregAll16);
    Max(g0, g0, g2, pregAll16);
    if constexpr (!clear_gmax) {
        LoadAlign(gOld, globalMax);
        Max(g0, g0, gOld, pregAll16);
    }
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(globalMax, g0, pregAll16);

    Gather(g0, ulmaxStage, idxT0, pregAll16);
    Gather(g1, ulmaxStage, idxT1, pregAll16);
    Gather(g2, ulmaxStage, idxT2, pregAll16);
    Gather(g3, ulmaxStage, idxT3, pregAll16);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(ulmax, g0, pregAll16);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(ulmax + 128, g1, pregAll16);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(ulmax + 256, g2, pregAll16);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(ulmax + 384, g3, pregAll16);
}

template <bool clear_gmax, bool hasAtten, typename T, typename T2>
__aicore__ inline void softmaxNdCastNzMxfp4CallVF(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& srcTensor,
                                                  const LocalTensor<T>& ulmaxStage, const LocalTensor<T>& ulmaxTensor,
                                                  const LocalTensor<T>& globalMaxTensor,
                                                  const LocalTensor<uint8_t>& indexesBuf, uint32_t actM, uint32_t actS2,
                                                  int32_t maskBase, int32_t causalScale, uint32_t s2ChunkOff)
{
    softmax_nd_cast_nz_mxfp4_vf<clear_gmax, hasAtten, T, T2>(
        (__ubuf__ T2*)dstTensor.GetPhyAddr(), (__ubuf__ T*)srcTensor.GetPhyAddr(), (__ubuf__ T*)ulmaxStage.GetPhyAddr(),
        (__ubuf__ T*)ulmaxTensor.GetPhyAddr(), (__ubuf__ T*)globalMaxTensor.GetPhyAddr(),
        (__ubuf__ uint8_t*)indexesBuf.GetPhyAddr(), actM, actS2, maskBase, causalScale, s2ChunkOff);
    softmax_nd_gmax_transpose_vf<clear_gmax, T>(
        (__ubuf__ T*)ulmaxStage.GetPhyAddr(), (__ubuf__ T*)ulmaxTensor.GetPhyAddr(),
        (__ubuf__ T*)globalMaxTensor.GetPhyAddr(), (__ubuf__ uint8_t*)indexesBuf.GetPhyAddr());
}

} // namespace Mxfp4Api
#endif // VF_SOFTMAX_ND_CAST_NZ_MXFP4_H_

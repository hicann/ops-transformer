/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L1_TO_L0_MX_FP8_A5_HPP
#define GEMM_TILE_COPY_L1_TO_L0_MX_FP8_A5_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"

namespace NpuArch::Gemm::Tile {

// MXFP8 L1→L0：数据侧 C0=32，scale 分形仍按 64（与 fp4 / matmul.h yStep 一致）

// V: nZ L1 [S2,D] + scale → zN L0A Vᵀ=[D,S2]（可沿 S2 切段）
struct CopyL1ToL0AMxFp8A5 {
    __aicore__ inline CopyL1ToL0AMxFp8A5() {}

    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                      uint32_t dAct, uint32_t s2Act, uint32_t s2Start = 0)
    {
        constexpr uint32_t NZ_C0 = Block::MXFP8::NZ_C0_ELEMS;
        constexpr uint32_t FP8_C0 = Block::MXFP8::FP8_C0_ELEMS;
        constexpr uint32_t MX_FRAC = Block::MXFP8::CONST_64;

        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = s2Start / NZ_C0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = s2Act / NZ_C0;
        // fp8 转置要求 mStep 为偶数
        loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        loadData2DParamsA.kStep = dAct / FP8_C0;
        loadData2DParamsA.srcStride = (s2Start + s2Act + NZ_C0 - 1) / NZ_C0; // full tile stride if needed
        // 简化：srcStride 仍用整段 S2 对齐（调用方保证 L1 按整 tile 布局）
        loadData2DParamsA.srcStride = ((s2Start + s2Act) >= Block::MXFP8::S2_BASE_TILE_SIZE) ?
                                          (Block::MXFP8::S2_BASE_TILE_SIZE / NZ_C0) :
                                          ((s2Start + s2Act + NZ_C0 - 1) / NZ_C0);
        // 调用方传完整 L1 S2 对齐时更稳妥——用 s2Act 作为本段，srcStride 由外层全长决定
        // 这里用 max(s2Start+s2Act, s2Act) 的 16 对齐；PV 侧会再传 s2Full
        loadData2DParamsA.dstStride = (dAct + NZ_C0 - 1) / NZ_C0 + 1; // +1 for rowsum pad fractal
        // mxfp4: `if (dAct <= FP4_C0)` 且 FP4_C0=64，把 D=64 的 dstStride 从 5 pad 到 9
        // （与 D=128 / PV_MMAD_M_DIM=144 同构）。fp8 不能用 FP8_C0=32 当阈值，否则 D=64 不进分支。
        if (dAct <= Block::MXFP8::CONST_64) {
            loadData2DParamsA.dstStride += Block::MXFP8::CONST_64 / NZ_C0;
        }
        loadData2DParamsA.ifTranspose = true;

        AscendC::LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;
        load2DMxParamsA.yStartPosition = s2Start / MX_FRAC;
        load2DMxParamsA.xStep = (dAct + NZ_C0 - 1) / NZ_C0;
        load2DMxParamsA.yStep = (s2Act + MX_FRAC - 1) / MX_FRAC;
        load2DMxParamsA.srcStride = load2DMxParamsA.yStep;
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;

        AscendC::LoadData(dst, src, scale, loadData2DParamsA, load2DMxParamsA);
    }

    // 带完整 L1 S2 行 stride 的重载（k-split 时 srcStride 必须是整 tile）。
    // scaleS2Start / scaleS2FullAlign16 允许 scale 与 data 不在同一 S2 窗口（P1b 512 行 scale）。
    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                      uint32_t dAct, uint32_t s2Act, uint32_t s2Start, uint32_t s2FullAlign16,
                                      uint32_t scaleS2Start, uint32_t scaleS2FullAlign16)
    {
        constexpr uint32_t NZ_C0 = Block::MXFP8::NZ_C0_ELEMS;
        constexpr uint32_t FP8_C0 = Block::MXFP8::FP8_C0_ELEMS;
        constexpr uint32_t MX_FRAC = Block::MXFP8::CONST_64;

        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = s2Start / NZ_C0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = s2Act / NZ_C0;
        loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        loadData2DParamsA.kStep = dAct / FP8_C0;
        loadData2DParamsA.srcStride = s2FullAlign16 / NZ_C0;
        loadData2DParamsA.dstStride = (dAct + NZ_C0 - 1) / NZ_C0 + 1;
        // 同 mxfp4 D=64：dstStride pad 到 9，PV mmad M 仍为 144。见上一重载注释。
        if (dAct <= Block::MXFP8::CONST_64) {
            loadData2DParamsA.dstStride += Block::MXFP8::CONST_64 / NZ_C0;
        }
        loadData2DParamsA.ifTranspose = true;

        AscendC::LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;
        load2DMxParamsA.yStartPosition = scaleS2Start / MX_FRAC;
        load2DMxParamsA.xStep = (dAct + NZ_C0 - 1) / NZ_C0;
        load2DMxParamsA.yStep = (s2Act + MX_FRAC - 1) / MX_FRAC;
        load2DMxParamsA.srcStride = (scaleS2FullAlign16 + MX_FRAC - 1) / MX_FRAC;
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;

        AscendC::LoadData(dst, src, scale, loadData2DParamsA, load2DMxParamsA);
    }

    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                      uint32_t dAct, uint32_t s2Act, uint32_t s2Start, uint32_t s2FullAlign16)
    {
        operator()(dst, src, scale, dAct, s2Act, s2Start, s2FullAlign16, s2Start, s2FullAlign16);
    }
};

// P: zN L1 [M,S2] + scale → nZ L0B Pᵀ=[S2,M]
struct CopyL1ToL0BMxFp8A5 {
    __aicore__ inline CopyL1ToL0BMxFp8A5() {}

    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                      uint32_t s2Align64, uint32_t dAct, uint32_t s2Base, uint32_t actScaleSrcStride,
                                      uint32_t s1Align64, uint32_t s2Start = 0)
    {
        constexpr uint32_t NZ_C0 = Block::MXFP8::NZ_C0_ELEMS;
        constexpr uint32_t FP8_C0 = Block::MXFP8::FP8_C0_ELEMS;
        constexpr uint32_t MX_FRAC = Block::MXFP8::CONST_64;

        AscendC::LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = s2Start / NZ_C0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.mStep = (s2Align64 + NZ_C0 - 1) / NZ_C0;
        loadData2DParamsB.mStep = (loadData2DParamsB.mStep + 1) >> 1 << 1;
        loadData2DParamsB.kStep = (s1Align64 + FP8_C0 - 1) / FP8_C0;
        loadData2DParamsB.srcStride = (s2Base + NZ_C0 - 1) / NZ_C0;
        loadData2DParamsB.dstStride = s1Align64 / NZ_C0;
        loadData2DParamsB.ifTranspose = true;

        AscendC::LoadData2DMxParams load2DMxParamsB;
        load2DMxParamsB.xStartPosition = 0;
        load2DMxParamsB.yStartPosition = s2Start / MX_FRAC;
        load2DMxParamsB.xStep = (s1Align64 + NZ_C0 - 1) / NZ_C0;
        load2DMxParamsB.yStep = (s2Align64 + MX_FRAC - 1) / MX_FRAC;
        load2DMxParamsB.srcStride = (actScaleSrcStride == 0) ? (s2Base / MX_FRAC + 1) : actScaleSrcStride;
        load2DMxParamsB.dstStride = load2DMxParamsB.yStep;

        AscendC::LoadData(dst, src, scale, loadData2DParamsB, load2DMxParamsB);
    }
};

// QK: K → L0A（可沿 D 切段，kStart 单位为元素）
struct CopyL1ToL0AMxFp8QKA5 {
    __aicore__ inline CopyL1ToL0AMxFp8QKA5() {}

    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                      uint32_t nSubRowStart, uint32_t nCur, uint32_t alignK, uint32_t s2Align16Full,
                                      uint32_t kStart = 0, uint32_t scaleRowStart = 0)
    {
        constexpr uint32_t NZ_C0 = Block::MXFP8::NZ_C0_ELEMS;
        constexpr uint32_t FP8_C0 = Block::MXFP8::FP8_C0_ELEMS;
        uint32_t nCurAlign16 = (nCur + NZ_C0 - 1) / NZ_C0 * NZ_C0;
        // scaleK = Ceil(D/32)；半段 D=64 → scale 组数 = alignK/32，yStep = alignK/64
        uint32_t scaleRowHalf = (alignK + 63) / 64;

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = nSubRowStart / NZ_C0;
        loadDataParams.kStartPosition = kStart / FP8_C0; // 32B / fp8 C0
        loadDataParams.mStep = nCurAlign16 / NZ_C0;
        loadDataParams.kStep = (alignK + FP8_C0 - 1) / FP8_C0;
        loadDataParams.srcStride = s2Align16Full / NZ_C0;
        loadDataParams.dstStride = nCurAlign16 / NZ_C0;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = scaleRowStart / NZ_C0;
        loadMxDataParams.yStartPosition = kStart / Block::MXFP8::CONST_64;
        loadMxDataParams.xStep = nCurAlign16 / NZ_C0;
        loadMxDataParams.yStep = scaleRowHalf;
        loadMxDataParams.srcStride = scaleRowHalf;
        loadMxDataParams.dstStride = scaleRowHalf;

        AscendC::LoadData(dst, src, scale, loadDataParams, loadMxDataParams);
    }
};

struct CopyL1ToL0BMxFp8QKA5 {
    __aicore__ inline CopyL1ToL0BMxFp8QKA5() {}

    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, TensorScale const& scale,
                                      uint32_t mAlignL1, uint32_t alignK, uint32_t mAlignL0, uint32_t scaleMAlignL1,
                                      uint32_t kStart = 0)
    {
        constexpr uint32_t NZ_C0 = Block::MXFP8::NZ_C0_ELEMS;
        constexpr uint32_t FP8_C0 = Block::MXFP8::FP8_C0_ELEMS;
        uint32_t scaleRowHalf = (alignK + 63) / 64;

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = kStart / FP8_C0;
        loadDataParams.mStep = (mAlignL1 + NZ_C0 - 1) / NZ_C0;
        loadDataParams.kStep = (alignK + FP8_C0 - 1) / FP8_C0;
        loadDataParams.srcStride = (mAlignL1 + NZ_C0 - 1) / NZ_C0;
        loadDataParams.dstStride = (mAlignL0 + NZ_C0 - 1) / NZ_C0;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = 0;
        loadMxDataParams.yStartPosition = kStart / Block::MXFP8::CONST_64;
        loadMxDataParams.xStep = (scaleMAlignL1 + NZ_C0 - 1) / NZ_C0;
        loadMxDataParams.yStep = scaleRowHalf;
        loadMxDataParams.srcStride = scaleRowHalf;
        loadMxDataParams.dstStride = scaleRowHalf;

        AscendC::LoadData(dst, src, scale, loadDataParams, loadMxDataParams);
    }
};

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L1_TO_L0_MX_FP8_A5_HPP

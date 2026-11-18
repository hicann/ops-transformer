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
 * \file msa_block_mmad_arch35.h
 * \brief Atlas A5 BlockMmad：Q 驻留 + K pingpong；C_to_UB 时 L0C Fixpipe dual-dst → AIV UB。
 *
 * 基于 Catlass FullLoadA：同一 M-tile 只搬一次 Q；相邻 K page 的 FIXPIPE 与 Cube 重叠。
 * Atlas A5：计算路径照搬 A2；L0C=256KB，两块 128x128 fp32 C 各 64KB。
 * C_to_UB：int8 dual-dst 写 fp32。非量化 dual-dst 禁止 quantPre!=NoQuant（CANN 断言）；
 * 改为两次 dualDstCtl=0 + subBlockId 的 F322F16，写出 packed [M/2,128] fp16（对齐 FA）。
 */

#ifndef MSA_BLOCK_MMAD_ARCH35_H
#define MSA_BLOCK_MMAD_ARCH35_H

#include <type_traits>

#include "../catlass/msa_catlass.hpp"
#include "../catlass/arch/msa_arch.hpp"
#include "../catlass/arch/msa_resource.hpp"
#include "../catlass/msa_coord.hpp"
#include "../catlass/msa_matrix_coord.hpp"
#include "../catlass/msa_gemm_coord.hpp"
#include "../catlass/layout/msa_layout.hpp"
#include "../catlass/gemm/msa_gemm_helper.hpp"
#include "../catlass/gemm/tile/msa_tile_copy.hpp"
#include "../catlass/gemm/tile/msa_tile_mmad.hpp"

#include "../msa_index_score_common.h"
#include "../msa_index_score_task.h"

namespace MsaIndexScoreNs {

template <class AType_, class BType_, class CType_, uint32_t STAGES_IN = MSA_L1B_STAGES_FP16,
          bool USE_UNIT_FLAG_ = false, bool IS_MXFP4_ = false, bool MXFP4_N512_ = false, bool MX_FIXED_TILE_ = false,
          bool GROUP_N256_ = false>
class MsaBlockMmad {
public:
    static constexpr bool IS_MXFP4 = IS_MXFP4_;
    // Core only enables this for MXFP4 block128 C2UB: packed K<=64,
    // Mround<=128 and grouped Nround<=the selected L1_N (256 or512).
    static constexpr bool MX_FIXED_TILE = MX_FIXED_TILE_;
    static constexpr bool GROUP_N256 = GROUP_N256_;
    static_assert(!MX_FIXED_TILE || IS_MXFP4, "fixed-tile specialization requires MXFP4");
    // 只有四页 fp16 行放得进每个 AIV 的 32KB ping 时才用 N=512，并把 L0 第二级让出去。
    static constexpr bool MXFP4_WIDE = IS_MXFP4 && MXFP4_N512_;
    using ArchTag = Catlass::Arch::AtlasA5;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using BiasType = void;
    using TileCopy = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType>;
    using TileMmad = Catlass::Gemm::Tile::TileMmad<ArchTag, AType_, BType_, BiasType>;
    using CopyGmToL1A = typename TileCopy::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy::CopyL0CToGm;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = Catlass::layout::zN;
    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    // K (B) L1 stages are selected by the caller. The int8 scratch path uses two stages
    // to avoid racing the next vector cast; other paths use the configured FP16 stages.
    static constexpr uint32_t STAGES = STAGES_IN;
    // 非量化路径开 unit flag：mmad 与 fixpipe 的依赖交给硬件互锁，省掉每页
    // M_FIX/FIX_M 四次标量 set/wait。标量因此能在 cube 完成前就发出下一页的
    // MTE2/MTE1，把 GM→L1 的搬运藏到 cube/fixpipe 后面。
    static constexpr bool USE_UNIT_FLAG = USE_UNIT_FLAG_;
    // 宽路径 N=512：L0C 用满 256KB 单级放下 M=128×N=512（128KB×2 会超），
    // 换掉 M-split（双 MMAD + 双 L0B 装载），保留 Q 整片驻留。代价是下一窗 MMAD
    // 无法与本窗 Fixpipe 叠 L0C；实测 M-split 的 mte1/同步气泡更贵。
    // L0B 仍单级：N=512 的 B tile 占满 64KB。窄路径继续 128 行、N=256、双 L0C。
    static constexpr uint32_t L1_M = MSA_ROW_TILE_M;
    static constexpr uint32_t L1_N =
        MXFP4_WIDE ? (MSA_BLOCK_SIZE * 4U) : (IS_MXFP4 || GROUP_N256 ? (MSA_BLOCK_SIZE * 2U) : MSA_BLOCK_SIZE);
    static constexpr uint32_t L0C_M = L1_M;
    static constexpr uint32_t L0C_STAGES = MXFP4_WIDE ? 1U : 2U;
    static constexpr uint32_t L1_K = MSA_K_TILE;
    static constexpr uint32_t L1A_SIZE = L1_M * L1_K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1_N * L1_K * sizeof(ElementB);
    static constexpr uint32_t L1_SCALE_A_SIZE = (L1_M * (L1_K / MSA_MX_GROUP) + 511U) & ~511U;
    static constexpr uint32_t L1_SCALE_B_SIZE = (L1_N * (L1_K / MSA_MX_GROUP) + 511U) & ~511U;
    // L0A/L0B 各仅 64KB，B tile 32KB → 乒乓固定 2 级（32KB/级）；与 L1B STAGES 解耦。
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0A_RESIDENT_ID = 0;
    static constexpr uint32_t L0A_FALLBACK_ID = 1;
    static constexpr uint32_t L0B_STAGES = MXFP4_WIDE ? 1U : 2U;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0A_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0B_STAGES;
    static constexpr uint32_t L0C_TILE_SIZE = L0C_M * L1_N * sizeof(ElementAccumulator);

    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "LayoutC must be RowMajor (fixpipe nz2nd)");
    // A2 的 L0A/L0B 各仅 64KB：A/B tile 32KB，乒乓最多 2 级（32KB/级）。
    static_assert(L0A_PINGPONG_BUF_SIZE >= L1A_SIZE, "L0A pingpong buf smaller than A tile");
    static_assert(L0B_PINGPONG_BUF_SIZE >= L1B_SIZE, "L0B pingpong buf smaller than B tile");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0C pingpong exceeds L0C");
    static_assert(L1A_SIZE + L1B_SIZE * STAGES <= ArchTag::L1_SIZE, "L1 A+B exceeds L1");

    __aicore__ inline MsaBlockMmad(Catlass::Arch::Resource<ArchTag>& resource)
    {
        l1ATensor_ = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        const uint32_t l1BOffset = L1A_SIZE;
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1BTensorList_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l1BEventList_[i] = static_cast<int32_t>(i + STAGES);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
        }

        if constexpr (IS_MXFP4) {
            static_assert(L1A_SIZE + L1B_SIZE * STAGES == MsaMxL1Layout<STAGES, MXFP4_WIDE>::SCALE_A_OFFSET,
                          "AIC/AIV MX scale handoff layout must match");
            const uint32_t scaleBase = MsaMxL1Layout<STAGES, MXFP4_WIDE>::SCALE_A_OFFSET;
            l1ScaleA_ = resource.l1Buf.template GetBufferByByte<fp8_e8m0_t>(scaleBase);
            for (uint32_t i = 0; i < STAGES; ++i) {
                l1ScaleBList_[i] = resource.l1Buf.template GetBufferByByte<fp8_e8m0_t>(scaleBase + L1_SCALE_A_SIZE +
                                                                                       L1_SCALE_B_SIZE * i);
            }
        }
        // L0A 槽 0 专供常驻 Q：不预置 M_MTE1，改用 LoadA 内「Set+Wait 成对」自平衡，
        // 避免计数型 flag 让 MTE1 用构造时的余额提前放行、覆盖仍被 mmad 读取的 Q。
        // 槽 1 仅供窄路径的多 K-tile 回退；宽路径不使用该槽，故不预置其同步事件。
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0AEventList_[i] = static_cast<int32_t>(i);
        }
        if constexpr (!MXFP4_WIDE) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0BEventList_[i] = static_cast<int32_t>(i + STAGES + L0A_STAGES);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList_[i] = static_cast<int32_t>(i);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
    }

    __aicore__ inline void SetBnsdQueryGather(uint32_t heads, uint32_t seq, uint32_t row)
    {
        qGatherHeads_ = heads;
        qGatherSeq_ = seq;
        qGatherRow_ = row;
    }

    __aicore__ inline ~MsaBlockMmad()
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
        }

        if constexpr (!MXFP4_WIDE) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[i]);
        }
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<ElementA>& gmA, const LayoutA& layoutA,
                                      const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                      const AscendC::GlobalTensor<ElementC>& gmC, const LayoutC& layoutC,
                                      const Catlass::GemmCoord& actualShape, bool needLoadL1, uint32_t loadN = 0U,
                                      bool bLoaded = false)
    {
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        const uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        // 尾块只从 GM 读有效列。Mmad 的 N 仍按整块，多余列由 epilogue 在取 max 前盖掉。
        const uint32_t nLoad = (loadN == 0U || loadN > actualShape.n()) ? actualShape.n() : loadN;
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1_M, actualShape.k());
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1_K, L1_N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRound, nRound));
        uint32_t kActual = MX_FIXED_TILE ? actualShape.k() : Min(actualShape.k(), L1_K);

        const uint32_t kTileCount = MX_FIXED_TILE ? 1U : CeilDiv<L1_K>(actualShape.k());
        const uint32_t mPartLoop = MX_FIXED_TILE ? 1U : CeilDiv<L1_M>(mRound);
        const uint32_t nPartLoop = MX_FIXED_TILE ? 1U : CeilDiv<L1_N>(nRound);
        // headDim <= L1_K（实际场景恒成立）时整块 Q 一次进 L0A，可跨 page 复用；
        // 否则退回逐 page 重搬 A 的通用路径。
        const bool qResident =
            MX_FIXED_TILE ? true : ((kTileCount == 1U) && (mPartLoop == 1U) && (CeilDiv<L1_K>(kActual) == 1U));

        if (needLoadL1) {
            // L1A overwrite is protected by MTE1_MTE2, L0A by M_MTE1 below.
            auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), actualShape.k()));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_);
            if constexpr (IS_MXFP4) {
                if (qGatherHeads_ != 0U) {
                    // BNSD head segments are gathered into token-major rows directly in L1.
                    // Segment boundaries are multiples of16 rows, including E8M0 DN2NZ tiles.
                    const uint32_t groups = kActual * 2U / MSA_MX_GROUP;
                    for (uint32_t r = 0U; r < actualShape.m();) {
                        const uint32_t logical = qGatherRow_ + r;
                        const uint32_t token = logical / qGatherHeads_;
                        const uint32_t head = logical % qGatherHeads_;
                        const uint32_t rows = Min(qGatherHeads_ - head, actualShape.m() - r);
                        const uint64_t srcRow = static_cast<uint64_t>(head) * qGatherSeq_ + token;
                        const LayoutA source(rows, kActual, qGatherSeq_ * kActual);
                        const Catlass::MatrixCoord dstRow{r, 0U};
                        copyGmToL1A_(l1ATensor_[layoutAInL1.GetOffset(dstRow)], gmA[srcRow * kActual], layoutAInL1,
                                     source);
                        if (qGatherHeads_ % 16U == 0U) {
                            CopyScaleGmToL1(l1ScaleA_[r * groups], gmScaleA_[srcRow * groups], rows, kActual * 2U,
                                            qGatherSeq_ * (groups / 2U));
                        } // Other head counts receive their complete NZ scale tile from AIV UB.

                        r += rows;
                    }
                } else {
                    copyGmToL1A_(l1ATensor_, gmA, layoutAInL1, layoutTileA);
                    CopyScaleGmToL1(l1ScaleA_, gmScaleA_, actualShape.m(), kActual * 2U);
                }
            } else {
                copyGmToL1A_(l1ATensor_, gmA, layoutAInL1, layoutTileA);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_);
            if (qResident) {
                // 本 M-tile 的 Q 一次搬进 L0A 槽 0 并驻留：省掉每页 32KB 的 L1→L0A
                // 搬运与两次标量同步。先让 MTE1 等上一轮 mmad 读完该槽。
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_RESIDENT_ID]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_RESIDENT_ID]);
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
                if constexpr (IS_MXFP4) {
                    LoadMxL1ToL0A(l0ATensorList_[L0A_RESIDENT_ID], l1ATensor_, l1ScaleA_, mRound, kActual);
                } else {
                    copyL1ToL0A_(l0ATensorList_[L0A_RESIDENT_ID], l1ATensor_, layoutAInL0, layoutAInL1);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[L0A_RESIDENT_ID]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[L0A_RESIDENT_ID]);
            }
        }

        if (!bLoaded) {
            LoadBTile(gmB, layoutB, kActual, nLoad);
        }

        // 等本槽位上一轮 FIXPIPE 完成后再写 L0C，从而与另一槽位的 FIXPIPE 重叠。
        // unit flag 下该依赖由硬件维护，省掉这次标量等待。
        if constexpr (!USE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            const uint32_t l1ListIdNext = (l1ListId_ + 1 < STAGES) ? (l1ListId_ + 1) : 0;
            uint32_t kActualNext = 0;

            if (kLoopIdx < kTileCount - 1) {
                const uint32_t kLoopIdxNext = kLoopIdx + 1;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ? L1_K : (actualShape.k() - kLoopIdxNext * L1_K);
                Catlass::MatrixCoord gmTileBOffset{kLoopIdxNext * L1_K, 0};
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B_(l1BTensorList_[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListIdNext]);
            }

            auto l1ATensor = l1ATensor_;
            auto l1BTensor = l1BTensorList_[l1ListId_];
            const uint32_t kPartLoop = MX_FIXED_TILE ? 1U : CeilDiv<L1_K>(kActual);

            for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
                const uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L1_M : (mRound - mPartIdx * L1_M);
                for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                    const uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L1_K : (kActual - kPartIdx * L1_K);
                    auto l0ATile = l0ATensorList_[qResident ? L0A_RESIDENT_ID : L0A_FALLBACK_ID];
                    if (!qResident) {
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                        Catlass::MatrixCoord l1AOffset{mPartIdx * L1_M, kPartIdx * L1_K + kLoopIdx * L1_K};
                        auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];
                        if constexpr (MXFP4_WIDE) {
                            // 宽路径未预置 FALLBACK，回退时用成对 Set+Wait。
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
                        copyL1ToL0A_(l0ATile, l1ATile, layoutAInL0, layoutAInL1);
                    }

                    for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                        const uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L1_N : (nRound - nPartIdx * L1_N);
                        auto l0BTile = l0BTensorList_[l0BListId_];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        Catlass::MatrixCoord l1BOffset{kPartIdx * L1_K, nPartIdx * L1_N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);

                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);
                        }

                        if constexpr (IS_MXFP4) {
                            LoadMxL1ToL0B(l0BTile, l1BTile, l1ScaleBList_[l1ListId_], nPartActual, kPartActual);
                        } else {
                            copyL1ToL0B_(l0BTile, l1BTile, layoutBInL0, layoutBInL1);
                        }
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        Catlass::MatrixCoord l0COffset{mPartIdx * L1_M, nPartIdx * L1_N};
                        auto l0CTile = l0CTensorList_[l0CListId_][layoutInL0C.GetOffset(l0COffset)];
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        const bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        uint8_t unitFlag = 0b00;
                        if constexpr (USE_UNIT_FLAG) {
                            const bool last = (kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                              (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1);
                            unitFlag = last ? 0b11 : 0b10;
                        }
                        if constexpr (IS_MXFP4) {
                            AscendC::MmadParams mmadParams;
                            mmadParams.m = mPartActual;
                            mmadParams.n = nPartActual;
                            mmadParams.k = kPartActual * 2U;
                            mmadParams.unitFlag = unitFlag;
                            mmadParams.cmatrixInitVal = initC;
                            mmadParams.disableGemv = true;
                            AscendC::MmadMx(l0CTile, l0ATile.template ReinterpretCast<fp4x2_e2m1_t>(),
                                            l0BTile.template ReinterpretCast<fp4x2_e2m1_t>(), mmadParams);
                            if ((mPartActual / 16U) * (nPartActual / 16U) < 10U) {
                                AscendC::PipeBarrier<PIPE_M>();
                            }
                        } else {
                            tileMmad_(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC,
                                      unitFlag);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
                        l0BListId_ = (l0BListId_ + 1 < L0B_STAGES) ? (l0BListId_ + 1) : 0;
                    }
                    if (!qResident) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[L0A_FALLBACK_ID]);
                    }
                }
            }
            l1ListId_ = l1ListIdNext;
            kActual = kActualNext;
        }

        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
        if (storeToUb_) {
            if (deferStore_) {
                // VC 下沉：MMAD 已写入当前 L0C 槽，等调用方拿到 VC 后再 FlushStoreToUb。
                pendingStoreShape_ = actualShape;
                pendingL0CListId_ = l0CListId_;
            } else {
                StoreToUb(actualShape);
                l0CListId_ = (l0CListId_ + 1 < L0C_STAGES) ? (l0CListId_ + 1) : 0;
            }
        } else if constexpr (USE_UNIT_FLAG) {
            copyL0CToGm_(gmC, l0CTensorList_[l0CListId_], layoutBlock, layoutInL0C, 0b11);
            l0CListId_ = (l0CListId_ + 1 < L0C_STAGES) ? (l0CListId_ + 1) : 0;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
            copyL0CToGm_(gmC, l0CTensorList_[l0CListId_], layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
            l0CListId_ = (l0CListId_ + 1 < L0C_STAGES) ? (l0CListId_ + 1) : 0;
        }
    }

    /// VC 下沉配套：把挂起的 L0C 结果 Fixpipe 到 UB，再释放 FIX_M / 推进 L0C 槽。
    /// earlyCv=true 时在 fp16 双路 Fixpipe 之间提前发 CV0/CV1；调用方勿再重复 SetFlag CV。
    __aicore__ inline void FlushStoreToUb(const AscendC::LocalTensor<float>& ubC, uint32_t cvPing = 0,
                                          bool earlyCv = false)
    {
        ubC_ = ubC;
        earlyCv_ = earlyCv;
        syncPing_ = cvPing;
        l0CListId_ = pendingL0CListId_;
        StoreToUb(pendingStoreShape_);
        l0CListId_ = (pendingL0CListId_ + 1 < L0C_STAGES) ? (pendingL0CListId_ + 1) : 0;
        earlyCv_ = false;
    }

    /// L0C → 配对 AIV UB（dualDstCtl=1，按 M 对半）。ubC 已是 pingpong slot 起点。
    /// splitGroup 为两个 AIV 之间的行分组粒度：默认 1 即原有的偶数对半；
    /// epilogue 必须用同一粒度调 MsaC2UbGroupSplit，否则两侧对不上。
    __aicore__ inline void ComputeToUbNative(const AscendC::GlobalTensor<ElementA>& gmA, const LayoutA& layoutA,
                                             const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                             const AscendC::LocalTensor<float>& ubC,
                                             const Catlass::GemmCoord& actualShape, bool needLoadL1,
                                             uint32_t splitGroup = 1U, int32_t vcPing = -1)
    {
        ComputeToUb(gmA, layoutA, gmB, layoutB, ubC, actualShape, needLoadL1, splitGroup, 0U, false, false,
                    static_cast<uint32_t>(vcPing));
    }

    __aicore__ inline void ComputeToUb(const AscendC::GlobalTensor<ElementA>& gmA, const LayoutA& layoutA,
                                       const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                       const AscendC::LocalTensor<float>& ubC, const Catlass::GemmCoord& actualShape,
                                       bool needLoadL1, uint32_t splitGroup = 1U, uint32_t loadN = 0U,
                                       bool bLoaded = false, bool earlyCv = false,
                                       uint32_t cvPing = (IS_MXFP4 ? 0U : ~0U))
    {
        storeToUb_ = true;
        deferStore_ = false;
        ubC_ = ubC;
        splitGroup_ = splitGroup;
        earlyCv_ = earlyCv;
        syncPing_ = cvPing;
        AscendC::GlobalTensor<ElementC> dummyGm;
        const LayoutC dummyLayout(actualShape.m(), actualShape.n(), actualShape.n());
        operator()(gmA, layoutA, gmB, layoutB, dummyGm, dummyLayout, actualShape, needLoadL1, loadN, bLoaded);
        storeToUb_ = false;
        earlyCv_ = false;
    }

    /// 只做到 MMAD 写入空闲 L0C，不碰 UB；调用方等 VC 后再 FlushStoreToUb。
    __aicore__ inline void MmadToL0C(const AscendC::GlobalTensor<ElementA>& gmA, const LayoutA& layoutA,
                                     const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                     const Catlass::GemmCoord& actualShape, bool needLoadL1, uint32_t splitGroup = 1U,
                                     uint32_t loadN = 0U, bool bLoaded = false)
    {
        storeToUb_ = true;
        deferStore_ = true;
        splitGroup_ = splitGroup;
        AscendC::GlobalTensor<ElementC> dummyGm;
        const LayoutC dummyLayout(actualShape.m(), actualShape.n(), actualShape.n());
        operator()(gmA, layoutA, gmB, layoutB, dummyGm, dummyLayout, actualShape, needLoadL1, loadN, bLoaded);
        storeToUb_ = false;
        deferStore_ = false;
    }

    __aicore__ inline void ComputeToUbMx(const AscendC::GlobalTensor<ElementA>& gmA, const LayoutA& layoutA,
                                         const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                         const AscendC::GlobalTensor<fp8_e8m0_t>& gmScaleA,
                                         const AscendC::GlobalTensor<fp8_e8m0_t>& gmScaleB,
                                         const AscendC::LocalTensor<float>& ubC, const Catlass::GemmCoord& actualShape,
                                         bool needLoadL1, uint32_t splitGroup = 1U, uint32_t loadN = 0U,
                                         bool bLoaded = false, bool earlyCv = false, uint32_t cvPing = 0U)
    {
        gmScaleA_ = gmScaleA;
        gmScaleB_ = gmScaleB;
        ComputeToUb(gmA, layoutA, gmB, layoutB, ubC, actualShape, needLoadL1, splitGroup, loadN, bLoaded, earlyCv,
                    cvPing);
    }

    __aicore__ inline void MmadToL0CMx(const AscendC::GlobalTensor<ElementA>& gmA, const LayoutA& layoutA,
                                       const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                       const AscendC::GlobalTensor<fp8_e8m0_t>& gmScaleA,
                                       const AscendC::GlobalTensor<fp8_e8m0_t>& gmScaleB,
                                       const Catlass::GemmCoord& actualShape, bool needLoadL1, uint32_t splitGroup = 1U,
                                       uint32_t loadN = 0U, bool bLoaded = false)
    {
        gmScaleA_ = gmScaleA;
        gmScaleB_ = gmScaleB;
        MmadToL0C(gmA, layoutA, gmB, layoutB, actualShape, needLoadL1, splitGroup, loadN, bLoaded);
    }

    // 把一页 K 追加到当前 L1B 槽的 nOffset 列。first 申请槽，last 才发布 MTE2_MTE1。
    // 随后 bLoaded 的 MMAD 一次消费这一组，槽号由 MMAD 推进。
    __aicore__ inline void AppendBPage(const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutPage,
                                       uint32_t nOffset, bool first, bool last)
    {
        if (first) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
        }
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1_K, L1_N);
        const Catlass::MatrixCoord coord{0U, nOffset};
        auto dstB = l1BTensorList_[l1ListId_][layoutBInL1.GetOffset(coord)];
        copyGmToL1B_(dstB, gmB, layoutBInL1, layoutPage);
        if (last) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);
        }
    }

    // 只把当前槽的 K（和 scale）从 GM 搬进 L1，不占 UB、不启动 MMAD。
    // 调用方可以在等 AIV 交还上一组 UB 之前发出，让 MTE2 叠在 Vector epilogue 上。
    __aicore__ inline void LoadBToL1(const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                     const AscendC::GlobalTensor<fp8_e8m0_t>& gmScaleB,
                                     const Catlass::GemmCoord& actualShape, uint32_t loadN = 0U)
    {
        gmScaleB_ = gmScaleB;
        const uint32_t nLoad = (loadN == 0U || loadN > actualShape.n()) ? actualShape.n() : loadN;
        const uint32_t kActual = Min(actualShape.k(), L1_K);
        LoadBTile(gmB, layoutB, kActual, nLoad);
    }

    // A logical block128 group can contain unrelated physical PA pages.
    // Acquire the slot once, append each page in logical N order, publish only after the final copy.
    __aicore__ inline void LoadBPageToL1(const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                         const AscendC::GlobalTensor<fp8_e8m0_t>& gmScaleB, uint32_t kActual,
                                         uint32_t nOffset, bool first, bool last, uint32_t rows = MSA_BLOCK_SIZE)
    {
        static_assert(IS_MXFP4, "PA gather is an MXFP4-only loader");
        if (first) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
        }
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1_K, L1_N);
        const Catlass::MatrixCoord coord{0U, nOffset};
        auto dstB = l1BTensorList_[l1ListId_][layoutBInL1.GetOffset(coord)];
        copyGmToL1B_(dstB, gmB, layoutBInL1, layoutB);
        // DN2NZ scale segments start at16-row boundaries (physical PA pages are16/128 rows).
        const uint32_t scaleOffset = nOffset * (kActual * 2U / MSA_MX_GROUP);
        CopyScaleGmToL1(l1ScaleBList_[l1ListId_][scaleOffset], gmScaleB, rows, kActual * 2U);
        if (last) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);
        }
    }

private:
    static constexpr AscendC::FixpipeConfig MSA_CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void LoadBTile(const AscendC::GlobalTensor<ElementB>& gmB, const LayoutB& layoutB,
                                     uint32_t kActual, uint32_t nLoad)
    {
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1_K, L1_N);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListId_]);
        auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActual, nLoad));
        copyGmToL1B_(l1BTensorList_[l1ListId_], gmB, layoutBInL1, layoutTileB);
        if constexpr (IS_MXFP4) {
            CopyScaleGmToL1(l1ScaleBList_[l1ListId_], gmScaleB_, nLoad, kActual * 2U);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListId_]);
    }

    __aicore__ inline void CopyScaleGmToL1(const AscendC::LocalTensor<fp8_e8m0_t>& l1Scale,
                                           const AscendC::GlobalTensor<fp8_e8m0_t>& gmScale, uint32_t rows, uint32_t k,
                                           uint32_t srcRowStride = 0U)
    {
        const uint32_t halfCols = k / (MSA_MX_GROUP * 2U);
        auto gmHalf = gmScale.template ReinterpretCast<half>();
        auto l1Half = l1Scale.template ReinterpretCast<half>();
        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = rows;
        dn2nzParams.nValue = halfCols;
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = srcRowStride == 0U ? halfCols : srcRowStride;
        dn2nzParams.dstNzC0Stride = halfCols;
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1Half, gmHalf, dn2nzParams);
    }

    __aicore__ inline void LoadMxL1ToL0A(const AscendC::LocalTensor<ElementA>& l0A,
                                         const AscendC::LocalTensor<ElementA>& l1A,
                                         const AscendC::LocalTensor<fp8_e8m0_t>& l1ScaleA, uint32_t mRound, uint32_t k)
    {
        // MXFP4：调用方传入的 k 是打包字节数（逻辑 D/2）。LoadData 的 kStep 按 64 个逻辑 E2M1。
        const uint32_t kLogical = IS_MXFP4 ? (k * 2U) : k;
        const uint32_t halfCols = kLogical / (MSA_MX_GROUP * 2U);
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = mRound / 16U;
        loadDataParams.kStep = IS_MXFP4 ? (kLogical / 64U) : (k / 32U);
        loadDataParams.srcStride = L1_M / 16U;
        loadDataParams.dstStride = mRound / 16U;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = 0;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.xStep = mRound / 16U;
        loadMxDataParams.yStep = halfCols;
        loadMxDataParams.srcStride = halfCols;
        loadMxDataParams.dstStride = halfCols;
        if constexpr (IS_MXFP4) {
            AscendC::LoadData(l0A.template ReinterpretCast<fp4x2_e2m1_t>(),
                              l1A.template ReinterpretCast<fp4x2_e2m1_t>(), l1ScaleA, loadDataParams, loadMxDataParams);
        } else {
            AscendC::LoadData(l0A, l1A, l1ScaleA, loadDataParams, loadMxDataParams);
        }
    }

    __aicore__ inline void LoadMxL1ToL0B(const AscendC::LocalTensor<ElementB>& l0B,
                                         const AscendC::LocalTensor<ElementB>& l1B,
                                         const AscendC::LocalTensor<fp8_e8m0_t>& l1ScaleB, uint32_t nRound, uint32_t k)
    {
        const uint32_t kLogical = IS_MXFP4 ? (k * 2U) : k;
        const uint32_t halfCols = kLogical / (MSA_MX_GROUP * 2U);
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = nRound / 16U;
        loadDataParams.kStep = IS_MXFP4 ? (kLogical / 64U) : (k / 32U);
        loadDataParams.srcStride = L1_N / 16U;
        loadDataParams.dstStride = nRound / 16U;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = 0;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.xStep = nRound / 16U;
        loadMxDataParams.yStep = halfCols;
        loadMxDataParams.srcStride = halfCols;
        loadMxDataParams.dstStride = halfCols;
        if constexpr (IS_MXFP4) {
            AscendC::LoadData(l0B.template ReinterpretCast<fp4x2_e2m1_t>(),
                              l1B.template ReinterpretCast<fp4x2_e2m1_t>(), l1ScaleB, loadDataParams, loadMxDataParams);
        } else {
            AscendC::LoadData(l0B, l1B, l1ScaleB, loadDataParams, loadMxDataParams);
        }
    }

    __aicore__ inline void StoreToUb(const Catlass::GemmCoord& actualShape)
    {
        if constexpr (!IS_MXFP4) {
            if (syncPing_ != ~0U) {
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + syncPing_);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + syncPing_ +
                                                                        MSA_A5_AIV_FLAG_OFFSET);
            }
        }

        const uint32_t mEven = (actualShape.m() + 1U) & ~1U;
        const uint32_t nAlign = (actualShape.n() + 7U) & ~7U;
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        if constexpr (!USE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[l0CListId_]);
        }
        if constexpr (sizeof(ElementC) == 2) {
            uint32_t m0 = 0;
            uint32_t m1 = 0;
            MsaC2UbGroupSplit(actualShape.m(), splitGroup_, m0, m1);
            auto ubH = ubC_.template ReinterpretCast<half>();
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.srcStride = mRound;
            fp.dualDstCtl = 0;
            fp.quantPre = QuantMode_t::F322F16;
            fp.nSize = nAlign;
            fp.dstStride = nAlign;
            fp.params.ndNum = 1;
            fp.params.srcNdStride = 0;
            fp.params.dstNdStride = 0;
            if (m0 > 0U) {
                fp.mSize = m0;
                fp.subBlockId = false;
                AscendC::Fixpipe<half, float, MSA_CFG_ROW_MAJOR_UB>(ubH, l0CTensorList_[l0CListId_], fp);
            }
            // 提前 CV0：AIV0 数据就位后立即放行，可与第二路 Fixpipe 重叠归约。
            if (earlyCv_) {
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + syncPing_);
            }
            if (m1 > 0U) {
                fp.mSize = m1;
                fp.subBlockId = true;
                AscendC::Fixpipe<half, float, MSA_CFG_ROW_MAJOR_UB>(ubH, l0CTensorList_[l0CListId_][m0 * 16U], fp);
            }
            if (earlyCv_) {
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + syncPing_ +
                                                                       MSA_A5_AIV_FLAG_OFFSET);
            }
        } else {
            // 950 L0C→UB：nSize 上限 64。ndNum=2 用 panel 布局（QLI）：每 AIV 先 [M/2,64]
            // 再紧接第二块 [M/2,64]，总 32KB ping。
            constexpr uint32_t N_PER_ND = 64;
            const uint32_t rowsPerAiv = mEven / MSA_AIV_PER_AIC;
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = mEven;
            fp.srcStride = mRound;
            fp.dstStride = N_PER_ND;
            fp.dualDstCtl = 1;
            fp.quantPre = QuantMode_t::NoQuant;
            fp.nSize = N_PER_ND;
            fp.params.ndNum = (nAlign > N_PER_ND) ? (nAlign / N_PER_ND) : 1;
            if (fp.params.ndNum > 1) {
                fp.params.srcNdStride = ((mEven + 15U) / 16U) * N_PER_ND;
                fp.params.dstNdStride = rowsPerAiv * N_PER_ND;
            } else {
                fp.nSize = nAlign;
                fp.params.srcNdStride = 0;
                fp.params.dstNdStride = 0;
            }
            AscendC::Fixpipe<float, float, MSA_CFG_ROW_MAJOR_UB>(ubC_, l0CTensorList_[l0CListId_], fp);
        }
        if constexpr (!USE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[l0CListId_]);
        }
    }
    AscendC::LocalTensor<ElementA> l1ATensor_;
    AscendC::LocalTensor<ElementB> l1BTensorList_[STAGES];
    AscendC::LocalTensor<fp8_e8m0_t> l1ScaleA_;
    AscendC::LocalTensor<fp8_e8m0_t> l1ScaleBList_[STAGES];
    AscendC::GlobalTensor<fp8_e8m0_t> gmScaleA_;
    AscendC::GlobalTensor<fp8_e8m0_t> gmScaleB_;
    AscendC::LocalTensor<ElementA> l0ATensorList_[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList_[L0C_STAGES];
    int32_t l1AEvent_{0};
    int32_t l1BEventList_[STAGES]{};
    int32_t l0AEventList_[L0A_STAGES]{};
    int32_t l0BEventList_[L0B_STAGES]{};
    int32_t l0CEventList_[L0C_STAGES]{};
    uint32_t qGatherHeads_{0U};
    uint32_t qGatherSeq_{0U};
    uint32_t qGatherRow_{0U};
    uint32_t l1ListId_{0};
    uint32_t l0BListId_{0};
    uint32_t l0CListId_{0};
    TileMmad tileMmad_;
    CopyGmToL1A copyGmToL1A_;
    CopyGmToL1B copyGmToL1B_;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    CopyL0CToGm copyL0CToGm_;
    bool storeToUb_{false};
    bool deferStore_{false};
    bool earlyCv_{false};
    // MX uses this ping for early CV; native uses it for deferred VC.
    uint32_t syncPing_{IS_MXFP4 ? 0U : ~0U};
    uint32_t splitGroup_{1U};
    uint32_t pendingL0CListId_{0};
    Catlass::GemmCoord pendingStoreShape_{};
    AscendC::LocalTensor<float> ubC_;
};

} // namespace MsaIndexScoreNs

#endif // MSA_BLOCK_MMAD_ARCH35_H

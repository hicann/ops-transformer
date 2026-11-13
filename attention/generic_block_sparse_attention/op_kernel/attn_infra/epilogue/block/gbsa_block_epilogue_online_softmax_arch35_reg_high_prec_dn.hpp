/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_DN_HPP
#define GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_DN_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"

namespace NpuArch::Epilogue::Block {

// DN (transposed-S) variant of the high-precision online softmax epilogue,
// selected by EpilogueOnlineSoftmaxUnion. The QK mmad computes K * Q^T, so
// S^T lands row-major in UB: one row is one KV token across this sub-block's
// query columns, and the per-query softmax state (gm/dm/gl) is a plain vector
// over the row lanes (no horizontal reductions). Structure follows BSA's
// ComputeOnlineSoftmaxDN: four KV rows per iteration with independent
// max/sum accumulators, and P is stored straight into a padded nZ staging
// (strided block stores) that two DataCopy calls move to L1. The GBSA union
// exact mask is folded into pass 1: scores are scaled, masked to MIN_VALUE
// and written back, so pass 2 needs no selects.
template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueOnlineSoftmaxUnion, OutputType_, Gemm::GemmType<float, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxUnion;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = float;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutSType = LayoutS_;

    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr float MIN_VALUE = -3.402823466e+38F;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, float scaleValue_,
                                    UBufTileHelper& uBufTileHelper)
        : scaleValue(scaleValue_)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        sPitch_ = uBufTileHelper.qBaseTilePerSubCore;
        // One P staging stage spans (kvTile/2+1)*128 half elements (BSA
        // strided-store pitch plus the second copy's source offset); this
        // must match the kernel's CalcUBufTileInfo stage sizing.
        const uint32_t pStageElemNum = (uBufTileHelper.kvBaseTilePerSubCore / 2U + 1U) * 128U;
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(float) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset + pStageElemNum * sizeof(ElementOutput) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; ++i) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset + UBufTileHelper::STATS_ROW_NUM * sizeof(float) * i);
        }
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.lmStartOffset);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.llStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        // UB is shared between the two vector sub-blocks, and both issue the
        // exact-mask copy for their row half. Give each sub-block a private
        // pair of stages: otherwise the faster sub-block's copy for the next
        // task (issued while its sibling still prescales the current task)
        // overwrites the mask the sibling is reading.
        for (uint32_t i = 0; i < 2U; ++i) {
            exactMaskUbTensor[i] = resource.ubBuf.template GetBufferByByte<int32_t>(
                uBufTileHelper.exactMaskStartOffset +
                (subBlockIdx_ * 2U + i) * uBufTileHelper.kvBaseTilePerSubCore * sizeof(int32_t));
        }
        partialMaxUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.partialMaxStartOffset);
    }

    // Union exact-mask mode: gExactMask holds one int32 bitmask per KV token
    // (bit `slot` = visibility of the slot-th query of the group). Query rows
    // are laid out [slot][head], so each slot owns groupSize consecutive lanes.
    __aicore__ inline void SetExactMask(AscendC::GlobalTensor<int32_t>& exactMask, uint32_t groupSize)
    {
        gExactMask_ = exactMask;
        exactMaskGroupSize_ = groupSize;
        exactMaskEnabled_ = true;
        // Prime both stages' read-done events: neither stage has been read
        // yet, so the first refill of each must not block. From then on the
        // prescale of each tile re-arms the event of the stage it read.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ExactMaskReadDoneEventId(0U));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ExactMaskReadDoneEventId(1U));
    }

    // Fully-masked rows keep gm == MIN_VALUE after the last tile; give them
    // gl = 1 so RescaleO yields O = 0/1 = 0 and LSE = log(1) + MIN_VALUE
    // instead of 0/0 = NaN. Call after the last tile's softmax, before rescale.
    __aicore__ inline void FixEmptyRows(uint16_t totalRows)
    {
        if (!exactMaskEnabled_) {
            return;
        }
        uint32_t nCopyOffset = RoundUp(static_cast<uint32_t>(totalRows), 32U) / 2U;
        uint32_t m = totalRows < nCopyOffset ? totalRows : nCopyOffset;
        m = subBlockIdx_ == 0 ? m : totalRows - m;
        if (m == 0) {
            return;
        }
        __ubuf__ float* gmUbAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* glUbAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        FixEmptyRowsVf(gmUbAddr, glUbAddr, static_cast<uint16_t>(m), EmptyRowMax());
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Row-max value of a row with no visible column: masked scores are
    // selected to MIN_VALUE after scaling, so the accumulated max of a
    // fully-masked row is exactly MIN_VALUE regardless of the scale.
    __aicore__ inline float EmptyRowMax() const
    {
        return MIN_VALUE;
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    // Double-buffered exact-mask fetch: stage parity rotates per KV tile and
    // the next tile's segment is prefetched while this tile computes, so the
    // MTE2 copy overlaps the QK cross-core wait of the following tile.
    static __aicore__ inline event_t ExactMaskEventId(uint32_t stage)
    {
        return stage == 0U ? EVENT_ID7 : EVENT_ID5;
    }

    // Per-stage "V pipe finished reading this stage" event (V -> scalar):
    // a refill copy may only be issued once the prescale that last read the
    // stage has completed. EVENT_ID2/3 are unused elsewhere in this kernel
    // (init seeds V_MTE2 ID0/1; rescale-o uses ID6/7 on a non-union path).
    static __aicore__ inline event_t ExactMaskReadDoneEventId(uint32_t stage)
    {
        return stage == 0U ? EVENT_ID2 : EVENT_ID3;
    }

    __aicore__ inline void IssueExactMaskCopy(uint32_t stage, uint64_t gmOffset, uint32_t cols)
    {
        AscendC::DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(cols * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<int32_t> maskPadParams{false, 0U, 0U, 0};
        AscendC::DataCopyPad(exactMaskUbTensor[stage], gExactMask_[gmOffset], maskCopyParams, maskPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ExactMaskEventId(stage));
    }

    template <class TensorP>
    __aicore__ inline void operator()(TensorP& l1PTensorTla, GemmCoord actualBlockShape, uint32_t isFirstKvSTile,
                                      uint32_t ubSBufId, uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag, uint64_t exactMaskTileOffset = 0,
                                      uint64_t exactMaskNextTileOffset = 0, uint32_t exactMaskNextCols = 0)
    {
        uint32_t nCopyOffset = RoundUp(actualBlockShape.m(), 32) / 2;
        uint32_t n = actualBlockShape.m() < nCopyOffset ? actualBlockShape.m() : nCopyOffset;
        n = subBlockIdx_ == 0 ? n : actualBlockShape.m() - n;
        if (n == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }

        uint32_t m = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        // Alter the nd2nz dst stride (typically 64 -> 65) to alleviate bank conflict.
        const uint32_t blockStride = mRound / 2 + 1;

        __ubuf__ float* sUbAddr = (__ubuf__ float*)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput* pUbAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float* gmUbAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* glUbAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ float* dmUbAddr = (__ubuf__ float*)dmUbTensor[l1PBufId].GetPhyAddr();

        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        __ubuf__ int32_t* maskUbAddr = nullptr;
        uint32_t maskReadStage = 0;
        if (exactMaskEnabled_) {
            // The mask copies below are issued by the scalar side, which the
            // V-pipe cross-core wait above does not stall: on long tile
            // streams it can lap the V pipe and overwrite a stage whose
            // prescale has not run yet. Gate every refill on the read-done
            // event the prescale raises after consuming a stage (primed in
            // SetExactMask, re-armed after each tile's prescale below).
            if (isFirstKvSTile) {
                // First tile of a task has nothing prefetched yet. Advance
                // the stage rotation instead of resetting to stage 0, so
                // the rotation runs continuously across tasks and, like the
                // intra-task prefetch below, a refill only ever targets the
                // stage read two tiles ago.
                exactMaskStage_ ^= 1U;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ExactMaskReadDoneEventId(exactMaskStage_));
                IssueExactMaskCopy(exactMaskStage_, exactMaskTileOffset, m);
            }
            maskReadStage = exactMaskStage_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ExactMaskEventId(maskReadStage));
            maskUbAddr = (__ubuf__ int32_t*)exactMaskUbTensor[maskReadStage].GetPhyAddr();
            if (exactMaskNextCols != 0U) {
                const uint32_t nextStage = maskReadStage ^ 1U;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ExactMaskReadDoneEventId(nextStage));
                IssueExactMaskCopy(nextStage, exactMaskNextTileOffset, exactMaskNextCols);
                exactMaskStage_ = nextStage;
            }
        }
        const uint32_t rowBase = subBlockIdx_ == 0 ? 0U : nCopyOffset;
        // S^T rows are pitched by the UB S layout's column count
        // (RoundUp(qBaseTile / 2, 16) floats of the tiling, not of this
        // task's actual row count), not by the vector width: GBSA tiles
        // can give each sub-block fewer than 64 query columns.
        const uint32_t sPitch = sPitch_;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        uint32_t mAlignedTile = mRound / 4;
        uint32_t mFirstTile = m % mAlignedTile;
        uint32_t mAlignedTileNum = m / mAlignedTile;
        if (mAlignedTileNum == 4U) {
            mFirstTile = mAlignedTile;
        }
        if (exactMaskEnabled_) {
            if (isFirstKvSTile) {
                ComputeSoftmaxDnDispatch<true, true>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr, gmUbAddr, dmUbAddr,
                                                     glUbAddr, m, n, mRound, blockStride, rowBase, exactMaskGroupSize_,
                                                     sPitch);
            } else {
                ComputeSoftmaxDnDispatch<false, true>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr, gmUbAddr, dmUbAddr,
                                                      glUbAddr, m, n, mRound, blockStride, rowBase, exactMaskGroupSize_,
                                                      sPitch);
            }
        } else {
            if (isFirstKvSTile) {
                ComputeSoftmaxDnDispatch<true, false>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr, gmUbAddr, dmUbAddr,
                                                      glUbAddr, m, n, mRound, blockStride, rowBase, exactMaskGroupSize_,
                                                      sPitch);
            } else {
                ComputeSoftmaxDnDispatch<false, false>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr, gmUbAddr,
                                                       dmUbAddr, glUbAddr, m, n, mRound, blockStride, rowBase,
                                                       exactMaskGroupSize_, sPitch);
            }
        }
        if (exactMaskEnabled_) {
            // The prescale above was the last read of this tile's mask
            // stage; release it so its refill (next tile's prefetch, or the
            // next task's first-tile copy) may be issued.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ExactMaskReadDoneEventId(maskReadStage));
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = nRound / ELE_NUM_PER_C0;
        dataCopyParams.blockLen = mRound / 2;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = mRound / 2;
        AscendC::DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset], lpUbTensor[ubSBufId],
                          dataCopyParams);
        AscendC::DataCopy(l1PTensorTla.data()[mRound * 8 + subBlockIdx_ * mRound * nCopyOffset],
                          lpUbTensor[ubSBufId][blockStride * 64], dataCopyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    template <bool IsFirstKvSTile, bool Masked>
    __aicore__ inline void ComputeSoftmaxDnDispatch(uint32_t mAlignedTileNum, __ubuf__ float* sUb,
                                                    __ubuf__ int32_t* maskUb, __ubuf__ ElementOutput* pUb,
                                                    __ubuf__ float* gmUb, __ubuf__ float* dmUb, __ubuf__ float* glUb,
                                                    uint32_t m, uint32_t n, uint16_t mRound, uint32_t blockStride,
                                                    uint32_t rowBase, uint32_t groupSize, uint32_t sPitch)
    {
        // Scaling after centering avoids cancellation at large score offsets.
        // Retain the established signed-scale path for nonpositive scales.
        if (scaleValue > 0.0f) {
            ComputeSoftmaxDnDispatchImpl<IsFirstKvSTile, Masked, true>(mAlignedTileNum, sUb, maskUb, pUb, gmUb, dmUb,
                                                                       glUb, m, n, mRound, blockStride, rowBase,
                                                                       groupSize, sPitch);
            return;
        }
        ComputeSoftmaxDnDispatchImpl<IsFirstKvSTile, Masked, false>(
            mAlignedTileNum, sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, sPitch);
    }

    template <bool IsFirstKvSTile, bool Masked, bool StableCenter>
    __aicore__ inline void ComputeSoftmaxDnDispatchImpl(uint32_t mAlignedTileNum, __ubuf__ float* sUb,
                                                        __ubuf__ int32_t* maskUb, __ubuf__ ElementOutput* pUb,
                                                        __ubuf__ float* gmUb, __ubuf__ float* dmUb,
                                                        __ubuf__ float* glUb, uint32_t m, uint32_t n, uint16_t mRound,
                                                        uint32_t blockStride, uint32_t rowBase, uint32_t groupSize,
                                                        uint32_t sPitch)
    {
        // Non-quant union default tile is kvBaseTile=128 with qBase/2=64 lanes
        // (FP8 raises kv to 256; do not copy that geometry here).
        if (m == 128U && n == 64U && sPitch == 64U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 4U, true, StableCenter>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch);
            return;
        }
        if (mAlignedTileNum == 0U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 0U, false, StableCenter>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch);
        } else if (mAlignedTileNum == 1U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 1U, false, StableCenter>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch);
        } else if (mAlignedTileNum == 2U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 2U, false, StableCenter>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch);
        } else if (mAlignedTileNum == 3U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 3U, false, StableCenter>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch);
        } else {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 4U, false, StableCenter>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch);
        }
    }

    template <bool IsFirstKvSTile, bool StableCenter>
    static __simd_callee__ inline void ExpProbability(AscendC::Reg::RegTensor<float>& dst,
                                                      AscendC::Reg::RegTensor<float>& src,
                                                      AscendC::Reg::RegTensor<float>& maximum,
                                                      AscendC::Reg::RegTensor<float>& rawMaximum,
                                                      AscendC::Reg::RegTensor<float>& offset, float scale,
                                                      AscendC::Reg::MaskReg& mask)
    {
        using namespace AscendC::Reg;
        if constexpr (StableCenter) {
            Sub(dst, src, rawMaximum, mask);
            Muls(dst, dst, scale, mask);
            if constexpr (!IsFirstKvSTile) {
                Add(dst, dst, offset, mask);
            }
            Exp(dst, dst, mask);
        } else {
            FusedExpSub(dst, src, maximum, mask);
        }
    }

    // Two-pass DN softmax over the S^T tile (m KV rows x 64 query lanes).
    // Positive scales center raw scores before multiplication in pass 2.
    // The running max stays scaled for the existing rescale/LSE interface.
    // Nonpositive scales retain the established computation. Pass 2 re-reads the
    // rows in quarter order, exponentiates against the merged row max and
    // stores P into the padded nZ staging via strided block stores. The
    // MTileNum dispatch tells pass 2 how many of the four quarter row
    // groups hold real tokens, so garbage rows never pollute the sum.
    // Pass-1 only rewrites [0, m); rows [m, mRound) are filled with MIN_VALUE
    // so pass-2's unguarded P stores exp to 0 instead of leaking prior-tile
    // residue into MM2's fractal-aligned K.
    template <bool IsFirstKvSTile, bool Masked, uint32_t MTileNum, bool FullTile = false, bool StableCenter = false>
    static __simd_vf__ inline void ComputeSoftmaxDn(__ubuf__ float* sUb, __ubuf__ int32_t* maskUb,
                                                    __ubuf__ ElementOutput* pUb, __ubuf__ float* gmUb,
                                                    __ubuf__ float* dmUb, __ubuf__ float* glUb, uint32_t mArg,
                                                    uint32_t nArg, uint16_t mRoundArg, uint32_t blockStride,
                                                    uint32_t rowBase, uint32_t groupSize, float scaleValue,
                                                    uint32_t sPitchArg)
    {
        const uint32_t m = FullTile ? 128U : mArg;
        uint32_t n = FullTile ? 64U : nArg;
        const uint16_t mRound = FullTile ? 128U : mRoundArg;
        const uint32_t sPitch = FullTile ? 64U : sPitchArg;
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_ROUND};
        RegTensor<float> src0Vreg, src1Vreg, src2Vreg, src3Vreg, src4Vreg;
        RegTensor<float> exp0Vreg, exp1Vreg, exp2Vreg, exp3Vreg;
        RegTensor<float> max0Vreg, max1Vreg, max2Vreg, max3Vreg;
        RegTensor<float> sum0Vreg, sum1Vreg, sum2Vreg, sum3Vreg;
        RegTensor<float> hmVreg, gmVreg, dmVreg, glVreg, rawMaxVreg, offsetVreg;
        RegTensor<float> minVreg, hugeVreg;
        RegTensor<ElementOutput> expHalf0Vreg, expHalf1Vreg, expHalf2Vreg, expHalf3Vreg;
        RegTensor<ElementOutput> deInterleave0Vreg, deInterleave1Vreg, deInterleave2Vreg, deInterleave3Vreg;
        RegTensor<int32_t> wordVreg, slotVreg, idxVreg, boundVreg, oneVreg, slotBitVreg, zeroVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(n);
        uint32_t vlHalf = 2U * FLOAT_VECTOR_SIZE;
        MaskReg pregFp16VL128 = UpdateMask<ElementOutput>(vlHalf);
        MaskReg visPreg, boundPreg, pregEmpty, vis0, vis1, vis2, vis3;
        RegTensor<int32_t> mask0, mask1, mask2, mask3;
        MaskReg fullInt = CreateMask<int32_t, MaskPattern::ALL>();
        Duplicate(minVreg, MIN_VALUE);
        Duplicate(hugeVreg, 1.0e30F);
        if constexpr (Masked) {
            // slot(lane) = (rowBase + lane) / groupSize, built exactly by
            // stepping over the slot boundaries (lanes are [slot][head]).
            Arange<int32_t>(idxVreg, 0);
            Duplicate(oneVreg, 1);
            if (groupSize == 1U) {
                Adds(slotVreg, idxVreg, static_cast<int32_t>(rowBase), fullInt);
            } else {
                const uint32_t slot0 = rowBase / groupSize;
                Duplicate(slotVreg, static_cast<int32_t>(slot0));
                for (uint32_t boundary = (slot0 + 1U) * groupSize - rowBase; boundary < FLOAT_VECTOR_SIZE;
                     boundary += groupSize) {
                    Duplicate(boundVreg, static_cast<int32_t>(boundary));
                    Compare<int32_t, AscendC::CMPMODE::GE>(boundPreg, idxVreg, boundVreg, fullInt);
                    Add<int32_t, MaskMergeMode::MERGING>(slotVreg, slotVreg, oneVreg, boundPreg);
                }
            }
        }

        if constexpr (Masked) {
            // Decode each visibility word with a precomputed query bit.
            ShiftLeft(slotBitVreg, oneVreg, slotVreg, fullInt);
            Duplicate(zeroVreg, 0);
        }
        // ---- Pass 1: mask, store scores and reduce the per-query max ----
        Duplicate(max0Vreg, MIN_VALUE);
        Duplicate(max1Vreg, MIN_VALUE);
        Duplicate(max2Vreg, MIN_VALUE);
        Duplicate(max3Vreg, MIN_VALUE);
        for (uint32_t i = 0; i < m / 4U; ++i) {
            __ubuf__ float* row0 = sUb + i * sPitch * 4U;
            LoadAlign(src0Vreg, row0);
            LoadAlign(src1Vreg, row0 + sPitch);
            LoadAlign(src2Vreg, row0 + 2U * sPitch);
            LoadAlign(src3Vreg, row0 + 3U * sPitch);
            if constexpr (Masked) {
                // Keep four mask dependency chains independent so loads and
                // predicate generation overlap across the four score vectors.
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(mask0, maskUb + i * 4U + 0U);
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(mask1, maskUb + i * 4U + 1U);
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(mask2, maskUb + i * 4U + 2U);
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(mask3, maskUb + i * 4U + 3U);
                And(mask0, mask0, slotBitVreg, fullInt);
                And(mask1, mask1, slotBitVreg, fullInt);
                And(mask2, mask2, slotBitVreg, fullInt);
                And(mask3, mask3, slotBitVreg, fullInt);
                Compare<int32_t, AscendC::CMPMODE::NE>(vis0, mask0, zeroVreg, fullInt);
                Compare<int32_t, AscendC::CMPMODE::NE>(vis1, mask1, zeroVreg, fullInt);
                Compare<int32_t, AscendC::CMPMODE::NE>(vis2, mask2, zeroVreg, fullInt);
                Compare<int32_t, AscendC::CMPMODE::NE>(vis3, mask3, zeroVreg, fullInt);
                if constexpr (!StableCenter) {
                    Muls(src0Vreg, src0Vreg, scaleValue, pregTailN);
                    Muls(src1Vreg, src1Vreg, scaleValue, pregTailN);
                    Muls(src2Vreg, src2Vreg, scaleValue, pregTailN);
                    Muls(src3Vreg, src3Vreg, scaleValue, pregTailN);
                }
                Select(src0Vreg, src0Vreg, minVreg, vis0);
                Select(src1Vreg, src1Vreg, minVreg, vis1);
                Select(src2Vreg, src2Vreg, minVreg, vis2);
                Select(src3Vreg, src3Vreg, minVreg, vis3);
            } else {
                if constexpr (!StableCenter) {
                    Muls(src0Vreg, src0Vreg, scaleValue, pregTailN);
                    Muls(src1Vreg, src1Vreg, scaleValue, pregTailN);
                    Muls(src2Vreg, src2Vreg, scaleValue, pregTailN);
                    Muls(src3Vreg, src3Vreg, scaleValue, pregTailN);
                }
            }
            StoreAlign(row0, src0Vreg, pregTailN);
            StoreAlign(row0 + sPitch, src1Vreg, pregTailN);
            StoreAlign(row0 + 2U * sPitch, src2Vreg, pregTailN);
            StoreAlign(row0 + 3U * sPitch, src3Vreg, pregTailN);
            Max(max0Vreg, max0Vreg, src0Vreg, pregTailN);
            Max(max1Vreg, max1Vreg, src1Vreg, pregTailN);
            Max(max2Vreg, max2Vreg, src2Vreg, pregTailN);
            Max(max3Vreg, max3Vreg, src3Vreg, pregTailN);
        }
        for (uint32_t i = 0; i < m % 4U; ++i) {
            __ubuf__ float* row = sUb + (m / 4U) * sPitch * 4U + i * sPitch;
            LoadAlign(src4Vreg, row);
            if constexpr (Masked) {
                LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(wordVreg, maskUb + (m / 4U) * 4U + i);
                And(wordVreg, wordVreg, slotBitVreg, fullInt);
                Compare<int32_t, AscendC::CMPMODE::NE>(visPreg, wordVreg, zeroVreg, fullInt);
                if constexpr (!StableCenter) {
                    Muls(src4Vreg, src4Vreg, scaleValue, pregTailN);
                }
                Select(src4Vreg, src4Vreg, minVreg, visPreg);
            } else {
                if constexpr (!StableCenter) {
                    Muls(src4Vreg, src4Vreg, scaleValue, pregTailN);
                }
            }
            StoreAlign(row, src4Vreg, pregTailN);
            Max(max0Vreg, max0Vreg, src4Vreg, pregTailN);
        }
        // Pad unused S rows: pass-2 stores P for the full mRound fractal without
        // an m guard, and MM2 consumes that width on K.
        for (uint32_t i = m; i < static_cast<uint32_t>(mRound); ++i) {
            StoreAlign(sUb + i * sPitch, minVreg, pregTailN);
        }
        Max(max0Vreg, max0Vreg, max2Vreg, pregTailN);
        Max(max1Vreg, max1Vreg, max3Vreg, pregTailN);
        Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);

        if constexpr (StableCenter) {
            rawMaxVreg = max0Vreg;
            Compare<float, AscendC::CMPMODE::EQ>(pregEmpty, rawMaxVreg, minVreg, pregFull);
            Muls(max0Vreg, max0Vreg, scaleValue, pregFull);
            Select(max0Vreg, minVreg, max0Vreg, pregEmpty);
        }
        // ---- Merge with the running state; dm = exp(gm - hm) ----
        if constexpr (IsFirstKvSTile) {
            StoreAlign(gmUb, max0Vreg, pregFull);
            hmVreg = max0Vreg;
        } else {
            LoadAlign(gmVreg, gmUb);
            Max(hmVreg, max0Vreg, gmVreg, pregFull);
            FusedExpSub(dmVreg, gmVreg, hmVreg, pregFull);
            StoreAlign(gmUb, hmVreg, pregFull);
            StoreAlign(dmUb, dmVreg, pregFull);
        }
        // Rows with no visible token yet keep gm == MIN_VALUE; substituting a
        // huge finite max makes their exp underflow to 0 in pass 2 (gmUb
        // itself stays MIN_VALUE for FixEmptyRows).
        Compare<float, AscendC::CMPMODE::EQ>(pregEmpty, hmVreg, minVreg, pregFull);
        Select(hmVreg, hugeVreg, hmVreg, pregEmpty);
        if constexpr (StableCenter) {
            Select(rawMaxVreg, hugeVreg, rawMaxVreg, pregEmpty);
            if constexpr (!IsFirstKvSTile) {
                Sub(offsetVreg, max0Vreg, hmVreg, pregFull);
            }
        }

        // ---- Pass 2: exponentiate in quarter order, store P to staging ----
        __ubuf__ ElementOutput* exp0Ub = pUb;
        __ubuf__ ElementOutput* exp1Ub = pUb + mRound * 4U;
        Duplicate(sum0Vreg, 0.0f);
        Duplicate(sum1Vreg, 0.0f);
        Duplicate(sum2Vreg, 0.0f);
        Duplicate(sum3Vreg, 0.0f);
        const uint32_t mAlignedTile = mRound / 4U;
        uint32_t mFirstTile = m % mAlignedTile;
        if constexpr (MTileNum == 4U) {
            mFirstTile = mAlignedTile;
        }
        for (uint32_t i = 0; i < mFirstTile; ++i) {
            LoadAlign(src0Vreg, sUb + i * sPitch);
            LoadAlign(src1Vreg, sUb + (mRound / 4U) * sPitch + i * sPitch);
            LoadAlign(src2Vreg, sUb + (mRound / 2U) * sPitch + i * sPitch);
            LoadAlign(src3Vreg, sUb + (mRound / 2U + mRound / 4U) * sPitch + i * sPitch);
            ExpProbability<IsFirstKvSTile, StableCenter>(exp0Vreg, src0Vreg, hmVreg, rawMaxVreg, offsetVreg, scaleValue,
                                                         pregTailN);
            ExpProbability<IsFirstKvSTile, StableCenter>(exp1Vreg, src1Vreg, hmVreg, rawMaxVreg, offsetVreg, scaleValue,
                                                         pregTailN);
            ExpProbability<IsFirstKvSTile, StableCenter>(exp2Vreg, src2Vreg, hmVreg, rawMaxVreg, offsetVreg, scaleValue,
                                                         pregTailN);
            ExpProbability<IsFirstKvSTile, StableCenter>(exp3Vreg, src3Vreg, hmVreg, rawMaxVreg, offsetVreg, scaleValue,
                                                         pregTailN);
            Cast<ElementOutput, float, castTraitZeroRound>(expHalf0Vreg, exp0Vreg, pregTailN);
            Cast<ElementOutput, float, castTraitZeroRound>(expHalf2Vreg, exp2Vreg, pregTailN);
            DeInterleave(deInterleave0Vreg, deInterleave1Vreg, expHalf0Vreg, expHalf2Vreg);
            Cast<ElementOutput, float, castTraitZeroRound>(expHalf1Vreg, exp1Vreg, pregTailN);
            Cast<ElementOutput, float, castTraitZeroRound>(expHalf3Vreg, exp3Vreg, pregTailN);
            DeInterleave(deInterleave2Vreg, deInterleave3Vreg, expHalf1Vreg, expHalf3Vreg);
            StoreAlign<ElementOutput, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                       AscendC::Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ ElementOutput*&)exp0Ub, deInterleave0Vreg,
                                                                    blockStride, 1, pregFp16VL128);
            Add(sum0Vreg, exp0Vreg, sum0Vreg, pregTailN);
            if constexpr (MTileNum >= 2U) {
                Add(sum2Vreg, exp2Vreg, sum2Vreg, pregTailN);
            }
            StoreAlign<ElementOutput, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                       AscendC::Reg::PostLiteral::POST_MODE_UPDATE>((__ubuf__ ElementOutput*&)exp1Ub, deInterleave2Vreg,
                                                                    blockStride, 1, pregFp16VL128);
            if constexpr (MTileNum >= 1U) {
                Add(sum1Vreg, exp1Vreg, sum1Vreg, pregTailN);
            }
            if constexpr (MTileNum >= 3U) {
                Add(sum3Vreg, exp3Vreg, sum3Vreg, pregTailN);
            }
        }
        if constexpr (MTileNum < 4U) {
            for (uint32_t i = mFirstTile; i < mRound / 4U; ++i) {
                LoadAlign(src0Vreg, sUb + i * sPitch);
                LoadAlign(src1Vreg, sUb + (mRound / 4U) * sPitch + i * sPitch);
                LoadAlign(src2Vreg, sUb + (mRound / 2U) * sPitch + i * sPitch);
                LoadAlign(src3Vreg, sUb + (mRound / 2U + mRound / 4U) * sPitch + i * sPitch);
                ExpProbability<IsFirstKvSTile, StableCenter>(exp0Vreg, src0Vreg, hmVreg, rawMaxVreg, offsetVreg,
                                                             scaleValue, pregTailN);
                ExpProbability<IsFirstKvSTile, StableCenter>(exp1Vreg, src1Vreg, hmVreg, rawMaxVreg, offsetVreg,
                                                             scaleValue, pregTailN);
                ExpProbability<IsFirstKvSTile, StableCenter>(exp2Vreg, src2Vreg, hmVreg, rawMaxVreg, offsetVreg,
                                                             scaleValue, pregTailN);
                ExpProbability<IsFirstKvSTile, StableCenter>(exp3Vreg, src3Vreg, hmVreg, rawMaxVreg, offsetVreg,
                                                             scaleValue, pregTailN);
                Cast<ElementOutput, float, castTraitZeroRound>(expHalf0Vreg, exp0Vreg, pregTailN);
                Cast<ElementOutput, float, castTraitZeroRound>(expHalf2Vreg, exp2Vreg, pregTailN);
                DeInterleave(deInterleave0Vreg, deInterleave1Vreg, expHalf0Vreg, expHalf2Vreg);
                Cast<ElementOutput, float, castTraitZeroRound>(expHalf1Vreg, exp1Vreg, pregTailN);
                Cast<ElementOutput, float, castTraitZeroRound>(expHalf3Vreg, exp3Vreg, pregTailN);
                DeInterleave(deInterleave2Vreg, deInterleave3Vreg, expHalf1Vreg, expHalf3Vreg);
                StoreAlign<ElementOutput, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                           AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ ElementOutput*&)exp0Ub, deInterleave0Vreg, blockStride, 1, pregFp16VL128);
                if constexpr (MTileNum > 0U) {
                    Add(sum0Vreg, exp0Vreg, sum0Vreg, pregTailN);
                }
                if constexpr (MTileNum > 2U) {
                    Add(sum2Vreg, exp2Vreg, sum2Vreg, pregTailN);
                }
                StoreAlign<ElementOutput, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                           AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ ElementOutput*&)exp1Ub, deInterleave2Vreg, blockStride, 1, pregFp16VL128);
                if constexpr (MTileNum > 1U) {
                    Add(sum1Vreg, exp1Vreg, sum1Vreg, pregTailN);
                }
            }
        }
        Add(sum0Vreg, sum0Vreg, sum2Vreg, pregFull);
        Add(sum1Vreg, sum1Vreg, sum3Vreg, pregFull);
        Add(sum0Vreg, sum0Vreg, sum1Vreg, pregFull);
        if constexpr (IsFirstKvSTile) {
            StoreAlign(glUb, sum0Vreg, pregFull);
        } else {
            LoadAlign(dmVreg, dmUb);
            LoadAlign(glVreg, glUb);
            Mul(glVreg, glVreg, dmVreg, pregFull);
            Add(glVreg, glVreg, sum0Vreg, pregFull);
            StoreAlign(glUb, glVreg, pregFull);
        }
    }

    static __simd_vf__ inline void FixEmptyRowsVf(__ubuf__ float* gmUb, __ubuf__ float* glUb, uint16_t m,
                                                  float emptyRowMax)
    {
        using namespace AscendC::Reg;
        RegTensor<float> gmVreg;
        RegTensor<float> glVreg;
        RegTensor<float> minVreg;
        RegTensor<float> emptyMaxVreg;
        RegTensor<float> oneVreg;
        MaskReg pregEmpty;
        uint32_t rowCount = m;
        MaskReg pregRows = UpdateMask<float>(rowCount);
        Duplicate(minVreg, MIN_VALUE);
        Duplicate(emptyMaxVreg, emptyRowMax);
        Duplicate(oneVreg, 1.0F);
        LoadAlign(gmVreg, gmUb);
        Compare<float, AscendC::CMPMODE::EQ>(pregEmpty, gmVreg, emptyMaxVreg, pregRows);
        LoadAlign(glVreg, glUb);
        Select(glVreg, oneVreg, glVreg, pregEmpty);
        StoreAlign(glUb, glVreg, pregRows);
        // Restore the plain MIN_VALUE row max so LSE matches the empty-row
        // convention (log(1) + MIN_VALUE) used by WriteEmptyOutput.
        Select(gmVreg, minVreg, gmVreg, pregEmpty);
        StoreAlign(gmUb, gmVreg, pregRows);
    }

private:
    float scaleValue;
    AscendC::LocalTensor<float> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<int32_t> exactMaskUbTensor[2];
    AscendC::LocalTensor<float> partialMaxUbTensor;
    AscendC::GlobalTensor<int32_t> gExactMask_;
    uint32_t exactMaskGroupSize_ = 1;
    uint32_t exactMaskStage_ = 0;
    bool exactMaskEnabled_ = false;
    uint32_t subBlockIdx_;
    uint32_t sPitch_ = FLOAT_VECTOR_SIZE;
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_DN_HPP

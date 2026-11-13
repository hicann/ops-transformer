/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_DN_S_FP32_P_FP8_HPP
#define GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_DN_S_FP32_P_FP8_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"

namespace NpuArch::Epilogue::Block {

// Union DN Softmax emits FP8 directly using four KV-quarter stripes.
// Or/Gather packs the byte lanes; two strided copies produce PV's L1 nZ.
// P uses the existing GBSA scale 448; gl remains unscaled for output and LSE.
template <class LayoutP_, class LayoutS_>
class BlockEpilogue<EpilogueOnlineSoftmaxUnion, Gemm::GemmType<fp8_e4m3fn_t, LayoutP_>,
                    Gemm::GemmType<float, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxUnion;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = fp8_e4m3fn_t;
    using LayoutOutput = LayoutP_;
    using ElementInput = float;
    using LayoutSType = LayoutS_;

    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t ELE_NUM_PER_C0_FP8 = 32;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr float MIN_VALUE = -3.402823466e+38F;
    static constexpr float MAX_VALUE_FP8 = 448.0f;
    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, float scaleValue_,
                                    UBufTileHelper& uBufTileHelper)
        : scaleValue(scaleValue_)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        sPitch_ = uBufTileHelper.qBaseTilePerSubCore;
        // Eight stripes, each a KV quarter plus one padding datablock.
        const uint32_t pStageBytes = (uBufTileHelper.kvBaseTilePerSubCore / 4U + 1U) * 256U;
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(float) * i);
            lpUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<ElementOutput>(uBufTileHelper.pStartOffset + pStageBytes * i);
        }
        // FP8 byte-lane gather table for the four KV-quarter stripes.
        gatherPIdxUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(
            uBufTileHelper.partialMaxStartOffset + UBufTileHelper::STATS_ROW_NUM * sizeof(float));
        // Four byte lanes -> four KV quarters, 32 queries per stripe.
        for (uint32_t quarter = 0; quarter < 4U; ++quarter) {
            for (uint32_t query = 0; query < 64U; ++query) {
                gatherPIdxUbTensor.SetValue(quarter * 64U + query, query * 4U + quarter);
            }
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
        // Pass 2 emits FP8 stripes directly.
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0_FP8);
        constexpr uint32_t eleNumPerC0 = ELE_NUM_PER_C0_FP8;
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
        // Softmax DN emits FP8 stripes directly.
        if (exactMaskEnabled_) {
            if (isFirstKvSTile) {
                ComputeSoftmaxDnDispatch<true, true, ElementOutput>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr,
                                                                    gmUbAddr, dmUbAddr, glUbAddr, m, n, mRound,
                                                                    blockStride, rowBase, exactMaskGroupSize_, sPitch);
            } else {
                ComputeSoftmaxDnDispatch<false, true, ElementOutput>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr,
                                                                     gmUbAddr, dmUbAddr, glUbAddr, m, n, mRound,
                                                                     blockStride, rowBase, exactMaskGroupSize_, sPitch);
            }
        } else {
            if (isFirstKvSTile) {
                ComputeSoftmaxDnDispatch<true, false, ElementOutput>(mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr,
                                                                     gmUbAddr, dmUbAddr, glUbAddr, m, n, mRound,
                                                                     blockStride, rowBase, exactMaskGroupSize_, sPitch);
            } else {
                ComputeSoftmaxDnDispatch<false, false, ElementOutput>(
                    mAlignedTileNum, sUbAddr, maskUbAddr, pUbAddr, gmUbAddr, dmUbAddr, glUbAddr, m, n, mRound,
                    blockStride, rowBase, exactMaskGroupSize_, sPitch);
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
        // FA stripe layout: alternate 32-query halves, four KV quarters.
        const uint32_t stripeBlocks = mRound / 4U;
        const uint32_t stripeStride = stripeBlocks + 1U;
        const uint32_t pRowsPerSubCore = RoundUp(nCopyOffset, 32U);
        const uint32_t subCoreOffset = subBlockIdx_ * mRound * pRowsPerSubCore;
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 4U;
        dataCopyParams.blockLen = stripeBlocks;
        dataCopyParams.srcStride = 2U * stripeStride - stripeBlocks;
        dataCopyParams.dstStride = 0U;
        AscendC::DataCopy(l1PTensorTla.data()[subCoreOffset], lpUbTensor[ubSBufId], dataCopyParams);
        if (nRound > 32U) {
            AscendC::DataCopy(l1PTensorTla.data()[subCoreOffset + 32U * mRound],
                              lpUbTensor[ubSBufId][stripeStride * 32U], dataCopyParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    template <bool IsFirstKvSTile, bool Masked, class StageT>
    __aicore__ inline void ComputeSoftmaxDnDispatch(uint32_t mAlignedTileNum, __ubuf__ float* sUb,
                                                    __ubuf__ int32_t* maskUb, __ubuf__ StageT* pUb,
                                                    __ubuf__ float* gmUb, __ubuf__ float* dmUb, __ubuf__ float* glUb,
                                                    uint32_t m, uint32_t n, uint16_t mRound, uint32_t blockStride,
                                                    uint32_t rowBase, uint32_t groupSize, uint32_t sPitch)
    {
        // Keep the full KV256/Q64 geometry compile-time constant inside the VF.
        // Partial tiles and smaller query groups retain the generic path.
        if (m == 256U && n == 64U && sPitch == 64U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 4U, StageT, true>(
                sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound, blockStride, rowBase, groupSize, scaleValue, sPitch,
                (__ubuf__ uint8_t*)gatherPIdxUbTensor.GetPhyAddr());
            return;
        }
        if (mAlignedTileNum == 0U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 0U, StageT>(sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound,
                                                                 blockStride, rowBase, groupSize, scaleValue, sPitch,
                                                                 (__ubuf__ uint8_t*)gatherPIdxUbTensor.GetPhyAddr());
        } else if (mAlignedTileNum == 1U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 1U, StageT>(sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound,
                                                                 blockStride, rowBase, groupSize, scaleValue, sPitch,
                                                                 (__ubuf__ uint8_t*)gatherPIdxUbTensor.GetPhyAddr());
        } else if (mAlignedTileNum == 2U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 2U, StageT>(sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound,
                                                                 blockStride, rowBase, groupSize, scaleValue, sPitch,
                                                                 (__ubuf__ uint8_t*)gatherPIdxUbTensor.GetPhyAddr());
        } else if (mAlignedTileNum == 3U) {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 3U, StageT>(sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound,
                                                                 blockStride, rowBase, groupSize, scaleValue, sPitch,
                                                                 (__ubuf__ uint8_t*)gatherPIdxUbTensor.GetPhyAddr());
        } else {
            ComputeSoftmaxDn<IsFirstKvSTile, Masked, 4U, StageT>(sUb, maskUb, pUb, gmUb, dmUb, glUb, m, n, mRound,
                                                                 blockStride, rowBase, groupSize, scaleValue, sPitch,
                                                                 (__ubuf__ uint8_t*)gatherPIdxUbTensor.GetPhyAddr());
        }
    }

    // Two-pass DN softmax over the S^T tile (m KV rows x 64 query lanes).
    // Pass 1 scales + masks each row, writes it back and accumulates the
    // per-query max into four independent accumulators. Pass 2 re-reads the
    // rows in quarter order, exponentiates against the merged row max and
    // stores P into the padded nZ staging via strided block stores. The
    // MTileNum dispatch tells pass 2 how many of the four quarter row
    // groups hold real tokens, so garbage rows never pollute the sum.
    // StageT is the FP8 output type; no half staging is used.
    template <bool IsFirstKvSTile, bool Masked, uint32_t MTileNum, class StageT, bool FullTile = false>
    static __simd_vf__ inline void ComputeSoftmaxDn(__ubuf__ float* sUb, __ubuf__ int32_t* maskUb, __ubuf__ StageT* pUb,
                                                    __ubuf__ float* gmUb, __ubuf__ float* dmUb, __ubuf__ float* glUb,
                                                    uint32_t mArg, uint32_t nArg, uint16_t mRoundArg,
                                                    uint32_t blockStride, uint32_t rowBase, uint32_t groupSize,
                                                    float scaleValue, uint32_t sPitchArg, __ubuf__ uint8_t* gatherUb)
    {
        const uint32_t m = FullTile ? 256U : mArg;
        uint32_t n = FullTile ? 64U : nArg;
        const uint16_t mRound = FullTile ? 256U : mRoundArg;
        const uint32_t sPitch = FullTile ? 64U : sPitchArg;
        using namespace AscendC::Reg;
        RegTensor<float> src0Vreg, src1Vreg, src2Vreg, src3Vreg, src4Vreg;
        RegTensor<float> exp0Vreg, exp1Vreg, exp2Vreg, exp3Vreg;
        RegTensor<float> max0Vreg, max1Vreg, max2Vreg, max3Vreg;
        RegTensor<float> sum0Vreg, sum1Vreg, sum2Vreg, sum3Vreg;
        RegTensor<float> hmVreg, gmVreg, dmVreg, glVreg;
        RegTensor<float> minVreg, hugeVreg;
        RegTensor<int32_t> wordVreg, slotVreg, idxVreg, boundVreg, oneVreg, slotBitVreg, zeroVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(n);
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
        // ---- Pass 1: scale, mask, store back, per-query max ----
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
                Muls(src0Vreg, src0Vreg, scaleValue, pregTailN);
                Muls(src1Vreg, src1Vreg, scaleValue, pregTailN);
                Muls(src2Vreg, src2Vreg, scaleValue, pregTailN);
                Muls(src3Vreg, src3Vreg, scaleValue, pregTailN);
                Select(src0Vreg, src0Vreg, minVreg, vis0);
                Select(src1Vreg, src1Vreg, minVreg, vis1);
                Select(src2Vreg, src2Vreg, minVreg, vis2);
                Select(src3Vreg, src3Vreg, minVreg, vis3);
            } else {
                Muls(src0Vreg, src0Vreg, scaleValue, pregTailN);
                Muls(src1Vreg, src1Vreg, scaleValue, pregTailN);
                Muls(src2Vreg, src2Vreg, scaleValue, pregTailN);
                Muls(src3Vreg, src3Vreg, scaleValue, pregTailN);
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
                Muls(src4Vreg, src4Vreg, scaleValue, pregTailN);
                Select(src4Vreg, src4Vreg, minVreg, visPreg);
            } else {
                Muls(src4Vreg, src4Vreg, scaleValue, pregTailN);
            }
            StoreAlign(row, src4Vreg, pregTailN);
            Max(max0Vreg, max0Vreg, src4Vreg, pregTailN);
        }
        // A clipped dense KV tail leaves unused rows in the FP8 P fractal.
        if constexpr (!FullTile) {
            for (uint32_t i = m; i < mRound; ++i) {
                StoreAlign(sUb + i * sPitch, minVreg, pregTailN);
            }
        }
        Max(max0Vreg, max0Vreg, max2Vreg, pregTailN);
        Max(max1Vreg, max1Vreg, max3Vreg, pregTailN);
        Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);

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

        // Subtract the unshifted max before scaling P. Shifting a large max
        // by ln(448) loses FP32 bits and biases the P/gl normalization.
        Duplicate(sum0Vreg, 0.0f);
        Duplicate(sum1Vreg, 0.0f);
        Duplicate(sum2Vreg, 0.0f);
        Duplicate(sum3Vreg, 0.0f);
        constexpr static CastTrait fp8Zero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait fp8One = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait fp8Two = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait fp8Three = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::CAST_RINT};
        RegTensor<ElementOutput> p0, packed;
        auto& p1 = (RegTensor<ElementOutput>&)exp0Vreg;
        auto& p2 = (RegTensor<ElementOutput>&)exp1Vreg;
        auto& p3 = (RegTensor<ElementOutput>&)exp2Vreg;
        RegTensor<uint8_t> gatherIndex;
        MaskReg allBytes = CreateMask<uint8_t, MaskPattern::ALL>();
        LoadAlign(gatherIndex, gatherUb);
        const uint32_t quarterSize = mRound / 4U;
        const uint32_t stripeStride = quarterSize + 1U;
        __ubuf__ ElementOutput* pStore = pUb;
        uint32_t firstSize = m % quarterSize;
        if constexpr (MTileNum == 4U)
            firstSize = quarterSize;
        for (uint32_t i = 0; i < firstSize; ++i) {
            LoadAlign(src0Vreg, sUb + i * sPitch);
            LoadAlign(src1Vreg, sUb + (i + quarterSize) * sPitch);
            LoadAlign(src2Vreg, sUb + (i + 2U * quarterSize) * sPitch);
            LoadAlign(src3Vreg, sUb + (i + 3U * quarterSize) * sPitch);
            FusedExpSub(exp0Vreg, src0Vreg, hmVreg, pregTailN);
            FusedExpSub(exp1Vreg, src1Vreg, hmVreg, pregTailN);
            FusedExpSub(exp2Vreg, src2Vreg, hmVreg, pregTailN);
            FusedExpSub(exp3Vreg, src3Vreg, hmVreg, pregTailN);
            if constexpr (MTileNum >= 0U)
                Add(sum0Vreg, exp0Vreg, sum0Vreg, pregTailN);
            if constexpr (MTileNum >= 1U)
                Add(sum1Vreg, exp1Vreg, sum1Vreg, pregTailN);
            if constexpr (MTileNum >= 2U)
                Add(sum2Vreg, exp2Vreg, sum2Vreg, pregTailN);
            if constexpr (MTileNum >= 3U)
                Add(sum3Vreg, exp3Vreg, sum3Vreg, pregTailN);
            Muls(exp0Vreg, exp0Vreg, MAX_VALUE_FP8, pregTailN);
            Muls(exp1Vreg, exp1Vreg, MAX_VALUE_FP8, pregTailN);
            Muls(exp2Vreg, exp2Vreg, MAX_VALUE_FP8, pregTailN);
            Muls(exp3Vreg, exp3Vreg, MAX_VALUE_FP8, pregTailN);
            Cast<ElementOutput, float, fp8Zero>(p0, exp0Vreg, pregTailN);
            Cast<ElementOutput, float, fp8One>(p1, exp1Vreg, pregTailN);
            Cast<ElementOutput, float, fp8Two>(p2, exp2Vreg, pregTailN);
            Cast<ElementOutput, float, fp8Three>(p3, exp3Vreg, pregTailN);
            Or((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p1, allBytes);
            Or((RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p3, allBytes);
            Or((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p2, allBytes);
            Gather(packed, p0, gatherIndex);
            StoreAlign<ElementOutput, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                       AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(pStore, packed, stripeStride, 1, allBytes);
        }
        if constexpr (MTileNum < 4U) {
            for (uint32_t i = firstSize; i < quarterSize; ++i) {
                LoadAlign(src0Vreg, sUb + i * sPitch);
                LoadAlign(src1Vreg, sUb + (i + quarterSize) * sPitch);
                LoadAlign(src2Vreg, sUb + (i + 2U * quarterSize) * sPitch);
                LoadAlign(src3Vreg, sUb + (i + 3U * quarterSize) * sPitch);
                FusedExpSub(exp0Vreg, src0Vreg, hmVreg, pregTailN);
                FusedExpSub(exp1Vreg, src1Vreg, hmVreg, pregTailN);
                FusedExpSub(exp2Vreg, src2Vreg, hmVreg, pregTailN);
                FusedExpSub(exp3Vreg, src3Vreg, hmVreg, pregTailN);
                if constexpr (MTileNum > 0U)
                    Add(sum0Vreg, exp0Vreg, sum0Vreg, pregTailN);
                if constexpr (MTileNum > 1U)
                    Add(sum1Vreg, exp1Vreg, sum1Vreg, pregTailN);
                if constexpr (MTileNum > 2U)
                    Add(sum2Vreg, exp2Vreg, sum2Vreg, pregTailN);
                if constexpr (MTileNum > 3U)
                    Add(sum3Vreg, exp3Vreg, sum3Vreg, pregTailN);
                Muls(exp0Vreg, exp0Vreg, MAX_VALUE_FP8, pregTailN);
                Muls(exp1Vreg, exp1Vreg, MAX_VALUE_FP8, pregTailN);
                Muls(exp2Vreg, exp2Vreg, MAX_VALUE_FP8, pregTailN);
                Muls(exp3Vreg, exp3Vreg, MAX_VALUE_FP8, pregTailN);
                Cast<ElementOutput, float, fp8Zero>(p0, exp0Vreg, pregTailN);
                Cast<ElementOutput, float, fp8One>(p1, exp1Vreg, pregTailN);
                Cast<ElementOutput, float, fp8Two>(p2, exp2Vreg, pregTailN);
                Cast<ElementOutput, float, fp8Three>(p3, exp3Vreg, pregTailN);
                Or((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p1, allBytes);
                Or((RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p3, allBytes);
                Or((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p2, allBytes);
                Gather(packed, p0, gatherIndex);
                StoreAlign<ElementOutput, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                           AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(pStore, packed, stripeStride, 1, allBytes);
            }
        }
        Add(sum0Vreg, sum0Vreg, sum2Vreg, pregFull);
        Add(sum1Vreg, sum1Vreg, sum3Vreg, pregFull);
        Add(sum0Vreg, sum0Vreg, sum1Vreg, pregFull);
        // gl was accumulated from unscaled FP32 exponentials.
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
    AscendC::LocalTensor<uint8_t> gatherPIdxUbTensor;
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

#endif // GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_DN_S_FP32_P_FP8_HPP

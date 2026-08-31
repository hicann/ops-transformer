/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/gemm_type.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

enum class KvBaseTileRegSplitStagesBf16 {
    ONE,
    TWO
};

template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueOnlineSoftmaxBsa, OutputType_, Gemm::GemmType<bfloat16_t, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = bfloat16_t;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t ELE_NUM_PER_C0_FP8 = 32;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t SM_UB_STAGES = 2; // gmUb/glUb 2-deep by ubSBufId
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);

    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementOutput, fp8_e4m3fn_t>::value;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Prefill local copy: multi-block online recursion stripped (each block computes
        // its own rowMax/rowSum and outputs gl/gm to GM per-slot; cross-block combine is
        // done by Phase2 FlashDecode). So lmUb/dmUb/llUb (used only by UpdateMax/
        // UpdateExpSumAndExpMax) are gone, and gmUb/glUb are 2-deep by ubSBufId to kill
        // the single-buffer reuse race (prefill rotates different qTokens on one AIV and
        // reads gmUb out via CopyPartialStatsToGm every block).
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        // 2-deep stats staging. Stage stride = 64 floats (per-AIV rowMax/rowSum cap,
        // SM_ROW_MAX_ELEM_NUM=64). gmUb[0..1] then glUb[0..1], packed in the freed tail.
        constexpr uint32_t STATS_UB_STAGE_BYTES = SM_ROW_MAX_ELEM_NUM * sizeof(float);
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;                        // 229376
        constexpr uint32_t GL_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 2 * STATS_UB_STAGE_BYTES; // 229888

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = AscendC::ToBfloat16(scaleValue_);
        MIN_VALUE = AscendC::ToBfloat16(-3.389531390315715675e+38);

        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        for (uint32_t i = 0; i < SM_UB_STAGES; i++) {
            gmUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET + i * STATS_UB_STAGE_BYTES);
            glUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET + i * STATS_UB_STAGE_BYTES);
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::DataCopyParams repeatParams;

        uint32_t elementNumPerC0;
        if constexpr (FULL_QUANT_FP8) {
            elementNumPerC0 = ELE_NUM_PER_C0_FP8;
        } else {
            elementNumPerC0 = ELE_NUM_PER_C0;
        }
        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / elementNumPerC0 - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / elementNumPerC0 - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    // Batched softmax: M = batchM (up to 128 = BATCH_GROUPS * groupSize). The batched QK
    // produces one S[batchM, validSize]; the 8 qTokens in the batch share the same KV block
    // (same validSize) but each group g (groupSize rows) has its own causalValidLens[g].
    // PV reduces over a uniform K = validSize (the block's valid size, shared by every
    // qToken in the batch), so P must be rectangular [batchM, validSize] with
    // [cvl[g], validSize) == 0. The per-row compute writes the FULL row [0, validSize):
    // FusedExpSub (masked by tailN=cvl[g]) produces real exp in [0, cvl[g]); a Select then
    // zeroes [cvl[g], 128) in-register and StoreAlign persists [0, validSize) -- so the causal
    // tail is 0 by construction. No pre-zero pass / maxK needed. validSize = actualBlockShape.n().
    template <class TensorP>
    __aicore__ inline void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape, uint32_t ubSBufId,
                                      uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag, const uint32_t *causalValidLens,
                                      uint32_t groupCount, uint32_t groupRows)
    {
        (void)l1PBufId;
        uint32_t M = actualBlockShape.m();
        // Split by GROUP (not M/2): each AIV owns whole groups, no group straddles a sub-core.
        // mCopyOffset = AIV0's row count = ceil(groupCount/2)*groupRows (groupRows-aligned), which
        // is also AIV1's L1-P zN write offset -- M/2 would misalign the zN fractal for odd
        // groupCount tail batches. Matches BlockMmadQK's group-dim S split.
        uint32_t mCopyOffset = CeilDiv(groupCount, 2U) * groupRows;
        uint32_t mHalf = mCopyOffset < M ? mCopyOffset : M;
        uint32_t m = subBlockIdx_ == 0 ? mHalf : (M - mHalf);
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound;
        uint32_t startRow = subBlockIdx_ * mCopyOffset;
        uint32_t endRow = startRow + m;
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput *)lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput *)lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        // gmUb/glUb are 2-deep by ubSBufId; this AIV's rowMax/rowSum land in gmUb[ubSBufId]/
        // glUb[ubSBufId] and are copied to GM per-slot by the kernel's CopyPartialStatsToGm.
        __ubuf__ float *nowMaxFloatAddr = (__ubuf__ float *)gmUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *nowSumAddr = (__ubuf__ float *)glUbTensor[ubSBufId].GetPhyAddr();

        // wait QK Fixpipe finish
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);

        uint32_t gStart = startRow / groupRows;
        uint32_t gEnd = (endRow == 0) ? 0 : (endRow - 1) / groupRows;
        for (uint32_t g = gStart; g <= gEnd; g++) {
            uint32_t grpLo = g * groupRows;
            uint32_t grpHi = grpLo + groupRows;
            uint32_t lo = grpLo > startRow ? grpLo : startRow;
            uint32_t hi = grpHi < endRow ? grpHi : endRow;
            uint32_t rows = hi - lo;
            if (rows == 0) {
                continue;
            }
            // S/P are densely packed (QK fixpipe / CopyPUbToPL1 use contiguous mSub rows), so their
            // slice base uses dense localOff = lo - startRow. rowMax/rowSum instead need a per-group
            // 8-float-aligned base: the per-row store (StoreUnAlign) and the kernel's DataCopyPad
            // read (CopyPartialStatsToGm) both require 32B-aligned group bases; with groupRows<8
            // (local groupSize=4 cases) a dense groupRows stride would straddle an 8-float block and
            // DataCopyPad would read from an unaligned UB address. grpStride = RoundUp(groupRows,8)
            // is a no-op for groupRows>=8 (production groupSize=16 -> stride 16 == groupRows).
            uint32_t localOff = lo - startRow;
            uint32_t grpStride = RoundUp(groupRows, 8U);
            uint32_t statsOff = (g - gStart) * grpStride;
            uint32_t tailN = causalValidLens[g];
            uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
            uint32_t tailNOdd = tailN / 2;
            uint32_t tailNEven = tailNOdd + tailN % 2;
            __ubuf__ ElementInput *sSlice = sAddr + localOff * nRound;
            __ubuf__ float *maxSlice = nowMaxFloatAddr + statsOff;
            __ubuf__ float *sumSlice = nowSumAddr + statsOff;
            constexpr uint32_t P_C0 = FULL_QUANT_FP8 ? ELE_NUM_PER_C0_FP8 : ELE_NUM_PER_C0;
            __ubuf__ ElementOutput *pSlice = pAddr + localOff * P_C0;

            uint32_t kvBaseTileRegStages = CeilDiv(n, SM_VREG_SIZE);
            if (kvBaseTileRegStages == 1) {
                ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::ONE>(sSlice, maxSlice, static_cast<uint16_t>(rows),
                                                                      tailN, nPadding, scaleValue, nRound);
            } else if (kvBaseTileRegStages == 2) {
                ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::TWO>(sSlice, maxSlice, static_cast<uint16_t>(rows),
                                                                      tailN, nPadding, scaleValue, nRound);
            }
            if (kvBaseTileRegStages == 1) {
                if constexpr (FULL_QUANT_FP8) {
                    ComputeExpSubSum16FP8<KvBaseTileRegSplitStagesBf16::ONE>(pSlice, sSlice, maxSlice, sumSlice,
                                                                             static_cast<uint16_t>(rows), tailN,
                                                                             blockStride, nRound, tailNOdd, tailNEven);
                } else {
                    ComputeExpSubSum16<KvBaseTileRegSplitStagesBf16::ONE>(pSlice, sSlice, maxSlice, sumSlice,
                                                                          static_cast<uint16_t>(rows), tailN,
                                                                          blockStride, nRound, tailNOdd, tailNEven, n);
                }
            } else if (kvBaseTileRegStages == 2) {
                if constexpr (FULL_QUANT_FP8) {
                    ComputeExpSubSum16FP8<KvBaseTileRegSplitStagesBf16::TWO>(pSlice, sSlice, maxSlice, sumSlice,
                                                                             static_cast<uint16_t>(rows), tailN,
                                                                             blockStride, nRound, tailNOdd, tailNEven);
                } else {
                    ComputeExpSubSum16<KvBaseTileRegSplitStagesBf16::TWO>(pSlice, sSlice, maxSlice, sumSlice,
                                                                          static_cast<uint16_t>(rows), tailN,
                                                                          blockStride, nRound, tailNOdd, tailNEven, n);
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        uint32_t curNRound = FULL_QUANT_FP8 ? RoundUp(n, ELE_NUM_PER_C0_FP8) : RoundUp(n, ELE_NUM_PER_C0);
        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, curNRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM], ubPLayoutTla, Arch::PositionUB{});
        // P output extent = validSize (uniform batch K). Per-group compute wrote the full
        // [0, validSize) row (real [0, cvl[g]) + Select-zeroed [cvl[g], validSize)), so PV
        // (K=validSize) reads valid exp + zero tail.
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> gmUbTensor[SM_UB_STAGES];
    AscendC::LocalTensor<float> glUbTensor[SM_UB_STAGES];
    uint32_t subBlockIdx_;
    ElementInput MIN_VALUE;

    template <KvBaseTileRegSplitStagesBf16 kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeScaleAndMax(__ubuf__ ElementInput *srcUb, __ubuf__ float *newMaxUb, uint16_t m,
                                               uint32_t tailN, uint32_t nPadding, ElementInput dScale,
                                               uint16_t S2BaseSize)
    {
        static_assert(kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::ONE ||
                          kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::TWO,
                      "ComputeScaleAndMax only supports ONE Or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::ONE>(__ubuf__ ElementInput *srcUb,
                                                                                  __ubuf__ float *newMaxUb, uint16_t m,
                                                                                  uint32_t tailN, uint32_t nPadding,
                                                                                  ElementInput dScale,
                                                                                  uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg;
        // RegTensor<ElementInput> maxSrcVreg;
        RegTensor<ElementInput> maxTmpVreg;
        RegTensor<ElementInput> scaleVreg;
        RegTensor<float> maxFloatVreg0;
        RegTensor<float> maxFloatVreg1;
        RegTensor<float> maxTmpFloatVreg;
        RegTensor<float> maxTmpFloatVreg0;
        RegTensor<float> maxTmpFloatVreg1;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        MaskReg pregFloatTailN = UpdateMask<float>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        Duplicate(scaleVreg, dScale);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            Mul(srcVreg, srcVreg, scaleVreg, pregFull);
            Select(srcVreg, srcVreg, minVreg, pregTailN);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * S2BaseSize, srcVreg, pregTailN);
            Cast<float, ElementInput, castTraitZero>(maxFloatVreg0, srcVreg, pregFull);
            Cast<float, ElementInput, castTraitOne>(maxFloatVreg1, srcVreg, pregFull);
            ReduceMax(maxTmpFloatVreg0, maxFloatVreg0, pregFull);
            ReduceMax(maxTmpFloatVreg1, maxFloatVreg1, pregFull);
            Max(maxTmpFloatVreg, maxTmpFloatVreg0, maxTmpFloatVreg1, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpFloatVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStagesBf16::TWO>(__ubuf__ ElementInput *srcUb,
                                                                                  __ubuf__ float *newMaxUb, uint16_t m,
                                                                                  uint32_t tailN, uint32_t nPadding,
                                                                                  ElementInput dScale,
                                                                                  uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        // RegTensor<ElementInput> maxSrcVreg;
        RegTensor<ElementInput> maxTmpVreg;
        RegTensor<ElementInput> scaleVreg;
        RegTensor<float> maxFloatVreg0;
        RegTensor<float> maxFloatVreg1;
        RegTensor<float> maxTmpFloatVreg;
        RegTensor<float> maxTmpFloatVreg0;
        RegTensor<float> maxTmpFloatVreg1;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        MaskReg pregFloatTailN = UpdateMask<float>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        Duplicate(scaleVreg, dScale);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Mul(srcVreg0, srcVreg0, scaleVreg, pregFull);
            Mul(srcVreg1, srcVreg1, scaleVreg, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + i * S2BaseSize + HALF_REP_SIZE, srcVreg1,
                                                               pregTailN);
            Max<ElementInput, MaskMergeMode::MERGING>(srcVreg0, srcVreg0, srcVreg1, pregTailN);

            Cast<float, ElementInput, castTraitZero>(maxFloatVreg0, srcVreg0, pregFull);
            Cast<float, ElementInput, castTraitOne>(maxFloatVreg1, srcVreg0, pregFull);
            ReduceMax(maxTmpFloatVreg0, maxFloatVreg0, pregFull);
            ReduceMax(maxTmpFloatVreg1, maxFloatVreg1, pregFull);
            Max(maxTmpFloatVreg, maxTmpFloatVreg0, maxTmpFloatVreg1, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpFloatVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <typename ElementS>
    __simd_vf__ inline void CastMax(__ubuf__ ElementS *nowMaxUb, __ubuf__ float *nowMaxFloatUb, uint16_t mLoops,
                                    uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementS> nowMaxVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<ElementS> maxVreg;

        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<ElementS>(tailM);
        MaskReg pregFloatTailM = UpdateMask<float>(tailM);
        for (uint16_t i = 0; i < mLoops; ++i) {
            LoadAlign(nowMaxVreg, nowMaxUb + i * HALF_REP_SIZE);
            Cast<float, ElementS, castTraitZero>(nowMaxFloatVreg, nowMaxVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxFloatUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg,
                                                        pregFloatFull);
        }
        LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
        Cast<float, ElementS, castTraitZero>(nowMaxFloatVreg, nowMaxVreg, pregFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxFloatUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg,
                                                    pregFloatTailM);
    }

    __simd_vf__ inline void UpdateMax(__ubuf__ float *nowMaxUb, __ubuf__ float *lastMaxUb, uint16_t mLoops,
                                      uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> nowMaxVreg;
        RegTensor<float> lastMaxFloatVreg;
        RegTensor<float> maxVreg;

        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFloatTailM = UpdateMask<float>(tailM);
        for (uint16_t i = 0; i < mLoops; ++i) {
            LoadAlign(lastMaxFloatVreg, lastMaxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxFloatVreg, pregFloatFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxUb + i * FLOAT_REP_SIZE, maxVreg, pregFloatFull);
        }
        LoadAlign(lastMaxFloatVreg, lastMaxUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(nowMaxVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Max(maxVreg, nowMaxVreg, lastMaxFloatVreg, pregFloatFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxUb + mLoops * FLOAT_REP_SIZE, maxVreg, pregFloatTailM);
    }

    template <KvBaseTileRegSplitStagesBf16 kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeExpSubSum16(__ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
                                               __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb, uint16_t m,
                                               uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize,
                                               uint32_t tailNOdd, uint32_t tailNEven, uint32_t storeN)
    {
        static_assert(kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::ONE ||
                          kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::TWO,
                      "ComputeExpSubSum16 only supports ONE Or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16<KvBaseTileRegSplitStagesBf16::ONE>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb, __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize, uint32_t tailNOdd, uint32_t tailNEven,
        uint32_t storeN)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitZeroDown = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOneDown = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementInput> expVreg;
        RegTensor<float> expFloatVreg0;
        RegTensor<float> expFloatVreg1;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<float> expDstFloatVreg0;
        RegTensor<float> expDstFloatVreg1;
        RegTensor<ElementInput> expDstVreg;
        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;

        UnalignReg expSumUreg;

        RegTensor<ElementInput> zeroBf16Vreg;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        MaskReg pregtailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg pregtailNEven = UpdateMask<float>(tailNEven);
        MaskReg pregStoreN = UpdateMask<ElementOutput>(storeN);
        Duplicate(zeroBf16Vreg, static_cast<ElementInput>(0));
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            LoadAlign(expVreg, srcUb + i * S2BaseSize);
            Cast<float, ElementInput, castTraitZero>(expFloatVreg0, expVreg, pregFull);
            Cast<float, ElementInput, castTraitOne>(expFloatVreg1, expVreg, pregFull);
            FusedExpSub(expDstFloatVreg0, expFloatVreg0, maxVreg, pregtailNEven);
            FusedExpSub(expDstFloatVreg1, expFloatVreg1, maxVreg, pregtailNOdd);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg0, pregtailNEven);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg1, pregtailNOdd);
            Cast<ElementInput, float, castTraitZeroDown>(expDstVreg0, expDstFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOneDown>(expDstVreg1, expDstFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t> &)expDstVreg, (RegTensor<uint16_t> &)expDstVreg0,
               (RegTensor<uint16_t> &)expDstVreg1, pregFull);
            // [0, cvl) holds real exp (FusedExpSub masked by tailN); zero [cvl, 128) so the
            // store of [0, validSize) writes 0 into the per-group causal tail [cvl, validSize).
            Select(expDstVreg, expDstVreg, zeroBf16Vreg, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0, expDstVreg,
                                                                     blockStride, pregStoreN);
            ReduceSum(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16<KvBaseTileRegSplitStagesBf16::TWO>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb, __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize, uint32_t tailNOdd, uint32_t tailNEven,
        uint32_t storeN)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitZeroDown = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOneDown = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        RegTensor<ElementInput> expVreg0;
        RegTensor<ElementInput> expVreg1;
        RegTensor<float> expFloatVreg0;
        RegTensor<float> expFloatVreg1;
        RegTensor<float> expFloatVreg2;
        RegTensor<float> expFloatVreg3;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<float> expDstFloatVreg0;
        RegTensor<float> expDstFloatVreg1;
        RegTensor<float> expDstFloatVreg2;
        RegTensor<float> expDstFloatVreg3;
        RegTensor<ElementInput> expOutVreg0;
        RegTensor<ElementInput> expOutVreg1;
        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;
        RegTensor<ElementInput> expDstVreg2;
        RegTensor<ElementInput> expDstVreg3;

        UnalignReg expSumUreg;

        // TWO stage: n (=validSize) in (128, 256]. First half covers cols [0,128), second
        // half [128,256). Per-row P must be [0, storeN) with [cvl, storeN) == 0. Each half
        // register is 128 bf16; zero the per-half causal tail in-register, then store the
        // per-half slice of [0, storeN). firstHalfValid = min(cvl,128); secondHalfValid =
        // max(cvl-128,0); secondStoreN = max(storeN-128,0) (storeN>128 in TWO stage).
        uint32_t firstHalfValid = tailN < 128U ? tailN : 128U;
        uint32_t secondHalfValid = tailN > 128U ? (tailN - 128U) : 0U;
        uint32_t secondStoreN = storeN > 128U ? (storeN - 128U) : 0U;

        RegTensor<ElementInput> zeroBf16Vreg;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregtailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg pregtailNEven = UpdateMask<float>(tailNEven);
        MaskReg pregFirstZero = UpdateMask<ElementInput>(firstHalfValid);
        MaskReg pregSecondZero = UpdateMask<ElementOutput>(secondHalfValid);
        MaskReg pregSecondStore = UpdateMask<ElementOutput>(secondStoreN);
        Duplicate(zeroBf16Vreg, static_cast<ElementInput>(0));
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            LoadAlign(expVreg0, srcUb + i * S2BaseSize);
            LoadAlign(expVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Cast<float, ElementInput, castTraitZero>(expFloatVreg0, expVreg0, pregFull);
            Cast<float, ElementInput, castTraitOne>(expFloatVreg1, expVreg0, pregFull);
            Cast<float, ElementInput, castTraitZero>(expFloatVreg2, expVreg1, pregFull);
            Cast<float, ElementInput, castTraitOne>(expFloatVreg3, expVreg1, pregFull);
            FusedExpSub(expDstFloatVreg0, expFloatVreg0, maxVreg, pregFloatFull);
            FusedExpSub(expDstFloatVreg1, expFloatVreg1, maxVreg, pregFloatFull);
            FusedExpSub(expDstFloatVreg2, expFloatVreg2, maxVreg, pregtailNEven);
            FusedExpSub(expDstFloatVreg3, expFloatVreg3, maxVreg, pregtailNOdd);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg0, pregFloatFull);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg1, pregFloatFull);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg2, pregtailNEven);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg3, pregtailNOdd);
            Cast<ElementInput, float, castTraitZeroDown>(expDstVreg0, expDstFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOneDown>(expDstVreg1, expDstFloatVreg1, pregFloatFull);
            Cast<ElementInput, float, castTraitZeroDown>(expDstVreg2, expDstFloatVreg2, pregFloatFull);
            Cast<ElementInput, float, castTraitOneDown>(expDstVreg3, expDstFloatVreg3, pregFloatFull);
            Or((RegTensor<uint16_t> &)expOutVreg0, (RegTensor<uint16_t> &)expDstVreg0,
               (RegTensor<uint16_t> &)expDstVreg1, pregFull);
            Or((RegTensor<uint16_t> &)expOutVreg1, (RegTensor<uint16_t> &)expDstVreg2,
               (RegTensor<uint16_t> &)expDstVreg3, pregFull);
            // Zero per-half causal tail: first half [min(cvl,128), 128), second half
            // [cvl-128, 128). Then store first half full (storeN>128) and second half
            // [0, storeN-128) -- together covering [0, storeN) with [cvl, storeN) == 0.
            Select(expOutVreg0, expOutVreg0, zeroBf16Vreg, pregFirstZero);
            Select(expOutVreg1, expOutVreg1, zeroBf16Vreg, pregSecondZero);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0, expOutVreg0,
                                                                     blockStride, pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0 + blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE, expOutVreg1, blockStride,
                pregSecondStore);

            ReduceSum(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <KvBaseTileRegSplitStagesBf16 kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeExpSubSum16FP8(__ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
                                                  __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb, uint16_t m,
                                                  uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize,
                                                  uint32_t tailNOdd, uint32_t tailNEven)
    {
        static_assert(kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::ONE ||
                          kvBaseTileRegSplitStages == KvBaseTileRegSplitStagesBf16::TWO,
                      "ComputeExpSubSum16FP8 only supports ONE Or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16FP8<KvBaseTileRegSplitStagesBf16::ONE>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb, __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize, uint32_t tailNOdd, uint32_t tailNEven)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> expVreg;
        RegTensor<float> expFloatVreg0;
        RegTensor<float> expFloatVreg1;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<float> expDstFloatVreg0;
        RegTensor<float> expDstFloatVreg1;
        RegTensor<ElementInput> expDstVreg;
        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;

        UnalignReg expSumUreg;

        constexpr static CastTrait castTraitZeroRINT = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitOneRINT = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitTwoRINT = {
            RegLayout::TWO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitThreeRINT = {
            RegLayout::THREE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        RegTensor<float> deInterleaveVreg0;
        RegTensor<float> deInterleaveVreg1;
        RegTensor<float> deInterleaveVreg2;
        RegTensor<float> deInterleaveVreg3;
        RegTensor<ElementOutput> pVreg0;
        RegTensor<ElementOutput> pVreg1;
        RegTensor<ElementOutput> pVreg2;
        RegTensor<ElementOutput> pVreg3;

        MaskReg pRegUint8All = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pRegUint8VL128 = CreateMask<uint8_t, MaskPattern::VL128>();
        MaskReg pRegFp16All = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pRegFp32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFp32tailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg pregFp32tailNEven = UpdateMask<float>(tailNEven);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            LoadAlign(expVreg, srcUb + i * S2BaseSize);
            Cast<float, ElementInput, castTraitZeroUNKNOWN>(expFloatVreg0, expVreg, pRegFp16All);
            Cast<float, ElementInput, castTraitOneUNKNOWN>(expFloatVreg1, expVreg, pRegFp16All);
            FusedExpSub(expDstFloatVreg0, expFloatVreg0, maxVreg, pregFp32tailNEven);
            FusedExpSub(expDstFloatVreg1, expFloatVreg1, maxVreg, pregFp32tailNOdd);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg0, pregFp32tailNEven);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg1, pregFp32tailNOdd);

            constexpr float maxValueFP8 = 448.0f;
            Muls(expDstFloatVreg0, expDstFloatVreg0, maxValueFP8, pregFp32tailNEven);
            Muls(expDstFloatVreg1, expDstFloatVreg1, maxValueFP8, pregFp32tailNOdd);

            DeInterleave(deInterleaveVreg0, deInterleaveVreg1, expDstFloatVreg0, expDstFloatVreg0);
            DeInterleave(deInterleaveVreg2, deInterleaveVreg3, expDstFloatVreg1, expDstFloatVreg1);

            Cast<ElementOutput, float, castTraitZeroRINT>(pVreg0, deInterleaveVreg0, pRegFp32All);
            Cast<ElementOutput, float, castTraitOneRINT>(pVreg1, deInterleaveVreg2, pRegFp32All);
            Cast<ElementOutput, float, castTraitTwoRINT>(pVreg2, deInterleaveVreg1, pRegFp32All);
            Cast<ElementOutput, float, castTraitThreeRINT>(pVreg3, deInterleaveVreg3, pRegFp32All);

            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg1, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg2, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg3, pRegUint8All);

            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0_FP8, pVreg0,
                                                                     blockStride, pRegUint8VL128);
            ReduceSum(expSumVreg, expSumVreg, pRegFp16All);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16FP8<KvBaseTileRegSplitStagesBf16::TWO>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb, __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize, uint32_t tailNOdd, uint32_t tailNEven)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> expVreg0;
        RegTensor<ElementInput> expVreg1;
        RegTensor<float> expFloatVreg0;
        RegTensor<float> expFloatVreg1;
        RegTensor<float> expFloatVreg2;
        RegTensor<float> expFloatVreg3;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;

        RegTensor<float> expDstFloatVreg0;
        RegTensor<float> expDstFloatVreg1;
        RegTensor<float> expDstFloatVreg2;
        RegTensor<float> expDstFloatVreg3;
        RegTensor<ElementInput> expOutVreg0;
        RegTensor<ElementInput> expOutVreg1;
        RegTensor<ElementInput> expDstVreg0;
        RegTensor<ElementInput> expDstVreg1;
        RegTensor<ElementInput> expDstVreg2;
        RegTensor<ElementInput> expDstVreg3;

        UnalignReg expSumUreg;

        constexpr static CastTrait castTraitZeroRINT = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitOneRINT = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitTwoRINT = {
            RegLayout::TWO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitThreeRINT = {
            RegLayout::THREE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        RegTensor<float> deInterleaveVreg0;
        RegTensor<float> deInterleaveVreg1;
        RegTensor<float> deInterleaveVreg2;
        RegTensor<float> deInterleaveVreg3;
        RegTensor<float> deInterleaveVreg4;
        RegTensor<float> deInterleaveVreg5;
        RegTensor<float> deInterleaveVreg6;
        RegTensor<float> deInterleaveVreg7;
        RegTensor<ElementOutput> pVreg0;
        RegTensor<ElementOutput> pVreg1;
        RegTensor<ElementOutput> pVreg2;
        RegTensor<ElementOutput> pVreg3;
        RegTensor<ElementOutput> pVreg4;
        RegTensor<ElementOutput> pVreg5;
        RegTensor<ElementOutput> pVreg6;
        RegTensor<ElementOutput> pVreg7;

        MaskReg pRegUint8All = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pRegUint8VL128 = CreateMask<uint8_t, MaskPattern::VL128>();
        MaskReg pRegFp16All = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFp16TailN = UpdateMask<ElementInput>(tailN);
        MaskReg pRegFp32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFp32TailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg pregFp32tailNEven = UpdateMask<float>(tailNEven);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            LoadAlign(expVreg0, srcUb + i * S2BaseSize);
            LoadAlign(expVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Cast<float, ElementInput, castTraitZeroUNKNOWN>(expFloatVreg0, expVreg0, pRegFp16All);
            Cast<float, ElementInput, castTraitOneUNKNOWN>(expFloatVreg1, expVreg0, pRegFp16All);
            Cast<float, ElementInput, castTraitZeroUNKNOWN>(expFloatVreg2, expVreg1, pRegFp16All);
            Cast<float, ElementInput, castTraitOneUNKNOWN>(expFloatVreg3, expVreg1, pRegFp16All);
            FusedExpSub(expDstFloatVreg0, expFloatVreg0, maxVreg, pRegFp32All);
            FusedExpSub(expDstFloatVreg1, expFloatVreg1, maxVreg, pRegFp32All);
            FusedExpSub(expDstFloatVreg2, expFloatVreg2, maxVreg, pregFp32tailNEven);
            FusedExpSub(expDstFloatVreg3, expFloatVreg3, maxVreg, pregFp32TailNOdd);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg0, pRegFp32All);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg1, pRegFp32All);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg2, pregFp32tailNEven);
            Add<float, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstFloatVreg3, pregFp32TailNOdd);

            constexpr float maxValueFP8 = 448.0f;
            Muls(expDstFloatVreg0, expDstFloatVreg0, maxValueFP8, pRegFp32All);
            Muls(expDstFloatVreg1, expDstFloatVreg1, maxValueFP8, pRegFp32All);
            Muls(expDstFloatVreg2, expDstFloatVreg2, maxValueFP8, pregFp32tailNEven);
            Muls(expDstFloatVreg3, expDstFloatVreg3, maxValueFP8, pregFp32TailNOdd);

            DeInterleave(deInterleaveVreg0, deInterleaveVreg1, expDstFloatVreg0, expDstFloatVreg0);
            DeInterleave(deInterleaveVreg2, deInterleaveVreg3, expDstFloatVreg1, expDstFloatVreg1);
            DeInterleave(deInterleaveVreg4, deInterleaveVreg5, expDstFloatVreg2, expDstFloatVreg2);
            DeInterleave(deInterleaveVreg6, deInterleaveVreg7, expDstFloatVreg3, expDstFloatVreg3);

            Cast<ElementOutput, float, castTraitZeroRINT>(pVreg0, deInterleaveVreg0, pRegFp32All);
            Cast<ElementOutput, float, castTraitOneRINT>(pVreg1, deInterleaveVreg2, pRegFp32All);
            Cast<ElementOutput, float, castTraitTwoRINT>(pVreg2, deInterleaveVreg1, pRegFp32All);
            Cast<ElementOutput, float, castTraitThreeRINT>(pVreg3, deInterleaveVreg3, pRegFp32All);
            Cast<ElementOutput, float, castTraitZeroRINT>(pVreg4, deInterleaveVreg4, pRegFp32All);
            Cast<ElementOutput, float, castTraitOneRINT>(pVreg5, deInterleaveVreg6, pRegFp32All);
            Cast<ElementOutput, float, castTraitTwoRINT>(pVreg6, deInterleaveVreg5, pRegFp32All);
            Cast<ElementOutput, float, castTraitThreeRINT>(pVreg7, deInterleaveVreg7, pRegFp32All);

            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg1, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg2, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg0, (RegTensor<uint8_t> &)pVreg3, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg5, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg6, pRegUint8All);
            Or((RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg4, (RegTensor<uint8_t> &)pVreg7, pRegUint8All);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0_FP8, pVreg0,
                                                                     blockStride, pRegUint8VL128);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0_FP8 + blockStride * HALF_VECTOR_SIZE, pVreg4, blockStride, pRegUint8VL128);
            ReduceSum(expSumVreg, expSumVreg, pRegFp16All);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    __simd_vf__ inline void UpdateExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *expMaxUb,
                                                  __ubuf__ float *maxUb, __ubuf__ float *expSumUb,
                                                  __ubuf__ float *nowMaxUb, uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<float> brcExpSumFloatVreg;
        RegTensor<float> updateExpSumVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(lastMaxVreg, maxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxFloatVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + i * FLOAT_REP_SIZE, expMaxVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(lastExpSumVreg, sumUb + i * FLOAT_REP_SIZE);
            LoadAlign(brcExpSumFloatVreg, expSumUb + i * FLOAT_REP_SIZE);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, updateExpSumVreg, pregFull);
        }
        LoadAlign(lastMaxVreg, maxUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(nowMaxFloatVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + mLoops * FLOAT_REP_SIZE, expMaxVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(brcExpSumFloatVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregTailM);
        Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, updateExpSumVreg, pregTailM);
    }
};
} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_HPP

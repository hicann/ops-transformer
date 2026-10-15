/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_S_TRANS_HIF8_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_S_TRANS_HIF8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"

namespace NpuArch::Epilogue::Block {

namespace Hif8VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace AscendC::Reg;

// Build the exact QFA HIF8 DN_INDEX table with vector instructions:
// index[i * 64 + j] = i + 4 * j, i in [0, 4), j in [0, 64).
__simd_vf__ inline void InitDnIndexVF(__ubuf__ uint8_t* indexDn)
{
    RegTensor<uint8_t> baseIndexReg;
    RegTensor<uint8_t> indexReg;
    RegTensor<uint8_t> indexHighReg;
    MaskReg mask32 = CreateMask<int8_t, MaskPattern::VL32>();

    Arange((RegTensor<int8_t>&)baseIndexReg, 0);
    ShiftLefts(baseIndexReg, baseIndexReg, static_cast<int16_t>(2), mask32);
    for (uint16_t i = 0; i < 4; ++i) {
        Adds(indexReg, baseIndexReg, i, mask32);
        StoreAlign(indexDn + i * 64, indexReg, mask32);
        Adds(indexHighReg, indexReg, static_cast<uint8_t>(128), mask32);
        StoreAlign(indexDn + i * 64 + 32, indexHighReg, mask32);
    }
}

__aicore__ inline void InitDnIndexCallVF(LocalTensor<uint8_t>& indexTensor)
{
    __ubuf__ uint8_t* indexDn = (__ubuf__ uint8_t*)indexTensor.GetPhyAddr();
    InitDnIndexVF(indexDn);
}
} // namespace Hif8VF

template <class LayoutP_, class LayoutS_>
class BlockEpilogue<
    // S is transposed, has shape [kvSTile, qSTile], softmax is done columnwise
    // P would be rearranged from [kvSTile, qSTile] to zN,
    // which can be viewed as from [qSTile, kvSTile] to nZ
    EpilogueOnlineSoftmaxBsa<true>, Gemm::GemmType<hifloat8_t, LayoutP_>, Gemm::GemmType<float, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = hifloat8_t;
    using ElementInput = float;

    using LayoutOutput = LayoutP_;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr ElementInput MIN_VALUE = -3.389531390315715675e+38f;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, float scaleValue_, float pScaleValue_,
                                    UBufTileHelper& uBufTileHelper)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = static_cast<ElementInput>(scaleValue_);
        pScaleValue = pScaleValue_;
        uint32_t pExtraElemNum = Max(uBufTileHelper.qBaseTilePerSubCore, uBufTileHelper.kvBaseTilePerSubCore);
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; i++) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementInput) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset +
                (uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore + pExtraElemNum) *
                    sizeof(ElementOutput) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; i++) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset + uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.lmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.llStartOffset);

        // QFA HIF8 DN_INDEX: gather the four byte lanes produced by ZERO/ONE/TWO/THREE Cast
        // into the nZ layout consumed by MM2. maskStartOffset is free after Mask2Idx completes.
        pGatherIndexUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(uBufTileHelper.maskStartOffset);
        Hif8VF::InitDnIndexCallVF(pGatherIndexUbTensor);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorP>
    __aicore__ inline void operator()(TensorP& l1PTensorTla, GemmCoord actualBlockShape, uint32_t isFirstKvSTile,
                                      uint32_t ubSBufId, uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag)
    {
        uint32_t nSplit = RoundUp(actualBlockShape.m(), 32) / 2;
        // The HIF8 VF always materializes a fixed 64-Q-lane P slot per AIV,
        // including zero-filled lanes on a Q tail.  Keep the two L1 regions
        // 64 rows apart exactly as QFA HIF8 does; compacting them with the
        // active split (for example 32/31 when rowNum=63) makes subcore 1
        // overwrite the second half of subcore 0's P tile.
        constexpr uint32_t nCopyOffset = 64;
        uint32_t n = actualBlockShape.m() < nSplit ? actualBlockShape.m() : nSplit;
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
        // HIF8 DN uses the fixed QFA stage layout even for a KV tail.  Each of
        // the eight byte-lane stripes has a 65-DataBlock pitch (64 data blocks
        // plus one bank-conflict pad).  A tail only shortens the valid prefix
        // copied from each stripe; changing the pitch to mRound/4+1 would make
        // L1 P use a different physical layout from the Cube nZ consumer.
        constexpr uint32_t HIF8_STRIPE_BLOCKS = 64;
        constexpr uint32_t HIF8_BLOCK_STRIDE = HIF8_STRIPE_BLOCKS + 1;
        uint32_t blockStride = HIF8_BLOCK_STRIDE;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        uint32_t tailN = (n - 1) % vlSize + 1;
        __ubuf__ ElementOutput* pAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementInput* sAddr = (__ubuf__ ElementInput*)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float* lastMaxAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* lastMaxStartAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* lastSumAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput* nowMaxAddr = (__ubuf__ float*)lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput* nowSumAddr = (__ubuf__ float*)llUbTensor.GetPhyAddr();
        __ubuf__ float* expMaxUbAddr = (__ubuf__ float*)dmUbTensor[l1PBufId].GetPhyAddr();
        __ubuf__ uint8_t* pGatherIndexUbAddr = (__ubuf__ uint8_t*)pGatherIndexUbTensor.GetPhyAddr();
        // wait QK Fixpipe finish
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);

        uint32_t mAlignedTile = mRound / 4;
        uint32_t mFirstTile = m % mAlignedTile;        // Valid rows in the final tile.
        uint32_t mAligned16TileNum = m / mAlignedTile; // Number of full tiles.
        if (mAligned16TileNum == 4) {
            mFirstTile = mAlignedTile;
        }
        if (isFirstKvSTile) {
            if (mAligned16TileNum == 0) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, false, MAlignedTileNum::Zero>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 1) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, false, MAlignedTileNum::One>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 2) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, false, MAlignedTileNum::Two>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 3) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, false, MAlignedTileNum::Three>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, false, MAlignedTileNum::Four>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            }
        } else {
            if (mAligned16TileNum == 0) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, true, MAlignedTileNum::Zero>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 1) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, true, MAlignedTileNum::One>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 2) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, true, MAlignedTileNum::Two>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 3) {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, true, MAlignedTileNum::Three>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            } else {
                ComputeOnlineSoftmaxDN<ElementInput, ElementOutput, true, MAlignedTileNum::Four>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue,
                    pScaleValue, pGatherIndexUbAddr, 64, blockStride, expMaxUbAddr, lastSumAddr);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * nCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        AscendC::DataCopyParams dataCopyParams;
        // The packed HIF8 register is stored as eight UB stripes.  Each
        // stripe contains mRound / 4 DataBlocks followed by one padding
        // DataBlock.  The two copies below gather the four even and four odd
        // stripes into the two contiguous 32-Q-row portions in L1.
        //
        // For a full mRound=256 tile this is the QFA 4/64/66 setting.  Keep
        // the stripe length dynamic so a KV tail does not copy past its P
        // region and overwrite the neighbouring subcore/portion in L1.
        uint32_t pStripeLen = mRound / 4;
        dataCopyParams.blockCount = 4;
        dataCopyParams.blockLen = pStripeLen;
        // DataCopy's srcStride is the gap after each copied block.  The next
        // even stripe starts two fixed 65-DB pitches later.
        dataCopyParams.srcStride = 2 * blockStride - pStripeLen;
        dataCopyParams.dstStride = 0;
        DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset], lpUbTensor[ubSBufId], dataCopyParams);
        DataCopy(l1PTensorTla.data()[mRound * 32 + subBlockIdx_ * mRound * nCopyOffset],
                 lpUbTensor[ubSBufId][blockStride * 32], dataCopyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
    }

private:
    ElementInput scaleValue;
    float pScaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    AscendC::LocalTensor<uint8_t> pGatherIndexUbTensor;
    uint32_t subBlockIdx_;

    enum class MAlignedTileNum {
        Zero = 0,
        One = 1,
        Two = 2,
        Three = 3,
        Four = 4
    };

    template <typename ElementS, typename ElementP, bool isUpdate, MAlignedTileNum mTileNum>
    __simd_vf__ inline void ComputeOnlineSoftmaxDN(__ubuf__ ElementS* srcUb, __ubuf__ ElementS* newMaxUb,
                                                   __ubuf__ ElementS* LastMaxUbStart, __ubuf__ ElementP* expUb,
                                                   __ubuf__ ElementS* expSumUb, uint16_t mRound, uint16_t m,
                                                   uint32_t tailN, uint32_t mFirstTile, ElementInput dScale,
                                                   ElementInput pScale, __ubuf__ uint8_t* pGatherIndexUb,
                                                   uint16_t s2BaseSize, uint32_t blockStride, __ubuf__ float* expMaxUb,
                                                   __ubuf__ ElementS* lastExpSumUb)
    {
        using namespace AscendC::Reg;
        RegTensor<float> src0Vreg;
        RegTensor<float> src1Vreg;
        RegTensor<float> src2Vreg;
        RegTensor<float> src3Vreg;
        RegTensor<float> src4Vreg;

        RegTensor<float> exp0Vreg32;
        RegTensor<float> exp1Vreg32;
        RegTensor<float> exp2Vreg32;
        RegTensor<float> exp3Vreg32;

        RegTensor<float> max0Vreg;
        RegTensor<float> max1Vreg;
        RegTensor<float> max2Vreg;
        RegTensor<float> max3Vreg;
        RegTensor<float> pScaleVreg;
        RegTensor<float> lnPScaleVreg;

        RegTensor<float> expMaxVreg;
        RegTensor<float> updateExpSumVreg;

        RegTensor<float> sum0Vreg;
        RegTensor<float> sum1Vreg;
        RegTensor<float> sum2Vreg;
        RegTensor<float> sum3Vreg;
        RegTensor<float> lastExpSumVreg;

        RegTensor<ElementP> exp0Vreg16;
        RegTensor<ElementP> exp1Vreg16;
        RegTensor<ElementP> exp2Vreg16;
        RegTensor<ElementP> exp3Vreg16;
        RegTensor<ElementP> merge01Vreg;
        RegTensor<ElementP> merge23Vreg;
        RegTensor<ElementP> packedPVreg;
        RegTensor<uint8_t> pGatherIndexVreg;

        constexpr static CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitOneRound = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitTwoRound = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitThreeRound = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_ROUND};

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        MaskReg pregFullB8 = CreateMask<uint8_t, MaskPattern::ALL>();
        Duplicate(pScaleVreg, pScale);
        Ln(lnPScaleVreg, pScaleVreg, pregTailN);
        LoadAlign(pGatherIndexVreg, pGatherIndexUb);
        // Each HIF8 packed vector contributes one sequential 32B DataBlock
        // to the flattened destination stream.  Compute that stream address
        // directly from the loop index so the VF path has no runtime branch.
        __ubuf__ float* srcUb0 = srcUb;
        __ubuf__ float* srcUb1 = srcUb0 + s2BaseSize;
        __ubuf__ float* srcUb2 = srcUb0 + s2BaseSize * 2;
        __ubuf__ float* srcUb3 = srcUb0 + s2BaseSize * 3;
        __ubuf__ float* srcUb4 = srcUb0 + s2BaseSize * (m / 4) * 4;
        Duplicate(max0Vreg, MIN_VALUE);
        Duplicate(max1Vreg, MIN_VALUE);
        Duplicate(max2Vreg, MIN_VALUE);
        Duplicate(max3Vreg, MIN_VALUE);
        for (int32_t i = 0; i < int32_t(m / 4); i++) {
            LoadAlign(src0Vreg, srcUb0 + i * s2BaseSize * 4);
            LoadAlign(src1Vreg, srcUb1 + i * s2BaseSize * 4);
            LoadAlign(src2Vreg, srcUb2 + i * s2BaseSize * 4);
            LoadAlign(src3Vreg, srcUb3 + i * s2BaseSize * 4);
            Max(max0Vreg, max0Vreg, src0Vreg, pregTailN);
            Max(max1Vreg, max1Vreg, src1Vreg, pregTailN);
            Max(max2Vreg, max2Vreg, src2Vreg, pregTailN);
            Max(max3Vreg, max3Vreg, src3Vreg, pregTailN);
        }
        Max(max0Vreg, max0Vreg, max2Vreg, pregTailN);
        Max(max1Vreg, max1Vreg, max3Vreg, pregTailN);
        Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);
        for (int32_t i = 0; i < int32_t(m % 4); i++) {
            LoadAlign(src4Vreg, srcUb4 + i * s2BaseSize);
            Max(max0Vreg, max0Vreg, src4Vreg, pregTailN);
        }
        Muls(max0Vreg, max0Vreg, dScale, pregTailN);

        if constexpr (isUpdate) {
            LoadAlign(max1Vreg, LastMaxUbStart);
            LoadAlign(lastExpSumVreg, lastExpSumUb);
            Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);
            FusedExpSub(expMaxVreg, max1Vreg, max0Vreg, pregTailN);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(LastMaxUbStart, max0Vreg, pregTailN);
        }

        StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)newMaxUb, max0Vreg, pregFull);
        // Keep the persisted online max as the real m. Only the temporary exp baseline uses m-ln(pScale),
        // so both HIF8 P and the FP32 sum are scaled by the same per-tensor pScale.
        Sub(max0Vreg, max0Vreg, lnPScaleVreg, pregTailN);

        Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(sum0Vreg, 0, pregFull);
        Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(sum1Vreg, 0, pregFull);
        Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(sum2Vreg, 0, pregFull);
        Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(sum3Vreg, 0, pregFull);

        for (int32_t i = 0; i < mFirstTile; i++) {
            LoadAlign(src0Vreg, srcUb + i * s2BaseSize);
            LoadAlign(src1Vreg, srcUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
            LoadAlign(src2Vreg, srcUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
            LoadAlign(src3Vreg, srcUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

            Muls(src0Vreg, src0Vreg, dScale, pregTailN);
            Muls(src1Vreg, src1Vreg, dScale, pregTailN);
            Muls(src2Vreg, src2Vreg, dScale, pregTailN);
            Muls(src3Vreg, src3Vreg, dScale, pregTailN);

            FusedExpSub(exp0Vreg32, src0Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp1Vreg32, src1Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp2Vreg32, src2Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp3Vreg32, src3Vreg, max0Vreg, pregTailN);

            Cast<ElementP, float, castTraitZeroRound>(exp0Vreg16, exp0Vreg32, pregFull);
            Cast<ElementP, float, castTraitOneRound>(exp1Vreg16, exp1Vreg32, pregFull);
            Cast<ElementP, float, castTraitTwoRound>(exp2Vreg16, exp2Vreg32, pregFull);
            Cast<ElementP, float, castTraitThreeRound>(exp3Vreg16, exp3Vreg32, pregFull);
            Or((RegTensor<uint8_t>&)merge01Vreg, (RegTensor<uint8_t>&)exp0Vreg16, (RegTensor<uint8_t>&)exp1Vreg16,
               pregFullB8);
            Or((RegTensor<uint8_t>&)merge23Vreg, (RegTensor<uint8_t>&)exp2Vreg16, (RegTensor<uint8_t>&)exp3Vreg16,
               pregFullB8);
            Or((RegTensor<uint8_t>&)packedPVreg, (RegTensor<uint8_t>&)merge01Vreg, (RegTensor<uint8_t>&)merge23Vreg,
               pregFullB8);
            Gather(packedPVreg, packedPVreg, pGatherIndexVreg);
            __ubuf__ ElementP* dstUb = expUb + static_cast<uint32_t>(i) * 32;
            StoreAlign<ElementP, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(dstUb, packedPVreg, blockStride,
                                                                              pregFullB8);

            if constexpr (mTileNum >= MAlignedTileNum::Zero) {
                Add(sum0Vreg, exp0Vreg32, sum0Vreg, pregTailN);
            }
            if constexpr (mTileNum >= MAlignedTileNum::Two) {
                Add(sum2Vreg, exp2Vreg32, sum2Vreg, pregTailN);
            }
            if constexpr (mTileNum >= MAlignedTileNum::One) {
                Add(sum1Vreg, exp1Vreg32, sum1Vreg, pregTailN);
            }
            if constexpr (mTileNum >= MAlignedTileNum::Three) {
                Add(sum3Vreg, exp3Vreg32, sum3Vreg, pregTailN);
            }
        }
        if constexpr (mTileNum < MAlignedTileNum::Four) {
            for (int32_t i = mFirstTile; i < int32_t(mRound / 4); i++) {
                LoadAlign(src0Vreg, srcUb + i * s2BaseSize);
                LoadAlign(src1Vreg, srcUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
                LoadAlign(src2Vreg, srcUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
                LoadAlign(src3Vreg, srcUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

                Muls(src0Vreg, src0Vreg, dScale, pregTailN);
                Muls(src1Vreg, src1Vreg, dScale, pregTailN);
                Muls(src2Vreg, src2Vreg, dScale, pregTailN);
                Muls(src3Vreg, src3Vreg, dScale, pregTailN);

                FusedExpSub(exp0Vreg32, src0Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp1Vreg32, src1Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp2Vreg32, src2Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp3Vreg32, src3Vreg, max0Vreg, pregTailN);

                Cast<ElementP, float, castTraitZeroRound>(exp0Vreg16, exp0Vreg32, pregFull);
                Cast<ElementP, float, castTraitOneRound>(exp1Vreg16, exp1Vreg32, pregFull);
                Cast<ElementP, float, castTraitTwoRound>(exp2Vreg16, exp2Vreg32, pregFull);
                Cast<ElementP, float, castTraitThreeRound>(exp3Vreg16, exp3Vreg32, pregFull);
                Or((RegTensor<uint8_t>&)merge01Vreg, (RegTensor<uint8_t>&)exp0Vreg16, (RegTensor<uint8_t>&)exp1Vreg16,
                   pregFullB8);
                Or((RegTensor<uint8_t>&)merge23Vreg, (RegTensor<uint8_t>&)exp2Vreg16, (RegTensor<uint8_t>&)exp3Vreg16,
                   pregFullB8);
                Or((RegTensor<uint8_t>&)packedPVreg, (RegTensor<uint8_t>&)merge01Vreg, (RegTensor<uint8_t>&)merge23Vreg,
                   pregFullB8);
                Gather(packedPVreg, packedPVreg, pGatherIndexVreg);
                __ubuf__ ElementP* dstUb = expUb + static_cast<uint32_t>(i) * 32;
                StoreAlign<ElementP, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(dstUb, packedPVreg, blockStride,
                                                                                  pregFullB8);

                if constexpr (mTileNum > MAlignedTileNum::Zero) {
                    Add(sum0Vreg, exp0Vreg32, sum0Vreg, pregTailN);
                }
                if constexpr (mTileNum > MAlignedTileNum::Two) {
                    Add(sum2Vreg, exp2Vreg32, sum2Vreg, pregTailN);
                }
                if constexpr (mTileNum > MAlignedTileNum::One) {
                    Add(sum1Vreg, exp1Vreg32, sum1Vreg, pregTailN);
                }
            }
        }
        Add(sum0Vreg, sum0Vreg, sum2Vreg, pregFull);
        Add(sum1Vreg, sum1Vreg, sum3Vreg, pregFull);
        Add(sum0Vreg, sum0Vreg, sum1Vreg, pregFull);
        if constexpr (isUpdate) {
            Add(updateExpSumVreg, updateExpSumVreg, sum0Vreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(lastExpSumUb, updateExpSumVreg, pregFull);
        }
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)expSumUb, sum0Vreg, pregFull);
    }
};
} // namespace NpuArch::Epilogue::Block

#endif

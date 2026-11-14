/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_A8C4_HPP
#define GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_A8C4_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"

namespace NpuArch::Epilogue::Block {

class A8C4OnlineSoftmax {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = fp8_e4m3fn_t;
    using ElementInput = float;
    using LayoutOutput = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = 32;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 512;
    static constexpr uint32_t SM_VREG_SIZE = 64;
    static constexpr uint32_t P_STORE_COL_COUNT = 2U * FLOAT_VECTOR_SIZE;
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t P_UB_EXTRA_ROWS = 1U;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr float MIN_VALUE = -3.402823466e+38F;

    __aicore__ inline A8C4OnlineSoftmax(Arch::Resource<ArchTag>& resource, float scaleValue_,
                                        UBufTileHelper& uBufTileHelper)
        : scaleValue(scaleValue_)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(float) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset + (uBufTileHelper.qBaseTilePerSubCore + P_UB_EXTRA_ROWS) *
                                                  uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementOutput) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; ++i) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset + UBufTileHelper::STATS_ROW_NUM * sizeof(float) * i);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
    }

    __aicore__ inline ~A8C4OnlineSoftmax() {}

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

    struct NoAfterSoftmax {
        __aicore__ inline void operator()() const {}
    };

    template <class TensorP, class AfterSoftmax = NoAfterSoftmax>
    __aicore__ inline void operator()(TensorP& l1PTensorTla, GemmCoord actualBlockShape, uint32_t isFirstKvSTile,
                                      uint32_t ubSBufId, uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag, AfterSoftmax afterSoftmax = {})
    {
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 2) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            afterSoftmax();
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }

        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        const uint16_t sRound = RoundUp(n, 16U);
        uint32_t blockStride = mRound + P_UB_EXTRA_ROWS;
        uint16_t nLoop = n / P_STORE_COL_COUNT;
        const uint32_t tailCols = n % P_STORE_COL_COUNT;
        const uint16_t hasNTail = tailCols != 0U;
        const uint16_t hasSecondTail = tailCols > FLOAT_VECTOR_SIZE;
        const uint32_t firstTail = min(tailCols, FLOAT_VECTOR_SIZE);
        const uint32_t secondTail = tailCols - firstTail;

        __ubuf__ float* sUbAddr = (__ubuf__ float*)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput* pUbAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float* gmUbAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* glUbAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ float* dmUbAddr = (__ubuf__ float*)dmUbTensor[l1PBufId].GetPhyAddr();

        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        if (isFirstKvSTile) {
            ComputeAndUpdateMax<true>(sUbAddr, gmUbAddr, dmUbAddr, m, firstTail, secondTail, scaleValue, sRound, nLoop,
                                      hasNTail, hasSecondTail);
        } else {
            ComputeAndUpdateMax<false>(sUbAddr, gmUbAddr, dmUbAddr, m, firstTail, secondTail, scaleValue, sRound, nLoop,
                                       hasNTail, hasSecondTail);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (isFirstKvSTile) {
            ComputePAndUpdateSum<true>(pUbAddr, sUbAddr, gmUbAddr, glUbAddr, dmUbAddr, m, firstTail, secondTail,
                                       tailCols, blockStride, sRound, nLoop, hasNTail, hasSecondTail);
        } else {
            ComputePAndUpdateSum<false>(pUbAddr, sUbAddr, gmUbAddr, glUbAddr, dmUbAddr, m, firstTail, secondTail,
                                        tailCols, blockStride, sRound, nLoop, hasNTail, hasSecondTail);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        afterSoftmax();

        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        const uint32_t l1Rows = RoundUp(actualBlockShape.m(), 16U);
        auto l1P = l1PTensorTla.data();
        AscendC::DataCopyParams copy{static_cast<uint16_t>(nRound / 32U), static_cast<uint16_t>(m),
                                     static_cast<uint16_t>(blockStride - m), static_cast<uint16_t>(l1Rows - m)};
        AscendC::DataCopy(l1P[subBlockIdx_ * mCopyOffset * 32U], lpUbTensor[ubSBufId], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    // Each column iteration processes two FP32 registers from one row.
    // A partial 128-column pair may contain one or two 64-column tails.
    template <bool IsFirstKvSTile>
    static __simd_vf__ inline void ComputeAndUpdateMax(__ubuf__ float* sUb, __ubuf__ float* gmUb, __ubuf__ float* dmUb,
                                                       uint16_t m, uint32_t firstTail, uint32_t secondTail,
                                                       float scaleValue, uint16_t s2BaseSize, uint16_t nLoop,
                                                       uint16_t hasNTail, uint16_t hasSecondTail)
    {
        using namespace AscendC::Reg;
        RegTensor<float> sVreg0;
        RegTensor<float> sVreg1;
        RegTensor<float> maxVreg;
        RegTensor<float> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> hmVreg;
        RegTensor<float> dmVreg;
        UnalignReg gmUreg;
        UnalignReg dmUreg;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg tail0 = UpdateMask<float>(firstTail);
        MaskReg tail1 = UpdateMask<float>(secondTail);
        for (uint16_t i = 0; i < m; ++i) {
            Duplicate(maxVreg, MIN_VALUE);
            for (uint16_t j = 0; j < nLoop; ++j) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize + j * P_STORE_COL_COUNT);
                LoadAlign(sVreg1, sUb + i * s2BaseSize + j * P_STORE_COL_COUNT + FLOAT_VECTOR_SIZE);
                Muls(sVreg0, sVreg0, scaleValue, full);
                Muls(sVreg1, sVreg1, scaleValue, full);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(sUb + i * s2BaseSize + j * P_STORE_COL_COUNT, sVreg0, full);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    sUb + i * s2BaseSize + j * P_STORE_COL_COUNT + FLOAT_VECTOR_SIZE, sVreg1, full);
                Max(sVreg0, sVreg0, sVreg1, full);
                Max(maxVreg, maxVreg, sVreg0, full);
            }
            for (uint16_t j = 0; j < hasNTail; ++j) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize + (nLoop + j) * P_STORE_COL_COUNT);
                Muls(sVreg0, sVreg0, scaleValue, tail0);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(sUb + i * s2BaseSize + (nLoop + j) * P_STORE_COL_COUNT,
                                                            sVreg0, tail0);
                Max<float, MaskMergeMode::MERGING>(maxVreg, maxVreg, sVreg0, tail0);
                for (uint16_t k = 0; k < hasSecondTail; ++k) {
                    LoadAlign(sVreg1,
                              sUb + i * s2BaseSize + (nLoop + j) * P_STORE_COL_COUNT + (k + 1U) * FLOAT_VECTOR_SIZE);
                    Muls(sVreg1, sVreg1, scaleValue, tail1);
                    StoreAlign<float, StoreDist::DIST_NORM_B32>(
                        sUb + i * s2BaseSize + (nLoop + j) * P_STORE_COL_COUNT + (k + 1U) * FLOAT_VECTOR_SIZE, sVreg1,
                        tail1);
                    Max<float, MaskMergeMode::MERGING>(maxVreg, maxVreg, sVreg1, tail1);
                }
            }
            ReduceMax(lmVreg, maxVreg, full);
            if constexpr (IsFirstKvSTile) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(gmUb, lmVreg, gmUreg, 1);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gmVreg, gmUb);
                Max(hmVreg, lmVreg, gmVreg, full);
                FusedExpSub(dmVreg, gmVreg, hmVreg, full);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(gmUb, hmVreg, gmUreg, 1);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dmUb, dmVreg, dmUreg, 1);
            }
        }
        vstas(gmUreg, gmUb, 0, POST_UPDATE);
        if constexpr (!IsFirstKvSTile) {
            vstas(dmUreg, dmUb, 0, POST_UPDATE);
        }
    }

    template <bool IsFirstKvSTile>
    static __simd_vf__ inline void ComputePAndUpdateSum(__ubuf__ ElementOutput* pUb, __ubuf__ float* sUb,
                                                        __ubuf__ float* gmUb, __ubuf__ float* glUb,
                                                        __ubuf__ float* dmUb, uint16_t m, uint32_t firstTail,
                                                        uint32_t secondTail, uint32_t tailCols, uint32_t blockStride,
                                                        uint16_t s2BaseSize, uint16_t nLoop, uint16_t hasNTail,
                                                        uint16_t hasSecondTail)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait castTraitOneRound = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};
        RegTensor<float> sVreg0;
        RegTensor<float> sVreg1;
        RegTensor<float> pVreg0;
        RegTensor<float> pVreg1;
        RegTensor<float> sumVreg;
        RegTensor<float> llVreg;
        RegTensor<float> hmVreg;
        RegTensor<float> dmVreg;
        RegTensor<float> glVreg;
        RegTensor<ElementOutput> pVreg80;
        RegTensor<ElementOutput> pVreg81;
        RegTensor<ElementOutput> packedPVreg;
        RegTensor<uint8_t> laneVreg;
        RegTensor<uint8_t> indexLeftVreg;
        RegTensor<uint8_t> indexRightVreg;
        RegTensor<uint8_t> pGatherIndexVreg;
        UnalignReg glUreg;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg fullB8 = CreateMask<uint8_t, MaskPattern::ALL>();
        uint32_t fullStore = P_STORE_COL_COUNT;
        MaskReg outputFull = UpdateMask<uint8_t>(fullStore);
        MaskReg tail0 = UpdateMask<float>(firstTail);
        MaskReg tail1 = UpdateMask<float>(secondTail);
        const uint32_t tailRound = RoundUp(tailCols, ELE_NUM_PER_C0);
        uint32_t tailStore = tailRound;
        MaskReg outputTail = UpdateMask<uint8_t>(tailStore);
        // DN index[d] = 4 * (d % 64) + d / 64, generated entirely in registers.
        Arange(reinterpret_cast<RegTensor<int8_t>&>(laneVreg), static_cast<int8_t>(0));
        ShiftLefts(indexLeftVreg, laneVreg, static_cast<int16_t>(2), fullB8);
        ShiftRights(indexRightVreg, laneVreg, static_cast<int16_t>(6), fullB8);
        Or(pGatherIndexVreg, indexLeftVreg, indexRightVreg, fullB8);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(hmVreg, gmUb + i);
            Duplicate(sumVreg, 0.0f);
            for (uint16_t j = 0; j < nLoop; ++j) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize + j * P_STORE_COL_COUNT);
                LoadAlign(sVreg1, sUb + i * s2BaseSize + j * P_STORE_COL_COUNT + FLOAT_VECTOR_SIZE);
                FusedExpSub(pVreg0, sVreg0, hmVreg, full);
                FusedExpSub(pVreg1, sVreg1, hmVreg, full);
                Cast<ElementOutput, float, castTraitZeroRound>(pVreg80, pVreg0, full);
                Cast<ElementOutput, float, castTraitOneRound>(pVreg81, pVreg1, full);
                Or(reinterpret_cast<RegTensor<uint8_t>&>(packedPVreg), reinterpret_cast<RegTensor<uint8_t>&>(pVreg80),
                   reinterpret_cast<RegTensor<uint8_t>&>(pVreg81), fullB8);
                Gather(packedPVreg, packedPVreg, pGatherIndexVreg);
                StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                    pUb + i * ELE_NUM_PER_C0 + j * P_STORE_COL_COUNT * blockStride, packedPVreg, blockStride,
                    outputFull);
                // The sum still uses FP32 probabilities, never rounded FP8 P.
                Add(pVreg0, pVreg0, pVreg1, full);
                Add(sumVreg, sumVreg, pVreg0, full);
            }
            for (uint16_t j = 0; j < hasNTail; ++j) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize + (nLoop + j) * P_STORE_COL_COUNT);
                // FP8 NZ padding columns must be zero, including unselected cast lanes.
                Duplicate(pVreg0, 0.0f);
                FusedExpSub(pVreg0, sVreg0, hmVreg, tail0);
                Cast<ElementOutput, float, castTraitZeroRound>(pVreg80, pVreg0, full);
                Duplicate(reinterpret_cast<RegTensor<uint8_t>&>(pVreg81), static_cast<uint8_t>(0), fullB8);
                for (uint16_t k = 0; k < hasSecondTail; ++k) {
                    LoadAlign(sVreg1,
                              sUb + i * s2BaseSize + (nLoop + j) * P_STORE_COL_COUNT + (k + 1U) * FLOAT_VECTOR_SIZE);
                    Duplicate(pVreg1, 0.0f);
                    FusedExpSub(pVreg1, sVreg1, hmVreg, tail1);
                    Cast<ElementOutput, float, castTraitOneRound>(pVreg81, pVreg1, full);
                    Add<float, MaskMergeMode::MERGING>(pVreg0, pVreg0, pVreg1, tail1);
                }
                Or(reinterpret_cast<RegTensor<uint8_t>&>(packedPVreg), reinterpret_cast<RegTensor<uint8_t>&>(pVreg80),
                   reinterpret_cast<RegTensor<uint8_t>&>(pVreg81), fullB8);
                Gather(packedPVreg, packedPVreg, pGatherIndexVreg);
                StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                    pUb + i * ELE_NUM_PER_C0 + (nLoop + j) * P_STORE_COL_COUNT * blockStride, packedPVreg, blockStride,
                    outputTail);
                Add<float, MaskMergeMode::MERGING>(sumVreg, sumVreg, pVreg0, tail0);
            }
            ReduceSum(llVreg, sumVreg, full);
            if constexpr (IsFirstKvSTile) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(glUb, llVreg, glUreg, 1);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(glVreg, glUb);
                Mul(glVreg, glVreg, dmVreg, full);
                Add(glVreg, glVreg, llVreg, full);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(glUb, glVreg, glUreg, 1);
            }
        }
        vstas(glUreg, glUb, 0, POST_UPDATE);
    }

private:
    float scaleValue;
    AscendC::LocalTensor<float> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<float> glUbTensor;
    uint32_t subBlockIdx_;
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_A8C4_HPP

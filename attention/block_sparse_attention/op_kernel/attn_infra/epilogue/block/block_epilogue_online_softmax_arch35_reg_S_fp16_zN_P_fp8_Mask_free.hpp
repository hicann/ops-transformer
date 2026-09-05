/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_S_FP16_ZN_P_FP8_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_S_FP16_ZN_P_FP8_HPP

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

template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueB8FullQuantOnlineSoftmaxBsa<false, true>, OutputType_, Gemm::GemmType<half, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueB8FullQuantOnlineSoftmaxBsa<false, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element; // b8
    using ElementInput = half;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue, UBufTileHelper &uBufTileHelper)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        subBlockNum_ = AscendC::GetSubBlockNum();
        scaleValue_ = static_cast<ElementInput>(scaleValue);
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; i++) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementInput) * i);
            // P pingpong shares space with S pingpong coresbondingly
            pUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementInput) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; i++) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset + uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.gmStartOffset);
        gmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.lmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.llStartOffset);
        // Init idx table for gatherP, only when zNOnlineSoftmax is true
        uint32_t gatherPIdxOffset =
            uBufTileHelper.lseStartOffset + uBufTileHelper.qBaseTilePerSubCore * 8 * sizeof(float);
        gatherPIdxUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(gatherPIdxOffset);
        uint32_t i1 = 0;
        uint32_t i2 = 1;
        for (uint32_t i = 0; i < 256; i += 32) {
            for (uint32_t j = i; j < i + 16; j++) {
                gatherPIdxUbTensor.SetValue(j, i1);
                i1 += 2;
            }
            for (uint32_t j = i + 16; j < i + 32; j++) {
                gatherPIdxUbTensor.SetValue(j, i2);
                i2 += 2;
            }
        }
        // Init tail mask, only when zNOnlineSoftmax is true
        uint32_t tailMaskOffset = gatherPIdxOffset + 256;
        tailMaskUbTensor = resource.ubBuf.template GetBufferByByte<int32_t>(tailMaskOffset);
        tailMaskUbTensor16 = resource.ubBuf.template GetBufferByByte<int16_t>(tailMaskOffset);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t rowNum)
    {
        AscendC::DataCopyParams repeatParams;
        uint32_t elementNumPerC0 = 32;
        repeatParams.blockCount = tla::get<1, 1>(srcTensor.shape());
        repeatParams.blockLen = rowNum;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / elementNumPerC0 - rowNum;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / elementNumPerC0 - rowNum;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <class TensorP>
    __aicore__ inline void SubCoreCompute(TensorP &l1PTensorTlaTile, uint32_t rowNumCurSubCore,
                                          uint32_t colNumCurSubCore, uint32_t zNRowNumSplit, uint32_t isFirstKvSTile,
                                          uint32_t isLastKvSTile, uint32_t ubSBufId, uint32_t ubDmBufId,
                                          Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {
        __ubuf__ ElementInput *sUbAddr = (__ubuf__ ElementInput *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput *pUbAddr = (__ubuf__ ElementOutput *)pUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementInput *gmUbAddr = (__ubuf__ ElementInput *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *gmUbAddr32 = (__ubuf__ float *)gmUbTensor32.GetPhyAddr();
        __ubuf__ float *glUbAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *lmUbAddr = (__ubuf__ ElementInput *)lmUbTensor.GetPhyAddr();
        __ubuf__ float *llUbAddr = (__ubuf__ float *)llUbTensor.GetPhyAddr();
        __ubuf__ float *dmUbAddr = (__ubuf__ float *)dmUbTensor[ubDmBufId].GetPhyAddr();
        __ubuf__ uint8_t *gatherPIdxUbAddr = (__ubuf__ uint8_t *)gatherPIdxUbTensor.GetPhyAddr();
        __ubuf__ int32_t *tailMaskUbAddr = (__ubuf__ int32_t *)tailMaskUbTensor.GetPhyAddr();
        __ubuf__ int16_t *tailMaskUbAddr16 = (__ubuf__ int16_t *)tailMaskUbTensor16.GetPhyAddr();

        uint32_t rowAligned16 = (zNRowNumSplit + 15) / 16 * 16;
        uint32_t colAligned16 = (colNumCurSubCore + 15) / 16 * 16;
        uint32_t colAligned32 = (colNumCurSubCore + 31) / 32 * 32;
        // for unaligned col
        uint32_t colMainLoopNum16 = colAligned16 / 16 - 1;
        uint32_t colMainLoopNum32 = colAligned32 / 32 - 1;
        uint32_t tailSizePerB16Fractal = colNumCurSubCore % 16; // zN fractal size of b16 is 16
        uint32_t tailSizePerB8Fractal = colNumCurSubCore % 32;  // zN fractal size of b8 is 32
        ElementInput minValue = -65504.0f;
        float pScale = 448.0f; // per tensor static quant

        // wait QK fixPipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (isFirstKvSTile) {
            if (tailSizePerB16Fractal == 0) {
                if (tailSizePerB8Fractal == 0) {
                    RowmaxImplzNAligned<false>(sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, rowNumCurSubCore, rowAligned16,
                                               colAligned16, minValue, pScale);
                    PAndRowSumImplzNAligned<false>(pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr,
                                                   gatherPIdxUbAddr, rowNumCurSubCore, rowAligned16, colAligned32);
                } else {
                    RowmaxImplzNUnAligned<false, ColzNFractalTailSizeStatus::EQ_16>(
                        sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, tailMaskUbAddr16, rowNumCurSubCore, rowAligned16,
                        colMainLoopNum16, minValue, pScale);
                    PAndRowSumImplzNUnAligned<false, ColzNFractalTailSizeStatus::EQ_16>(
                        pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr, gatherPIdxUbAddr, tailMaskUbAddr16,
                        rowNumCurSubCore, rowAligned16, colMainLoopNum32, minValue);
                }
            } else if (tailSizePerB8Fractal < 16) {
                GenB16zNTailMask(tailMaskUbAddr, tailSizePerB16Fractal);
                RowmaxImplzNUnAligned<false, ColzNFractalTailSizeStatus::GT_0_LT_16>(
                    sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, tailMaskUbAddr16, rowNumCurSubCore, rowAligned16,
                    colMainLoopNum16, minValue, pScale);
                PAndRowSumImplzNUnAligned<false, ColzNFractalTailSizeStatus::GT_0_LT_16>(
                    pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr, gatherPIdxUbAddr, tailMaskUbAddr16,
                    rowNumCurSubCore, rowAligned16, colMainLoopNum32, minValue);
            } else {
                GenB16zNTailMask(tailMaskUbAddr, tailSizePerB16Fractal);
                RowmaxImplzNUnAligned<false, ColzNFractalTailSizeStatus::GT_16_LT_32>(
                    sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, tailMaskUbAddr16, rowNumCurSubCore, rowAligned16,
                    colMainLoopNum16, minValue, pScale);
                PAndRowSumImplzNUnAligned<false, ColzNFractalTailSizeStatus::GT_16_LT_32>(
                    pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr, gatherPIdxUbAddr, tailMaskUbAddr16,
                    rowNumCurSubCore, rowAligned16, colMainLoopNum32, minValue);
            }
        } else {
            if (tailSizePerB16Fractal == 0) {
                if (tailSizePerB8Fractal == 0) {
                    RowmaxImplzNAligned<true>(sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, rowNumCurSubCore, rowAligned16,
                                              colAligned16, minValue, pScale);
                    UpdateRowMaxImpl(dmUbAddr, gmUbAddr, lmUbAddr, rowNumCurSubCore);
                    PAndRowSumImplzNAligned<true>(pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr,
                                                  gatherPIdxUbAddr, rowNumCurSubCore, rowAligned16, colAligned32);
                } else {
                    RowmaxImplzNUnAligned<true, ColzNFractalTailSizeStatus::EQ_16>(
                        sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, tailMaskUbAddr16, rowNumCurSubCore, rowAligned16,
                        colMainLoopNum16, minValue, pScale);
                    UpdateRowMaxImpl(dmUbAddr, gmUbAddr, lmUbAddr, rowNumCurSubCore);
                    PAndRowSumImplzNUnAligned<true, ColzNFractalTailSizeStatus::EQ_16>(
                        pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr, gatherPIdxUbAddr, tailMaskUbAddr16,
                        rowNumCurSubCore, rowAligned16, colMainLoopNum32, minValue);
                }
            } else if (tailSizePerB8Fractal < 16) {
                GenB16zNTailMask(tailMaskUbAddr, tailSizePerB16Fractal);
                RowmaxImplzNUnAligned<true, ColzNFractalTailSizeStatus::GT_0_LT_16>(
                    sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, tailMaskUbAddr16, rowNumCurSubCore, rowAligned16,
                    colMainLoopNum16, minValue, pScale);
                UpdateRowMaxImpl(dmUbAddr, gmUbAddr, lmUbAddr, rowNumCurSubCore);
                PAndRowSumImplzNUnAligned<true, ColzNFractalTailSizeStatus::GT_0_LT_16>(
                    pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr, gatherPIdxUbAddr, tailMaskUbAddr16,
                    rowNumCurSubCore, rowAligned16, colMainLoopNum32, minValue);
            } else {
                GenB16zNTailMask(tailMaskUbAddr, tailSizePerB16Fractal);
                RowmaxImplzNUnAligned<true, ColzNFractalTailSizeStatus::GT_16_LT_32>(
                    sUbAddr, gmUbAddr, lmUbAddr, dmUbAddr, tailMaskUbAddr16, rowNumCurSubCore, rowAligned16,
                    colMainLoopNum16, minValue, pScale);
                UpdateRowMaxImpl(dmUbAddr, gmUbAddr, lmUbAddr, rowNumCurSubCore);
                PAndRowSumImplzNUnAligned<true, ColzNFractalTailSizeStatus::GT_16_LT_32>(
                    pUbAddr, sUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr, gatherPIdxUbAddr, tailMaskUbAddr16,
                    rowNumCurSubCore, rowAligned16, colMainLoopNum32, minValue);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        // copy P to L1
        // wait till P L1 space is fully released
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(rowAligned16, colAligned32);
        auto ubPTensorTla = tla::MakeTensor(pUbTensor[ubSBufId], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile =
            GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, rowNumCurSubCore);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // trigger PV mm
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        // release S UB space. S/P UB reuse, thus the trigger pipe is MTE3
        SetCrossCoreSync<4, PIPE_MTE3>(mm1ToSmFlag);
        if (!isFirstKvSTile) {
            if (!isLastKvSTile) {
                UpdateExpSumImpl(dmUbAddr, gmUbAddr, lmUbAddr, glUbAddr, llUbAddr);
            } else {
                UpdateExpSumLastTileImpl(dmUbAddr, gmUbAddr, gmUbAddr32, lmUbAddr, glUbAddr, llUbAddr);
            }
        } else if (isLastKvSTile) {
            RestoreRowMaxLastAndFirstTileImpl(gmUbAddr, gmUbAddr32);
        }
    }

    template <class TensorP>
    __aicore__ inline void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape, uint32_t isFirstKvSTile,
                                      uint32_t isLastKvSTile, uint32_t ubSBufId, uint32_t ubDmBufId,
                                      Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {
        uint32_t rowNum = actualBlockShape[0];
        uint32_t colNum = actualBlockShape[1];
        // data split stays consistent with QK fixPipe config
        uint32_t rowNumAligned8 = RoundUp(rowNum, 8);
        uint32_t colNumAligned16 = RoundUp(colNum, 16);
        uint32_t rowNumSplit = rowNumAligned8 / subBlockNum_;
        uint32_t zNRowNumSplit = rowNumSplit;
        rowNumSplit = (rowNum < rowNumSplit) ? rowNum : rowNumSplit;
        uint32_t rowNumCurSubCore = (subBlockIdx_ == 0) ? rowNumSplit : (rowNum - rowNumSplit);
        uint32_t rowOffsetCurSubCore = rowNumSplit * subBlockIdx_;
        uint32_t colNumCurSubCore = colNum;
        uint32_t colStrideCurSubCore = colNumAligned16;

        auto l1PTensorTlaTile = GetTile(l1PTensorTla, tla::MakeCoord(rowOffsetCurSubCore, 0),
                                        tla::MakeShape(rowNumCurSubCore, colNumCurSubCore));
        if (rowNumCurSubCore == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        } else {
            SubCoreCompute(l1PTensorTlaTile, rowNumCurSubCore, colNumCurSubCore, zNRowNumSplit, isFirstKvSTile,
                           isLastKvSTile, ubSBufId, ubDmBufId, mm1ToSmFlag, smToMm2Flag);
        }
    }

private:
    ElementInput scaleValue_;
    AscendC::LocalTensor<ElementInput> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> pUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementInput> gmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<uint8_t> gatherPIdxUbTensor;
    AscendC::LocalTensor<int32_t> tailMaskUbTensor;
    AscendC::LocalTensor<int16_t> tailMaskUbTensor16;
    uint32_t subBlockIdx_;
    uint32_t subBlockNum_;

    enum class ColzNFractalTailSizeStatus {
        GT_0_LT_16,
        EQ_16,
        GT_16_LT_32,
    };

    __simd_vf__ inline void GenB16zNTailMask(__ubuf__ int32_t *tailMaskUb, uint32_t tailSizePerB16Fractal)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<int32_t> tailMaskElemVreg;
        UnalignReg tailMaskUreg;
        int32_t tailMaskElemVal = (1 << (2 * tailSizePerB16Fractal - 1)) - 1;
        Duplicate(tailMaskElemVreg, tailMaskElemVal);
        // store 8*int32_t data to tailMaskUb, which is 256bit
        StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(tailMaskUb, tailMaskElemVreg, tailMaskUreg, 8);
        StoreUnAlignPost<int32_t, PostLiteral::POST_MODE_UPDATE>(tailMaskUb, tailMaskUreg, 8);
    }

    template <bool NeedUpdate>
    __simd_vf__ inline void RowmaxImplzNAligned(__ubuf__ ElementInput *sUbAddr, __ubuf__ ElementInput *gmUbAddr,
                                                __ubuf__ ElementInput *lmUbAddr, __ubuf__ float *dmUbAddr, uint32_t row,
                                                uint32_t rowAligned16, uint32_t colAligned16, ElementInput minValue,
                                                float pScale)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<half> vreg_p_scale;
        RegTensor<half> vreg_ln_p_scale;
        RegTensor<ElementInput> vreg_input_x_1;
        RegTensor<ElementInput> vreg_input_x_unroll_1;

        RegTensor<half> vreg_max_tmp;
        RegTensor<half> vreg_max_tmp_unroll;

        UnalignRegForStore ureg_max;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg preg_real_m = UpdateMask<uint16_t>(row);

        Duplicate(vreg_p_scale, static_cast<half>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);
        for (uint16_t i = 0; i < (rowAligned16 / 16); ++i) { // 一次循环处理[16, 16, 16]
            Duplicate(vreg_max_tmp, minValue);
            Duplicate(vreg_max_tmp_unroll, minValue);
            for (uint16_t j = 0; j < (colAligned16 / 16); ++j) { // 每个kvTile内部，第j个[16, 16]
                LoadAlign(vreg_input_x_1, sUbAddr + i * 16 * 16 + j * rowAligned16 * 16); // 搬入第j个[16, 16]的前8行
                LoadAlign(vreg_input_x_unroll_1,
                          sUbAddr + i * 16 * 16 + j * rowAligned16 * 16 + 8 * 16); // 搬入第j个[16, 16]的后8行
                Max(vreg_max_tmp, vreg_max_tmp, vreg_input_x_1, preg_all); // 和已经读入的前j-1个[16, 16]的max值再取max
                Max(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_1, preg_all);
            }
            ReduceDataBlock<ReduceType::MAX>(vreg_max_tmp, vreg_max_tmp, preg_all); // 8行的8个max值
            ReduceDataBlock<ReduceType::MAX>(vreg_max_tmp_unroll, vreg_max_tmp_unroll, preg_all);
            Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
            Sub(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_ln_p_scale, preg_all);
            if constexpr (NeedUpdate) {
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)lmUbAddr), vreg_max_tmp, ureg_max,
                                                                  8); // 写回8行max到lmUb
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)lmUbAddr), vreg_max_tmp_unroll,
                                                                  ureg_max, 8);
            } else {
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)gmUbAddr), vreg_max_tmp, ureg_max,
                                                                  8); // 写回8行max到gmUb
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)gmUbAddr), vreg_max_tmp_unroll,
                                                                  ureg_max, 8);
            }
        }
        if constexpr (NeedUpdate) {
            StoreUnAlignPost<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)lmUbAddr), ureg_max, 0);
        } else {
            StoreUnAlignPost<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)gmUbAddr), ureg_max, 0);
        }
    }

    template <bool NeedUpdate>
    __simd_vf__ inline void PAndRowSumImplzNAligned(__ubuf__ ElementOutput *pUbAddr, __ubuf__ ElementInput *sUbAddr,
                                                    __ubuf__ ElementInput *gmUbAddr, __ubuf__ ElementInput *lmUbAddr,
                                                    __ubuf__ float *glUbAddr, __ubuf__ float *llUbAddr,
                                                    __ubuf__ uint8_t *gatherPIdxUbAddr, uint32_t row,
                                                    uint32_t rowAligned16, uint32_t colAligned32)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> vreg_input_x_1;
        RegTensor<ElementInput> vreg_input_x_unroll_1;
        RegTensor<half> vreg_input_x_2;
        RegTensor<half> vreg_input_x_unroll_2;

        RegTensor<half> vreg_max;
        RegTensor<half> vreg_max_2;

        RegTensor<float> vreg_exp_sum_1;
        RegTensor<float> vreg_exp_sum_2;

        RegTensor<float> vreg_exp_0_1;
        RegTensor<float> vreg_exp_1_1;
        RegTensor<float> vreg_exp_2_1;
        RegTensor<float> vreg_exp_3_1;
        RegTensor<float> vreg_exp_0_2;
        RegTensor<float> vreg_exp_1_2;
        RegTensor<float> vreg_exp_2_2;
        RegTensor<float> vreg_exp_3_2;

        RegTensor<ElementOutput> vreg_exp_0_f8_1;
        RegTensor<ElementOutput> vreg_exp_2_f8_1;
        RegTensor<ElementOutput> vreg_exp_1_f8_1;
        RegTensor<ElementOutput> vreg_exp_3_f8_1;
        RegTensor<ElementOutput> vreg_exp_0_f8_2;
        RegTensor<ElementOutput> vreg_exp_2_f8_2;
        RegTensor<ElementOutput> vreg_exp_1_f8_2;
        RegTensor<ElementOutput> vreg_exp_3_f8_2;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_1_1;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_1_2;
        RegTensor<ElementOutput> vreg_exp_merge_f8_1;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_2_1;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_2_2;
        RegTensor<ElementOutput> vreg_exp_merge_f8_2;

        RegTensor<uint8_t> vreg_exp_merge_f8_indexes;

        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b8 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg preg_real_m = UpdateMask<uint16_t>(row);
        for (uint16_t i = 0; i < (rowAligned16 / 16); ++i) {
            if constexpr (NeedUpdate) {
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max, lmUbAddr + i * 16); // 从gmUb读入8行的max并广播
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max_2, lmUbAddr + i * 16 + 8);
            } else {
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max, gmUbAddr + i * 16); // 从gmUb读入8行的max并广播
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max_2, gmUbAddr + i * 16 + 8);
            }

            Duplicate(vreg_exp_sum_1, 0, preg_all_b32); // sum清零
            Duplicate(vreg_exp_sum_2, 0, preg_all_b32);
            for (uint16_t j = 0; j < (colAligned32 / 32); ++j) {
                // 两个fp16的分形（一行16个元素）合一个fp8的分形（一行32个元素），第j个[64, 32]
                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_1, sUbAddr + i * 16 * 16 + j * rowAligned16 * 32); // 第一个分形的前8行
                LoadAlign<half, LoadDist::DIST_NORM>(vreg_input_x_unroll_1,
                                                     sUbAddr + i * 16 * 16 + j * rowAligned16 * 32 +
                                                         rowAligned16 * 16); // 第二个分形的前8行（第二个[64,16]）

                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_2, sUbAddr + i * 16 * 16 + j * rowAligned16 * 32 + 8 * 16); // 第一个分形的后8行
                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_unroll_2, sUbAddr + i * 16 * 16 + j * rowAligned16 * 32 + rowAligned16 * 16 + 8 * 16);

                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_1, vreg_input_x_1, vreg_max, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_1, vreg_input_x_1, vreg_max, preg_all);
                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_1, vreg_input_x_unroll_1, vreg_max, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_1, vreg_input_x_unroll_1, vreg_max, preg_all);

                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_2, vreg_input_x_2, vreg_max_2, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_2, vreg_input_x_2, vreg_max_2, preg_all);
                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);

                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_0_1, preg_all_b32);
                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_2_1, preg_all_b32);
                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_1_1, preg_all_b32);
                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_3_1, preg_all_b32);

                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_0_2, preg_all_b32);
                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_2_2, preg_all_b32);
                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_1_2, preg_all_b32);
                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_3_2, preg_all_b32);

                Cast<ElementOutput, float, castTraitRintZero>(vreg_exp_0_f8_1, vreg_exp_0_1, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintTwo>(vreg_exp_2_f8_1, vreg_exp_2_1, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintOne>(vreg_exp_1_f8_1, vreg_exp_1_1, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintThree>(vreg_exp_3_f8_1, vreg_exp_3_1, preg_all_b32);

                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_1,
                   (RegTensor<uint8_t> &)vreg_exp_2_f8_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_1,
                   (RegTensor<uint8_t> &)vreg_exp_3_f8_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1,
                   (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, preg_all_b8);

                LoadAlign(vreg_exp_merge_f8_indexes, gatherPIdxUbAddr);
                Gather(vreg_exp_merge_f8_1, vreg_exp_merge_f8_1, vreg_exp_merge_f8_indexes);
                StoreAlign(pUbAddr + i * 16 * 32 + j * rowAligned16 * 32, vreg_exp_merge_f8_1, preg_all_b8);

                // 16行中的后8行
                Cast<ElementOutput, float, castTraitRintZero>(vreg_exp_0_f8_2, vreg_exp_0_2, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintTwo>(vreg_exp_2_f8_2, vreg_exp_2_2, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintOne>(vreg_exp_1_f8_2, vreg_exp_1_2, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintThree>(vreg_exp_3_f8_2, vreg_exp_3_2, preg_all_b32);

                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_2,
                   (RegTensor<uint8_t> &)vreg_exp_2_f8_2, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_2,
                   (RegTensor<uint8_t> &)vreg_exp_3_f8_2, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1,
                   (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, preg_all_b8);

                Gather(vreg_exp_merge_f8_2, vreg_exp_merge_f8_2, vreg_exp_merge_f8_indexes);
                StoreAlign(pUbAddr + i * 16 * 32 + j * rowAligned16 * 32 + 8 * 32, vreg_exp_merge_f8_2, preg_all_b8);
            }
            ReduceDataBlock<ReduceType::SUM>(vreg_exp_sum_1, vreg_exp_sum_1, preg_all_b32);
            ReduceDataBlock<ReduceType::SUM>(vreg_exp_sum_2, vreg_exp_sum_2, preg_all_b32);
            if constexpr (NeedUpdate) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)llUbAddr), vreg_exp_sum_1,
                                                                   ureg_exp_sum, 8);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)llUbAddr), vreg_exp_sum_2,
                                                                   ureg_exp_sum, 8);
            } else {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)glUbAddr), vreg_exp_sum_1,
                                                                   ureg_exp_sum, 8);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)glUbAddr), vreg_exp_sum_2,
                                                                   ureg_exp_sum, 8);
            }
        }
        if constexpr (NeedUpdate) {
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)llUbAddr), ureg_exp_sum, 0);
        } else {
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)glUbAddr), ureg_exp_sum, 0);
        }
    }

    template <bool NeedUpdate, ColzNFractalTailSizeStatus TailSizeStatus>
    __simd_vf__ inline void RowmaxImplzNUnAligned(__ubuf__ ElementInput *sUbAddr, __ubuf__ ElementInput *gmUbAddr,
                                                  __ubuf__ ElementInput *lmUbAddr, __ubuf__ float *dmUbAddr,
                                                  __ubuf__ int16_t *tailMaskUbAddr, uint32_t row, uint32_t rowAligned16,
                                                  uint32_t colMainLoopNum16, ElementInput minValue, float pScale)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<half> vreg_p_scale;
        RegTensor<half> vreg_ln_p_scale;
        RegTensor<half> vreg_input_x_1;
        RegTensor<half> vreg_input_x_unroll_1;

        RegTensor<half> vreg_max_tmp;
        RegTensor<half> vreg_max_tmp_unroll;

        UnalignRegForStore ureg_max;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg preg_real_m = UpdateMask<uint16_t>(row);
        MaskReg preg_tail_fractal;

        Duplicate(vreg_p_scale, static_cast<half>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);
        if constexpr (TailSizeStatus != ColzNFractalTailSizeStatus::EQ_16) {
            LoadAlign(preg_tail_fractal, tailMaskUbAddr);
        }
        for (uint16_t i = 0; i < (rowAligned16 / 16); ++i) { // 一次循环处理[16, 16, 16]
            Duplicate(vreg_max_tmp, minValue);
            Duplicate(vreg_max_tmp_unroll, minValue);
            for (uint16_t j = 0; j < colMainLoopNum16; ++j) { // 每个kvTile内部，第j个[16, 16]
                LoadAlign(vreg_input_x_1, sUbAddr + i * 16 * 16 + j * rowAligned16 * 16); // 搬入第j个[16, 16]的前8行
                LoadAlign(vreg_input_x_unroll_1,
                          sUbAddr + i * 16 * 16 + j * rowAligned16 * 16 + 8 * 16); // 搬入第j个[16, 16]的后8行
                Max(vreg_max_tmp, vreg_max_tmp, vreg_input_x_1, preg_all); // 和已经读入的前j-1个[16, 16]的max值再取max
                Max(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_1, preg_all);
            }
            // tail col loop
            LoadAlign(vreg_input_x_1, sUbAddr + i * 16 * 16 + colMainLoopNum16 * rowAligned16 * 16);
            LoadAlign(vreg_input_x_unroll_1, sUbAddr + i * 16 * 16 + colMainLoopNum16 * rowAligned16 * 16 + 8 * 16);
            if constexpr (TailSizeStatus == ColzNFractalTailSizeStatus::EQ_16) {
                // 尾块填满16分型但填不满32分型
                Max(vreg_max_tmp, vreg_max_tmp, vreg_input_x_1,
                    preg_all); // 和已经读入的前colMainLoopNum16个[16, 16]的max值再取max
                Max(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_1, preg_all);
            } else {
                Max<half, MaskMergeMode::MERGING>(
                    vreg_max_tmp, vreg_max_tmp, vreg_input_x_1,
                    preg_tail_fractal); // 和已经读入的前colMainLoopNum16个[16, 16]的max值再取max
                Max<half, MaskMergeMode::MERGING>(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_input_x_unroll_1,
                                                  preg_tail_fractal);
            }
            // final fractal reduce
            ReduceDataBlock<ReduceType::MAX>(vreg_max_tmp, vreg_max_tmp, preg_all); // 8行的8个max值
            ReduceDataBlock<ReduceType::MAX>(vreg_max_tmp_unroll, vreg_max_tmp_unroll, preg_all);
            Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
            Sub(vreg_max_tmp_unroll, vreg_max_tmp_unroll, vreg_ln_p_scale, preg_all);
            if constexpr (NeedUpdate) {
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)lmUbAddr), vreg_max_tmp, ureg_max,
                                                                  8); // 写回8行max到lmUb
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)lmUbAddr), vreg_max_tmp_unroll,
                                                                  ureg_max, 8);
            } else {
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)gmUbAddr), vreg_max_tmp, ureg_max,
                                                                  8); // 写回8行max到gmUb
                StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ half *&)gmUbAddr), vreg_max_tmp_unroll,
                                                                  ureg_max, 8);
            }
        }
        if constexpr (NeedUpdate) {
            StoreUnAlignPost<half, PostLiteral::POST_MODE_UPDATE>(lmUbAddr, ureg_max, 0);
        } else {
            StoreUnAlignPost<half, PostLiteral::POST_MODE_UPDATE>(gmUbAddr, ureg_max, 0);
        }
    }

    template <bool NeedUpdate, ColzNFractalTailSizeStatus TailSizeStatus>
    __simd_vf__ inline void PAndRowSumImplzNUnAligned(__ubuf__ ElementOutput *pUbAddr, __ubuf__ ElementInput *sUbAddr,
                                                      __ubuf__ ElementInput *gmUbAddr, __ubuf__ ElementInput *lmUbAddr,
                                                      __ubuf__ float *glUbAddr, __ubuf__ float *llUbAddr,
                                                      __ubuf__ uint8_t *gatherPIdxUbAddr,
                                                      __ubuf__ int16_t *tailMaskUbAddr, uint32_t row,
                                                      uint32_t rowAligned16, uint32_t colMainLoopNum32,
                                                      ElementInput minValue)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<half> vreg_input_x_1;
        RegTensor<half> vreg_input_x_unroll_1;
        RegTensor<half> vreg_input_x_2;
        RegTensor<half> vreg_input_x_unroll_2;

        RegTensor<half> vreg_min_value;
        RegTensor<half> vreg_min_value_unroll;

        RegTensor<half> vreg_max;
        RegTensor<half> vreg_max_2;

        RegTensor<float> vreg_exp_sum_1;
        RegTensor<float> vreg_exp_sum_2;

        RegTensor<float> vreg_exp_0_1;
        RegTensor<float> vreg_exp_1_1;
        RegTensor<float> vreg_exp_2_1;
        RegTensor<float> vreg_exp_3_1;
        RegTensor<float> vreg_exp_0_2;
        RegTensor<float> vreg_exp_1_2;
        RegTensor<float> vreg_exp_2_2;
        RegTensor<float> vreg_exp_3_2;

        RegTensor<ElementOutput> vreg_exp_0_f8_1;
        RegTensor<ElementOutput> vreg_exp_2_f8_1;
        RegTensor<ElementOutput> vreg_exp_1_f8_1;
        RegTensor<ElementOutput> vreg_exp_3_f8_1;
        RegTensor<ElementOutput> vreg_exp_0_f8_2;
        RegTensor<ElementOutput> vreg_exp_2_f8_2;
        RegTensor<ElementOutput> vreg_exp_1_f8_2;
        RegTensor<ElementOutput> vreg_exp_3_f8_2;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_1_1;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_1_2;
        RegTensor<ElementOutput> vreg_exp_merge_f8_1;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_2_1;
        RegTensor<ElementOutput> vreg_exp_merge_tmp_f8_2_2;
        RegTensor<ElementOutput> vreg_exp_merge_f8_2;

        RegTensor<uint8_t> vreg_exp_merge_f8_indexes;

        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b8 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg preg_tail_fractal;

        Duplicate(vreg_min_value, minValue);
        Duplicate(vreg_min_value_unroll, minValue);
        if constexpr (TailSizeStatus != ColzNFractalTailSizeStatus::EQ_16) {
            LoadAlign(preg_tail_fractal, tailMaskUbAddr);
        }
        for (uint16_t i = 0; i < (rowAligned16 / 16); ++i) {
            if constexpr (NeedUpdate) {
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max, lmUbAddr + i * 16); // 从gmUb读入8行的max并广播
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max_2, lmUbAddr + i * 16 + 8);
            } else {
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max, gmUbAddr + i * 16); // 从gmUb读入8行的max并广播
                LoadAlign<half, LoadDist::DIST_E2B_B16>(vreg_max_2, gmUbAddr + i * 16 + 8);
            }

            Duplicate(vreg_exp_sum_1, 0, preg_all); // sum清零
            Duplicate(vreg_exp_sum_2, 0, preg_all);
            for (uint16_t j = 0; j < colMainLoopNum32; ++j) {
                // 两个fp16的分形（一行16个元素）合一个fp8的分形（一行32个元素），第j个[64, 32]
                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_1, sUbAddr + i * 16 * 16 + j * rowAligned16 * 32); // 第一个分形的前8行
                LoadAlign<half, LoadDist::DIST_NORM>(vreg_input_x_unroll_1,
                                                     sUbAddr + i * 16 * 16 + j * rowAligned16 * 32 +
                                                         rowAligned16 * 16); // 第二个分形的前8行（第二个[64,16]）

                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_2, sUbAddr + i * 16 * 16 + j * rowAligned16 * 32 + 8 * 16); // 第一个分形的后8行
                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_unroll_2, sUbAddr + i * 16 * 16 + j * rowAligned16 * 32 + rowAligned16 * 16 + 8 * 16);

                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_1, vreg_input_x_1, vreg_max, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_1, vreg_input_x_1, vreg_max, preg_all);
                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_1, vreg_input_x_unroll_1, vreg_max, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_1, vreg_input_x_unroll_1, vreg_max, preg_all);

                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_2, vreg_input_x_2, vreg_max_2, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_2, vreg_input_x_2, vreg_max_2, preg_all);
                ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
                ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);

                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_0_1, preg_all_b32);
                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_2_1, preg_all_b32);
                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_1_1, preg_all_b32);
                Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_3_1, preg_all_b32);

                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_0_2, preg_all_b32);
                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_2_2, preg_all_b32);
                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_1_2, preg_all_b32);
                Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_3_2, preg_all_b32);

                Cast<ElementOutput, float, castTraitRintZero>(vreg_exp_0_f8_1, vreg_exp_0_1, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintTwo>(vreg_exp_2_f8_1, vreg_exp_2_1, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintOne>(vreg_exp_1_f8_1, vreg_exp_1_1, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintThree>(vreg_exp_3_f8_1, vreg_exp_3_1, preg_all_b32);

                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_1,
                   (RegTensor<uint8_t> &)vreg_exp_2_f8_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_1,
                   (RegTensor<uint8_t> &)vreg_exp_3_f8_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1,
                   (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, preg_all_b8);

                LoadAlign(vreg_exp_merge_f8_indexes, gatherPIdxUbAddr);
                Gather(vreg_exp_merge_f8_1, vreg_exp_merge_f8_1, vreg_exp_merge_f8_indexes);
                StoreAlign(pUbAddr + i * 16 * 32 + j * rowAligned16 * 32, vreg_exp_merge_f8_1, preg_all_b8);

                // 16行中的后8行
                Cast<ElementOutput, float, castTraitRintZero>(vreg_exp_0_f8_2, vreg_exp_0_2, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintTwo>(vreg_exp_2_f8_2, vreg_exp_2_2, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintOne>(vreg_exp_1_f8_2, vreg_exp_1_2, preg_all_b32);
                Cast<ElementOutput, float, castTraitRintThree>(vreg_exp_3_f8_2, vreg_exp_3_2, preg_all_b32);

                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_2,
                   (RegTensor<uint8_t> &)vreg_exp_2_f8_2, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_2,
                   (RegTensor<uint8_t> &)vreg_exp_3_f8_2, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1,
                   (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, preg_all_b8);

                Gather(vreg_exp_merge_f8_2, vreg_exp_merge_f8_2, vreg_exp_merge_f8_indexes);
                StoreAlign(pUbAddr + i * 16 * 32 + j * rowAligned16 * 32 + 8 * 32, vreg_exp_merge_f8_2, preg_all_b8);
            }
            // tail col loop
            // 两个fp16的分形（一行16个元素）合一个fp8的分形（一行32个元素）
            LoadAlign<half, LoadDist::DIST_NORM>(
                vreg_input_x_1, sUbAddr + i * 16 * 16 + colMainLoopNum32 * rowAligned16 * 32); // 第一个分形的前8行
            LoadAlign<half, LoadDist::DIST_NORM>(
                vreg_input_x_2,
                sUbAddr + i * 16 * 16 + colMainLoopNum32 * rowAligned16 * 32 + 8 * 16); // 第一个分形的后8行
            if constexpr (TailSizeStatus == ColzNFractalTailSizeStatus::EQ_16) {
                // 第一个分型完整，第二个分型全置为min
                Duplicate(vreg_input_x_unroll_1, minValue, preg_all);
                Duplicate(vreg_input_x_unroll_2, minValue, preg_all);
            } else if constexpr (TailSizeStatus == ColzNFractalTailSizeStatus::GT_0_LT_16) {
                // 第一个分型用tailMask更新，第二个分型全置为min
                Select(vreg_input_x_1, vreg_input_x_1, vreg_min_value, preg_tail_fractal);
                Select(vreg_input_x_2, vreg_input_x_2, vreg_min_value_unroll, preg_tail_fractal);
                Duplicate(vreg_input_x_unroll_1, minValue, preg_all);
                Duplicate(vreg_input_x_unroll_2, minValue, preg_all);
            } else {
                // 第一个分型完整，第二个分型用tailMask更新
                LoadAlign<half, LoadDist::DIST_NORM>(vreg_input_x_unroll_1,
                                                     sUbAddr + i * 16 * 16 + colMainLoopNum32 * rowAligned16 * 32 +
                                                         rowAligned16 * 16); // 第二个分形的前8行（第二个[64,16]）
                LoadAlign<half, LoadDist::DIST_NORM>(
                    vreg_input_x_unroll_2,
                    sUbAddr + i * 16 * 16 + colMainLoopNum32 * rowAligned16 * 32 + rowAligned16 * 16 + 8 * 16);
                Select(vreg_input_x_unroll_1, vreg_input_x_unroll_1, vreg_min_value, preg_tail_fractal);
                Select(vreg_input_x_unroll_2, vreg_input_x_unroll_2, vreg_min_value_unroll, preg_tail_fractal);
            }

            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_1, vreg_input_x_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_1, vreg_input_x_unroll_1, vreg_max, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_1, vreg_input_x_unroll_1, vreg_max, preg_all);

            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_0_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_2_2, vreg_input_x_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ZERO>(vreg_exp_1_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);
            ExpSub<float, half, RegLayout::ONE>(vreg_exp_3_2, vreg_input_x_unroll_2, vreg_max_2, preg_all);

            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_0_1, preg_all_b32);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_2_1, preg_all_b32);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_1_1, preg_all_b32);
            Add(vreg_exp_sum_1, vreg_exp_sum_1, vreg_exp_3_1, preg_all_b32);

            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_0_2, preg_all_b32);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_2_2, preg_all_b32);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_1_2, preg_all_b32);
            Add(vreg_exp_sum_2, vreg_exp_sum_2, vreg_exp_3_2, preg_all_b32);

            Cast<ElementOutput, float, castTraitRintZero>(vreg_exp_0_f8_1, vreg_exp_0_1, preg_all_b32);
            Cast<ElementOutput, float, castTraitRintTwo>(vreg_exp_2_f8_1, vreg_exp_2_1, preg_all_b32);
            Cast<ElementOutput, float, castTraitRintOne>(vreg_exp_1_f8_1, vreg_exp_1_1, preg_all_b32);
            Cast<ElementOutput, float, castTraitRintThree>(vreg_exp_3_f8_1, vreg_exp_3_1, preg_all_b32);

            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_1,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_1, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_1_2, preg_all_b8);

            LoadAlign(vreg_exp_merge_f8_indexes, gatherPIdxUbAddr);
            Gather(vreg_exp_merge_f8_1, vreg_exp_merge_f8_1, vreg_exp_merge_f8_indexes);
            StoreAlign(pUbAddr + i * 16 * 32 + colMainLoopNum32 * rowAligned16 * 32, vreg_exp_merge_f8_1, preg_all_b8);

            // 16行中的后8行
            Cast<ElementOutput, float, castTraitRintZero>(vreg_exp_0_f8_2, vreg_exp_0_2, preg_all_b32);
            Cast<ElementOutput, float, castTraitRintTwo>(vreg_exp_2_f8_2, vreg_exp_2_2, preg_all_b32);
            Cast<ElementOutput, float, castTraitRintOne>(vreg_exp_1_f8_2, vreg_exp_1_2, preg_all_b32);
            Cast<ElementOutput, float, castTraitRintThree>(vreg_exp_3_f8_2, vreg_exp_3_2, preg_all_b32);

            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1, (RegTensor<uint8_t> &)vreg_exp_0_f8_2,
               (RegTensor<uint8_t> &)vreg_exp_2_f8_2, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, (RegTensor<uint8_t> &)vreg_exp_1_f8_2,
               (RegTensor<uint8_t> &)vreg_exp_3_f8_2, preg_all_b8);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_f8_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_1,
               (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8_2_2, preg_all_b8);

            Gather(vreg_exp_merge_f8_2, vreg_exp_merge_f8_2, vreg_exp_merge_f8_indexes);
            StoreAlign(pUbAddr + i * 16 * 32 + colMainLoopNum32 * rowAligned16 * 32 + 8 * 32, vreg_exp_merge_f8_2,
                       preg_all_b8);

            ReduceDataBlock<ReduceType::SUM>(vreg_exp_sum_1, vreg_exp_sum_1, preg_all_b32);
            ReduceDataBlock<ReduceType::SUM>(vreg_exp_sum_2, vreg_exp_sum_2, preg_all_b32);
            if constexpr (NeedUpdate) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)llUbAddr), vreg_exp_sum_1,
                                                                   ureg_exp_sum, 8);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)llUbAddr), vreg_exp_sum_2,
                                                                   ureg_exp_sum, 8);
            } else {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)glUbAddr), vreg_exp_sum_1,
                                                                   ureg_exp_sum, 8);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)glUbAddr), vreg_exp_sum_2,
                                                                   ureg_exp_sum, 8);
            }
        }
        if constexpr (NeedUpdate) {
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)llUbAddr), ureg_exp_sum, 0);
        } else {
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)glUbAddr), ureg_exp_sum, 0);
        }
    }

    __simd_vf__ inline void UpdateRowMaxImpl(__ubuf__ float *dmUbAddr, __ubuf__ ElementInput *gmUbAddr,
                                             __ubuf__ ElementInput *lmUbAddr, uint32_t row)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<half> vreg_in_max;
        RegTensor<half> vreg_input_max;
        RegTensor<half> vreg_max_new;
        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg preg_real_m = UpdateMask<uint16_t>(row);
        LoadAlign(vreg_in_max, gmUbAddr);                            // 历史gm, 有效元素不超过64
        LoadAlign(vreg_input_max, lmUbAddr);                         // 当前基本块的lm
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_real_m); // hm
        StoreAlign<half, StoreDist::DIST_NORM_B16>(lmUbAddr, vreg_max_new, preg_all); // hm
    }

    __simd_vf__ inline void UpdateExpSumImpl(__ubuf__ float *dmUbAddr, __ubuf__ ElementInput *gmUbAddr,
                                             __ubuf__ ElementInput *lmUbAddr, __ubuf__ float *glUbAddr,
                                             __ubuf__ float *llUbAddr)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_sum_update;

        RegTensor<half> vreg_in_max;
        RegTensor<half> vreg_input_max;

        RegTensor<float> vreg_exp_max_even_fp32;
        RegTensor<float> vreg_exp_max_odd_fp32;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_exp_max_tmp;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<half, MaskPattern::ALL>();

        LoadAlign(vreg_in_max, gmUbAddr);    // 历史gm, 有效元素不超过64
        LoadAlign(vreg_input_max, lmUbAddr); // hm
        ExpSub<float, half, RegLayout::ZERO>(vreg_exp_max_even_fp32, vreg_in_max, vreg_input_max,
                                             preg_all); // dm = exp(gm - hm)
        ExpSub<float, half, RegLayout::ONE>(vreg_exp_max_odd_fp32, vreg_in_max, vreg_input_max, preg_all);
        Interleave(vreg_exp_max, vreg_exp_max_tmp, vreg_exp_max_even_fp32, vreg_exp_max_odd_fp32); // dm
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUbAddr, vreg_exp_max, preg_all_b32);
        StoreAlign<half, StoreDist::DIST_NORM_B16>(gmUbAddr, vreg_input_max, preg_all); // gm = hm

        // gl = dm * gl + ll
        LoadAlign(vreg_in_exp_sum, (__ubuf__ float *&)llUbAddr);
        LoadAlign(vreg_exp_sum_brc, (__ubuf__ float *&)glUbAddr);
        Mul(vreg_exp_sum_update, vreg_exp_max, vreg_exp_sum_brc, preg_all_b32);
        Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_in_exp_sum, preg_all_b32);
        StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float *&)glUbAddr, vreg_exp_sum_update, preg_all_b32);
    }

    __simd_vf__ inline void UpdateExpSumLastTileImpl(__ubuf__ float *dmUbAddr, __ubuf__ ElementInput *gmUbAddr,
                                                     __ubuf__ float *gmUbAddr32, __ubuf__ ElementInput *lmUbAddr,
                                                     __ubuf__ float *glUbAddr, __ubuf__ float *llUbAddr)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_sum_update;

        RegTensor<half> vreg_in_max;
        RegTensor<half> vreg_input_max;

        RegTensor<float> vreg_exp_max_even_fp32;
        RegTensor<float> vreg_exp_max_odd_fp32;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_exp_max_tmp;

        RegTensor<float> vreg_gmax_odd;
        RegTensor<float> vreg_gmax_even;
        RegTensor<float> vreg_gmax;
        RegTensor<float> vreg_gmax_tmp;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<half, MaskPattern::ALL>();

        LoadAlign(vreg_in_max, gmUbAddr);    // 历史gm, 有效元素不超过64
        LoadAlign(vreg_input_max, lmUbAddr); // hm
        ExpSub<float, half, RegLayout::ZERO>(vreg_exp_max_even_fp32, vreg_in_max, vreg_input_max,
                                             preg_all); // dm = exp(gm - hm)
        ExpSub<float, half, RegLayout::ONE>(vreg_exp_max_odd_fp32, vreg_in_max, vreg_input_max, preg_all);
        Interleave(vreg_exp_max, vreg_exp_max_tmp, vreg_exp_max_even_fp32, vreg_exp_max_odd_fp32); // dm
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUbAddr, vreg_exp_max, preg_all_b32);
        // 复原gm，给lse运算使用
        Cast<float, ElementInput, castTraitZero>(vreg_gmax_odd, vreg_input_max, preg_all);
        Cast<float, ElementInput, castTraitOne>(vreg_gmax_even, vreg_input_max, preg_all);
        Interleave(vreg_gmax, vreg_gmax_tmp, vreg_gmax_odd, vreg_gmax_even);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUbAddr32, vreg_gmax, preg_all_b32);

        // gl = dm * gl + ll
        LoadAlign(vreg_in_exp_sum, (__ubuf__ float *&)llUbAddr);
        LoadAlign(vreg_exp_sum_brc, (__ubuf__ float *&)glUbAddr);
        Mul(vreg_exp_sum_update, vreg_exp_max, vreg_exp_sum_brc, preg_all_b32);
        Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_in_exp_sum, preg_all_b32);
        StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float *&)glUbAddr, vreg_exp_sum_update, preg_all_b32);
    }

    __simd_vf__ inline void RestoreRowMaxLastAndFirstTileImpl(__ubuf__ ElementInput *gmUbAddr,
                                                              __ubuf__ float *gmUbAddr32)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<half> vreg_in_max;
        RegTensor<float> vreg_gmax;
        RegTensor<float> vreg_gmax_odd;
        RegTensor<float> vreg_gmax_even;
        RegTensor<float> vreg_gmax_tmp;

        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg_all_b32 = CreateMask<half, MaskPattern::ALL>();

        LoadAlign(vreg_in_max, gmUbAddr);
        Cast<float, ElementInput, castTraitZero>(vreg_gmax_odd, vreg_in_max, preg_all);
        Cast<float, ElementInput, castTraitOne>(vreg_gmax_even, vreg_in_max, preg_all);
        Interleave(vreg_gmax, vreg_gmax_tmp, vreg_gmax_odd, vreg_gmax_even);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUbAddr32, vreg_gmax, preg_all_b32);
    }
};

} // namespace NpuArch::Epilogue::Block
#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_S_FP16_ZN_P_FP8_HPP

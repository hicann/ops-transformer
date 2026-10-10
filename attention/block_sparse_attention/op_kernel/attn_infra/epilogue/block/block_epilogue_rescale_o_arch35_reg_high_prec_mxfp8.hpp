/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP8
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP8

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp8_vf/vf_updateScale_dn_mxfp8.h"
#include "mxfp8_vf/vf_attenOut_dn_mxfp8.h"
#include "mxfp8_vf/vf_common_def_mxfp8.h"

namespace NpuArch::Epilogue::Block {
using namespace MXFP8Kernel;
template <class ElementO_, class ElementOTmp_, class ElementDm_, class TileCopy_, class OTmpSrcPos_, LseMode LSE_MODE_,
          LseFormat LSE_FORMAT_>
class BlockEpilogue<EpilogueAtlasA5BsaRescaleOMxfp8<LSE_MODE_, LSE_FORMAT_, true>, ElementO_, ElementOTmp_, ElementDm_,
                    TileCopy_, OTmpSrcPos_> {
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleOMxfp8<LSE_MODE_, LSE_FORMAT_, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = float;
    using ElementDm = ElementDm_;
    using ElementSum = ElementOTmp;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;
    using LayoutO = typename TileCopy::LayoutO;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr float LN2_FP32 = 0.69314718056f;
    // snap 7.8046875 vs pscale emax=8：ℓ 含 2^Δ。切卡混用 LSE 时补 -Δ·ln2。
    static constexpr float P_SNAP_EMAX_FP32 = 7.8046875f;
    static constexpr float P_PSCALE_EMAX_FP32 = 8.0f;
    static constexpr float LSE_SNAP_PSCALE_CORR = (P_PSCALE_EMAX_FP32 - P_SNAP_EMAX_FP32) * LN2_FP32;
    static constexpr uint32_t NEG_INF_BITS = 0xFF800000u;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource)
    {
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP8::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP8::UB_LOCAL_ROW_SUM_BUF_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<ElementSum>(MXFP8::UB_GLOBAL_ROW_SUM_BUF_OFFSET);
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<ElementO>(MXFP8::UB_O_TRANS_BUF_OFFSET);
        oUBTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(MXFP8::UB_O_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP8::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(MXFP8::UB_UPDATE_SCALE_BUF_OFFSET +
                                                                               MXFP8::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorLseGm, class TensorLseUb>
    __aicore__ inline void CopyUbToGmLse(TensorLseGm const& gLseTensorTlaTile, TensorLseUb const& ubLseTensorTla)
    {
        AscendC::DataCopyExtParams repeatParams;
        if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                      (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
            repeatParams.blockCount = tla::get<0>(ubLseTensorTla.shape());
            repeatParams.blockLen = sizeof(float);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = (tla::get<0>(gLseTensorTlaTile.stride()) - 1) * sizeof(float);
        } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = tla::get<0>(ubLseTensorTla.shape()) * sizeof(float);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
        }
        auto dstOffset = gLseTensorTlaTile.layout()(gLseTensorTlaTile.coord());
        auto srcOffset = ubLseTensorTla.layout()(ubLseTensorTla.coord());
        AscendC::DataCopyPad(gLseTensorTlaTile.data()[dstOffset], ubLseTensorTla.data()[srcOffset], repeatParams);
    }

    // softmaxMax 是 128-half even/odd pack；SPLIT_CONSEC64 后 AIV0=qs[0:64)、AIV1=qs[64:128)。
    // lse = K_g·ln2 + ln(ℓ) - Δ·ln2，Δ=7.8046875-8；ℓ=0 → -inf。TN1/BSN1 再 Expand。
    template <uint16_t coreIndex>
    __simd_vf__ inline void ComputeMxfp8LseVF(__ubuf__ half* kgHalf, __ubuf__ float* rowsum, __ubuf__ float* lseUb,
                                              uint32_t row)
    {
        using namespace AscendC::Reg;
        using namespace AscendC::MicroAPI;
        MaskReg pregAll16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(row);
        MaskReg pregZero;
        static constexpr LnSpecificMode lnMode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};

        RegTensor<half> kg128;
        LoadAlign(kg128, kgHalf);

        RegTensor<float> kgLo;
        RegTensor<float> kgHi;
        Cast<float, half, Mxfp8VF::castTraitZero>(kgLo, kg128, pregAll16);
        Cast<float, half, Mxfp8VF::castTraitOne>(kgHi, kg128, pregAll16);
        Interleave(kgLo, kgHi, kgLo, kgHi);

        RegTensor<float> kgSel;
        if constexpr (coreIndex == 0) {
            Muls(kgSel, kgLo, LN2_FP32, pregTail);
        } else {
            Muls(kgSel, kgHi, LN2_FP32, pregTail);
        }

        RegTensor<float> ell;
        RegTensor<float> logEll;
        RegTensor<float> lse;
        RegTensor<float> zeros;
        RegTensor<float> negInf;
        LoadAlign(ell, rowsum);
        Duplicate(zeros, 0.0f);
        float negInfVal = *((float*)&NEG_INF_BITS);
        Duplicate(negInf, negInfVal);
        Ln<float, &lnMode>(logEll, ell, pregTail);
        Add(lse, logEll, kgSel, pregTail);
        Adds(lse, lse, LSE_SNAP_PSCALE_CORR, pregTail);
        Compare<float, AscendC::CMPMODE::EQ>(pregZero, ell, zeros, pregTail);
        Select(lse, negInf, lse, pregZero);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lseUb, lse, pregTail);
    }

    template <uint16_t coreIndex>
    __aicore__ inline void ComputeMxfp8Lse(const AscendC::LocalTensor<half>& kgSnap,
                                           const AscendC::LocalTensor<float>& rowsum,
                                           const AscendC::LocalTensor<float>& lseUb, uint32_t row)
    {
        __ubuf__ half* kgHalf = (__ubuf__ half*)kgSnap.GetPhyAddr();
        __ubuf__ float* rowsumPtr = (__ubuf__ float*)rowsum.GetPhyAddr();
        __ubuf__ float* lsePtr = (__ubuf__ float*)lseUb.GetPhyAddr();
        ComputeMxfp8LseVF<coreIndex>(kgHalf, rowsumPtr, lsePtr, row);
        if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                      (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
            AscendC::PipeBarrier<PIPE_V>();
            ExpandLseToUnalign(lsePtr, row);
        }
    }

    // 连续 64-float lse → 每行 32B 槽（与 regular Incontinuous 写出一致）
    __simd_vf__ inline void ExpandLseToUnalignVF(__ubuf__ float* lseUb, uint32_t row)
    {
        using namespace AscendC::Reg;
        UnalignReg ureg0;
        UnalignReg ureg1;
        static constexpr uint32_t postUpdateStride = 32 / sizeof(float);
        uint16_t rowEven = static_cast<uint16_t>((row + 1u) & ~1u);
        // 先拷到 scratch：lse 连续区在 [0, row)，broadcast 写到 [64, ...) 再不回头踩。
        // 这里 lseUb 调用方保证 [0,64) 是连续结果、[64,64+row*8) 可写。
        __ubuf__ float* dst = lseUb + 64;
        for (uint16_t i = 0; i < rowEven; i += 2) {
            RegTensor<float> v0;
            RegTensor<float> v1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v0, lseUb + i);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v1, lseUb + (i + 1));
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, v0, ureg0, postUpdateStride);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, v1, ureg1, postUpdateStride);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(dst, ureg0, postUpdateStride);
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(dst, ureg1, postUpdateStride);
    }

    __aicore__ inline void ExpandLseToUnalign(__ubuf__ float* lsePtr, uint32_t row)
    {
        ExpandLseToUnalignVF(lsePtr, row);
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline void operator()(TensorO& gOTensor, TensorLse& gLseTensor, AscendC::LocalTensor<half>& kgSnap,
                                      GemmCoord& actualOriShape, TaskInfo& taskInfo, TileInfo& tileInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_ATTN_BUF_FLAG);
            Mxfp8VF::processUpdate<false>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        } else {
            Mxfp8VF::processUpdate<true>(oUBTensor, oTmpUBTensor, dm, globalRowSumUBTensor, localRowSumUBTensor);
        }

        if (tileInfo.isLastKvsTile) {
            const uint32_t actQsTile = actualOriShape.n();
            const uint32_t splitMSizePerCore = (MXFP8::QS_BASE_SIZE + 1) / 2;
            uint32_t actMSizeThisCore = actQsTile < splitMSizePerCore ? actQsTile : splitMSizePerCore;
            if (subBlockIdx != VEC0) {
                actMSizeThisCore = actQsTile < splitMSizePerCore ? 0 : (actQsTile - splitMSizePerCore);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (actMSizeThisCore != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
                AscendC::LocalTensor<ElementOTmp> outTensorFp32Ub =
                    oTransUBTensor.template ReinterpretCast<ElementOTmp>(); // 空间复用P, bf16 vf写出是强解释fp32
                Mxfp8VF::processOut<MXFP8::QS_BASE_SIZE, ElementO>(outTensorFp32Ub, oUBTensor, globalRowSumUBTensor);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<ElementO> oTransUBTensorB16 = oTransUBTensor.template ReinterpretCast<ElementO>();
                AscendC::LocalTensor<ElementO> oUBTensorB16 = oUBTensor.template ReinterpretCast<ElementO>();
                // embding size actualOriShape.m()
                uint32_t embedColumnCnt = EMB_ALIGN128 + MXFP8::DATA_BLOCK_BYTE / sizeof(ElementO);
                // 搬回 oUBTensorB16
                AscendC::DataCopy(oUBTensorB16, oTransUBTensorB16, actMSizeThisCore * embedColumnCnt);
                AscendC::PipeBarrier<PIPE_V>();

                // 无 LSE 时 oTrans 已拷完，可以还给 P。有 LSE 时 oTrans 还要做 LSE 的 UB→GM，信号放到搬出之后。
                if constexpr (DispatchPolicy::LSE_MODE != LseMode::OUT_ONLY) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_ATTN_BUF_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_ATTN_BUF_FLAG);

                // copyUBOToGm
                uint32_t embdingSize = actualOriShape.m();
                uint32_t rowOffsetCurSubCore = subBlockIdx * splitMSizePerCore;
                uint32_t colNumCurSubCore = tla::get<1>(gOTensor.shape());
                auto gOTensorTlaTile = GetTile(gOTensor, tla::MakeCoord(rowOffsetCurSubCore, 0),
                                               tla::MakeShape(actMSizeThisCore, colNumCurSubCore));
                auto ubOLayoutTla = tla::MakeLayout(tla::MakeShape(actMSizeThisCore, embdingSize),
                                                    tla::MakeStride(embedColumnCnt, tla::Int<1>{}));
                auto ubOTensorTla = tla::MakeTensor(oUBTensorB16, ubOLayoutTla, Arch::PositionUB{});
                copyUbToGmO(gOTensorTlaTile, ubOTensorTla);

                if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
                    // oTrans 已搬到 oUB，复用为 lse workspace：连续结果 [0,64)，TN1/BSN1 broadcast 从 +64
                    AscendC::LocalTensor<float> lseTmp = oTransUBTensor.template ReinterpretCast<float>();
                    if (subBlockIdx == VEC0) {
                        ComputeMxfp8Lse<0>(kgSnap, globalRowSumUBTensor, lseTmp, actMSizeThisCore);
                    } else {
                        ComputeMxfp8Lse<1>(kgSnap, globalRowSumUBTensor, lseTmp, actMSizeThisCore);
                    }
                    uint32_t colNumLseUb = 1;
                    uint32_t colStrideLseUb = 1;
                    uint32_t lseUbOffset = 0;
                    if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                                  (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                        colNumLseUb = 8;
                        colStrideLseUb = 8;
                        lseUbOffset = 64;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_ATTN_BUF_FLAG);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_ATTN_BUF_FLAG);
                    auto ubLseLayoutTla = tla::MakeLayout(tla::MakeShape(actMSizeThisCore, colNumLseUb),
                                                          tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
                    auto ubLseTensorTla = tla::MakeTensor(lseTmp[lseUbOffset], ubLseLayoutTla, Arch::PositionUB{});
                    auto gLseTensorTlaTile = GetTile(gLseTensor, tla::MakeCoord(rowOffsetCurSubCore, 0),
                                                     tla::MakeShape(actMSizeThisCore, 1));
                    CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_ATTN_BUF_FLAG);
        }
    }

    // NONE 路径：不碰 LSE GM / K_g snap，保持原调用。
    template <class TensorO>
    __aicore__ inline void operator()(TensorO& gOTensor, GemmCoord& actualOriShape, TaskInfo& taskInfo,
                                      TileInfo& tileInfo)
    {
        auto dummySnap = oTransUBTensor.template ReinterpretCast<half>();
        operator()(gOTensor, gOTensor, dummySnap, actualOriShape, taskInfo, tileInfo);
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementOTmp> oTmpUBTensor;                         // mm2Res(PV)
    AscendC::LocalTensor<ElementSum> localRowSumUBTensor;                   // LocalRowSum
    AscendC::LocalTensor<ElementSum> globalRowSumUBTensor;                  // GlobalRowSum
    AscendC::LocalTensor<ElementO> oTransUBTensor;                          // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementOTmp> oUBTensor;                            // attentionOut
    AscendC::LocalTensor<ElementDm> dmUBTensor[MXFP8::UB_UPDATE_SCALE_CNT]; // updateScale
    CopyUbToGmO copyUbToGmO;
    static constexpr uint32_t VEC0 = 0;
    static constexpr uint32_t EMB_ALIGN128 = 128;
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP8

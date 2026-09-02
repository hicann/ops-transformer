/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tla_epilogue_dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/tla_detail_callback.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag_, class CType_, class ScaleType_, class PerTokenScaleType_, class BiasType_, class DType_,
          class TileRowBroadcastMul_, class TileRowBroadcastAdd_, class TileBroadcastOneBlk_,
          class TileOneBlkColumnBroadcastMul_, class TileCopy_, class EpilogueTileSwizzle_>
class BlockEpilogue<ArchTag_, CType_, ScaleType_, PerTokenScaleType_, BiasType_, DType_, TileRowBroadcastMul_,
                    TileRowBroadcastAdd_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_, TileCopy_,
                    EpilogueTileSwizzle_> {
public:
    static constexpr uint32_t UB_STAGES = 2;

    // Data infos
    using ArchTag = ArchTag_;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementScale = typename ScaleType_::Element;
    using LayoutScale = typename ScaleType_::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType_::Element;
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;
    using ElementBias = typename BiasType_::Element;
    using LayoutBias = typename BiasType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(std::is_same_v<ElementC, int32_t> &&
                      (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>) &&
                      std::is_same_v<ElementScale, float> && std::is_same_v<ElementPerTokenScale, float>,
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::RowMajor>,
                  "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileRowBroadcastAdd = TileRowBroadcastAdd_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;
    using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasType_>;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyGmToUbD = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<ElementD, layout::RowMajor>>;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static_assert(TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
                      std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
                  "TileShape must be consistent for all tile compute ops");

    static_assert((UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale) +
                                TileShape::ROW * sizeof(ElementPerTokenScale) +
                                TileShape::COLUMN * sizeof(ElementBias) + TileShape::COUNT * sizeof(ElementD)) +
                   (TileShape::COUNT + TileShape::COUNT) * sizeof(float) + TileShape::ROW * BYTE_PER_BLK) <=
                      ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubBiasList[i] = resource.ubBuf.template GetBufferByByte<ElementBias>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementBias);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbBiasVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbBiasMTE2VList[i] = eventMTE2V++;
            eventUbScaleMTE2VList[i] = eventMTE2V++;
            eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
        }
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                      AscendC::IsSameType<ElementBias, half>::value) {
            ubBiasFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(float);
        }
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubMul;
        ubBiasAdd = ubMul;
    }

    CATLASS_DEVICE
    void WaitFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void InitFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    // perChannel、perToken
    CATLASS_DEVICE
    void operator()(__gm__ ElementScale *ptrScale, LayoutScale layoutScale,
                    __gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
                    __gm__ ElementBias *ptrBias, LayoutBias layoutBias, __gm__ ElementC *ptrC, LayoutC layoutC,
                    __gm__ ElementD *ptrD, LayoutD layoutD, GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        gmScale.SetGlobalBuffer(ptrScale);
        gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
        gmC.SetGlobalBuffer(ptrC);
        gmD.SetGlobalBuffer(ptrD);
        if (ptrBias != nullptr) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)ptrBias);
        }

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle swizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = swizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = swizzle.GetTileCoord(loopIdx);
            auto actualTileShape = swizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            // C: GM -> UB
            auto &ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmC[layoutC.GetOffset(tileOffset)], layoutUbC, layoutC.GetTileLayout(actualTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // perChannel scale: GM -> UB
            auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();
            auto &ubScale = ubScaleList[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);
            copyGmToUbScale(ubScale, gmScale[layoutScale.GetOffset(tileOffset.template GetCoordByAxis<1>())],
                            layoutUbScale, layoutScale.GetTileLayout(scaleTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            // perToken scale: GM -> UB
            auto perTokenTileShape = actualTileShape.template GetCoordByAxis<0>();
            auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale = LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(perTokenTileShape);

            copyGmToUbPerTokenScale(
                ubPerTokenScale,
                gmPerTokenScale[layoutPerTokenScale.GetOffset(tileOffset.template GetCoordByAxis<0>())],
                layoutUbPerTokenScale, layoutPerTokenScale.GetTileLayout(perTokenTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // bias: GM -> UB (optional)
            CopyBiasToUb(tileOffset, actualTileShape, layoutBias, ptrBias);

            // C(int32) -> FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubCFp32, ubScale);
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubMul, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            AddBias(ubPerTokenMul, ptrBias);
            // 选择要cast的源tensor：如果有bias，使用ubBiasAdd；否则直接使用ubPerTokenMul
            AscendC::LocalTensor<float> &castSrc = ptrBias != nullptr ? ubBiasAdd : ubPerTokenMul;

            // FP32 -> D -> GM
            CastToDAndStore(castSrc, tileOffset, actualTileShape, layoutD, gmD);
            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

    // perChannel
    CATLASS_DEVICE
    void operator()(__gm__ ElementScale *ptrScale, LayoutScale layoutScale, __gm__ ElementBias *ptrBias,
                    LayoutBias layoutBias, __gm__ ElementC *ptrC, LayoutC layoutC, __gm__ ElementD *ptrD,
                    LayoutD layoutD, GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        gmScale.SetGlobalBuffer(ptrScale);
        gmC.SetGlobalBuffer(ptrC);
        gmD.SetGlobalBuffer(ptrD);
        if (ptrBias != nullptr) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)ptrBias);
        }

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle swizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = swizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = swizzle.GetTileCoord(loopIdx);
            auto actualTileShape = swizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            // C: GM -> UB
            auto &ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmC[layoutC.GetOffset(tileOffset)], layoutUbC, layoutC.GetTileLayout(actualTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // perChannel scale: GM -> UB
            auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();
            auto &ubScale = ubScaleList[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);
            copyGmToUbScale(ubScale, gmScale[layoutScale.GetOffset(tileOffset.template GetCoordByAxis<1>())],
                            layoutUbScale, layoutScale.GetTileLayout(scaleTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            // bias: GM -> UB (optional)
            CopyBiasToUb(tileOffset, actualTileShape, layoutBias, ptrBias);

            // C(int32) -> FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubCFp32, ubScale);
            AscendC::PipeBarrier<PIPE_V>();

            AddBias(ubMul, ptrBias);
            // 选择要cast的源tensor：如果有bias，使用ubBiasAdd；否则直接使用ubMul
            AscendC::LocalTensor<float> &castSrc = ptrBias != nullptr ? ubBiasAdd : ubMul;

            // FP32 -> D -> GM
            CastToDAndStore(castSrc, tileOffset, actualTileShape, layoutD, gmD);
            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

    // perToken
    CATLASS_DEVICE
    void operator()(__gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
                    __gm__ ElementBias *ptrBias, LayoutBias layoutBias, __gm__ ElementD *ptrIn, LayoutD layoutIn,
                    __gm__ ElementD *ptrOut, LayoutD layoutOut, GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        AscendC::GlobalTensor<ElementD> gmIn;
        AscendC::GlobalTensor<ElementD> gmOut;
        gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
        gmIn.SetGlobalBuffer(ptrIn);
        gmOut.SetGlobalBuffer(ptrOut);
        if (ptrBias != nullptr) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)ptrBias);
        }

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle swizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = swizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = swizzle.GetTileCoord(loopIdx);
            auto actualTileShape = swizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            // D(input): GM -> UB, reuse ubDList and C's event slots
            auto &ubIn = ubDList[ubListId];
            LayoutD layoutUbIn{actualTileShape, ubTileStride};
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbD(ubIn, gmIn[layoutIn.GetOffset(tileOffset)], layoutUbIn,
                        layoutIn.GetTileLayout(actualTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // perToken scale: GM -> UB
            auto perTokenTileShape = actualTileShape.template GetCoordByAxis<0>();
            auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale = LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(perTokenTileShape);
            copyGmToUbPerTokenScale(
                ubPerTokenScale,
                gmPerTokenScale[layoutPerTokenScale.GetOffset(tileOffset.template GetCoordByAxis<0>())],
                layoutUbPerTokenScale, layoutPerTokenScale.GetTileLayout(perTokenTileShape));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // bias: GM -> UB (optional)
            CopyBiasToUb(tileOffset, actualTileShape, layoutBias, ptrBias);

            // D(fp16/bf16) -> FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubIn, AscendC::RoundMode::CAST_NONE, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubCFp32, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            AddBias(ubPerTokenMul, ptrBias);
            // 选择要cast的源tensor：如果有bias，使用ubBiasAdd；否则直接使用ubPerTokenMul
            AscendC::LocalTensor<float> &castSrc = ptrBias != nullptr ? ubBiasAdd : ubPerTokenMul;

            // FP32 -> D -> GM (writes back to ubIn = ubDList[ubListId])
            CastToDAndStore(castSrc, tileOffset, actualTileShape, layoutOut, gmOut);
            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

private:
    // bias 从 GM 拷贝到 UB；half/bfloat16 时同时转成 FP32。ptrBias 为 nullptr 时直接返回。
    CATLASS_DEVICE
    void CopyBiasToUb(MatrixCoord tileOffset, MatrixCoord actualTileShape, LayoutBias layoutBias,
                      __gm__ ElementBias *ptrBias)
    {
        if (ptrBias == nullptr) {
            return;
        }
        auto biasTileOffset = tileOffset.template GetCoordByAxis<1>();
        auto biasTileShape = actualTileShape.template GetCoordByAxis<1>();
        auto &ubBias = ubBiasList[ubListId];
        auto layoutUbBias = LayoutBias::template MakeLayoutInUb<ElementBias>(biasTileShape);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
        copyGmToUbBias(ubBias, gmBias[layoutBias.GetOffset(biasTileOffset)], layoutUbBias,
                       layoutBias.GetTileLayout(biasTileShape));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);

        if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                      AscendC::IsSameType<ElementBias, half>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
            AscendC::Cast(ubBiasFp32, ubBias, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
        }
    }

    // bias 加法：ubBiasAdd = src + bias。ptrBias 为 nullptr 时直接返回。
    // half/bfloat16 使用预转好的 ubBiasFp32；其它类型从 ubBias 读取（含事件同步）。
    // 末尾保证 PipeBarrier<PIPE_V>。
    CATLASS_DEVICE
    void AddBias(AscendC::LocalTensor<float> &src, __gm__ ElementBias *ptrBias)
    {
        if (ptrBias == nullptr) {
            return;
        }
        if constexpr (AscendC::IsSameType<ElementBias, bfloat16_t>::value ||
                      AscendC::IsSameType<ElementBias, half>::value) {
            tileRowBroadcastAdd(ubBiasAdd, src, ubBiasFp32);
        } else {
            auto &ubBias = ubBiasList[ubListId];
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBiasMTE2VList[ubListId]);
            tileRowBroadcastAdd(ubBiasAdd, src, ubBias);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBiasVMTE2List[ubListId]);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 将 FP32 结果 cast 到 D 并拷贝回 GM。op1/op2 传 gmD，perToken 场景传 gmOut。
    CATLASS_DEVICE
    void CastToDAndStore(AscendC::LocalTensor<float> &castSrc, MatrixCoord tileOffset, MatrixCoord actualTileShape,
                         LayoutD layoutD, AscendC::GlobalTensor<ElementD> &gmDst)
    {
        auto &ubD = ubDList[ubListId];
        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        LayoutD layoutUbD{actualTileShape, ubTileStride};

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
        AscendC::Cast(ubD, castSrc, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
        copyUbToGmD(gmDst[layoutD.GetOffset(tileOffset)], ubD, layoutD.GetTileLayout(actualTileShape), layoutUbD);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
    }

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementBias> ubBiasList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbScaleMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];
    int32_t eventUbBiasVMTE2List[UB_STAGES];
    int32_t eventUbBiasMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubBiasFp32;
    AscendC::LocalTensor<float> ubMul;
    AscendC::LocalTensor<float> ubPerTokenScaleBrcb;
    AscendC::LocalTensor<float> ubPerTokenMul;
    AscendC::LocalTensor<float> ubBiasAdd;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileRowBroadcastAdd tileRowBroadcastAdd;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyGmToUbBias copyGmToUbBias;
    CopyUbToGmD copyUbToGmD;
    CopyGmToUbD copyGmToUbD;

    AscendC::GlobalTensor<ElementScale> gmScale;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementD> gmD;
    AscendC::GlobalTensor<ElementBias> gmBias;
};
} // namespace Catlass::Epilogue::Block

#endif // MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H

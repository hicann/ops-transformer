/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_PV_ARCH35_MXFP8_HPP
#define GEMM_BLOCK_PV_ARCH35_MXFP8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "block_mmad_arch35_utils.hpp"

namespace NpuArch::Gemm::Block {
template <bool transposedMm1_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_,
          class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaPVMxfp8<transposedMm1_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
                    ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaPVMxfp8<transposedMm1_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    template <class TensorA>
    using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
    template <class TensorB>
    using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr bool IS_MXFP8 = AscendC::IsSameType<ElementA, fp8_e4m3fn_t>::value;
    static_assert(IS_MXFP8, "BlockMmadTla<MmadAtlasA5BsaPVMxfp8> requires ElementA = fp8_e4m3fn_t");

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});

    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;

    // Event ID：QK 用 2-3，PV 用 4-7；同 ID 不同 HardEvent 类型不冲突
    static constexpr uint32_t KV_EVENT0 = 4;
    static constexpr uint32_t KV_EVENT1 = 5;
    // PV L0A/B 双缓冲，与 KV_EVENT 复用（不同 HardEvent 类型，flag 独立）
    static constexpr uint32_t PV_L0AB_EVENT0 = KV_EVENT0;
    static constexpr uint32_t PV_L0AB_EVENT1 = KV_EVENT1;
    // PV L0C 单缓冲
    static constexpr uint32_t PV_L0C_EVENT0 = KV_EVENT0;

    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t& kvBufId)
        : l1ABufId(kvBufId)
    {
        for (uint32_t i = 0; i < MXFP8::L1_P_BUF_CNT; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(MXFP8::L1_P_BUF_OFFSET + MXFP8::L1_P_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP8::L1_P_SCALE_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_P_SCALE_BUF_OFFSET +
                                                                                 MXFP8::L1_P_SCALE_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP8::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(MXFP8::L1_KV_BUF_OFFSET + MXFP8::L1_KV_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP8::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_KV_DESCALE_BUF_OFFSET +
                                                                                 MXFP8::L1_KV_DESCALE_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP8::L0A_PV_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(MXFP8::L0A_PV_BUF_OFFSET +
                                                                              MXFP8::L0A_PV_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP8::L0B_PV_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(MXFP8::L0B_PV_BUF_OFFSET +
                                                                              MXFP8::L0B_PV_BUF_SIZE * i);
        }

        l0CTensor[0] = resource.l0CBuf.template GetBufferByByte<float>(MXFP8::L0C_PV_BUF_OFFSET);

        l1RowsumSeedTensor = resource.l1Buf.template GetBufferByByte<ElementA>(MXFP8::L1_ROW_SUM_SEED_OFFSET);
        l1RowsumSeedScaleTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_ROW_SUM_SEED_SCALE_OFFSET);

        AllocEventID();
        InitL0BufferForReduceSum();
    }
    __aicore__ inline ~BlockMmadTla() {}

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PV_L0C_EVENT0);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PV_L0C_EVENT0);
    }

    // V: nZ L1 + V-scale → zN L0A（转置）；s2Start/s2Act 沿 S2 切段
    __aicore__ inline void LoadVToL0(uint32_t l1ABufId, uint32_t l0ABufId, uint32_t embedReal, uint32_t s2Act,
                                     uint32_t s2Start, uint32_t s2FullAlign16, uint32_t scaleBufId,
                                     uint32_t scaleS2Start, uint32_t scaleS2FullAlign16)
    {
        copyL1ToL0AMx(l0ATensor[l0ABufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                      l1ATensor[l1ABufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                      l1AScaleTensor[scaleBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), embedReal, s2Act,
                      s2Start, s2FullAlign16, scaleS2Start, scaleS2FullAlign16);
    }

    // P: zN L1 + P-scale → nZ L0B（转置）
    __aicore__ inline void LoadPToL0(uint32_t l1BBufId, uint32_t l0BBufId, uint32_t s2Act, uint32_t embedReal,
                                     uint32_t s1Align64, uint32_t s2Start)
    {
        constexpr uint32_t s2Base = MXFP8::S2_BASE_TILE_SIZE;
        constexpr uint32_t scaleSrcStride = s2Base / MXFP8::CONST_64 + 1;
        copyL1ToL0BMx(l0BTensor[l0BBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                      l1BTensor[l1BBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                      l1BScaleTensor[l1BBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), s2Act, embedReal,
                      s2Base, scaleSrcStride, s1Align64, s2Start);
    }

    // C = Vᵀ @ P = Oᵀ，initC 由 isTileGroupFirstTile 控制
    __aicore__ inline void MatmulPV(uint32_t l0ABufId, uint32_t l0BBufId, uint32_t m, uint32_t n, uint32_t k,
                                    bool initC)
    {
        auto l0ALayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0A>(m, k);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0B>(k, n);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(m, n);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, m, n, k, initC);
    }

    // V 数据 + V-scale 融合稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1。
    // 两者的分段边界必然一致，故只遍历一遍 index，每段发 V / V-scale 两条 copy。
    // ori 连续递增即代表 GM 上物理相邻，向前偷看合并成单条 Nd2Nz。
    // 段推进天然落在 y-block 边界上（仅首段 innerStart 可能非 0），故循环内无除法/取模。
    // firstYBlockIdx / firstOriYBlockIdx 由调用方提前预取，避免首个 index 的阻塞访存落在此处。
    template <class TensorV, class TensorL1A, class TensorVScale, class TensorL1AScale>
    __aicore__ inline void SparseVFusedBaseTileL1FullLoad(
        TensorV& gVTensor, TensorL1A& l1ATensorTla, TensorVScale& gVScaleTensor, TensorL1AScale& l1AScaleTensorTla,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t scaleRows, uint32_t embed, uint32_t firstYBlockIdx,
        uint32_t firstOriYBlockIdx)
    {
        using CopyGmToL1A = Tile::TileCopyTla<ArchTag, TensorV, TensorL1A>;
        using CopyGmToL1AScale = Tile::TileCopyTla<ArchTag, TensorVScale, TensorL1AScale>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1AScale copyGmToL1AScale;

        const uint32_t kvSeqlenU = static_cast<uint32_t>(kvSeqlen);
        uint32_t gatheredYBlockIdx = firstYBlockIdx;
        uint32_t yBlockInnerStart = gatheredKvSTileIdx * kvSBaseTile - firstYBlockIdx * blockShapeY;
        uint32_t oriYBlockIdx = firstOriYBlockIdx;
        uint32_t dealtLenAccum = 0;
        uint32_t walkRows = (scaleRows > curBaseTileSize) ? scaleRows : curBaseTileSize;

        while (dealtLenAccum < walkRows && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval) {
            uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            if (oriStartOffset >= kvSeqlenU) {
                break;
            }

            uint32_t curYBlockSize =
                (oriYBlockIdx == yBlockNumAval - 1) ? (kvSeqlenU - oriYBlockIdx * blockShapeY) : blockShapeY;
            if (curYBlockSize <= yBlockInnerStart) {
                break;
            }
            uint32_t availLen = curYBlockSize - yBlockInnerStart;
            // 尾部不满块之后 gather 流即终止（与原实现的 curDealtLen==0 退出等价）
            bool runEndsShort = (curYBlockSize < blockShapeY);

            // 向前偷看：ori 连续则 GM 地址连续，拼进同一条 Nd2Nz
            uint32_t peekYBlockIdx = gatheredYBlockIdx + 1;
            uint32_t peekOriIdx = oriYBlockIdx + 1;
            while (!runEndsShort && dealtLenAccum + availLen < walkRows && peekYBlockIdx < yBlockNumRsvd &&
                   peekOriIdx < yBlockNumAval &&
                   static_cast<uint32_t>(gSparseBlockIdx.GetValue(peekYBlockIdx)) == peekOriIdx) {
                uint32_t peekBlockSize =
                    (peekOriIdx == yBlockNumAval - 1) ? (kvSeqlenU - peekOriIdx * blockShapeY) : blockShapeY;
                availLen += peekBlockSize;
                runEndsShort = (peekBlockSize < blockShapeY);
                ++peekYBlockIdx;
                ++peekOriIdx;
            }

            uint32_t curDealtLen = min(availLen, walkRows - dealtLenAccum);
            uint32_t dataLen = 0;
            if (dealtLenAccum < curBaseTileSize) {
                dataLen = min(curDealtLen, curBaseTileSize - dealtLenAccum);
            }
            uint32_t scaleLen = 0;
            if (dealtLenAccum < scaleRows) {
                scaleLen = min(curDealtLen, scaleRows - dealtLenAccum);
            }

            uint32_t pairLen = 0;
            uint32_t pairOriStart = 0;
            bool pairEndsShort = false;
            uint32_t pairPeekEnd = peekYBlockIdx;
            if (!runEndsShort && dataLen > 0 && dataLen == curDealtLen && dataLen == scaleLen &&
                dealtLenAccum + curDealtLen < walkRows && peekYBlockIdx < yBlockNumRsvd) {
                uint32_t nextOri = static_cast<uint32_t>(gSparseBlockIdx.GetValue(peekYBlockIdx));
                if (nextOri < yBlockNumAval) {
                    uint32_t nextStart = nextOri * blockShapeY;
                    uint32_t nextSize =
                        (nextOri == yBlockNumAval - 1) ? (kvSeqlenU - nextOri * blockShapeY) : blockShapeY;
                    uint32_t remain = walkRows - dealtLenAccum - curDealtLen;
                    uint32_t nextDealt = min(nextSize, remain);
                    if (nextDealt == dataLen && nextStart > oriStartOffset) {
                        pairLen = nextDealt;
                        pairOriStart = nextStart;
                        pairEndsShort = (nextSize < blockShapeY);
                        pairPeekEnd = peekYBlockIdx + 1;
                    }
                }
            }
            uint32_t srcNdStride = 0;
            uint32_t dstNzStride = 0;
            bool fusedNd2 = false;
            if (pairLen > 0) {
                auto gV0 = GetTile(gVTensor, tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, dataLen));
                auto l1V0 = GetTile(l1ATensorTla, tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, dataLen));
                auto gV1 = GetTile(gVTensor, tla::MakeCoord(0, pairOriStart), tla::MakeShape(embed, dataLen));
                auto l1V1 =
                    GetTile(l1ATensorTla, tla::MakeCoord(0, dealtLenAccum + dataLen), tla::MakeShape(embed, dataLen));
                fusedNd2 = MXFP8::Mxfp8U32Delta(static_cast<int64_t>(gV1.layout()(gV1.coord())) -
                                                    static_cast<int64_t>(gV0.layout()(gV0.coord())),
                                                srcNdStride) &&
                           MXFP8::Mxfp8U32Delta(static_cast<int64_t>(l1V1.layout()(l1V1.coord())) -
                                                    static_cast<int64_t>(l1V0.layout()(l1V0.coord())),
                                                dstNzStride);
                if (fusedNd2) {
                    copyGmToL1A(l1V0, gV0, 2, srcNdStride, dstNzStride);
                }
            }
            if (!fusedNd2 && dataLen > 0) {
                auto gVTensorTile =
                    GetTile(gVTensor, tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, dataLen));
                auto l1ATensorTile =
                    GetTile(l1ATensorTla, tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, dataLen));
                copyGmToL1A(l1ATensorTile, gVTensorTile);
            }
            if (scaleLen > 0) {
                uint32_t curScaleLen = CeilDiv(scaleLen, MX_SCALE_GROUP_NUM);
                auto gVScaleTile = GetTile(gVScaleTensor, tla::MakeCoord(0, oriStartOffset / MX_SCALE_GROUP_NUM),
                                           tla::MakeShape(embed, curScaleLen));
                auto l1AScaleTile = GetTile(l1AScaleTensorTla, tla::MakeCoord(0, dealtLenAccum / MX_SCALE_GROUP_NUM),
                                            tla::MakeShape(embed, curScaleLen));
                copyGmToL1AScale(l1AScaleTile, gVScaleTile);
                if (fusedNd2) {
                    uint32_t pairScaleLen = CeilDiv(pairLen, MX_SCALE_GROUP_NUM);
                    auto gS1 = GetTile(gVScaleTensor, tla::MakeCoord(0, pairOriStart / MX_SCALE_GROUP_NUM),
                                       tla::MakeShape(embed, pairScaleLen));
                    auto l1S1 =
                        GetTile(l1AScaleTensorTla, tla::MakeCoord(0, (dealtLenAccum + dataLen) / MX_SCALE_GROUP_NUM),
                                tla::MakeShape(embed, pairScaleLen));
                    copyGmToL1AScale(l1S1, gS1);
                }
            }

            dealtLenAccum += curDealtLen + (fusedNd2 ? pairLen : 0);
            if (runEndsShort || (fusedNd2 && pairEndsShort)) {
                break;
            }
            gatheredYBlockIdx = fusedNd2 ? pairPeekEnd : peekYBlockIdx;
            yBlockInnerStart = 0;
            if (dealtLenAccum < walkRows) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
            }
        }
    }

    // L0C → UB fixpipe（SPLIT_N 模式）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm2(uint32_t s1Align64, TensorUB& ubOTmpTensor)
    {
        auto l0CTensorTla =
            tla::MakeTensor(l0CTensor[0], tla::MakeLayoutL0C(MXFP8::PV_MMAD_M_DIM, s1Align64), Arch::PositionL0C{});
        using CopyL0CToDst =
            Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB, Tile::CopyL0CToUBMode::SPLIT_N,
                                 Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，s1 ≤ 64 时单 vector 核搬出）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm2SingleVect(uint32_t s1Align64, TensorUB& ubOTmpTensor)
    {
        auto l0CTensorTla =
            tla::MakeTensor(l0CTensor[0], tla::MakeLayoutL0C(MXFP8::PV_MMAD_M_DIM, s1Align64), Arch::PositionL0C{});
        using CopyL0CToDst =
            Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB, Tile::CopyL0CToUBMode::NO_SPLIT,
                                 Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);

        // seed fill：V data = e4m3 1.0 (0x38), V scale = e8m0 1.0 (0x7F) → 有效值 1.0
        constexpr uint16_t vFillBlocks = static_cast<uint16_t>(MXFP8::L0A_PV_BUF_SIZE / MXFP8::BLOCK_SIZE);
        AscendC::InitConstValueParams<uint16_t> vFillParams(1, vFillBlocks, 0, MXFP8::SEED_V_DATA_FILL);
        AscendC::Fill(l1ATensor[l1ABufId].template ReinterpretCast<uint16_t>(), vFillParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        constexpr uint16_t vScaleFillBlocks = static_cast<uint16_t>(MXFP8::V_SCALE_L0A_SIZE / MXFP8::BLOCK_SIZE);
        AscendC::InitConstValueParams<uint16_t> vscaleFillParams(1, vScaleFillBlocks, 0, MXFP8::SEED_V_SCALE_FILL);
        AscendC::Fill(l1AScaleTensor[l1ABufId].template ReinterpretCast<uint16_t>(), vscaleFillParams);

        constexpr uint16_t seedPadBlocks = static_cast<uint16_t>(MXFP8::L1_ROW_SUM_SEED_SIZE / MXFP8::BLOCK_SIZE);
        AscendC::InitConstValueParams<uint16_t> seedPadParams(1, seedPadBlocks, 0, MXFP8::SEED_V_DATA_FILL);
        AscendC::Fill(l1RowsumSeedTensor.template ReinterpretCast<uint16_t>(), seedPadParams);
        constexpr uint16_t seedPadScaleBlocks =
            static_cast<uint16_t>(MXFP8::L1_ROW_SUM_SEED_SCALE_SIZE / MXFP8::BLOCK_SIZE);
        AscendC::InitConstValueParams<uint16_t> seedPadScaleParams(1, seedPadScaleBlocks, 0, MXFP8::SEED_V_SCALE_FILL);
        AscendC::Fill(l1RowsumSeedScaleTensor.template ReinterpretCast<uint16_t>(), seedPadScaleParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);

        // seed 装载：把 V 的 D+16 行（含 rowsum pad）非转置装入 L0A，循环 seed 两个 ping-pong 槽
        constexpr uint32_t mStepVal = MXFP8::PV_MMAD_M_DIM / MXFP8::NZ_C0_ELEMS;
        constexpr uint32_t kStepVal = MXFP8::S2_BASE_TILE_SIZE / MXFP8::FP8_C0_ELEMS;

        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = mStepVal;
        loadData2DParamsA.kStep = kStepVal;
        loadData2DParamsA.srcStride = mStepVal;
        loadData2DParamsA.dstStride = mStepVal;
        loadData2DParamsA.ifTranspose = false;

        AscendC::LoadData2DMxParams loadData2DMXParamsA;
        loadData2DMXParamsA.xStartPosition = 0;
        loadData2DMXParamsA.yStartPosition = 0;
        loadData2DMXParamsA.xStep = mStepVal;
        loadData2DMXParamsA.yStep = kStepVal;
        loadData2DMXParamsA.srcStride = kStepVal;
        loadData2DMXParamsA.dstStride = kStepVal;

        for (uint32_t i = 0; i < MXFP8::L0A_PV_BUF_CNT; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            LoadData(l0ATensor[pvL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                     l1ATensor[l1ABufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                     l1AScaleTensor[l1ABufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), loadData2DParamsA,
                     loadData2DMXParamsA);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            pvL0abBufId = (pvL0abBufId + 1) % MXFP8::L0A_PV_BUF_CNT;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        l1ABufId = (l1ABufId + 1) % MXFP8::L1_KV_BUF_CNT;
    }

    __aicore__ inline void PadPS2NotAlign64(uint32_t s2Align64, uint32_t s2Align32, uint32_t pSlot)
    {
        // L1 P 是 128 行 NZ、qs 方向 4 个 FP8 C0。pad 的是每面 kvs[align32, align64)。
        AscendC::InitConstValueParams<uint16_t> PL1InitParams(1, static_cast<uint16_t>(s2Align64 - s2Align32), 0,
                                                              MXFP8::ZERO_FILL_PATTERN);
        constexpr uint32_t planeElems = MXFP8::S2_BASE_TILE_SIZE * MXFP8::FP8_C0_ELEMS;
        constexpr uint32_t planeCnt = MXFP8::CONST_128 / MXFP8::FP8_C0_ELEMS;
        uint32_t validElems = s2Align32 * MXFP8::FP8_C0_ELEMS;
        for (uint32_t p = 0; p < planeCnt; ++p) {
            AscendC::Fill(l1BTensor[pSlot][p * planeElems + validElems].template ReinterpretCast<uint16_t>(),
                          PL1InitParams);
        }
    }

    __aicore__ inline void PadVS2NotAlign64(uint32_t s2Align64, uint32_t curBaseTileSize, uint32_t vSlot)
    {
        // halfEmbed = ROW_SUM_NUM/2，始终按 D=128 两面清尾。
        // fp8 C0=32 → 4 面覆盖 D=128；D=64 的 GM 只占前 2 面，后 2 面写在 32KB KV 槽余量上。
        // LoadV kStep 仍按实际 dAct。
        AscendC::InitConstValueParams<uint16_t> kvL1InitParams(1, static_cast<uint16_t>(s2Align64 - curBaseTileSize), 0,
                                                               MXFP8::ZERO_FILL_PATTERN);
        uint32_t planeElems = s2Align64 * MXFP8::FP8_C0_ELEMS;
        constexpr uint32_t planeCnt = MXFP8::CONST_128 / MXFP8::FP8_C0_ELEMS;
        uint32_t validElems = curBaseTileSize * MXFP8::FP8_C0_ELEMS;
        for (uint32_t p = 0; p < planeCnt; ++p) {
            AscendC::Fill(l1ATensor[vSlot][p * planeElems + validElems].template ReinterpretCast<uint16_t>(),
                          kvL1InitParams);
        }
    }

    // 只发 GM→L1（even 256 / 尾块 128），不 Wait MTE2、不占 L0。odd 直接 return。
    // kernel 在 Wait(V2_C2)/pscale 之前调用，让 V 的 MTE2 和 AIV rescale 重叠。
    template <class TensorV, class TaskInfoT, class TileInfoT>
    __aicore__ inline void PrefetchV(TensorV& gV, AscendC::GlobalTensor<uint8_t> gVDequantScale,
                                     AscendC::GlobalTensor<int32_t> gSparseIdx, TileInfoT& delay20TileInfo,
                                     TaskInfoT& delay20TaskInfo, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                     uint32_t kvHeadMul, GemmCoord actualBlockShapePV)
    {
        uint32_t embedReal = actualBlockShapePV[1];
        uint32_t curBaseTileSize = actualBlockShapePV[2];
        uint32_t gatheredKvSTileIdx = delay20TileInfo.pvGatheredKvSTileIdx;

        const bool pairOdd = (gatheredKvSTileIdx & 1u) != 0u;
        if (pairOdd) {
            return;
        }
        const bool pairEven = !delay20TileInfo.isLastKvsTile;
        uint32_t pairRows = curBaseTileSize;
        if (pairEven) {
            pairRows += MXFP8::Mxfp8NextKvsTileRows(gatheredKvSTileIdx, kvSBaseTile, delay20TaskInfo.gatheredKvSeqlen);
        }
        uint32_t pairAlign64 = MXFP8::Mxfp8Align64(pairRows);
        uint32_t vSlot = l1ABufId;
        uint32_t scaleRows = pairRows;
        uint32_t vScaleAlign64 = pairAlign64;

        AscendC::GlobalTensor<uint8_t> gVScale = gVDequantScale[delay20TaskInfo.gmOffsetVScale];
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx = gSparseIdx[delay20TaskInfo.gmOffsetSparseIdx];
        uint32_t firstYBlockIdx = (gatheredKvSTileIdx * kvSBaseTile) / blockShapeY;
        uint32_t firstOriYBlockIdx = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + vSlot);
        if (pairRows != pairAlign64) {
            PadVS2NotAlign64(pairAlign64, pairRows, vSlot);
        }

        auto l1ATensorTla = tla::MakeTensor(
            l1ATensor[vSlot], tla::MakeLayout<ElementA, layout::nZ>(embedReal, pairAlign64), Arch::PositionL1{});
        auto gVScaleTensorTla = tla::MakeTensor(gVScale.ReinterpretCast<AscendC::fp8_e8m0_t>(),
                                                tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::ColumnMajor, false>(
                                                    kvHeadMul * embedReal, vScaleAlign64 / MX_SCALE_GROUP_NUM),
                                                Arch::PositionGM{});
        auto l1AScaleTensorTla = tla::MakeTensor(l1AScaleTensor[vSlot].ReinterpretCast<AscendC::fp8_e8m0_t>(),
                                                 tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::zZ, false>(
                                                     embedReal, vScaleAlign64 / MX_SCALE_GROUP_NUM),
                                                 Arch::PositionL1{});

        SparseVFusedBaseTileL1FullLoad(gV, l1ATensorTla, gVScaleTensorTla, l1AScaleTensorTla, gSparseBlockIdx,
                                       gatheredKvSTileIdx, delay20TaskInfo.kvSeqlen, kvSBaseTile, blockShapeY,
                                       delay20TaskInfo.yBlockNumAval, delay20TaskInfo.yBlockNumRsvd, pairRows,
                                       scaleRows, embedReal, firstYBlockIdx, firstOriYBlockIdx);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + vSlot);
        pairVSlot_ = vSlot;
        pairVL1Align64_ = pairAlign64;
        l1ABufId = (l1ABufId + 1) % MXFP8::L1_KV_BUF_CNT;
    }

    // 等本拍 V 的 MTE2 完成后 LoadV/mmad。P pad 每 128-tile 一次。odd 复用 even 的 L1。
    template <class TensorC, class TaskInfoT, class TileInfoT>
    __aicore__ inline void ComputePV(TensorC& ubOTmpTensor, GemmCoord actualBlockShapePV, TileInfoT& delay20TileInfo,
                                     TaskInfoT& delay20TaskInfo)
    {
        uint32_t embedReal = actualBlockShapePV[1];
        uint32_t gatheredKvSTileIdx = delay20TileInfo.pvGatheredKvSTileIdx;
        uint32_t pSlot = delay20TileInfo.pSlot;

        const bool pairOdd = (gatheredKvSTileIdx & 1u) != 0u;
        const bool pairEven = ((gatheredKvSTileIdx & 1u) == 0u) && !delay20TileInfo.isLastKvsTile;
        uint32_t vSlot = pairVSlot_;
        uint32_t pairAlign64 = pairVL1Align64_;
        uint32_t mMmad = MXFP8::PV_MMAD_M_DIM;

        if (delay20TileInfo.kvsActBaseTileAlign32 != delay20TileInfo.kvsActBaseTileAlign64) {
            PadPS2NotAlign64(delay20TileInfo.kvsActBaseTileAlign64, delay20TileInfo.kvsActBaseTileAlign32, pSlot);
        }

        if (!pairOdd) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + vSlot);
        }

        uint32_t l0CEventId = PV_L0C_EVENT0;
        uint32_t l0ABufId = pvL0abBufId;
        uint32_t l0BBufId = pvL0abBufId;
        uint32_t l0ABEventId = PV_L0AB_EVENT0 + pvL0abBufId;

        if (delay20TileInfo.isTileGoupFirstTile) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);

        uint32_t s2Full = delay20TileInfo.kvsActBaseTileAlign64;
        uint32_t s2Start = pairOdd ? MXFP8::QK_HALF_KVS : 0;
        bool initC = delay20TileInfo.isTileGoupFirstTile;
        LoadPToL0(pSlot, l0BBufId, s2Full, embedReal, delay20TaskInfo.qsActBaseTileAlign64, 0);
        LoadVToL0(vSlot, l0ABufId, embedReal, s2Full, s2Start, pairAlign64, vSlot, s2Start, pairAlign64);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);

        MatmulPV(l0ABufId, l0BBufId, mMmad, delay20TaskInfo.qsActBaseTileAlign64, s2Full, initC);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
        pvL0abBufId = (pvL0abBufId + 1) % MXFP8::L0A_PV_BUF_CNT;

        if (delay20TileInfo.isUpdatePScale) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            if (delay20TaskInfo.qsActBaseTileAlign8 <= MXFP8::CONST_64) {
                FixpipeMm2SingleVect(delay20TaskInfo.qsActBaseTileAlign64, ubOTmpTensor);
            } else {
                FixpipeMm2(delay20TaskInfo.qsActBaseTileAlign64, ubOTmpTensor);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }

        if (!pairEven) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + vSlot);
        }

        if (delay20TileInfo.isLastSecondKvsTile) {
            InitL0BufferForReduceSum();
        }
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensor[MXFP8::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementB> l1BTensor[MXFP8::L1_P_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0ATensor[MXFP8::L0A_PV_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0BTensor[MXFP8::L0B_PV_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP8::L0C_PV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP8::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP8::L1_P_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1RowsumSeedTensor;
    AscendC::LocalTensor<uint8_t> l1RowsumSeedScaleTensor;

    TileMmad tileMmad;
    Tile::CopyL1ToL0AMxFp8A5 copyL1ToL0AMx;
    Tile::CopyL1ToL0BMxFp8A5 copyL1ToL0BMx;

    uint32_t pvL0abBufId = 0;
    uint32_t& l1ABufId;
    // 偶 tile V gather 所用的 L1 槽和整槽 S2 对齐，奇 tile LoadV 复用。
    uint32_t pairVSlot_ = 0;
    uint32_t pairVL1Align64_ = 0;
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_PV_ARCH35_MXFP8_HPP

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_QK_ARCH35_MXFP8_HPP
#define GEMM_BLOCK_QK_ARCH35_MXFP8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz_a5.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Gemm::Block {

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaQKMxfp8<true>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
                    ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaQKMxfp8<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});

    // Q(M) 方向统一 pad 到 128，满足 Fixpipe nSize 32B 倍数约束
    static constexpr uint32_t Q_M_PAD = 128;

    // blockShapeX=64 的满块 qs=64。QK 的 N 用 64，S 写进 128 宽 UB 行的前 64 列。
    // softmax / pscale / P 拷贝仍按 128 宽；后 64 列不参与有效列。短尾不是 64，继续 align128。
    template <class TaskInfoT>
    __aicore__ inline uint32_t QsMmadN(const TaskInfoT& info) const
    {
        if (info.qsActBaseTile == 64) {
            return 64;
        }
        return (info.qsActBaseTileAlign128 != 0) ? info.qsActBaseTileAlign128 : info.qsActBaseTileAlign16;
    }

    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    // Event ID：QK 用 2-3，L1 K 复用 MXFP8::KV_EVENT0..3（与 PV 共享）
    static constexpr uint32_t Q_EVENT0 = 2;
    static constexpr uint32_t Q_EVENT1 = 3;
    static constexpr uint32_t QK_L0AB_EVENT0 = 2;
    static constexpr uint32_t QK_L0AB_EVENT1 = 3;
    static constexpr uint32_t QK_L0C_EVENT0 = 2;
    static constexpr uint32_t QK_L0C_EVENT1 = 3;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t& kvBufId, uint64_t softmaxScale)
        : kBufId(kvBufId)
    {
        softmaxScale_ = softmaxScale;
        for (uint32_t i = 0; i < MXFP8::L1_Q_BUF_CNT; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(MXFP8::L1_Q_BUF_OFFSET + MXFP8::L1_Q_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L1_Q_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_Q_DESCALE_BUF_OFFSET +
                                                                                 MXFP8::L1_Q_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(MXFP8::L1_KV_BUF_OFFSET + MXFP8::L1_KV_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP8::L1_KV_DESCALE_BUF_OFFSET +
                                                                                 MXFP8::L1_KV_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L0A_QK_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(MXFP8::L0A_QK_BUF_OFFSET +
                                                                              MXFP8::L0A_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L0B_QK_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(MXFP8::L0B_QK_BUF_OFFSET +
                                                                              MXFP8::L0B_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::L0C_QK_BUF_CNT; i++) {
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<float>(MXFP8::L0C_QK_BUF_OFFSET + MXFP8::L0C_QK_BUF_SIZE * i);
        }

        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP8::KV_EVENT3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline ~BlockMmadTla() {}

    __aicore__ inline uint32_t CeilDivision(uint32_t numerator, uint32_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
    {
        return (a < b) ? a : b;
    }

    // qs64：L1 Q / Q-scale 按 128 行布局，先清零再只搬有效 64 行，避免 pad 行脏数据进 S。
    __aicore__ inline void PadQL1Zero(uint32_t qBufId)
    {
        constexpr uint32_t BLOCK = 32;
        AscendC::InitConstValueParams<uint16_t> qFill(1, static_cast<uint16_t>(MXFP8::L1_Q_BUF_SIZE / BLOCK), 0,
                                                      MXFP8::ZERO_FILL_PATTERN);
        AscendC::Fill(l1BTensor[qBufId].template ReinterpretCast<uint16_t>(), qFill);
        AscendC::InitConstValueParams<uint16_t> qScaleFill(
            1, static_cast<uint16_t>(MXFP8::L1_Q_DESCALE_BUF_SIZE / BLOCK), 0, MXFP8::ZERO_FILL_PATTERN);
        AscendC::Fill(l1BScaleTensor[qBufId].template ReinterpretCast<uint16_t>(), qScaleFill);
    }

    // Q 稠密 GM → L1（RowMajor → zN）
    template <class TensorQ, class TensorL1Q>
    __aicore__ inline void CopyQGmToL1(uint32_t qsActBaseTile, uint32_t embed, TensorQ gQTensorTla,
                                       TensorL1Q l1QTensorTla)
    {
        auto l1QTileTla = GetTile(l1QTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));
        // BSND: GetTile 确保 dValue = embed 而非 qHeadMul * embed
        auto gQTileTla = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));

        using CopyGmToL1Q = Tile::TileCopyTla<ArchTag, decltype(gQTileTla), decltype(l1QTensorTla)>;
        CopyGmToL1Q copyGmToL1Q;
        copyGmToL1Q(l1QTileTla, gQTileTla);
    }

    // Q-scale 稠密 GM → L1（Host 沿 D Dn2Nz）
    __aicore__ inline void CopyQScaleGmToL1(uint32_t qsActBaseTile, uint32_t scaleK,
                                            AscendC::GlobalTensor<uint8_t>& gQDequantScale, uint32_t qBufId,
                                            uint32_t qHeadMul)
    {
        copyGmToL1MxScaleDn2Nz(l1BScaleTensor[qBufId], gQDequantScale, qsActBaseTile, scaleK, 0, 0, qHeadMul);
    }

    // 中间 y-block 恒满 blockShapeY；只有 ori==aval-1 可能短。
    __aicore__ inline uint32_t YBlockRows(uint32_t ori, uint32_t blockShapeY, uint32_t yBlockNumAval,
                                          uint32_t kvSeqlenU)
    {
        if (ori == yBlockNumAval - 1u) {
            return kvSeqlenU - ori * blockShapeY;
        }
        return blockShapeY;
    }

    // 一段连续 GM：一条 Nd2Nz + 一条 scale。中间无 GetValue。
    template <class TensorK, class TensorL1K>
    __aicore__ inline void CopyKDataAndScale(TensorK gKTensorTla, TensorL1K l1KTensorTla,
                                             AscendC::GlobalTensor<uint8_t> gKScale, uint32_t oriStart, uint32_t nRows,
                                             uint32_t l1RowOff, uint32_t embed, uint32_t scaleK, uint32_t scaleBufId,
                                             uint32_t kvHeadMul)
    {
        using CopyGmToL1K = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;
        auto gKTile = GetTile(gKTensorTla, tla::MakeCoord(oriStart, 0), tla::MakeShape(nRows, embed));
        auto l1KTile = GetTile(l1KTensorTla, tla::MakeCoord(l1RowOff, 0), tla::MakeShape(nRows, embed));
        copyGmToL1K(l1KTile, gKTile);
        uint32_t scaleRowHalf = scaleK / 2;
        copyGmToL1MxScaleDn2Nz(l1AScaleTensor[scaleBufId], gKScale, nRows, scaleK, oriStart * kvHeadMul * scaleRowHalf,
                               l1RowOff * scaleRowHalf, kvHeadMul);
    }

    // 等长非连续两段：K 数据 stride 合法时一条 ndNum=2，否则两条 copy。scale 固定两条 Dn2Nz。
    template <class TensorK, class TensorL1K>
    __aicore__ inline void CopyKDataNdNum2(TensorK gKTensorTla, TensorL1K l1KTensorTla,
                                           AscendC::GlobalTensor<uint8_t> gKScale, uint32_t oriStart0,
                                           uint32_t oriStart1, uint32_t nRowsEach, uint32_t l1RowOff, uint32_t embed,
                                           uint32_t scaleK, uint32_t scaleBufId, uint32_t kvHeadMul)
    {
        using CopyGmToL1K = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;
        auto gKTile0 = GetTile(gKTensorTla, tla::MakeCoord(oriStart0, 0), tla::MakeShape(nRowsEach, embed));
        auto l1KTile0 = GetTile(l1KTensorTla, tla::MakeCoord(l1RowOff, 0), tla::MakeShape(nRowsEach, embed));
        auto gKTile1 = GetTile(gKTensorTla, tla::MakeCoord(oriStart1, 0), tla::MakeShape(nRowsEach, embed));
        auto l1KTile1 =
            GetTile(l1KTensorTla, tla::MakeCoord(l1RowOff + nRowsEach, 0), tla::MakeShape(nRowsEach, embed));
        uint32_t srcNdStride = 0;
        uint32_t dstNzStride = 0;
        bool fusedNd2 = MXFP8::Mxfp8U32Delta(static_cast<int64_t>(gKTile1.layout()(gKTile1.coord())) -
                                                 static_cast<int64_t>(gKTile0.layout()(gKTile0.coord())),
                                             srcNdStride) &&
                        MXFP8::Mxfp8U32Delta(static_cast<int64_t>(l1KTile1.layout()(l1KTile1.coord())) -
                                                 static_cast<int64_t>(l1KTile0.layout()(l1KTile0.coord())),
                                             dstNzStride);
        if (fusedNd2) {
            copyGmToL1K(l1KTile0, gKTile0, 2, srcNdStride, dstNzStride);
        } else {
            copyGmToL1K(l1KTile0, gKTile0);
            copyGmToL1K(l1KTile1, gKTile1);
        }
        uint32_t scaleRowHalf = scaleK / 2;
        copyGmToL1MxScaleDn2Nz(l1AScaleTensor[scaleBufId], gKScale, nRowsEach, scaleK,
                               oriStart0 * kvHeadMul * scaleRowHalf, l1RowOff * scaleRowHalf, kvHeadMul);
        copyGmToL1MxScaleDn2Nz(l1AScaleTensor[scaleBufId], gKScale, nRowsEach, scaleK,
                               oriStart1 * kvHeadMul * scaleRowHalf, (l1RowOff + nRowsEach) * scaleRowHalf, kvHeadMul);
    }

    // K+K-scale gather。偶 tile 一次最多 256 行，起点是 256 的倍数。
    // yBlockInnerStart = 起点 % blockShapeY。Y 整除 256（64/128/256）时为 0，快路径与原先一致。
    // 非 0 时第一段从块内偏移起、长度是该块剩余行，其后的块仍从块头搬；数据和 scale 用同一行偏移。
    // PAIR256 满对 nRows=256：Y=64 → 4 块，Y=128 → 2 块。先读齐 ori，连续则一条 copy；
    // nY==2 非连续满块直发 ndNum=2（O5，仅 innerStart=0）；
    // 其余高稀疏 gap 等长满块走下方通用 NDNUM2。跨块数不超过 4。
    template <class TensorK, class TensorL1K>
    __aicore__ inline void SparseKFusedBaseTileL1FullLoad(
        TensorK gKTensorTla, TensorL1K l1KTensorTla, AscendC::GlobalTensor<uint8_t> gKScale,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, int64_t kvSeqlen, uint32_t blockShapeY, uint32_t yBlockNumAval,
        uint32_t yBlockNumRsvd, uint32_t nRows, uint32_t embed, uint32_t scaleBufId, uint32_t kvHeadMul,
        uint32_t firstYBlockIdx, uint32_t firstOriYBlockIdx, uint32_t yBlockInnerStart, uint32_t scaleK)
    {
        if (nRows == 0) {
            return;
        }
        const uint32_t kvSeqlenU = static_cast<uint32_t>(kvSeqlen);
        constexpr uint32_t MAX_NY = 4;
        // 第一块只贡献 Y-innerStart 行，跨块数按起点到终点算。innerStart=0 时与 ceil(nRows/Y) 相同。
        uint32_t span = nRows + yBlockInnerStart;
        uint32_t nY = (span + blockShapeY - 1u) / blockShapeY;
        if (nY > MAX_NY) {
            nY = MAX_NY;
        }
        uint32_t ori[MAX_NY];
        ori[0] = firstOriYBlockIdx;
        if (ori[0] >= yBlockNumAval) {
            return;
        }
        for (uint32_t i = 1; i < nY; ++i) {
            uint32_t yi = firstYBlockIdx + i;
            if (yi >= yBlockNumRsvd) {
                nY = i;
                break;
            }
            uint32_t o = static_cast<uint32_t>(gSparseBlockIdx.GetValue(yi));
            if (o >= yBlockNumAval) {
                nY = i;
                break;
            }
            ori[i] = o;
        }

        // O5：nY==2 非连续满块（现网 Y=128 + PAIR256 的 keep<1 主路径）。
        // 语义等同于下方 NDNUM2 分支，跳过 consecutive 扫描和两层 while。
        // 尾块 / Y=64(nY=4) / 非升序 sparseIdx / 含短 y-block / 块内偏移 全部回落通用路径。
        if (yBlockInnerStart == 0 && nY == 2 && nRows == (blockShapeY << 1) && ori[1] != ori[0] + 1u &&
            ori[1] > ori[0] && YBlockRows(ori[0], blockShapeY, yBlockNumAval, kvSeqlenU) == blockShapeY &&
            YBlockRows(ori[1], blockShapeY, yBlockNumAval, kvSeqlenU) == blockShapeY) {
            CopyKDataNdNum2(gKTensorTla, l1KTensorTla, gKScale, ori[0] * blockShapeY, ori[1] * blockShapeY, blockShapeY,
                            0, embed, scaleK, scaleBufId, kvHeadMul);
            return;
        }

        bool consecutive = true;
        uint32_t avail = 0;
        for (uint32_t i = 0; i < nY; ++i) {
            if (i > 0 && ori[i] != ori[i - 1] + 1u) {
                consecutive = false;
            }
            uint32_t br = YBlockRows(ori[i], blockShapeY, yBlockNumAval, kvSeqlenU);
            bool shortBlock = br < blockShapeY;
            if (i == 0) {
                if (br <= yBlockInnerStart) {
                    return;
                }
                br -= yBlockInnerStart;
            }
            avail += br;
            if (shortBlock) {
                nY = i + 1u;
                break;
            }
        }
        uint32_t copyRows = MinU32(nRows, avail);
        if (consecutive && copyRows > 0) {
            CopyKDataAndScale(gKTensorTla, l1KTensorTla, gKScale, ori[0] * blockShapeY + yBlockInnerStart, copyRows, 0,
                              embed, scaleK, scaleBufId, kvHeadMul);
            return;
        }

        uint32_t l1Off = 0;
        uint32_t remain = nRows;
        uint32_t i = 0;
        while (i < nY && remain > 0) {
            uint32_t b0 = YBlockRows(ori[i], blockShapeY, yBlockNumAval, kvSeqlenU);
            uint32_t rowOff = (i == 0) ? yBlockInnerStart : 0;
            if (b0 <= rowOff) {
                break;
            }
            uint32_t runLen = MinU32(b0 - rowOff, remain);
            bool hitShort = b0 < blockShapeY;
            uint32_t j = i + 1u;
            while (!hitShort && j < nY && remain > runLen && ori[j] == ori[j - 1] + 1u) {
                uint32_t bj = YBlockRows(ori[j], blockShapeY, yBlockNumAval, kvSeqlenU);
                runLen += MinU32(bj, remain - runLen);
                hitShort = bj < blockShapeY;
                ++j;
            }
            uint32_t oriStart0 = ori[i] * blockShapeY + rowOff;
            if (!hitShort && j < nY && remain >= (runLen << 1) && ori[j] != ori[j - 1] + 1u) {
                uint32_t b1 = YBlockRows(ori[j], blockShapeY, yBlockNumAval, kvSeqlenU);
                uint32_t oriStart1 = ori[j] * blockShapeY;
                if (b1 == runLen && oriStart1 > oriStart0) {
                    CopyKDataNdNum2(gKTensorTla, l1KTensorTla, gKScale, oriStart0, oriStart1, runLen, l1Off, embed,
                                    scaleK, scaleBufId, kvHeadMul);
                    l1Off += runLen << 1;
                    remain -= runLen << 1;
                    i = j + 1u;
                    if (b1 < blockShapeY) {
                        break;
                    }
                    continue;
                }
            }
            CopyKDataAndScale(gKTensorTla, l1KTensorTla, gKScale, oriStart0, runLen, l1Off, embed, scaleK, scaleBufId,
                              kvHeadMul);
            l1Off += runLen;
            remain -= runLen;
            i = j;
            if (hitShort) {
                break;
            }
        }
    }

    // Q → L0B（非转置，带 mx scale；kStart/kCur 沿 D 切段）
    __aicore__ inline void LoadQToL0B(uint32_t qsActBaseTileAlign16, uint32_t kCur, uint32_t qsActBaseTileAlign16L0,
                                      uint32_t scaleMAlign16, uint32_t qBufId, uint32_t qkL0abBufId, uint32_t kStart)
    {
        copyL1ToL0BMxQk(l0BTensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                        l1BTensor[qBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                        l1BScaleTensor[qBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), qsActBaseTileAlign16,
                        kCur, qsActBaseTileAlign16L0, scaleMAlign16, kStart);
    }

    // K → L0A（非转置，带 mx scale，n 方向子切分 + D 方向切段）
    __aicore__ inline void LoadKToL0A(uint32_t nSubRowStart, uint32_t nCur, uint32_t kCur,
                                      uint32_t kvsActBaseTileAlign16, uint32_t kBufId, uint32_t scaleBufId,
                                      uint32_t scaleRowStart, uint32_t qkL0abBufId, uint32_t kStart)
    {
        copyL1ToL0AMxQk(l0ATensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                        l1ATensor[kBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                        l1AScaleTensor[scaleBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), nSubRowStart, nCur,
                        kCur, kvsActBaseTileAlign16, kStart, scaleRowStart);
    }

    // C = K · Qᵀ；默认整段 D，initC 由调用方传入
    __aicore__ inline void MatmulQK(uint32_t n, uint32_t qsActBaseTileAlign16, uint32_t kCur, uint32_t qkL0aBufId,
                                    uint32_t qkL0bBufId, uint32_t qkL0cBufId, bool initC)
    {
        auto l0ALayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0A>(n, kCur);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0aBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0B>(kCur, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qkL0bBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, kCur, initC);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，subBlockId 选目标 AIV）
    // l0cM / l0cRowStart：256 mmad 后两段 Fixpipe 从同一块 L0C 切 M。
    template <class TensorUB>
    __aicore__ inline void FixpipeMm1(uint32_t qsActBaseTileAlign16, uint32_t n, TensorUB& ubSTensorTla,
                                      bool subBlockId, uint32_t qkL0cBufId, uint32_t l0cM, uint32_t l0cRowStart)
    {
        auto l0CFullTla =
            tla::MakeTensor(l0CTensor[qkL0cBufId], tla::MakeLayoutL0C(l0cM, qsActBaseTileAlign16), Arch::PositionL0C{});
        auto l0CTensorTla =
            GetTile(l0CFullTla, tla::MakeCoord(l0cRowStart, 0), tla::MakeShape(n, qsActBaseTileAlign16));

        using CopyL0CToUB =
            Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB, Tile::CopyL0CToUBMode::NO_SPLIT,
                                 Tile::ScaleGranularity::PER_TENSOR, false>;
        CopyL0CToUB copyL0CToUB;
        copyL0CToUB(ubSTensorTla, l0CTensorTla, softmaxScale_, subBlockId);
    }

    template <class TensorUB>
    __aicore__ inline void EmitFixpipeMm1(uint32_t l0cBufId, uint32_t nSubRowStart, uint32_t nCur, uint32_t qsMmadN,
                                          TensorUB& ubSTensorTla, uint32_t subBlockIdx, uint32_t l0cM,
                                          uint32_t l0cRowStart, bool waitMFix, bool setFixM)
    {
        if (waitMFix) {
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(QK_L0C_EVENT0 + l0cBufId);
        }
        uint32_t nCurEven = (nCur + 1) >> 1 << 1;
        // ubS 是 256×128 ping-pong 视图、stride 仍 256。GetTile(row=128) 只改 coord，
        // data() 不动，layout()(coord)=128*256。不要先切成 128 行再 MakeTensor(data()+off)
        // （会把 ping-pong 的 coord.col 清零）。
        auto ubSSubTla = GetTile(ubSTensorTla, tla::MakeCoord(nSubRowStart, 0), tla::MakeShape(nCurEven, qsMmadN));
        FixpipeMm1(qsMmadN, nCur, ubSSubTla, subBlockIdx != 0, l0cBufId, l0cM, l0cRowStart);
        if (setFixM) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0 + l0cBufId);
        }
    }

    // 只发 GM→L1（Q 首 tile + 每 tile K），不 Wait MTE2、不占 L0、不推进 kBufId。
    // kernel 在 Wait(V1_C1) 之前调用，让 K 的 MTE2 和 AIV softmax 重叠。
    template <class TensorQ, class TensorK, class TaskInfoT, class TileInfoT>
    __aicore__ inline void PrefetchK(TensorQ& gQTensorTla, TensorK& gKTensorTla,
                                     AscendC::GlobalTensor<uint8_t> gQDequantScale,
                                     AscendC::GlobalTensor<uint8_t> gKDequantScale,
                                     AscendC::GlobalTensor<int32_t> gSparseBlockIdx, TaskInfoT& curTaskInfo,
                                     TileInfoT& curTileInfo, uint32_t embed, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                     uint32_t kvHeadMul, uint32_t qHeadMul)
    {
        uint32_t qsActBaseTile = curTaskInfo.qsActBaseTile;
        uint32_t qsMmadN = QsMmadN(curTaskInfo);
        int64_t kvSeqlen = curTaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = curTaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = curTaskInfo.yBlockNumRsvd;

        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;
        bool isFirstKvsTile = curTileInfo.isFirstKvsTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;

        uint32_t scaleK = CeilDivision(embed, MXFP8::MX_GROUP_ELEMS);
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

        // 奇 tile 复用偶 tile 已搬进 L1 的下半 128 行，不再发 MTE2。
        const bool pairOdd = (gatheredKvSTileIdx & 1u) != 0u;
        if (pairOdd) {
            return;
        }
        const bool pairEven = !isLastKvsTile;
        uint32_t gatherRows = kvsActBaseTile;
        if (pairEven) {
            gatherRows += MXFP8::Mxfp8NextKvsTileRows(gatheredKvSTileIdx, kvSBaseTile, curTaskInfo.gatheredKvSeqlen);
        }
        pairKSlot_ = kBufId;
        pairKL1Align16_ = MXFP8::Mxfp8Align16(gatherRows);
        uint32_t l1KEventId = MXFP8::KV_EVENT0 + pairKSlot_;

        uint32_t gatheredRow = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t firstYBlockIdx = gatheredRow / blockShapeY;
        uint32_t yBlockInnerStart = gatheredRow - firstYBlockIdx * blockShapeY;
        uint32_t firstOriYBlockIdx = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx));

        // Q 只在 task 首 KV tile 搬。copy 后只 SetFlag；K 发出后再 Wait Q，与 K 的 MTE2 重叠。
        if (isFirstKvsTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            if (qsActBaseTile < Q_M_PAD) {
                PadQL1Zero(qBufId);
            }
            auto l1QLayout = tla::MakeLayout<ElementA, layout::zN>(qsMmadN, embed);
            auto l1QTensorTla = tla::MakeTensor(l1BTensor[qBufId], l1QLayout, Arch::PositionL1{});
            CopyQGmToL1(qsActBaseTile, embed, gQTensorTla, l1QTensorTla);
            CopyQScaleGmToL1(qsActBaseTile, scaleK, gQDequantScale, qBufId, qHeadMul);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        auto l1KLayout = tla::MakeLayout<ElementA, layout::zN>(gatherRows, embed);
        auto l1KTensorTla = tla::MakeTensor(l1ATensor[pairKSlot_], l1KLayout, Arch::PositionL1{});
        SparseKFusedBaseTileL1FullLoad(gKTensorTla, l1KTensorTla, gKDequantScale, gSparseBlockIdx, kvSeqlen,
                                       blockShapeY, yBlockNumAval, yBlockNumRsvd, gatherRows, embed, pairKSlot_,
                                       kvHeadMul, firstYBlockIdx, firstOriYBlockIdx, yBlockInnerStart, scaleK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        if (isFirstKvsTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
        }
        // 槽已占用到奇 tile LoadK 结束；立刻推进 kvBufId，让随后的 PV 用下一槽，避免和跨 softmax 的 K 对撞。
        kBufId = (kBufId + 1) % MXFP8::L1_KV_BUF_CNT;
    }

    // 偶拍：LoadK + mmad256 进同一块 L0C S，不碰 UB。kernel 可在 Wait(V1_C1) 之前调用。
    template <class TaskInfoT>
    __aicore__ inline void ComputeQKMmad256Even(TaskInfoT& curTaskInfo, uint32_t embed)
    {
        uint32_t qsMmadN = QsMmadN(curTaskInfo);
        uint32_t kSlot = pairKSlot_;
        uint32_t l1KAlign16 = pairKL1Align16_;
        uint32_t l1KEventId = MXFP8::KV_EVENT0 + kSlot;
        constexpr uint32_t L0B_Q_SLOT = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);

        uint32_t nCur = l1KAlign16;
        uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
        uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
        LoadQToL0B(qsMmadN, embed, qsMmadN, qsMmadN, qBufId, L0B_Q_SLOT, 0);
        LoadKToL0A(0, nCur, embed, l1KAlign16, kSlot, kSlot, 0, qkL0abBufId, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
        // 256 已全部进 L0，奇 tile 不再 LoadK。preload 偶拍在 Wait(V1_C1) 前，必须立刻放 L1。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
        MatmulQK(nCur, qsMmadN, embed, qkL0abBufId, L0B_Q_SLOT, qkL0cBufId, true);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
        qkL0abBufId = (qkL0abBufId + 1) % MXFP8::L0A_QK_BUF_CNT;
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        pairL0cM_ = nCur;
    }

    // 偶拍：把 L0C S 的前 128 行 Fixpipe 到 UB。不 Set FIX_M，槽留给奇拍第二段。
    template <class TensorUB, class TaskInfoT>
    __aicore__ inline void EmitFixpipeQK256Even(TensorUB& ubSTensorTla, TaskInfoT& curTaskInfo, uint32_t subBlockIdx)
    {
        uint32_t qsMmadN = QsMmadN(curTaskInfo);
        uint32_t nFix0 = MinU32(MXFP8::QK_HALF_KVS, pairL0cM_);
        EmitFixpipeMm1(qkL0cBufId, 0, nFix0, qsMmadN, ubSTensorTla, subBlockIdx, pairL0cM_, 0, true, false);
    }

    // 等本拍 K 的 MTE2 完成后 LoadK/mmad。最后一次 LoadK 立刻放 L1，不把槽占到 fixpipe。
    template <class TensorUB, class TaskInfoT, class TileInfoT>
    __aicore__ inline void ComputeQK(TensorUB& ubSTensorTla, TaskInfoT& curTaskInfo, TileInfoT& curTileInfo,
                                     uint32_t embed, uint32_t subBlockIdx)
    {
        uint32_t qsMmadN = QsMmadN(curTaskInfo);
        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;

        uint32_t l1QEventId = Q_EVENT0 + qBufId;

        const bool pairOdd = (gatheredKvSTileIdx & 1u) != 0u;
        const bool pairEven = ((gatheredKvSTileIdx & 1u) == 0u) && !isLastKvsTile;
        uint32_t kSlot = pairKSlot_;
        uint32_t l1KAlign16 = pairKL1Align16_;
        uint32_t pairRowOff = pairOdd ? MXFP8::QK_HALF_KVS : 0;
        uint32_t scaleSlot = kSlot;
        uint32_t scaleRowStart = pairRowOff;
        uint32_t l1KEventId = MXFP8::KV_EVENT0 + kSlot;

        constexpr uint32_t L0B_Q_SLOT = 0;

        if (pairOdd) {
            uint32_t nCur = kvsActBaseTile;
            EmitFixpipeMm1(qkL0cBufId, 0, nCur, qsMmadN, ubSTensorTla, subBlockIdx, pairL0cM_, MXFP8::QK_HALF_KVS,
                           false, true);
            if (unlikely(isLastKvsTile)) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
                qBufId = (qBufId + 1) % MXFP8::L1_Q_BUF_CNT;
            }
            return;
        }

        if (!pairOdd) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        }

        // 尾块或未开 MMAD256：一拍 128×128。
        uint32_t nLoopNum = CeilDivision(kvsActBaseTile, L0_TILE_N);
        bool fixPending = false;
        uint32_t pendL0cBufId = 0;
        uint32_t pendNSubRowStart = 0;
        uint32_t pendNCur = 0;
        for (uint32_t nSub = 0; nSub < nLoopNum; ++nSub) {
            uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
            uint32_t nSubRowStart = nSub * L0_TILE_N + pairRowOff;
            uint32_t nCur = MinU32(L0_TILE_N, kvsActBaseTile - nSub * L0_TILE_N);
            uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            LoadQToL0B(qsMmadN, embed, qsMmadN, qsMmadN, qBufId, L0B_Q_SLOT, 0);
            LoadKToL0A(nSubRowStart, nCur, embed, l1KAlign16, kSlot, scaleSlot, scaleRowStart, qkL0abBufId, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
            if (nSub + 1 == nLoopNum && !pairEven) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
            }

            if (fixPending) {
                EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qsMmadN, ubSTensorTla, subBlockIdx, pendNCur,
                               0, true, true);
                fixPending = false;
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
            MatmulQK(nCur, qsMmadN, embed, qkL0abBufId, L0B_Q_SLOT, qkL0cBufId, true);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            qkL0abBufId = (qkL0abBufId + 1) % MXFP8::L0A_QK_BUF_CNT;

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            pendL0cBufId = qkL0cBufId;
            pendNSubRowStart = nSub * L0_TILE_N;
            pendNCur = nCur;
            fixPending = true;
            qkL0cBufId = (qkL0cBufId + 1) % MXFP8::L0C_QK_BUF_CNT;
        }
        if (fixPending) {
            EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qsMmadN, ubSTensorTla, subBlockIdx, pendNCur, 0,
                           true, true);
        }

        if (unlikely(isLastKvsTile)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            qBufId = (qBufId + 1) % MXFP8::L1_Q_BUF_CNT;
        }
    }

protected:
    AscendC::LocalTensor<ElementB> l1ATensor[MXFP8::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1BTensor[MXFP8::L1_Q_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0ATensor[MXFP8::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0BTensor[MXFP8::L0B_QK_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP8::L0C_QK_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP8::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP8::L1_Q_BUF_CNT];

    Tile::CopyL1ToL0AMxFp8QKA5 copyL1ToL0AMxQk;
    Tile::CopyL1ToL0BMxFp8QKA5 copyL1ToL0BMxQk;
    TileMmad tileMmad;
    Tile::CopyGmToL1MxScaleDn2NzA5 copyGmToL1MxScaleDn2Nz;

    uint32_t qBufId = 0;
    uint32_t& kBufId;
    uint32_t qkL0abBufId = 0;
    uint32_t qkL0cBufId = 0;
    uint64_t softmaxScale_;
    // 偶 tile gather 所用的 L1 槽和整槽行对齐，奇 tile LoadK 复用。
    uint32_t pairKSlot_ = 0;
    uint32_t pairKL1Align16_ = 0;
    uint32_t pairL0cM_ = 0;
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_QK_ARCH35_MXFP8_HPP

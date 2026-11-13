/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GBSA_BLOCK_MMAD_FULL_QUANT_HPP
#define GBSA_BLOCK_MMAD_FULL_QUANT_HPP

#include "../../arch/gbsa_resource.hpp"
#include "../tile_common/gbsa_gemm_tile_copy.hpp"
#include "../../../arch22/gbsa_full_quant_utils.hpp"
#include "../../../arch22/kernel_utils.hpp"
#include "../../../arch22/gbsa_packed_kv.hpp"

namespace GbsaKernelArch22::Int8 {

using namespace NpuArch;

static_assert(FullQuantLayout::L1_END == 480 * FullQuantLayout::KIB);
static_assert(FullQuantLayout::L0A_END == 64 * FullQuantLayout::KIB);
static_assert(FullQuantLayout::DOUBLE_BUFFER * FullQuantLayout::L0B_SLOT_BYTES == 64 * FullQuantLayout::KIB);
static_assert(FullQuantLayout::L0C_END == 128 * FullQuantLayout::KIB);
static_assert(FullQuantLayout::DOUBLE_BUFFER * FullQuantLayout::L0C_SHARED_SLOT_BYTES == 128 * FullQuantLayout::KIB);
static_assert(FullQuantLayout::WORKSPACE_BYTES == 1344 * FullQuantLayout::KIB);
static_assert(FullQuantLayout::HEAD_DIM * 16384 <= INT32_MAX);
static_assert(FullQuantLayout::KV_TILE * 127 * 128 <= INT32_MAX);

// Owns all Cube-side buffers and events. QK and PV share one L0B cursor;
// separate cursors would allow PV to overwrite a QK operand still in use.
template <bool singleRowTile>
class GbsaFullQuantCubeArch22 {
    using Resource = Arch::Int8StaticResource;
    using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using KType = Gemm::GemmType<int8_t, layout::ColumnMajor>;
    using OType = Gemm::GemmType<int32_t, layout::RowMajor>;
    using A1Type = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>;
    using A2Type = Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>;
    using K1Type = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>;
    using V1Type = Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>;
    using B2Type = Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>;

public:
    __aicore__ inline explicit GbsaFullQuantCubeArch22(Resource& resource, uint32_t headDim = FullQuantLayout::HEAD_DIM,
                                                       uint32_t kvHeads = 1, bool kvHeadMajor = false,
                                                       bool packedKv = false, uint32_t packedBlockY = 128U)
        : resource_(resource),
          headDim_(headDim),
          kvHeads_(kvHeads),
          copyDim_(headDim),
          kvHeadMajor_(kvHeadMajor),
          packedKv_(packedKv),
          packedBlockY_(packedBlockY)
    {}

    __aicore__ inline void Init()
    {
        for (uint32_t i = 0; i < 2; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(i);
        }
        // Independent resident Q/P regions in L0A.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
        for (uint32_t i = 0; i < 4; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);
        }
        // K[0:2], V[2:4], P[4:7].
        for (uint32_t i = 0; i < 7; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
    }

    __aicore__ inline void Finish()
    {
        for (uint32_t i = 0; i < 4; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(i);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(i);
        }
        for (uint32_t i = 0; i < 7; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
    }

    struct BlockMmadQK;
    struct BlockMmadPV;

private:
    __aicore__ inline uint32_t HeadDim() const
    {
        return headDim_;
    }

    __aicore__ inline uint32_t QkDim() const
    {
        return (HeadDim() + 31U) / 32U * 32U;
    }

    __aicore__ inline uint32_t KvDim() const
    {
        return (copyDim_ + 31U) / 32U * 32U;
    }

    __aicore__ inline void ClearTailSlot(uint32_t offset, const Tile& tile, uint32_t half)
    {
        bool cleared = false;
        // NZ stores one 32-byte C0 per token. Clear only missing tokens in
        // every C0, rather than overwriting the complete 64 KiB slot.
        for (uint32_t page = 0; page < 2; ++page) {
            const uint32_t valid = tile.pages[half * 2 + page].validTokens;
            const uint32_t missing = FullQuantLayout::PAGE_TOKENS - valid;
            if (missing != 0) {
                AscendC::InitConstValueParams<int32_t> params(KvDim() / 32U, missing,
                                                              FullQuantLayout::L1_KV_ROWS - missing, 0);
                const uint32_t start = (page * FullQuantLayout::PAGE_TOKENS + valid) * 32U;
                AscendC::InitConstValue(L1(offset + start).template ReinterpretCast<int32_t>(), params);
                cleared = true;
            }
        }
        if ((copyDim_ % 32U) != 0U) {
            // Only the last C0 needs channel padding; complete C0s are overwritten by DMA.
            // DMA can include extra channels. Clear padding after that whole
            // window; Q padding is zero and only the real D outputs are stored.
            const uint32_t tailOffset = (copyDim_ / 32U) * FullQuantLayout::L1_KV_ROWS * 32U;
            AscendC::InitConstValueParams<int32_t> params(1, FullQuantLayout::L1_KV_ROWS, 0, 0);
            AscendC::InitConstValue(L1(offset + tailOffset).template ReinterpretCast<int32_t>(), params);
            cleared = true;
        }
        if (cleared) {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }

    template <AscendC::HardEvent EVENT>
    __aicore__ inline void Sync()
    {
        AscendC::SetFlag<EVENT>(7);
        AscendC::WaitFlag<EVENT>(7);
    }

    __aicore__ inline uint32_t RowsRound() const
    {
        return singleRowTile ? 16U : rowRound_;
    }
    __aicore__ inline uint32_t QBytes() const
    {
        return singleRowTile ? 16U * FullQuantLayout::HEAD_DIM : qBytes_;
    }
    __aicore__ inline uint32_t CBytes() const
    {
        return singleRowTile ? 16U * FullQuantLayout::QK_COLUMNS * sizeof(int32_t) : cBytes_;
    }

    __aicore__ inline void Mmad(AscendC::LocalTensor<int32_t> dst, AscendC::LocalTensor<int8_t> a,
                                AscendC::LocalTensor<int8_t> b, bool initialize, uint32_t reduction, uint32_t columns,
                                uint32_t matrixRows)
    {
        AscendC::MmadParams params;
        params.m = matrixRows;
        params.n = columns;
        params.k = reduction;
        params.cmatrixInitVal = initialize;
        AscendC::Mmad(dst, a, b, params);
        // Same short-MMAD guard as FP16 TileMmad.
        if ((matrixRows / 16U) * (columns / 16U) < 10U) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    __aicore__ inline AscendC::LocalTensor<int8_t> L1(uint32_t bytes)
    {
        return resource_.l1Buf.template GetBufferByByte<int8_t>(bytes);
    }
    __aicore__ inline AscendC::LocalTensor<int8_t> L0A(uint32_t bytes)
    {
        return resource_.l0ABuf.template GetBufferByByte<int8_t>(bytes);
    }
    __aicore__ inline AscendC::LocalTensor<int8_t> L0B(uint32_t slot)
    {
        return resource_.l0BBuf.template GetBufferByByte<int8_t>(slot * FullQuantLayout::L0B_SLOT_BYTES);
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> L0C(uint32_t bytes)
    {
        return resource_.l0CBuf.template GetBufferByByte<int32_t>(bytes);
    }

    Resource& resource_;
    uint32_t headDim_;
    uint32_t kvHeads_;
    uint32_t copyDim_;
    bool kvHeadMajor_;
    bool packedKv_;
    uint32_t packedBlockY_;
    uint32_t headOffset_ = 0;
    uint32_t rows_ = FullQuantLayout::ROWS;
    uint32_t rowRound_ = FullQuantLayout::ROWS;
    uint32_t qBytes_ = FullQuantLayout::Q_BYTES;
    uint32_t pBytes_ = FullQuantLayout::P_BYTES;
    uint32_t cBytes_ = FullQuantLayout::L0C_S_SLOT_BYTES;
    uint32_t qkColumns_ = FullQuantLayout::QK_COLUMNS;
    uint32_t qkReduction_ = FullQuantLayout::QK_REDUCTION;
    uint32_t pvReduction_ = FullQuantLayout::PV_REDUCTION;
    uint32_t pvRows_ = FullQuantLayout::PV_ROWS;
    uint32_t bCursor_ = 0;
    uint32_t cCursor_ = 0;
    uint32_t sCursor_ = 0;
    uint32_t oCursor_ = 0;
    Gemm::Tile::CopyGmToL1<Arch::AtlasA2, AType, A1Type> copyA_;
    Gemm::Tile::CopyGmToL1<Arch::AtlasA2, KType, K1Type> copyK_;
    Gemm::Tile::CopyGmToL1<Arch::AtlasA2, AType, V1Type> copyV_;
    Gemm::Tile::CopyL1ToL0A<Arch::AtlasA2, A1Type, A2Type> loadA_;
    Gemm::Tile::CopyL1ToL0B<Arch::AtlasA2, K1Type, B2Type> loadK_;
    Gemm::Tile::CopyL1ToL0B<Arch::AtlasA2, V1Type, B2Type> loadV_;
    Gemm::Tile::CopyL0CToGm<Arch::AtlasA2, int32_t, OType> store_;
};

} // namespace GbsaKernelArch22::Int8

#endif

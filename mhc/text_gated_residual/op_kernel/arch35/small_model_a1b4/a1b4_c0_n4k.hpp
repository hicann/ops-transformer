/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "a1b4_block_mmad_c0_fullb_separate_k.hpp"
namespace Catlass::Gemm::Kernel::C0N4K {
constexpr uint32_t L0K = 128u;
template <class E, class Tag, uint32_t BK, uint32_t BN, class Tensor>
CATLASS_DEVICE void Preload(Arch::Resource<Tag>& resource, Tensor& src, uint32_t offset)
{
    using Copy =
        Gemm::Tile::PackedTileCopyTla<Tag, E, layout::RowMajor, E, layout::ColumnMajor, float, layout::RowMajor, void>;
    auto dst = tla::MakeTensor(resource.l1Buf.template GetBufferByByte<E>(offset),
                               tla::MakeLayout<E, typename Copy::LayoutTagL1B>(tla::Int<BK>{}, tla::Int<BN>{}),
                               Arch::PositionL1{});
    typename Copy::template CopyGmToL1B<Tensor> copy;
    copy(dst, src);
}
template <class E, class Tag, uint32_t TM, uint32_t AK, uint32_t FullK>
CATLASS_DEVICE void Cube(Arch::Resource<Tag>& resource, GM_ADDR x, GM_ADDR wd, GM_ADDR wg, GM_ADDR partial,
                         GM_ADDR gatePartial, uint32_t M, uint32_t split, bool gate, bool dedicatedGate,
                         uint32_t gateCols, uint32_t gateRows, Callback const& ready)
{
    using namespace AscendC;
    constexpr uint32_t ABytes = TM * AK * 2u * 2u, BBytes = FullK * 80u * 2u;
    static_assert(ABytes + BBytes + FullK * 16u * 2u <= Tag::L1_SIZE, "N4K L1 capacity");
    const uint32_t id = GetBlockIdx(), tasks = 4u * split, blocks = 10240u / 32u;
    const bool down = id < tasks;
    const bool gateTask = gate && (dedicatedGate ? (id >= tasks && id < tasks + split) : (down && id % 4u == 0u));
    const uint32_t dk = down ? id / 4u : 0u, dn = id % 4u;
    const uint32_t gk = gateTask ? (dedicatedGate ? id - tasks : dk) : 0u;
    const uint32_t kb = blocks / split, extra = blocks % split;
    const uint32_t ko = (dk * kb + (dk < extra ? dk : extra)) * 32u, kn = (kb + (dk < extra ? 1u : 0u)) * 32u;
    const uint32_t gko = (gk * kb + (gk < extra ? gk : extra)) * 32u, gkn = (kb + (gk < extra ? 1u : 0u)) * 32u;
    GlobalTensor<E> gx, gd, gg;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(x));
    gd.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(wd));
    gg.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(wg));
    auto ta = tla::MakeTensor(gx, tla::MakeLayout<E, layout::RowMajor>(M, 10240u), Arch::PositionGM{});
    auto tb = tla::MakeTensor(gd, tla::MakeLayout<E, layout::ColumnMajor>(10240u, 320u), Arch::PositionGM{});
    auto tg = tla::MakeTensor(gg, tla::MakeLayout<E, layout::ColumnMajor>(10240u, 4u), Arch::PositionGM{});
    auto a = GetTile(ta, tla::MakeCoord(0u, ko), tla::MakeShape(M, kn));
    auto b = GetTile(tb, tla::MakeCoord(ko, dn * 80u), tla::MakeShape(kn, 80u));
    auto ga = GetTile(ta, tla::MakeCoord(0u, gko), tla::MakeShape(M, gkn));
    auto g = GetTile(tg, tla::MakeCoord(gko, 0u), tla::MakeShape(gkn, 4u));
    if (down)
        Preload<E, Tag, FullK, 80u>(resource, b, ABytes);
    if (gateTask)
        Preload<E, Tag, FullK, 16u>(resource, g, ABytes + BBytes);
    SetFlag<HardEvent::MTE2_MTE1>(6);
    WaitFlag<HardEvent::MTE2_MTE1>(6);
    ready();
    SetFlag<HardEvent::MTE1_MTE2>(7);
    WaitFlag<HardEvent::MTE1_MTE2>(7);
    using P = Gemm::MmadPingpongPreLoad<Tag, false, false, 1, false, 2, 1, 2, 2>;
    using Copy =
        Gemm::Tile::PackedTileCopyTla<Tag, E, layout::RowMajor, E, layout::ColumnMajor, float, layout::RowMajor, void>;
    using Tile = Gemm::Tile::TileMmadTla<Tag, E, typename Copy::LayoutTagL1A>;
    if (down) {
        GlobalTensor<float> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial) + static_cast<int64_t>(dk) * M * 320u + dn * 80u);
        auto dst = tla::MakeTensor(out, tla::MakeLayout<float, layout::RowMajor>(M, 320u), Arch::PositionGM{});
        auto tile = GetTile(dst, tla::MakeCoord(0u, 0u), tla::MakeShape(M, 80u));
        using L1 = tla::Shape<tla::Int<TM>, tla::Int<80>, tla::Int<AK>>;
        using L0 = tla::Shape<tla::Int<TM>, tla::Int<80>, tla::Int<L0K>>;
        Gemm::Block::BlockMmadC0FullBA1B4Tla<P, L1, L0, E, E, float, void, Copy, Tile, FullK> mm(resource, ABytes);
        mm(a, b, tile, GemmCoord{M, 80u, kn});
    }
    if (gateTask) {
        GlobalTensor<float> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gatePartial) +
                            static_cast<int64_t>(gk) * gateRows * gateCols);
        auto dst = tla::MakeTensor(out, tla::MakeLayout<float, layout::RowMajor>(M, gateCols), Arch::PositionGM{});
        auto tile = GetTile(dst, tla::MakeCoord(0u, 0u), tla::MakeShape(M, 4u));
        using L1 = tla::Shape<tla::Int<TM>, tla::Int<16>, tla::Int<AK>>;
        using L0 = tla::Shape<tla::Int<TM>, tla::Int<16>, tla::Int<L0K>>;
        Gemm::Block::BlockMmadC0FullBA1B4Tla<P, L1, L0, E, E, float, void, Copy, Tile, FullK> mm(resource,
                                                                                                 ABytes + BBytes);
        mm(ga, g, tile, GemmCoord{M, 4u, gkn});
    }
}
} // namespace Catlass::Gemm::Kernel::C0N4K

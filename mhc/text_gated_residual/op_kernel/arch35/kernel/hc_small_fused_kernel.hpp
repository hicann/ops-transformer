/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#pragma once
// Fused small-M C0/reduction/C1 with GM channel epilogue.
// Include after the latest generic kernel: its synchronization helpers remain authoritative.
#include "../block/block_epilogue_hc_v2_channel_tile.hpp"
namespace Catlass::Gemm::Kernel {
template <class Element>
CATLASS_DEVICE void HcReduceGateFast(Arch::Resource<Arch::Ascend950>& resource, GM_ADDR src, GM_ADDR dst, uint32_t M,
                                     uint32_t split)
{
    using namespace AscendC;
    uint32_t cores = GetBlockNum() * HcSubBlockNum(), id = GetBlockIdx();
    uint32_t groups = (M + 3u) / 4u, begin = (groups * id / cores) * 4u, end = (groups * (id + 1u) / cores) * 4u;
    end = end < M ? end : M;
    if (begin >= end)
        return;
    uint32_t rows = end - begin, len = rows * 8u;
    auto acc = resource.ubBuf.template GetBufferByByte<float>(0);
    auto out = resource.ubBuf.template GetBufferByByte<Element>(32u * 1024u);
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    GlobalTensor<Element> go;
    go.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(dst));
    for (uint32_t k = 0; k < split; ++k) {
        DataCopyExtParams cp(rows, 4u * static_cast<int64_t>(sizeof(float)),
                             (32u - 4u) * static_cast<int64_t>(sizeof(float)), 0, 0);
        DataCopyPadExtParams<float> pad(true, 0, 4, 0.0f);
        DataCopyPad(acc[k * len], gm[static_cast<int64_t>(k) * M * 32u + begin * 32u], cp, pad);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    for (uint32_t k = 1; k < split; ++k) {
        Add(acc, acc, acc[k * len], len);
        PipeBarrier<PIPE_V>();
    }
    auto outBits = resource.ubBuf.template GetBufferByByte<uint16_t>(32u * 1024u);
    Duplicate(outBits, static_cast<uint16_t>(0), rows * 16u);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < rows; ++row) {
        Cast(out[row * 16u], acc[row * 8u], RoundMode::CAST_RINT, 4u);
    }
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyExtParams cp(rows, 4u * static_cast<int64_t>(sizeof(Element)), 0, 0, 0);
    DataCopyPad(go[begin * 4u], out, cp);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t TM>
CATLASS_DEVICE uint32_t HcDownWorkFast(uint32_t core, uint32_t M, uint32_t sk, uint32_t sliceK)
{
    uint32_t tile = core / sk, mo = (tile / 2u) * TM, ko = (core % sk) * sliceK;
    if (mo >= M || ko >= 10240u)
        return 0u;
    uint32_t am = M - mo < TM ? M - mo : TM, ak = 10240u - ko < sliceK ? 10240u - ko : sliceK;
    return am * ak;
}

template <uint32_t TM, class Element, class Tiling>
CATLASS_DEVICE void HcDownGateStageFast(GM_ADDR x, GM_ADDR wd, GM_ADDR wg, GM_ADDR downPreAct, GM_ADDR gateOut,
                                        GM_ADDR c0, GM_ADDR partial, Tiling const& td)
{
    using namespace AscendC;
    using ArchTag = Arch::Ascend950;
    using LA = layout::RowMajor;
    using LB = layout::ColumnMajor;
    using LC = layout::RowMajor;
    uint32_t M = td.M, r = td.R, d = td.D, cores = td.usedAic;
    uint32_t id = GetBlockIdx();
    if ASCEND_IS_AIV {
        id = GetBlockIdx() / HcSubBlockNum();
    }
    if (id >= cores)
        return;
    uint32_t requested = td.pad0 >> 8u;
    bool gate = td.useCombine != 0u;
    bool reserve = gate && (td.pad0 & 1u) != 0;
    uint32_t mn = ((M + TM - 1u) / TM) * 2u;
    uint32_t sk = (cores - (reserve ? requested : 0u)) / mn;
    uint32_t downTasks = mn * sk, ks = ((d + sk - 1u) / sk + 31u) / 32u * 32u;
    uint32_t gateCores = reserve ? cores - downTasks : requested;
    uint32_t gateMn = (M + 127u) / 128u, gateSplit = gateCores / gateMn;
    if (gateSplit == 0)
        gateSplit = 1;
    uint32_t gateTasks = gateMn * gateSplit, gk = ((d + gateSplit - 1u) / gateSplit + 31u) / 32u * 32u;
    GM_ADDR gatePartial = partial + static_cast<int64_t>(M) * r * static_cast<int64_t>(sizeof(float)) * td.splitkFactor;
    Arch::Resource<ArchTag> resource;
    if ASCEND_IS_AIC {
        GlobalTensor<Element> ga, gb, gg;
        ga.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x));
        gb.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(wd));
        gg.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(wg));
        auto ta = tla::MakeTensor(ga, tla::MakeLayout<Element, LA>(M, d), Arch::PositionGM{});
        auto tb = tla::MakeTensor(gb, tla::MakeLayout<Element, LB>(d, r), Arch::PositionGM{});
        auto tg = tla::MakeTensor(gg, tla::MakeLayout<Element, LB>(d, 4u), Arch::PositionGM{});
        if (id < downTasks) {
            using P = Gemm::MmadPingpongPreLoad<ArchTag, false, false, 1, false, 2, 2, 2, 2>;
            using L1 = tla::Shape<tla::Int<TM>, tla::Int<160>, tla::Int<256>>;
            using L0 = tla::Shape<tla::Int<TM>, tla::Int<160>, tla::Int<64>>;
            using Copy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LA, Element, LB, float, LC, void>;
            using MM = Gemm::Block::BlockMmadTla<P, L1, L0, Element, Element, float, void, Copy>;
            MM mm(resource);
            uint32_t k = id % sk, tile = id / sk, mo = tile / 2u * TM, no = tile % 2u * 160u, ko = k * ks;
            uint32_t am = M - mo < TM ? M - mo : TM, ak = d - ko < ks ? d - ko : ks;
            GlobalTensor<float> gp;
            gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial) + static_cast<int64_t>(k) * M * r);
            auto tp = tla::MakeTensor(gp, tla::MakeLayout<float, LC>(M, r), Arch::PositionGM{});
            auto a = GetTile(ta, tla::MakeCoord(mo, ko), tla::MakeShape(am, ak));
            auto b = GetTile(tb, tla::MakeCoord(ko, no), tla::MakeShape(ak, 160u));
            auto c = GetTile(tp, tla::MakeCoord(mo, no), tla::MakeShape(am, 160u));
            mm(a, b, c, GemmCoord{am, 160u, ak});
        }
        uint32_t rank = 0;
        if (reserve) {
            rank = id >= downTasks ? id - downTasks : gateCores;
        } else {
            uint32_t own = HcDownWorkFast<TM>(id, M, sk, ks);
            for (uint32_t core = 0; core < cores; ++core) {
                uint32_t work = HcDownWorkFast<TM>(core, M, sk, ks);
                if (work < own || (work == own && core < id))
                    ++rank;
            }
        }
        if (gate && rank < gateTasks) {
            using P = Gemm::MmadPingpongPreLoad<ArchTag, false, false, 1, false, 2, 2, 2, 2>;
            using L1 = tla::Shape<tla::Int<128>, tla::Int<16>, tla::Int<512>>;
            using L0 = tla::Shape<tla::Int<128>, tla::Int<16>, tla::Int<128>>;
            using Copy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LA, Element, LB, float, LC, void>;
            using MM = Gemm::Block::BlockMmadTla<P, L1, L0, Element, Element, float, void, Copy>;
            MM mm(resource);
            uint32_t k = rank % gateSplit, mo = rank / gateSplit * 128u;
            uint32_t blocks = d / 32u, base = blocks / gateSplit, extra = blocks % gateSplit;
            uint32_t ko = (k * base + (k < extra ? k : extra)) * 32u;
            uint32_t am = M - mo < 128u ? M - mo : 128u, ak = (base + (k < extra ? 1u : 0u)) * 32u;
            GlobalTensor<float> gp;
            gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gatePartial) + static_cast<int64_t>(k) * M * 32u);
            auto tp = tla::MakeTensor(gp, tla::MakeLayout<float, LC>(M, 32u), Arch::PositionGM{});
            auto a = GetTile(ta, tla::MakeCoord(mo, ko), tla::MakeShape(am, ak));
            auto b = GetTile(tg, tla::MakeCoord(ko, 0), tla::MakeShape(ak, 4u));
            auto c = GetTile(tp, tla::MakeCoord(mo, 0), tla::MakeShape(am, 4u));
            mm(a, b, c, GemmCoord{am, 4u, ak});
        }
        PipeBarrier<PIPE_ALL>();
        HcAicSet(FLAG_C0_0);
    }
    if ASCEND_IS_AIV {
        HcAivWait(FLAG_C0_0);
        HcAivSyncAll();
        GlobalTensor<float> gp;
        gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
        GlobalTensor<Element> go;
        go.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(downPreAct != nullptr ? downPreAct : c0));
        using Reduce =
            Gemm::Kernel::SplitkReduceAdd<ArchTag, float, Element, 1, 192 * 1024 / static_cast<int64_t>(sizeof(float))>;
        Reduce reduce(resource);
        reduce(go, gp, static_cast<uint64_t>(M) * r, sk);
        if (gate)
            HcReduceGateFast<Element>(resource, gatePartial, gateOut, M, gateSplit);
    }
}

template <class Element>
CATLASS_DEVICE void HcReduceGateSmall(Arch::Resource<Arch::Ascend950>& resource, GM_ADDR src, GM_ADDR dst, uint32_t M,
                                      uint32_t split, uint32_t channels)
{
    using namespace AscendC;
    uint32_t begin = 0u, end = 0u;
    HcAivRowRange(M, channels * static_cast<int64_t>(sizeof(Element)), GetBlockNum(), begin, end);
    auto acc = resource.ubBuf.template GetBufferByByte<float>(0);
    auto out = resource.ubBuf.template GetBufferByByte<Element>(4096u);
    GlobalTensor<float> gi;
    gi.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    GlobalTensor<Element> go;
    go.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(dst));
    for (uint32_t row = begin; row < end; ++row) {
        DataCopy(acc, gi[row * 32u], 32u);
        for (uint32_t k = 1; k < split; ++k) {
            DataCopy(acc[k * 32u], gi[static_cast<int64_t>(k) * M * 32u + row * 32u], 32u);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        for (uint32_t k = 1; k < split; ++k) {
            Add(acc, acc, acc[k * 32u], channels);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same_v<Element, half>) {
            Cast(out, acc, RoundMode::CAST_NONE, channels);
        } else {
            Cast(out, acc, RoundMode::CAST_RINT, channels);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams cp(1, channels * static_cast<int64_t>(sizeof(Element)), 0, 0, 0);
        DataCopyPad(go[static_cast<int64_t>(row) * channels], out, cp);
        PipeBarrier<PIPE_ALL>();
    }
}

template <uint32_t TM, class Element, class Tiling>
CATLASS_DEVICE void HcDownGateStageSmall(GM_ADDR x, GM_ADDR wd, GM_ADDR wg, GM_ADDR downPreAct, GM_ADDR gateOut,
                                         GM_ADDR c0, GM_ADDR partial, Tiling const& td)
{
    using namespace AscendC;
    using ArchTag = Arch::Ascend950;
    using LA = layout::RowMajor;
    using LB = layout::ColumnMajor;
    using LC = layout::RowMajor;
    uint32_t M = td.M, r = td.R, d = td.D, cores = td.usedAic;
    uint32_t id = GetBlockIdx();
    if ASCEND_IS_AIV {
        id = GetBlockIdx() / HcSubBlockNum();
    }
    if (id >= cores)
        return;
    bool gate = td.useCombine != 0u;
    const uint32_t nTiles = (r + 159u) / 160u;
    uint32_t mn = ((M + TM - 1u) / TM) * nTiles;
    uint32_t sk = cores / mn;
    sk = sk == 0u ? 1u : sk;
    sk = sk > 14u ? 14u : sk;
    sk = sk > d / 32u ? d / 32u : sk;
    uint32_t downTasks = mn * sk;
    uint32_t gateMn = (M + 127u) / 128u;
    uint32_t gateSplit = cores / gateMn;
    gateSplit = gateSplit == 0u ? 1u : gateSplit;
    gateSplit = gateSplit > 14u ? 14u : gateSplit;
    gateSplit = gateSplit > d / 32u ? d / 32u : gateSplit;
    uint32_t gateTasks = gateMn * gateSplit;
    GM_ADDR gatePartial = partial + static_cast<int64_t>(M) * r * static_cast<int64_t>(sizeof(float)) * td.splitkFactor;
    Arch::Resource<ArchTag> resource;
    if ASCEND_IS_AIC {
        GlobalTensor<Element> ga, gb, gg;
        ga.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x));
        gb.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(wd));
        gg.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(wg));
        auto ta = tla::MakeTensor(ga, tla::MakeLayout<Element, LA>(M, d), Arch::PositionGM{});
        auto tb = tla::MakeTensor(gb, tla::MakeLayout<Element, LB>(d, r), Arch::PositionGM{});
        auto tg = tla::MakeTensor(gg, tla::MakeLayout<Element, LB>(d, td.C), Arch::PositionGM{});
        for (uint32_t task = id; task < downTasks; task += cores) {
            using P = Gemm::MmadPingpongPreLoad<ArchTag, false, false, 1, false, 2, 2, 2, 2>;
            using L1 = tla::Shape<tla::Int<TM>, tla::Int<160>, tla::Int<256>>;
            using L0 = tla::Shape<tla::Int<TM>, tla::Int<160>, tla::Int<64>>;
            using Copy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LA, Element, LB, float, LC, void>;
            using MM = Gemm::Block::BlockMmadTla<P, L1, L0, Element, Element, float, void, Copy>;
            MM mm(resource);
            uint32_t k = task % sk, tile = task / sk;
            uint32_t mo = tile / nTiles * TM, no = tile % nTiles * 160u;
            uint32_t blocks = d / 32u, base = blocks / sk, extra = blocks % sk;
            uint32_t ko = (k * base + (k < extra ? k : extra)) * 32u;
            uint32_t ak = (base + (k < extra ? 1u : 0u)) * 32u;
            uint32_t am = M - mo < TM ? M - mo : TM;
            uint32_t an = r - no < 160u ? r - no : 160u;
            GlobalTensor<float> gp;
            gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial) + static_cast<int64_t>(k) * M * r);
            auto tp = tla::MakeTensor(gp, tla::MakeLayout<float, LC>(M, r), Arch::PositionGM{});
            auto a = GetTile(ta, tla::MakeCoord(mo, ko), tla::MakeShape(am, ak));
            auto b = GetTile(tb, tla::MakeCoord(ko, no), tla::MakeShape(ak, an));
            auto c = GetTile(tp, tla::MakeCoord(mo, no), tla::MakeShape(am, an));
            mm(a, b, c, GemmCoord{am, an, ak});
        }
        for (uint32_t rank = id; gate && rank < gateTasks; rank += cores) {
            using P = Gemm::MmadPingpongPreLoad<ArchTag, false, false, 1, false, 2, 2, 2, 2>;
            using L1 = tla::Shape<tla::Int<128>, tla::Int<16>, tla::Int<512>>;
            using L0 = tla::Shape<tla::Int<128>, tla::Int<16>, tla::Int<128>>;
            using Copy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LA, Element, LB, float, LC, void>;
            using MM = Gemm::Block::BlockMmadTla<P, L1, L0, Element, Element, float, void, Copy>;
            MM mm(resource);
            uint32_t k = rank % gateSplit, mo = rank / gateSplit * 128u;
            uint32_t blocks = d / 32u, base = blocks / gateSplit, extra = blocks % gateSplit;
            uint32_t ko = (k * base + (k < extra ? k : extra)) * 32u;
            uint32_t am = M - mo < 128u ? M - mo : 128u, ak = (base + (k < extra ? 1u : 0u)) * 32u;
            GlobalTensor<float> gp;
            gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gatePartial) + static_cast<int64_t>(k) * M * 32u);
            auto tp = tla::MakeTensor(gp, tla::MakeLayout<float, LC>(M, 32u), Arch::PositionGM{});
            auto a = GetTile(ta, tla::MakeCoord(mo, ko), tla::MakeShape(am, ak));
            auto b = GetTile(tg, tla::MakeCoord(ko, 0), tla::MakeShape(ak, td.C));
            auto c = GetTile(tp, tla::MakeCoord(mo, 0), tla::MakeShape(am, td.C));
            mm(a, b, c, GemmCoord{am, td.C, ak});
        }
        PipeBarrier<PIPE_ALL>();
        HcAicSet(FLAG_C0_0);
    }
    if ASCEND_IS_AIV {
        HcAivWait(FLAG_C0_0);
        HcAivSyncAll();
        GlobalTensor<float> gp;
        gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
        GlobalTensor<Element> go;
        go.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(downPreAct != nullptr ? downPreAct : c0));
        using Reduce =
            Gemm::Kernel::SplitkReduceAdd<ArchTag, float, Element, 1, 192 * 1024 / static_cast<int64_t>(sizeof(float))>;
        Reduce reduce(resource);
        reduce(go, gp, static_cast<uint64_t>(M) * r, sk);
        if (gate)
            HcReduceGateSmall<Element>(resource, gatePartial, gateOut, M, gateSplit, td.C);
    }
}

template <class Element, uint32_t NH, uint32_t TM, uint32_t LK, uint32_t L0K, bool Resident>
CATLASS_DEVICE void HcC1TileEpilogue(GM_ADDR act, GM_ADDR wu, GM_ADDR up, GM_ADDR xn, GM_ADDR mixed, uint32_t M,
                                     uint32_t r, uint32_t d, uint32_t hidden, uint32_t cores, uint32_t channels = 4u)
{
    using namespace AscendC;
    using ArchTag = Arch::Ascend950;
    using LA = layout::RowMajor;
    using LB = layout::ColumnMajor;
    using LC = layout::RowMajor;
    uint32_t id = GetBlockIdx();
    if ASCEND_IS_AIV {
        id = GetBlockIdx() / HcSubBlockNum();
    }
    Arch::Resource<ArchTag> resource;
    using P = Gemm::MmadPingpongPreLoad<ArchTag, false, false, Resident ? 2 : 1, Resident, 2, Resident ? 1 : 2, 2, 2>;
    using L1 = tla::Shape<tla::Int<TM>, tla::Int<NH>, tla::Int<LK>>;
    using L0 = tla::Shape<tla::Int<TM>, tla::Int<NH>, tla::Int<L0K>>;
    using Copy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LA, Element, LB, Element, LC, void>;
    using MM = Gemm::Block::BlockMmadTla<P, L1, L0, Element, Element, Element, void, Copy>;
    MM mm(resource);
    Epilogue::BlockEpilogueHcV2ChannelTile<Element, NH> epi;
    GlobalTensor<Element> ga, gb, gc;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(act));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(wu));
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(up));
    auto ta = tla::MakeTensor(ga, tla::MakeLayout<Element, LA>(M, r), Arch::PositionGM{});
    auto tb = tla::MakeTensor(gb, tla::MakeLayout<Element, LB>(r, d), Arch::PositionGM{});
    auto tc = tla::MakeTensor(gc, tla::MakeLayout<Element, LC>(M, d), Arch::PositionGM{});
    if ASCEND_IS_AIV {
        HcAivSet(7u);
        HcAivSet(8u);
    }
    for (uint32_t hb = id; hb < (hidden + NH - 1u) / NH; hb += cores) {
        uint32_t h0 = hb * NH, hn = hidden - h0 < NH ? hidden - h0 : NH;
        for (uint32_t mo = 0; mo < M; mo += TM) {
            uint32_t am = M - mo < TM ? M - mo : TM;
            for (uint32_t c = 0; c < channels; ++c) {
                uint32_t slot = c & 1u;
                if ASCEND_IS_AIC {
                    HcAicWait(7u + slot);
                    auto a = GetTile(ta, tla::MakeCoord(mo, 0), tla::MakeShape(am, r));
                    auto b = GetTile(tb, tla::MakeCoord(0, c * hidden + h0), tla::MakeShape(r, hn));
                    auto out = GetTile(tc, tla::MakeCoord(mo, c * hidden + h0), tla::MakeShape(am, hn));
                    mm(a, b, out, GemmCoord{am, hn, r});
                    PipeBarrier<PIPE_ALL>();
                    HcAicSet(5u + slot);
                }
                if ASCEND_IS_AIV {
                    HcAivWait(5u + slot);
                    epi.Consume(resource, up, xn, mixed, mo, am, h0, hn, c, d, hidden, channels);
                    HcAivSet(7u + slot);
                }
            }
        }
    }
    if ASCEND_IS_AIC {
        HcAicWait(7u);
        HcAicWait(8u);
    }
    PipeBarrier<PIPE_ALL>();
}

template <class Element, class Tiling>
CATLASS_DEVICE void HcSmallFusedGeneric(GM_ADDR x, GM_ADDR wd, GM_ADDR wu, GM_ADDR wg, GM_ADDR gamma, GM_ADDR mixed,
                                        GM_ADDR inject, GM_ADDR invRms, GM_ADDR downCache, GM_ADDR upCache, GM_ADDR ws,
                                        Tiling const& td)
{
    using namespace AscendC;
    using ArchTag = Arch::Ascend950;
    const uint32_t M = td.M, d = td.D, r = td.R, c = td.C, cores = td.usedAic;
    const bool gate = td.useCombine != 0;
    GM_ADDR xn = ws;
    GM_ADDR act = xn + static_cast<int64_t>(M) * d * static_cast<int64_t>(sizeof(Element));
    // Physical workspace addresses never derive from external cache outputs.
    GM_ADDR workspaceRawBase = act + static_cast<int64_t>(M) * r * static_cast<int64_t>(sizeof(Element));
    GM_ADDR rawGate = workspaceRawBase + static_cast<int64_t>(M) * r * static_cast<int64_t>(sizeof(Element));
    GM_ADDR partial = workspaceRawBase + 2ul * M * (r + td.nWsGate) * static_cast<int64_t>(sizeof(Element));
    GM_ADDR rawDownRead = downCache != nullptr ? downCache : workspaceRawBase;
    GM_ADDR up = upCache != nullptr ?
                     upCache :
                     partial + 2ul * M * (r + td.nWsGate) * static_cast<int64_t>(sizeof(float)) * td.splitkFactor;
    Arch::Resource<ArchTag> resource;
    if ASCEND_IS_AIV {
        uint32_t begin = 0, end = 0;
        HcAivRowRange(
            M, invRms != nullptr ? c * static_cast<int64_t>(sizeof(float)) : d * static_cast<int64_t>(sizeof(Element)),
            cores, begin, end);
        if (begin < end) {
            Epilogue::BlockEpilogueHcV0<Element> v0;
            v0(resource, x + static_cast<int64_t>(begin) * d * static_cast<int64_t>(sizeof(Element)),
               xn + static_cast<int64_t>(begin) * d * static_cast<int64_t>(sizeof(Element)), inject, wg, gamma,
               end - begin, d, c, td.H, td.eps, gate, td.hasGamma != 0, true,
               invRms == nullptr ? nullptr :
                                   invRms + static_cast<int64_t>(begin) * c * static_cast<int64_t>(sizeof(float)),
               false);
        }
        HcAivSyncAll();
        HcAivSet(FLAG_V0);
    }
    if ASCEND_IS_AIC {
        HcAicWait(FLAG_V0);
    }
    if (M <= 256u)
        HcDownGateStageSmall<128u, Element>(xn, wd, wg, rawDownRead, rawGate, workspaceRawBase, partial, td);
    else
        HcDownGateStageSmall<256u, Element>(xn, wd, wg, rawDownRead, rawGate, workspaceRawBase, partial, td);
    if ASCEND_IS_AIV {
        HcAivSyncAll();
        uint32_t begin = 0, end = 0;
        HcAivRowRange(M, gate ? c * static_cast<int64_t>(sizeof(Element)) : r * static_cast<int64_t>(sizeof(Element)),
                      cores, begin, end);
        if (begin < end) {
            Epilogue::BlockEpilogueHcV1<Arch::Ascend950, false> v1;
            v1.template operator()<Arch::Resource<ArchTag>, Element>(
                resource, rawDownRead + static_cast<int64_t>(begin) * r * static_cast<int64_t>(sizeof(Element)),
                rawGate + static_cast<int64_t>(begin) * c * static_cast<int64_t>(sizeof(Element)),
                act + static_cast<int64_t>(begin) * r * static_cast<int64_t>(sizeof(Element)),
                inject == nullptr ? nullptr :
                                    inject + static_cast<int64_t>(begin) * c * static_cast<int64_t>(sizeof(Element)),
                end - begin, r, gate ? c : 0u, r, c, gate, true, nullptr);
        }
        HcAivSyncAll();
        HcAivSet(FLAG_V1);
    }
    if ASCEND_IS_AIC {
        HcAicWait(FLAG_V1);
    }
    if (M <= 128u)
        HcC1TileEpilogue<Element, 96u, 128u, 128u, 32u, false>(act, wu, up, xn, mixed, M, r, d, td.H, cores, c);
    else
        HcC1TileEpilogue<Element, 96u, 256u, 320u, 64u, true>(act, wu, up, xn, mixed, M, r, d, td.H, cores, c);
    if ASCEND_IS_AIV {
        HcAivSyncAll();
    }
}

template <class Element, class Tiling>
CATLASS_DEVICE void HcSmallFusedFast(GM_ADDR x, GM_ADDR wd, GM_ADDR wu, GM_ADDR wg, GM_ADDR gamma, GM_ADDR mixed,
                                     GM_ADDR inject, GM_ADDR invRms, GM_ADDR downCache, GM_ADDR upCache, GM_ADDR ws,
                                     Tiling const& td)
{
    if constexpr (std::is_same_v<Element, bfloat16_t>) {
        using namespace AscendC;
        using ArchTag = Arch::Ascend950;
        const uint32_t M = td.M, d = td.D, r = td.R, c = td.C, cores = td.usedAic;
        const bool gate = td.useCombine != 0;
        GM_ADDR xn = ws;
        GM_ADDR act = xn + static_cast<int64_t>(M) * d * static_cast<int64_t>(sizeof(Element));
        // Physical workspace addresses never derive from external cache outputs.
        GM_ADDR workspaceRawBase = act + static_cast<int64_t>(M) * r * static_cast<int64_t>(sizeof(Element));
        GM_ADDR rawGate = workspaceRawBase + static_cast<int64_t>(M) * r * static_cast<int64_t>(sizeof(Element));
        GM_ADDR partial = workspaceRawBase + 2ul * M * (r + td.nWsGate) * static_cast<int64_t>(sizeof(Element));
        GM_ADDR rawDownRead = downCache != nullptr ? downCache : workspaceRawBase;
        GM_ADDR up = upCache != nullptr ?
                         upCache :
                         partial + 2ul * M * (r + td.nWsGate) * static_cast<int64_t>(sizeof(float)) * td.splitkFactor;
        Arch::Resource<ArchTag> resource;
        if ASCEND_IS_AIV {
            uint32_t begin = 0, end = 0;
            HcAivRowRange(
                M,
                invRms != nullptr ? c * static_cast<int64_t>(sizeof(float)) : d * static_cast<int64_t>(sizeof(Element)),
                cores, begin, end);
            if (begin < end) {
                Epilogue::BlockEpilogueHcV0<Element> v0;
                v0(resource, x + static_cast<int64_t>(begin) * d * static_cast<int64_t>(sizeof(Element)),
                   xn + static_cast<int64_t>(begin) * d * static_cast<int64_t>(sizeof(Element)), inject, wg, gamma,
                   end - begin, d, c, td.H, td.eps, gate, td.hasGamma != 0, true,
                   invRms == nullptr ? nullptr :
                                       invRms + static_cast<int64_t>(begin) * c * static_cast<int64_t>(sizeof(float)),
                   false);
            }
            HcAivSyncAll();
            HcAivSet(FLAG_V0);
        }
        if ASCEND_IS_AIC {
            HcAicWait(FLAG_V0);
        }
        auto local = td;
        uint32_t tm = M <= 256u ? 128u : 256u, tail = M % tm;
        bool reserve = M <= 128u;
        uint32_t budget = reserve ? 8u : (tail > 0u && tail <= tm / 2u ? 14u : 28u);
        local.pad0 = (budget << 8u) | (reserve ? 1u : 0u);
        if (M <= 256u)
            HcDownGateStageFast<128u, Element>(xn, wd, wg, rawDownRead, rawGate, workspaceRawBase, partial, local);
        else
            HcDownGateStageFast<256u, Element>(xn, wd, wg, rawDownRead, rawGate, workspaceRawBase, partial, local);
        if ASCEND_IS_AIV {
            HcAivSyncAll();
            uint32_t begin = 0, end = 0;
            HcAivRowRange(M,
                          gate ? c * static_cast<int64_t>(sizeof(Element)) : r * static_cast<int64_t>(sizeof(Element)),
                          cores, begin, end);
            if (begin < end) {
                Epilogue::BlockEpilogueHcV1<Arch::Ascend950, false> v1;
                v1.template operator()<Arch::Resource<ArchTag>, Element>(
                    resource, rawDownRead + static_cast<int64_t>(begin) * r * static_cast<int64_t>(sizeof(Element)),
                    rawGate + static_cast<int64_t>(begin) * c * static_cast<int64_t>(sizeof(Element)),
                    act + static_cast<int64_t>(begin) * r * static_cast<int64_t>(sizeof(Element)),
                    inject == nullptr ?
                        nullptr :
                        inject + static_cast<int64_t>(begin) * c * static_cast<int64_t>(sizeof(Element)),
                    end - begin, r, gate ? 4u : 0u, r, c, gate, true, nullptr);
            }
            HcAivSyncAll();
            HcAivSet(FLAG_V1);
        }
        if ASCEND_IS_AIC {
            HcAicWait(FLAG_V1);
        }
        if (M <= 128u) {
            HcC1TileEpilogue<Element, 96u, 128u, 128u, 32u, false>(act, wu, up, xn, mixed, M, r, d, td.H, cores);
        } else {
            HcC1TileEpilogue<Element, 96u, 256u, 320u, 64u, true>(act, wu, up, xn, mixed, M, r, d, td.H, cores);
        }
        if ASCEND_IS_AIV {
            HcAivSyncAll();
        }
    }
}
template <class Element, class Tiling>
CATLASS_DEVICE void HcSmallFused(GM_ADDR x, GM_ADDR wd, GM_ADDR wu, GM_ADDR wg, GM_ADDR gamma, GM_ADDR mixed,
                                 GM_ADDR inject, GM_ADDR invRms, GM_ADDR downCache, GM_ADDR upCache, GM_ADDR ws,
                                 Tiling const& td)
{
    if constexpr (std::is_same_v<Element, bfloat16_t>) {
        if (td.M < 480u && td.C == 4u && td.R == 320u && td.H == 2560u && td.usedAic == 28u) {
            HcSmallFusedFast<Element>(x, wd, wu, wg, gamma, mixed, inject, invRms, downCache, upCache, ws, td);
            return;
        }
    }
    HcSmallFusedGeneric<Element>(x, wd, wu, wg, gamma, mixed, inject, invRms, downCache, upCache, ws, td);
}

} // namespace Catlass::Gemm::Kernel

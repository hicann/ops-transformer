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

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include <type_traits>
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../block/block_epilogue_hc_v0.hpp"
#include "../block/block_epilogue_hc_v1.hpp"
#include "../block/block_epilogue_hc_v2.hpp"
#include "../../catlass_hc_forward_mul_sum_tiling_data.h"

namespace Catlass::Gemm::Kernel {
constexpr uint32_t HC_A2_VECTOR_SUBBLOCK_COUNT = 2;
constexpr uint32_t HC_A2_SWIZZLE_TILE_GROUP = 3;
constexpr uint32_t HC_A2_L1_A_STAGES = 2;
constexpr uint32_t HC_A2_L1_B_STAGES = 2;
constexpr uint32_t HC_A2_L0_A_STAGES = 2;

constexpr int64_t HC_SYNC_MODE = 2;
constexpr int64_t FLAG_V0 = 0;
constexpr int64_t FLAG_C0_0 = 1;
constexpr int64_t FLAG_C0_1 = 2;
constexpr int64_t FLAG_V1 = 3;
constexpr int64_t FLAG_C1_0 = 4; // C1[k] Set slot=k&1；V2[k] Wait 同槽（delay-2，禁止与 C1[k+1] 串同一 flag）
constexpr int64_t FLAG_C1_1 = 5;
constexpr int64_t HC_FLAG_PAIR = 16;

// modeId=4：AIV0 的 flag N 对应 AIC N，AIV1 的同一 N 由硬件映射到 AIC N+16。
// 对齐 example 49 fai_kernel.h 与 grouped_matmul_slice_m_gelu.hpp：
//   AIV Set/Wait 只用基 id；AIC 连发/连等 id 与 id+16。
// 仅当 MIX 未拉起第二颗 AIV（GetSubBlockNum()<2）时，AIV0 空转补 +16，避免 AIC Wait(16+id) 挂死。
CATLASS_DEVICE
uint32_t HcSubBlockNum()
{
    uint32_t n = AscendC::GetSubBlockNum();
    return n == 0 ? 1 : n;
}

CATLASS_DEVICE
bool HcNeedDummyPair()
{
    return HcSubBlockNum() < HC_A2_VECTOR_SUBBLOCK_COUNT && AscendC::GetSubBlockIdx() == 0;
}

CATLASS_DEVICE
void HcAivSet(int64_t base)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::CrossCoreSetFlag<HC_SYNC_MODE, PIPE_MTE3>(base);
}

CATLASS_DEVICE
void HcAivWait(int64_t base)
{
    AscendC::CrossCoreWaitFlag<HC_SYNC_MODE, PIPE_V>(base);
}

CATLASS_DEVICE
void HcAicSet(int64_t base)
{
    AscendC::CrossCoreSetFlag<HC_SYNC_MODE, PIPE_FIX>(base);
}

CATLASS_DEVICE
void HcAicWait(int64_t base)
{
    AscendC::CrossCoreWaitFlag<HC_SYNC_MODE, PIPE_MTE1>(base);
}

// 仅 AIV 调用。硬同步占 flagId=14，禁止 AIC / SyncAll<false>。
CATLASS_DEVICE
void HcAivSyncAll()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<true>();
}

// 每个 AIV 领一段连续行。rowBytes 不是 32 的倍数时，一段至少盖住一个 32B 块，避免两颗 AIV 改同一块 inject。
CATLASS_DEVICE
void HcAivContig(uint32_t actualM, uint32_t rowBytes, uint32_t& begin, uint32_t& end)
{
    uint32_t grain = 1;
    uint32_t rem = rowBytes & 31u;
    if (rem != 0u) {
        grain = (32u % rem) == 0u ? (32u / rem) : 32u;
    }
    uint32_t id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    begin = id * grain;
    if (begin >= actualM) {
        begin = actualM;
        end = actualM;
        return;
    }
    end = begin + grain;
    if (end > actualM) {
        end = actualM;
    }
}

// 一行字节数不是 32 的倍数时，相邻行共用一个 32B GM 块。A2 上两颗 AIV 同时读改写会丢更新。
CATLASS_DEVICE
void HcAivRowRange(uint32_t actualM, uint32_t rowBytes, uint32_t usedAic, uint32_t& begin, uint32_t& end)
{
    uint32_t total = usedAic * HcSubBlockNum();
    if (total == 0)
        total = 1;
    uint32_t grain = 1;
    while (((rowBytes * grain) & 31u) != 0u && grain < 32u)
        grain *= 2u;
    uint32_t groups = (actualM + grain - 1u) / grain;
    uint32_t id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    begin = (groups * id / total) * grain;
    end = (groups * (id + 1u) / total) * grain;
    begin = begin < actualM ? begin : actualM;
    end = end < actualM ? end : actualM;
}

template <class Element, uint32_t C1SwizzleDir, bool SmallN = false, class ArchTag_ = Arch::Ascend950,
          class Tiling = HcForwardMulSumTilingData>
struct HcForwardCommon {
    using ArchTag = ArchTag_;
    static constexpr bool kIsA2 = std::is_same_v<ArchTag, Arch::AtlasA2>;
    // C1 L0 32×256×32: L0B = 32*256*4*stages; stages=2 → 64KB. PreLoad unitFlag 时 L0C 必须保持 1.
    static constexpr uint32_t L0B_STAGES_C1 = 2;
    static constexpr bool ENABLE_UNIT_FLAG_C1 = true;
    // A2 small-N 的 M 往往小于 16。unit flag 按分形搬出时最后一行会丢，inject 对不齐。
    static constexpr bool kSmallNC0UnitFlag = !kIsA2;
    static constexpr uint32_t L0B_STAGES_C0 = 2u;
    static constexpr bool ENABLE_UNIT_FLAG_C0 = SmallN ? kSmallNC0UnitFlag : false;
    // A2 小路径 mTask=128 时 C0 有 4×6=24 块（32×64）：host 侧 usedAic 已计入 mTile，
    // 24 块摊到 20 个 AIC；gate-only 块排在队首由低号核算，down 峰值 2 块/核。
    static constexpr uint32_t L1M0 = SmallN ? ((kIsA2) ? 32u : HC_C0_L1M_SMALL) : 128;
    static constexpr uint32_t kA2SmallN0 = 64u;
    static constexpr uint32_t L1N0 = SmallN ? (kIsA2 ? kA2SmallN0 : HC_C0_L1N0_SMALL) : (kIsA2 ? 256u : HC_L1N0_LARGE);
    // MatmulV3 大 K：baseK=64，stepK=4，L1 上一次带 256 的 K，双缓冲即 depth=8。
    static constexpr uint32_t L1K0 = SmallN ? HC_L1K0_SMALL_F16 : (kIsA2 ? 256u : 128u);
    // C1 tile M：A2 bf16 下 K=320 全载 L1。L1M1=32 时每核每 wave 发 8 个小 m-tile 的
    // mmad+fixpipe（16KB/次），发射开销主导。升到 128：L1 占用 2*80K(A)+2*160K(B)=480K/512K
    // 仍可行，fixpipe 粒度 64KB、次数 /4。fp32 与非 A2 保持原值。
    static constexpr uint32_t L1M1 = (SmallN && kIsA2) ? 128u : (SmallN ? HC_L1M_SMALL : 128);
    static constexpr uint32_t L1N1 = HC_L1N1;
    // C1 的 K 是 R=320，A2 上整段放进 L1，避免每个 N 块把 A 再搬一遍。
    static constexpr uint32_t L1K1 = (kIsA2) ? 320u : HC_C1_L1K_F16;
    static constexpr uint32_t L0K0 = SmallN ? HC_L0K0_SMALL : ((kIsA2) ? 64u : 32u);
    static constexpr uint32_t L0K1 = (kIsA2) ? 64u : 32u;
    static_assert(!SmallN || kIsA2 || (L1M0 == HC_C0_L1M_SMALL && L1N0 == HC_C0_L1N0_SMALL), "C0 L1_M/L1_N knobs");
    static_assert(L1M0 != 0 && L1N0 != 0, "C0 tile nonempty");
    static_assert(L1K0 % L0K0 == 0, "C0 L1.K must be a multiple of L0.K");
    static_assert(L0K0 * L1N0 * static_cast<int64_t>(sizeof(Element)) * L0B_STAGES_C0 <= 64u * 1024u,
                  "C0 L0B capacity");
    static_assert(L0K1 * L1N1 * static_cast<int64_t>(sizeof(Element)) * L0B_STAGES_C1 <= 64u * 1024u,
                  "C1 L0B capacity");

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;

    using DispatchPolicyC0 = Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG_C0, false, 1, false, HC_A2_L1_A_STAGES,
                                                HC_A2_L1_B_STAGES, HC_A2_L0_A_STAGES, L0B_STAGES_C0>;
    // C1 的 K=R 已整段在 L1。同一行的 A 在 N 块之间驻留，只重搬 B。
    // 两级 L1A 留给两条 128 行。只有一条行带时第二片 N 会落到空槽，改用一级。
    using DispatchPolicyC1 = Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG_C1, false, 1, true, HC_A2_L1_A_STAGES,
                                                HC_A2_L1_B_STAGES, HC_A2_L0_A_STAGES, L0B_STAGES_C1>;
    using DispatchPolicyC1A1 = Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG_C1, false, 1, true, 1, HC_A2_L1_B_STAGES,
                                                  HC_A2_L0_A_STAGES, L0B_STAGES_C1>;

    using L1C0 = tla::Shape<tla::Int<L1M0>, tla::Int<L1N0>, tla::Int<L1K0>>;
    using L0C0 = tla::Shape<tla::Int<L1M0>, tla::Int<L1N0>, tla::Int<L0K0>>;
    using L1C1 = tla::Shape<tla::Int<L1M1>, tla::Int<L1N1>, tla::Int<L1K1>>;
    using L0C1 = tla::Shape<tla::Int<L1M1>, tla::Int<L1N1>, tla::Int<L0K1>>;

    using TileCopyC0Packed =
        Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutTagA, Element, LayoutTagB, Element, LayoutTagC, void>;
    using TileCopyC0 = TileCopyC0Packed;
    using TileCopyC1 = TileCopyC0Packed;
    using BlockMmadC0 =
        Gemm::Block::BlockMmadTla<DispatchPolicyC0, L1C0, L0C0, Element, Element, Element, void, TileCopyC0>;
    using BlockMmadC1 =
        Gemm::Block::BlockMmadTla<DispatchPolicyC1, L1C1, L0C1, Element, Element, Element, void, TileCopyC1>;
    using BlockMmadC1A1 =
        Gemm::Block::BlockMmadTla<DispatchPolicyC1A1, L1C1, L0C1, Element, Element, Element, void, TileCopyC1>;
    using BlockSchedulerC1 = Gemm::Block::GemmIdentityBlockSwizzle<HC_A2_SWIZZLE_TILE_GROUP, C1SwizzleDir>;

    struct Params {
        GM_ADDR ptrX;
        GM_ADDR ptrWDown;
        GM_ADDR ptrWUp;
        GM_ADDR ptrWGate;
        GM_ADDR ptrGamma;
        GM_ADDR ptrMixed;
        GM_ADDR ptrInject;
        GM_ADDR ptrInvRms;
        GM_ADDR ptrDownPreAct;
        GM_ADDR ptrUpPreAct;
        GM_ADDR userWs;
        GM_ADDR tiling;
        // 1：registry GET_TILING_DATA 已解包；0：直调 POD GM（.asc memcpy）。
        uint32_t tilingFromUnpacked;
        const Tiling* tilingUnpacked;
    };

    template <class Mmad, class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void RunMmad(Mmad& mmad, TensorA const& tensorA, TensorB const& tensorB, TensorC const& tensorC,
                                uint32_t mOff, uint32_t nOffB, uint32_t nOffC, uint32_t actualM, uint32_t actualN,
                                uint32_t kFull)
    {
        auto tileC = GetTile(tensorC, tla::MakeCoord(mOff, nOffC), tla::MakeShape(actualM, actualN));
        auto tileA = GetTile(tensorA, tla::MakeCoord(mOff, 0), tla::MakeShape(actualM, kFull));
        auto tileB = GetTile(tensorB, tla::MakeCoord(0, nOffB), tla::MakeShape(kFull, actualN));
        mmad(tileA, tileB, tileC, GemmCoord{actualM, actualN, kFull});
    }

    // kTileM/kTileN 默认就是 A2 小路径的 32×64。超过 128 行改走 64×80：
    // down 的 N 能被 80 整除时 gate 单独占核。块数要两趟、改成 64×128 能收成一趟时才换。
    // 128 列时最后一块跨过 R，gate 并进这块 down，不再单独占一趟 K。fp32 的 L0B 放不下 128 列。
    CATLASS_DEVICE bool C0PreferN128(uint32_t actualM, uint32_t nConcat, uint32_t usedAic) const
    {
        if (actualM <= 128u || usedAic == 0u || nConcat == 0u) {
            return false;
        }
        uint32_t mTiles = (actualM + 63u) / 64u;
        uint32_t n80 = (nConcat + 79u) / 80u;
        uint32_t n128 = (nConcat + 127u) / 128u;
        uint32_t pass80 = (mTiles * n80 + usedAic - 1u) / usedAic;
        uint32_t pass128 = (mTiles * n128 + usedAic - 1u) / usedAic;
        return pass128 < pass80;
    }
    // 不超过 32 行时只有一条 M 块。32×64 对 R=320 只有 6 块，占不满 20 核。
    // 收到 32×32 是 10 块 down 加 1 块 gate，仍是一趟 K。再高的段收成 32 会变成两趟，不收。
    CATLASS_DEVICE bool C0PreferN32(uint32_t actualM, uint32_t nConcat, uint32_t usedAic) const
    {
        if (actualM == 0u || actualM > 32u || usedAic == 0u || nConcat == 0u) {
            return false;
        }
        uint32_t n64 = (nConcat + 63u) / 64u;
        uint32_t n32 = (nConcat + 31u) / 32u;
        if (n64 >= usedAic || n32 <= n64) {
            return false;
        }
        uint32_t pass32 = (n32 + usedAic - 1u) / usedAic;
        uint32_t pass64 = (n64 + usedAic - 1u) / usedAic;
        return pass32 == pass64;
    }
    CATLASS_DEVICE
    HcForwardCommon() {}

    template <class Body>
    CATLASS_DEVICE void Execute(Params const& params)
    {
        using namespace AscendC;
        // 禁止 tdGm->field：3510 对 tiling GM 做结构体标量读会 264。
        // registry 入口必须 GET_TILING_DATA 解包后置 tilingFromUnpacked=1。
        if (params.tilingFromUnpacked == 0 || params.tilingUnpacked == nullptr) {
            return;
        }
        const Tiling* td = params.tilingUnpacked;
        uint32_t usedAic = td->usedAic;
        uint32_t mTask = td->mTask;
        uint32_t taskNum = td->taskNum;
        uint32_t d = td->D;
        uint32_t r = td->R;
        uint32_t hidden = td->H;
        uint32_t channels = td->C;
        uint32_t M = td->M;
        float eps = td->eps;
        bool useCombine = td->useCombine != 0;
        bool hasGamma = td->hasGamma != 0;
        uint32_t szT = td->szT;

        uint32_t nConcat = td->nConcat;
        uint32_t nWs = td->nWs;
        if (nConcat == 0) {
            nConcat = useCombine ? (r + channels) : r;
        }
        if (nConcat > r && channels > 0) {
            useCombine = true;
        }
        if (nWs == 0) {
            nWs = ((nConcat + 15u) / 16u) * 16u;
        }

        uint32_t blockIdx = 0;
        if ASCEND_IS_AIC {
            blockIdx = GetBlockIdx();
        } else {
            blockIdx = GetBlockIdx() / HcSubBlockNum();
        }
        if (blockIdx >= usedAic) {
            return;
        }

        Arch::Resource<ArchTag> resource;

        GM_ADDR ws = params.userWs;
        const int64_t xnSlotBytes = static_cast<int64_t>(mTask) * d * szT;
        const int64_t downSlotBytes = static_cast<int64_t>(mTask) * r * szT;
        // 正常大路径 xn 3 拍、down 2 拍，按核环形槽。小路径按全量 M。
        int64_t s0;
        int64_t s1;
        if constexpr (SmallN) {
            s0 = static_cast<int64_t>(M) * d * szT;
            s1 = static_cast<int64_t>(M) * r * szT;
        } else {
            s0 = static_cast<int64_t>(usedAic) * 3u * xnSlotBytes;
            s1 = static_cast<int64_t>(usedAic) * 2u * downSlotBytes;
        }
        int64_t s2 = 0;
        const uint32_t gateCols = (useCombine && channels > 0) ? channels : 0u;
        const uint32_t slotRows = mTask;
        const int64_t c0SlotBytes = static_cast<int64_t>(slotRows) * (static_cast<int64_t>(r) + gateCols) *
                                    static_cast<int64_t>(sizeof(Element));
        if constexpr (SmallN) {
            s2 = static_cast<int64_t>(2) * c0SlotBytes;
        } else {
            s2 = static_cast<int64_t>(usedAic) * c0SlotBytes * 2;
        }
        GM_ADDR ptrXn = ws;
        GM_ADDR ptrDown = ws + s0;
        GM_ADDR ptrC0Base = ws + s0 + s1;
        GM_ADDR ptrAfterC0 = ptrC0Base + s2;
        // C1 writes the backward cache directly; inference keeps private workspace.
        GM_ADDR ptrUp = params.ptrUpPreAct != nullptr ? params.ptrUpPreAct : ptrAfterC0;
        GM_ADDR ptrC0Core = SmallN ? ptrC0Base : (ptrC0Base + static_cast<int64_t>(blockIdx) * c0SlotBytes * 2);

        GlobalTensor<Element> gmX;
        gmX.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrX));
        GlobalTensor<Element> gmXn;
        gmXn.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrXn));
        GlobalTensor<Element> gmDown;
        gmDown.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrDown));
        GlobalTensor<Element> gmWDown;
        gmWDown.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWDown));
        GlobalTensor<Element> gmWUp;
        gmWUp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWUp));
        GlobalTensor<Element> gmWGate;
        gmWGate.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWGate));
        GlobalTensor<Element> gmC0;
        gmC0.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrC0Core));
        GlobalTensor<Element> gmUp;
        gmUp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrUp));
        // up_pre_act 是反向专属保存：C1 写入后仅 V2 两 wave 后流式读一次，无前向复用，
        // 大 shape 不占 L2（与 x / down_pre_act / xn 同策略），把容量留给权重。
        // 仅大 shape 关闭：小 shape 的 up_pre_act 只有 KB~10MiB，对 L2 无压力，
        // 实测 M=32~512 全档噪声内（m128 +3.4% 偏劣），smallN 路径不设置。
        if (params.ptrUpPreAct != nullptr && td->smallN == 0u) {
            gmUp.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        auto layoutXnFull = tla::MakeLayout<Element, LayoutTagA>(M, d);
        auto layoutDownFull = tla::MakeLayout<Element, LayoutTagA>(M, r);
        auto layoutWDown = tla::MakeLayout<Element, LayoutTagB>(d, r);
        auto layoutWUp = tla::MakeLayout<Element, LayoutTagB>(r, d);
        auto layoutWGate = tla::MakeLayout<Element, LayoutTagB>(d, channels);
        auto tensorX = tla::MakeTensor(gmX, layoutXnFull, Arch::PositionGM{});
        auto tensorXn = tla::MakeTensor(gmXn, layoutXnFull, Arch::PositionGM{});
        auto tensorDown = tla::MakeTensor(gmDown, layoutDownFull, Arch::PositionGM{});
        auto tensorWDown = tla::MakeTensor(gmWDown, layoutWDown, Arch::PositionGM{});
        auto tensorWGate = tla::MakeTensor(gmWGate, layoutWGate, Arch::PositionGM{});
        auto tensorWUp = tla::MakeTensor(gmWUp, layoutWUp, Arch::PositionGM{});
        auto layoutUpFull = tla::MakeLayout<Element, LayoutTagC>(M, d);
        auto tensorUp = tla::MakeTensor(gmUp, layoutUpFull, Arch::PositionGM{});

        uint32_t localTasks = 0;
        if constexpr (SmallN) {
            localTasks = taskNum;
        } else if (usedAic > 0 && blockIdx < taskNum) {
            localTasks = (taskNum - blockIdx + usedAic - 1u) / usedAic;
        }

        Epilogue::BlockEpilogueHcV0<Element> blockV0;
        Epilogue::BlockEpilogueHcV1<Element> blockV1;
        Epilogue::BlockEpilogueHcV2<Element> blockV2;
        blockV0.Init(params.ptrX, ptrXn, hasGamma ? params.ptrGamma : nullptr, params.ptrInvRms, !SmallN, !SmallN);
        blockV1.Init(ptrC0Core, ptrDown, params.ptrInject, params.ptrDownPreAct, !SmallN, channels, r);
        blockV2.Init(ptrUp, ptrXn, params.ptrMixed, params.ptrUpPreAct != nullptr && !SmallN, !SmallN);

        const uint32_t nSliceC1 = (d + L1N1 - 1) / L1N1;

        struct ExecutionContext {
            decltype(params)& f_params;
            decltype(usedAic)& f_usedAic;
            decltype(mTask)& f_mTask;
            decltype(d)& f_d;
            decltype(r)& f_r;
            decltype(hidden)& f_hidden;
            decltype(channels)& f_channels;
            decltype(M)& f_M;
            decltype(eps)& f_eps;
            decltype(useCombine)& f_useCombine;
            decltype(szT)& f_szT;
            decltype(nConcat)& f_nConcat;
            decltype(nWs)& f_nWs;
            decltype(blockIdx)& f_blockIdx;
            decltype(resource)& f_resource;
            decltype(xnSlotBytes)& f_xnSlotBytes;
            decltype(downSlotBytes)& f_downSlotBytes;
            decltype(c0SlotBytes)& f_c0SlotBytes;
            decltype(ptrXn)& f_ptrXn;
            decltype(ptrDown)& f_ptrDown;
            decltype(ptrC0Core)& f_ptrC0Core;
            decltype(tensorXn)& f_tensorXn;
            decltype(tensorDown)& f_tensorDown;
            decltype(tensorWDown)& f_tensorWDown;
            decltype(tensorWGate)& f_tensorWGate;
            decltype(tensorWUp)& f_tensorWUp;
            decltype(tensorUp)& f_tensorUp;
            decltype(localTasks)& f_localTasks;
            decltype(blockV0)& f_blockV0;
            decltype(blockV1)& f_blockV1;
            decltype(blockV2)& f_blockV2;
            decltype(nSliceC1)& f_nSliceC1;
        };
        ExecutionContext context{params,   usedAic,     mTask,         d,           r,         hidden,   channels,
                                 M,        eps,         useCombine,    szT,         nConcat,   nWs,      blockIdx,
                                 resource, xnSlotBytes, downSlotBytes, c0SlotBytes, ptrXn,     ptrDown,  ptrC0Core,
                                 tensorXn, tensorDown,  tensorWDown,   tensorWGate, tensorWUp, tensorUp, localTasks,
                                 blockV0,  blockV1,     blockV2,       nSliceC1};
        Body body;
        body.RunBody(context);
    }
};

} // namespace Catlass::Gemm::Kernel

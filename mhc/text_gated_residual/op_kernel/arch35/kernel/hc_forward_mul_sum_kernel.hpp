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

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include <type_traits>
#include "catlass/gemm/block/block_mmad.hpp"
#include "../catlass_ext/hc_dispatch_policy.hpp"
#include "../catlass_ext/block_mmad_pingpong_preload_tla.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/splitk_matmul.hpp"
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
constexpr uint32_t HC_A5_VECTOR_SUBBLOCK_COUNT = 2;
constexpr uint32_t HC_A5_SWIZZLE_TILE_GROUP = 3;
constexpr uint32_t HC_A5_L1_A_STAGES = 2;
constexpr uint32_t HC_A5_L1_B_STAGES = 2;
constexpr uint32_t HC_A5_L0_A_STAGES = 2;
constexpr uint32_t HC_A5_C1_L0_K = 32;
constexpr uint32_t HC_A5_PIPELINE_DRAIN_WAVES = 2;

constexpr uint64_t HC_SYNC_MODE = 4;
constexpr uint64_t FLAG_V0 = 0;
constexpr uint64_t FLAG_C0_0 = 1;
constexpr uint64_t FLAG_C0_1 = 2;
constexpr uint64_t FLAG_V1 = 3;
constexpr uint64_t FLAG_C1_0 = 4; // C1[k] Set slot=k&1；V2[k] Wait 同槽（delay-2，禁止与 C1[k+1] 串同一 flag）
constexpr uint64_t FLAG_C1_1 = 5;
constexpr uint64_t HC_FLAG_PAIR = 16;

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
    return HcSubBlockNum() < HC_A5_VECTOR_SUBBLOCK_COUNT && AscendC::GetSubBlockIdx() == 0;
}

CATLASS_DEVICE
void HcAivSet(uint64_t base)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::CrossCoreSetFlag<HC_SYNC_MODE, PIPE_MTE3>(base);
    if (HcNeedDummyPair()) {
        AscendC::CrossCoreSetFlag<HC_SYNC_MODE, PIPE_MTE3>(base + HC_FLAG_PAIR);
    }
}

CATLASS_DEVICE
void HcAivWait(uint64_t base)
{
    AscendC::CrossCoreWaitFlag<HC_SYNC_MODE, PIPE_V>(base);
    if (HcNeedDummyPair()) {
        AscendC::CrossCoreWaitFlag<HC_SYNC_MODE, PIPE_V>(base + HC_FLAG_PAIR);
    }
}

CATLASS_DEVICE
void HcAicSet(uint64_t base)
{
    AscendC::CrossCoreSetFlag<HC_SYNC_MODE, PIPE_FIX>(base);
    AscendC::CrossCoreSetFlag<HC_SYNC_MODE, PIPE_FIX>(base + HC_FLAG_PAIR);
}

CATLASS_DEVICE
void HcAicWait(uint64_t base)
{
    AscendC::CrossCoreWaitFlag<HC_SYNC_MODE, PIPE_MTE1>(base);
    AscendC::CrossCoreWaitFlag<HC_SYNC_MODE, PIPE_MTE1>(base + HC_FLAG_PAIR);
}

// 仅 AIV 调用。硬同步占 flagId=14，禁止 AIC / SyncAll<false>。
CATLASS_DEVICE
void HcAivSyncAll()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<true>();
}

// Keep adjacent output rows that share a 32-byte GM block on one AIV.
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

template <class Element, uint32_t C1SwizzleDir, class Tiling = HcForwardMulSumTilingData>
struct HcForwardMulSumKernel {
    using ArchTag = Arch::Ascend950;
    // Retain the existing large-path tile configuration and resident policy.
    static constexpr uint32_t L0B_STAGES_C1 = 2;
    static constexpr bool ENABLE_UNIT_FLAG_C1 = false;
    static constexpr uint32_t L0B_STAGES_C0 = 2u;
    static constexpr bool ENABLE_UNIT_FLAG_C0 = false;
    static constexpr uint32_t kC1L0CStages = HC_C1_L0C_STAGES;
    static constexpr bool kC1L1Resident = true;
    static constexpr uint32_t L1M0 = 128u;
    static constexpr uint32_t L1N0 = HC_L1N0_LARGE;
    static constexpr uint32_t L1K0 = 128u;
    static constexpr uint32_t L1M1 = 128u;
    static constexpr uint32_t L1N1 = HC_L1N1;
    static constexpr uint32_t L0K0 = 32u;
    static_assert(L0K0 * L1N0 * static_cast<int64_t>(sizeof(Element)) * L0B_STAGES_C0 <= 64u * 1024u,
                  "C0 L0B capacity");
    static_assert(L1M0 * L0K0 * static_cast<int64_t>(sizeof(Element)) * 2u <= ArchTag::L0A_SIZE, "C0 L0A capacity");
    static_assert(32u * HC_L1N1 * static_cast<int64_t>(sizeof(Element)) * L0B_STAGES_C1 <= 64u * 1024u,
                  "C1 L0B capacity");

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;

    using DispatchPolicyC1 =
        Gemm::MmadPingpongPreLoad<ArchTag, ENABLE_UNIT_FLAG_C1, false, kC1L0CStages, kC1L1Resident, HC_A5_L1_A_STAGES,
                                  HC_A5_L1_B_STAGES, HC_A5_L0_A_STAGES, L0B_STAGES_C1>;
    using L1C0 = tla::Shape<tla::Int<L1M0>, tla::Int<L1N0>, tla::Int<L1K0>>;
    using L0C0 = tla::Shape<tla::Int<L1M0>, tla::Int<L1N0>, tla::Int<L0K0>>;
    using L1C1 = tla::Shape<tla::Int<L1M1>, tla::Int<HC_L1N1>, tla::Int<HC_C1_L1K_F16>>;
    using L0C1 = tla::Shape<tla::Int<L1M1>, tla::Int<HC_L1N1>, tla::Int<HC_A5_C1_L0_K>>;
    using TileCopyC1 =
        Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutTagA, Element, LayoutTagB, Element, LayoutTagC, void>;
    using TileCopyC0FinalPacked =
        Gemm::Tile::PackedTileCopyTla<ArchTag, Element, LayoutTagA, Element, LayoutTagB, Element, LayoutTagC, void>;
    using DispatchPolicyC0Final =
        Gemm::MmadPingpongPreLoad<ArchTag, ENABLE_UNIT_FLAG_C0, false, 1, false, HC_A5_L1_A_STAGES, HC_A5_L1_B_STAGES,
                                  HC_A5_L0_A_STAGES, L0B_STAGES_C0>;
    using BlockMmadC0Final = Gemm::Block::BlockMmadTla<DispatchPolicyC0Final, L1C0, L0C0, Element, Element, Element,
                                                       void, TileCopyC0FinalPacked>;
    using BlockMmadC1 =
        Gemm::Block::BlockMmadTla<DispatchPolicyC1, L1C1, L0C1, Element, Element, Element, void, TileCopyC1>;
    using BlockSchedulerC1 = Gemm::Block::GemmIdentityBlockSwizzle<HC_A5_SWIZZLE_TILE_GROUP, C1SwizzleDir>;

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

    template <class TensorA, class TensorBDown, class TensorBGate, class TensorCDown, class TensorCGate>
    CATLASS_DEVICE void RunLargeC0Two(Arch::Resource<ArchTag>& resource, TensorA const& tensorA,
                                      TensorBDown const& tensorBDown, TensorBGate const& tensorBGate,
                                      TensorCDown const& tensorCDown, TensorCGate const& tensorCGate, uint32_t row,
                                      uint32_t actualM, uint32_t rDim, uint32_t channels, uint32_t kDim,
                                      bool useCombine)
    {
        auto tileA = GetTile(tensorA, tla::MakeCoord(row, 0), tla::MakeShape(actualM, kDim));
        BlockMmadC0Final mmad(resource);
        for (uint32_t no = 0; no < rDim; no += L1N0) {
            uint32_t an = rDim - no < L1N0 ? rDim - no : L1N0;
            auto tileB = GetTile(tensorBDown, tla::MakeCoord(0, no), tla::MakeShape(kDim, an));
            auto tileC = GetTile(tensorCDown, tla::MakeCoord(0, no), tla::MakeShape(actualM, an));
            mmad(tileA, tileB, tileC, GemmCoord{actualM, an, kDim});
        }
        if (useCombine && channels > 0) {
            auto tileBg = GetTile(tensorBGate, tla::MakeCoord(0, 0), tla::MakeShape(kDim, channels));
            auto tileCg = GetTile(tensorCGate, tla::MakeCoord(0, 0), tla::MakeShape(actualM, channels));
            mmad(tileA, tileBg, tileCg, GemmCoord{actualM, channels, kDim});
        }
    }

    CATLASS_DEVICE
    HcForwardMulSumKernel() {}

    CATLASS_DEVICE
    void operator()(Params const& params)
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

        uint32_t nWsDown = td->nWs;
        uint32_t nWsGate = td->nWsGate;
        if (nWsDown == 0) {
            nWsDown = ((r + 15u) / 16u) * 16u;
        }
        if (useCombine && channels > 0 && nWsGate == 0) {
            nWsGate = ((channels + 15u) / 16u) * 16u;
        }
        if (!useCombine) {
            nWsGate = 0;
        }
        const uint32_t nWsAll = nWsDown + nWsGate;

        uint32_t blockIdx = 0;
        if ASCEND_IS_AIC {
            blockIdx = GetBlockIdx();
        } else {
            blockIdx = GetBlockIdx() / HcSubBlockNum();
        }
        if (blockIdx >= usedAic) {
            return;
        }
        // Isolated C1：无 Vector。Isolated C0：AIV 必须做 SplitkReduce，禁止提前 return。

        Arch::Resource<ArchTag> resource;

        GM_ADDR ws = params.userWs;
        // fused 大路径核内环形槽（与 host HcA5WorkspaceBytes 对齐）：xn 3 拍、down 2 拍。
        int64_t s0 = static_cast<int64_t>(usedAic) * 3u * mTask * d * szT;
        int64_t s1 = static_cast<int64_t>(usedAic) * 2u * mTask * r * szT;
        int64_t s2 = 0;
        s2 = static_cast<int64_t>(usedAic) * L1M0 * nWsAll * static_cast<int64_t>(sizeof(Element)) *
             HC_PING_PONG_SLOT_COUNT;
        GM_ADDR ptrXn = ws;
        GM_ADDR ptrDown = ws + s0;
        GM_ADDR ptrC0Base = ws + s0 + s1;
        GM_ADDR ptrUpWorkspace = ptrC0Base + s2;
        GM_ADDR ptrUp = params.ptrUpPreAct != nullptr ? params.ptrUpPreAct : ptrUpWorkspace;

        const int64_t c0SlotBytes = static_cast<int64_t>(L1M0) * nWsAll * static_cast<int64_t>(sizeof(Element));
        const int64_t c0DownBytes = static_cast<int64_t>(L1M0) * nWsDown * static_cast<int64_t>(sizeof(Element));
        GM_ADDR ptrC0Core = ptrC0Base + static_cast<int64_t>(blockIdx) * c0SlotBytes * 2;

        GlobalTensor<Element> gmWDown;
        gmWDown.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWDown));
        GlobalTensor<Element> gmWUp;
        gmWUp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWUp));
        GlobalTensor<Element> gmWGate;
        gmWGate.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrWGate));
        GlobalTensor<Element> gmUp;
        gmUp.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrUp));
        // up 全量只被 C1 写、V2 流式读一次（与 x 同类的流式数据），关 L2 分配
        // 避免挤掉 C0/C1 权重等高复用数据（V0 的 gX 同款优化）。
        gmUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        auto layoutWDown = tla::MakeLayout<Element, LayoutTagB>(d, r);
        auto layoutWUp = tla::MakeLayout<Element, LayoutTagB>(r, d);
        auto layoutWGate = tla::MakeLayout<Element, LayoutTagB>(d, channels);
        auto tensorWDown = tla::MakeTensor(gmWDown, layoutWDown, Arch::PositionGM{});
        auto tensorWGate = tla::MakeTensor(gmWGate, layoutWGate, Arch::PositionGM{});
        auto tensorWUp = tla::MakeTensor(gmWUp, layoutWUp, Arch::PositionGM{});
        auto layoutUpFull = tla::MakeLayout<Element, LayoutTagC>(M, d);
        auto tensorUp = tla::MakeTensor(gmUp, layoutUpFull, Arch::PositionGM{});

        uint32_t localTasks = 0;
        for (uint32_t task = blockIdx; task < taskNum; task += usedAic) {
            ++localTasks;
        }

        Epilogue::BlockEpilogueHcV0<Element> blockV0;
        Epilogue::BlockEpilogueHcV1<> blockV1;
        Epilogue::BlockEpilogueHcV2<Element> blockV2;

        // 核内私有环形槽基址：每核 xn 3 槽、down 2 槽，仅 fused 大路径使用。
        const int64_t xnSlotBytes = static_cast<int64_t>(mTask) * d * szT;
        const int64_t downSlotBytes = static_cast<int64_t>(mTask) * r * szT;
        GM_ADDR xnCoreBase = ptrXn + static_cast<int64_t>(blockIdx) * 3u * xnSlotBytes;
        GM_ADDR downCoreBase = ptrDown + static_cast<int64_t>(blockIdx) * 2u * downSlotBytes;

        // fused 默认 V2 delay-2：V0[t] → C0[t] → V1[t-1] → C1[t-1] → V2[t-2]
        // 尾部多排空 2 拍（最后 V1/C1 与最后 V2 各一拍）。
        for (uint32_t wave = 0; wave <= localTasks + 1; ++wave) {
            uint32_t task = blockIdx + wave * usedAic;
            if (wave < localTasks) {
                uint32_t row = task * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    GM_ADDR xTask = params.ptrX + static_cast<int64_t>(row) * d * static_cast<int64_t>(sizeof(Element));
                    // 环形槽：V0[w] 写槽 w%3，同核 C0[w]/V2[w] 串行读同槽，w+3 拍复用。
                    GM_ADDR xnTask = xnCoreBase + static_cast<int64_t>(wave % 3u) * xnSlotBytes;
                    GM_ADDR injTask =
                        params.ptrInject + static_cast<int64_t>(row) * channels * static_cast<int64_t>(sizeof(Element));
                    blockV0(resource, xTask, xnTask, injTask, params.ptrWGate, params.ptrGamma, actualM, d, channels,
                            hidden, eps, useCombine, hasGamma, false,
                            params.ptrInvRms == nullptr ? nullptr :
                                                          params.ptrInvRms + static_cast<int64_t>(row) * channels *
                                                                                 static_cast<int64_t>(sizeof(float)),
                            true, true);
                    HcAivSet(FLAG_V0);
                }
                if ASCEND_IS_AIC {
                    HcAicWait(FLAG_V0);
                    uint32_t slot = wave & 1;
                    GM_ADDR c0Slot = ptrC0Core + slot * c0SlotBytes;
                    // c0dp：C0 的 down 输出直写 down_pre_act 保存原始值（未除 C，反向自行重算 /C），
                    // gate 仍写 c0Slot；V1 改从 down_pre_act 读入且不再单独写保存。
                    // down_pre_act 行宽为紧凑 r，C0 fixpipe 行宽为 nWsDown，二者相等才可直写。
                    const bool c0dpLarge = params.ptrDownPreAct != nullptr && nWsDown == r;
                    GM_ADDR c0DownAddr = c0dpLarge ? params.ptrDownPreAct + static_cast<int64_t>(row) * r *
                                                                                static_cast<int64_t>(sizeof(Element)) :
                                                     c0Slot;
                    GlobalTensor<Element> gmC0Down;
                    gmC0Down.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(c0DownAddr));
                    auto layoutDown = tla::MakeLayout<Element, LayoutTagC>(L1M0, nWsDown);
                    auto tensorCDown = tla::MakeTensor(gmC0Down, layoutDown, Arch::PositionGM{});
                    GlobalTensor<Element> gmC0Gate;
                    gmC0Gate.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(c0Slot + c0DownBytes));
                    auto layoutGate = tla::MakeLayout<Element, LayoutTagC>(L1M0, nWsGate == 0 ? 16u : nWsGate);
                    auto tensorCGate = tla::MakeTensor(gmC0Gate, layoutGate, Arch::PositionGM{});
                    // 环形槽：C0[w] 读 xn 槽 w%3（本核私有），行坐标转槽内 0。
                    GlobalTensor<Element> gmXnWave;
                    gmXnWave.SetGlobalBuffer(
                        reinterpret_cast<__gm__ Element*>(xnCoreBase + static_cast<int64_t>(wave % 3u) * xnSlotBytes));
                    // 环形槽为核私有地址无人跨核回读，关闭 L2 分配避免 28 核并发
                    // 写分配挤占 C0/C1 权重驻留。
                    gmXnWave.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    auto tensorXnWave =
                        tla::MakeTensor(gmXnWave, tla::MakeLayout<Element, LayoutTagA>(mTask, d), Arch::PositionGM{});
                    RunLargeC0Two(resource, tensorXnWave, tensorWDown, tensorWGate, tensorCDown, tensorCGate, 0u,
                                  actualM, r, channels, d, useCombine);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    HcAicSet(slot == 0 ? FLAG_C0_0 : FLAG_C0_1);
                }
            }
            if (wave >= 1 && (wave - 1) < localTasks) {
                uint32_t prev = blockIdx + (wave - 1) * usedAic;
                uint32_t prevSlot = (wave - 1) & 1;
                uint32_t row = prev * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    HcAivWait(prevSlot == 0 ? FLAG_C0_0 : FLAG_C0_1);
                    GM_ADDR c0Slot = ptrC0Core + prevSlot * c0SlotBytes;
                    // c0dp：V1 从 down_pre_act 读 C0 原始输出，gmDownPreAct 传空跳过保存写。
                    const bool c0dpLarge = params.ptrDownPreAct != nullptr && nWsDown == r;
                    GM_ADDR v1C0Addr = c0dpLarge ? params.ptrDownPreAct + static_cast<int64_t>(row) * r *
                                                                              static_cast<int64_t>(sizeof(Element)) :
                                                   c0Slot;
                    // 环形槽：V1[t] 写 down 槽 t&1（t=wave-1），同核 C1 串行读同槽，t+2 拍复用。
                    GM_ADDR downTask = downCoreBase + static_cast<int64_t>(prevSlot) * downSlotBytes;
                    GM_ADDR injTask =
                        params.ptrInject + static_cast<int64_t>(row) * channels * static_cast<int64_t>(sizeof(Element));
                    blockV1.template operator()<Arch::Resource<ArchTag>, Element>(
                        resource, v1C0Addr, c0Slot + c0DownBytes, downTask, injTask, actualM, nWsDown, nWsGate, r,
                        channels, useCombine, false,
                        c0dpLarge ? nullptr :
                                    (params.ptrDownPreAct == nullptr ?
                                         nullptr :
                                         params.ptrDownPreAct +
                                             static_cast<int64_t>(row) * r * static_cast<int64_t>(sizeof(Element))));
                    HcAivSet(FLAG_V1);
                }
                if ASCEND_IS_AIC {
                    HcAicWait(FLAG_V1);
                    // 环形槽：C1[t] 读 down 槽 t&1（本核私有），行坐标转槽内 0。
                    GlobalTensor<Element> gmDownWave;
                    gmDownWave.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(
                        downCoreBase + static_cast<int64_t>(prevSlot) * downSlotBytes));
                    auto tensorDownWave =
                        tla::MakeTensor(gmDownWave, tla::MakeLayout<Element, LayoutTagA>(mTask, r), Arch::PositionGM{});
                    GemmCoord problem{actualM, d, r};
                    BlockSchedulerC1 scheduler(problem, MatrixCoord{L1M1, L1N1});
                    uint32_t nTiles = scheduler.GetCoreLoops();
                    BlockMmadC1 mmadC1(resource);
                    for (uint32_t nTile = 0; nTile < nTiles; ++nTile) {
                        GemmCoord coord = scheduler.GetBlockCoord(nTile);
                        GemmCoord actual = scheduler.GetActualBlockShape(coord);
                        uint32_t nOff = coord.n() * L1N1;
                        auto tileA = GetTile(tensorDownWave, tla::MakeCoord(0u, 0), tla::MakeShape(actual.m(), r));
                        auto tileB = GetTile(tensorWUp, tla::MakeCoord(0, nOff), tla::MakeShape(r, actual.n()));
                        auto tileC =
                            GetTile(tensorUp, tla::MakeCoord(row, nOff), tla::MakeShape(actual.m(), actual.n()));
                        mmadC1(tileA, tileB, tileC, actual);
                    }

                    AscendC::PipeBarrier<PIPE_ALL>();
                    HcAicSet(prevSlot == 0 ? FLAG_C1_0 : FLAG_C1_1);
                }
            }
            if (wave >= HC_A5_PIPELINE_DRAIN_WAVES && (wave - HC_A5_PIPELINE_DRAIN_WAVES) < localTasks) {
                uint32_t prev2 = blockIdx + (wave - HC_A5_PIPELINE_DRAIN_WAVES) * usedAic;
                uint32_t c1Slot = (wave - HC_A5_PIPELINE_DRAIN_WAVES) & 1;
                uint32_t row = prev2 * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    HcAivWait(c1Slot == 0 ? FLAG_C1_0 : FLAG_C1_1);
                    // 环形槽：V2[t] 读 xn 槽 t%3（t=wave-2，即写入后第 3 拍，本核 AIV 串行保证安全）。
                    GM_ADDR xnTask =
                        xnCoreBase + static_cast<int64_t>((wave - HC_A5_PIPELINE_DRAIN_WAVES) % 3u) * xnSlotBytes;
                    GM_ADDR upTask = ptrUp + static_cast<int64_t>(row) * d * static_cast<int64_t>(sizeof(Element));
                    GM_ADDR mixedTask =
                        params.ptrMixed + static_cast<int64_t>(row) * hidden * static_cast<int64_t>(sizeof(Element));
                    blockV2(
                        resource, upTask, xnTask, mixedTask, actualM, channels, hidden, false,
                        params.ptrUpPreAct == nullptr ?
                            nullptr :
                            params.ptrUpPreAct + static_cast<int64_t>(row) * d * static_cast<int64_t>(sizeof(Element)),
                        true);
                }
            }
        }
    }
};

} // namespace Catlass::Gemm::Kernel

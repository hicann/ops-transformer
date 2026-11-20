/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <new>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "distribute_barrier_extend_tiling_def.h"
#include "../../../../common/op_kernel/mc2_moe_context.h"

extern "C" __global__ __aicore__ void distribute_barrier_extend(GM_ADDR context, GM_ADDR xRef, GM_ADDR timeOut,
                                                                GM_ADDR elasticInfo, GM_ADDR xRefOut,
                                                                GM_ADDR workspaceGM, GM_ADDR tilingGM);

namespace Mc2BarrierUt {
CommState* g_commState = nullptr;
Channel* g_delayedChannel = nullptr;
uint32_t* g_drainGate = nullptr;
constexpr uint32_t TRANSPORT_ROUNDS = 32U;

bool MatchesProtocol(const Channel* channel, Transport protocol)
{
    if (channel == nullptr) {
        return false;
    }
    bool matches = channel->protocol == protocol;
#if ASC_DEVKIT_VERSION_NUM >= 90200000
    matches = matches || (protocol == Transport::CTP && channel->protocol == Transport::RTP);
#endif
    return matches;
}

int32_t Init(Transport protocol, uint32_t bytes)
{
    if (bytes != HCOMM_BUFFER_BYTES) {
        __atomic_add_fetch(&g_commState->invalidCalls, 1U, __ATOMIC_SEQ_CST);
        return -1;
    }
    __atomic_add_fetch(&g_commState->initCalls[protocol == Transport::ROCE], 1U, __ATOMIC_SEQ_CST);
    return g_commState->failure == Failure::INIT ? -1 : 0;
}

int32_t Lock(Transport protocol, uint64_t handle)
{
    auto* channel = reinterpret_cast<Channel*>(handle);
    if (!MatchesProtocol(channel, protocol) || g_commState->failure == Failure::LOCK) {
        return -1;
    }
    while (__atomic_exchange_n(&channel->locked, 1U, __ATOMIC_SEQ_CST) != 0U) {
        std::this_thread::yield();
    }
    channel->owner = AscendC::GetBlockIdx();
    ++channel->locks;
    return 0;
}

int32_t Unlock(Transport protocol, uint64_t handle)
{
    auto* channel = reinterpret_cast<Channel*>(handle);
    if (!MatchesProtocol(channel, protocol) || channel->locked == 0U || channel->owner != AscendC::GetBlockIdx()) {
        __atomic_add_fetch(&g_commState->invalidCalls, 1U, __ATOMIC_SEQ_CST);
        return -1;
    }
    ++channel->unlocks;
    __atomic_store_n(&channel->locked, 0U, __ATOMIC_SEQ_CST);
    return g_commState->failure == Failure::UNLOCK ? -1 : 0;
}

int32_t Write(Transport protocol, uint64_t handle, uint8_t* dst, uint8_t* src, uint64_t bytes)
{
    auto* channel = reinterpret_cast<Channel*>(handle);
    if (!MatchesProtocol(channel, protocol) || channel->source != src || bytes != FLAG_BYTES ||
        (dst != channel->destination[0] && dst != channel->destination[1]) || channel->pending != nullptr ||
        (protocol != Transport::ROCE && (channel->locked == 0U || channel->owner != AscendC::GetBlockIdx()))) {
        __atomic_add_fetch(&g_commState->invalidCalls, 1U, __ATOMIC_SEQ_CST);
        return -1;
    }
    const auto* flags = reinterpret_cast<const uint64_t*>(src);
    if (flags[0] == 0U || !std::all_of(flags, flags + FLAG_BYTES / sizeof(uint64_t),
                                       [flags](uint64_t value) { return value == flags[0]; })) {
        __atomic_add_fetch(&g_commState->invalidCalls, 1U, __ATOMIC_SEQ_CST);
        return -1;
    }
    channel->bank = dst == channel->destination[1];
    __atomic_add_fetch(&channel->writes[channel->bank], 1U, __ATOMIC_SEQ_CST);
    if (g_commState->failure == Failure::WRITE) {
        return -1;
    }
    std::memcpy(channel->payloads[channel->bank], src, FLAG_BYTES);
    channel->pending = dst;
    return 0;
}

int32_t Drain(Transport protocol, uint64_t handle)
{
    if (g_commState != nullptr && g_commState->failure == Failure::STALL_DRAIN) {
        __atomic_add_fetch(&g_commState->drainEntered, 1U, __ATOMIC_SEQ_CST);
        while (g_commState->failure == Failure::STALL_DRAIN) {
            std::this_thread::yield();
        }
    }

    auto* channel = reinterpret_cast<Channel*>(handle);
    if (!MatchesProtocol(channel, protocol) || channel->pending == nullptr ||
        (protocol != Transport::ROCE && (channel->locked == 0U || channel->owner != AscendC::GetBlockIdx()))) {
        __atomic_add_fetch(&g_commState->invalidCalls, 1U, __ATOMIC_SEQ_CST);
        return -1;
    }
    ++channel->drains[channel->bank];
    if (g_commState->failure == Failure::DRAIN) {
        return -1;
    }
    std::memcpy(channel->pending, channel->payloads[channel->bank], FLAG_BYTES);
    channel->pending = nullptr;
    if (channel == g_delayedChannel && channel->bank == 0U) {
        __atomic_store_n(g_drainGate + 1, 1U, __ATOMIC_SEQ_CST);
        while (__atomic_load_n(g_drainGate, __ATOMIC_SEQ_CST) == 0U) {
            std::this_thread::yield();
        }
    }
    return 0;
}
} // namespace Mc2BarrierUt

void DistributeBarrierTrap()
{
    if (Mc2BarrierUt::g_commState != nullptr) {
        Mc2BarrierUt::g_commState->trapCore = AscendC::GetBlockIdx();
        __atomic_add_fetch(&Mc2BarrierUt::g_commState->trapCalls, 1U, __ATOMIC_SEQ_CST);
    }
    _exit(1);
}

class DistributeBarrierExtendTest : public testing::Test {
protected:
    struct ElasticRound {
        uint8_t* info;
        uint32_t participants;
        uint32_t lateRank;
    };

    void SetUp() override
    {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
    }

    uint8_t* Allocate(size_t bytes)
    {
        auto* memory = static_cast<uint8_t*>(AscendC::GmAlloc(bytes));
        std::memset(memory, 0, bytes);
        allocations_.push_back(memory);
        return memory;
    }

    void TearDown() override
    {
        for (pid_t worker : workers_) {
            if (worker > 0) {
                kill(-worker, SIGKILL);
            }
        }
        for (pid_t worker : workers_) {
            if (worker > 0) {
                waitpid(worker, nullptr, 0);
            }
        }
        for (auto* memory : allocations_) {
            AscendC::GmFree(memory);
        }
        Mc2BarrierUt::g_commState = nullptr;
        Mc2BarrierUt::g_delayedChannel = nullptr;
        Mc2BarrierUt::g_drainGate = nullptr;
    }

    void InitContext(uint32_t local, uint32_t rail)
    {
        context_ = new (Allocate(Mc2Barrier::ContextBytes(local, rail))) Mc2Barrier::Context;
        context_->worldSize = local * rail;
        context_->localSize = local;
        context_->railSize = rail;
        context_->windowBytes = Mc2Barrier::WindowBytes(local, rail);
        context_->window = reinterpret_cast<uint64_t>(Allocate(context_->windowBytes));
        peers_ = reinterpret_cast<Mc2Barrier::Peer*>(context_ + 1);
        for (uint32_t i = 0; i < local + rail; ++i) {
            const uint32_t slot = i < local ? i : i - local;
            const uint64_t window =
                slot == 0 ? context_->window : reinterpret_cast<uint64_t>(Allocate(context_->windowBytes));
            peers_[i] = {window, 0, i < local ? slot : slot * local, Mc2Barrier::Protocol::UBMEM};
        }
    }

    uint8_t* InitTiling(uint32_t aivNum)
    {
        auto* tiling = Allocate(sizeof(DistributeBarrierTilingData));
        auto& info = reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo;
        info.worldSize = context_->worldSize;
        info.aivNum = aivNum;
        info.totalWinSize = Mc2Barrier::ContextElems(context_->localSize, context_->railSize);
        info.isMc2Context = true;
        info.isAuto = true;
        info.totalUbSize = 256U * 1024U;
        return tiling;
    }

    void CheckRound(uint32_t bank, uint8_t* tiling)
    {
        const uint32_t count = context_->localSize + context_->railSize;
        const uint64_t sequence = Mc2Barrier::SequenceOffset(context_->localSize, context_->railSize);
        const uint64_t epoch = *reinterpret_cast<uint64_t*>(context_->window + sequence) + 1U;
        const uint64_t offset = bank * Mc2Barrier::BankBytes(context_->localSize, context_->railSize);
        for (uint32_t i = 0; i < count; ++i) {
            auto* flag = reinterpret_cast<uint64_t*>(context_->window + offset + i * Mc2Barrier::SLOT_BYTES);
            std::fill(flag, flag + 4, epoch);
        }
        ICPU_RUN_KF(distribute_barrier_extend, 2, reinterpret_cast<GM_ADDR>(context_), nullptr, nullptr, nullptr,
                    nullptr, nullptr, tiling);
        for (uint32_t i = 0; i < context_->localSize; ++i) {
            EXPECT_EQ(*reinterpret_cast<uint64_t*>(context_->window + offset + i * Mc2Barrier::SLOT_BYTES), 0U);
            EXPECT_EQ(*reinterpret_cast<uint64_t*>(peers_[i].window + offset), i == 0 ? 0U : epoch);
        }
        if (context_->railSize > 1) {
            for (uint32_t i = 0; i < context_->railSize; ++i) {
                const uint64_t phase = Mc2Barrier::PhaseOffset(1, context_->localSize);
                EXPECT_EQ(*reinterpret_cast<uint64_t*>(context_->window + offset + phase + i * Mc2Barrier::SLOT_BYTES),
                          0U);
                EXPECT_EQ(*reinterpret_cast<uint64_t*>(peers_[context_->localSize + i].window + offset + phase),
                          i == 0 ? 0U : epoch);
            }
        }
        EXPECT_EQ(*reinterpret_cast<uint64_t*>(context_->window + sequence), epoch);
    }

    void CheckStaleSignals(uint8_t* tiling)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        auto* timeout = reinterpret_cast<int32_t*>(Allocate(sizeof(int32_t)));
        *timeout = 500000;
        reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputTimeOut = true;
        ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(context_, tiling, timeout, nullptr, 1));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (__atomic_load_n(&Mc2BarrierUt::g_commState->trapCalls, __ATOMIC_SEQ_CST) == 0U &&
               __atomic_load_n(&Mc2BarrierUt::g_commState->completed[0], __ATOMIC_SEQ_CST) == 0U &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        const pid_t worker = workers_.back();
        kill(-worker, SIGKILL);
        waitpid(worker, nullptr, 0);
        workers_.back() = 0;
        EXPECT_GT(__atomic_load_n(&Mc2BarrierUt::g_commState->trapCalls, __ATOMIC_SEQ_CST), 0U);
    }

    void CheckRejectedContext(uint8_t* tiling, uint8_t* elastic = nullptr)
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            const uint64_t offset = bank * Mc2Barrier::BankBytes(context_->localSize, context_->railSize);
            for (uint32_t i = 0; i < context_->localSize; ++i) {
                auto* flag = reinterpret_cast<float*>(context_->window + offset + i * Mc2Barrier::SLOT_BYTES);
                std::fill(flag, flag + 8, 1.0F);
            }
        }
        auto* memory = reinterpret_cast<uint8_t*>(context_->window);
        const std::vector<uint8_t> before(memory, memory + context_->windowBytes);
        ICPU_RUN_KF(distribute_barrier_extend, 1, reinterpret_cast<GM_ADDR>(context_), nullptr, nullptr, elastic,
                    nullptr, nullptr, tiling);
        EXPECT_EQ(std::memcmp(memory, before.data(), before.size()), 0);
    }

    Mc2BarrierUt::Channel* InitTransportChannel(const Mc2Barrier::Context& context, const Mc2Barrier::Peer& peer)
    {
        auto* channel = new (Allocate(sizeof(Mc2BarrierUt::Channel))) Mc2BarrierUt::Channel{};
        channel->protocol =
            peer.protocol == Mc2Barrier::Protocol::ROCE ? Mc2BarrierUt::Transport::ROCE : urmaTransport_;
        const uint32_t slots = context.localSize + context.railSize;
        channel->source = reinterpret_cast<uint8_t*>(context.window + 2U * slots * Mc2Barrier::SLOT_BYTES);
        for (uint32_t bank = 0; bank < 2; ++bank) {
            const uint64_t offset = (bank * slots + context.localSize + context.railSlot) * Mc2Barrier::SLOT_BYTES;
            channel->destination[bank] = reinterpret_cast<uint8_t*>(peer.window + offset);
        }
        transportChannels_.push_back(channel);
        return channel;
    }

    Mc2Barrier::Context* InitTransportRank(uint32_t rank, const std::vector<uint64_t>& windows,
                                           Mc2Barrier::Protocol protocol, bool mixed, uint32_t local = 2)
    {
        auto* context = new (Allocate(Mc2Barrier::ContextBytes(local, windows.size() / local))) Mc2Barrier::Context;
        context->rank = rank;
        context->worldSize = windows.size();
        context->localSize = local;
        context->railSize = windows.size() / local;
        context->localSlot = rank % local;
        context->railSlot = rank / local;
        context->window = windows[rank];
        context->windowBytes = Mc2Barrier::WindowBytes(local, context->railSize);
        auto* peers = reinterpret_cast<Mc2Barrier::Peer*>(context + 1);
        for (uint32_t i = 0; i < local + context->railSize; ++i) {
            const uint32_t remote = i < local ? rank / local * local + i : (i - local) * local + rank % local;
            new (peers + i) Mc2Barrier::Peer{windows[remote], 0, remote, Mc2Barrier::Protocol::UBMEM};
            if (i >= local && remote != rank) {
                peers[i].protocol = mixed ? ((rank / local + remote / local) % 2 == 0 ? Mc2Barrier::Protocol::URMA :
                                                                                        Mc2Barrier::Protocol::ROCE) :
                                            protocol;
                context->protocolMask |= 1U << static_cast<uint32_t>(peers[i].protocol);
                peers[i].handle = reinterpret_cast<uint64_t>(InitTransportChannel(*context, peers[i]));
            }
        }
        return context;
    }

    void RunTransportRank(Mc2Barrier::Context* context, uint8_t* tiling, int32_t* timeout, uint32_t rounds)
    {
        for (uint32_t round = 0; round < rounds; ++round) {
            ICPU_RUN_KF(distribute_barrier_extend, 2, reinterpret_cast<uint8_t*>(context), nullptr,
                        reinterpret_cast<uint8_t*>(timeout), elasticInfo_, nullptr, nullptr, tiling);
            auto* completed = &Mc2BarrierUt::g_commState->completed[round % 2];
            __atomic_add_fetch(completed, 1U, __ATOMIC_SEQ_CST);
        }
    }

    void LaunchTransportRank(Mc2Barrier::Context* context, uint8_t* tiling, int32_t* timeout, uint32_t* gate = nullptr,
                             uint32_t rounds = Mc2BarrierUt::TRANSPORT_ROUNDS)
    {
        const pid_t worker = fork();
        ASSERT_GE(worker, 0);
        if (worker == 0) {
            setpgid(0, 0);
            if (gate != nullptr) {
                while (context->rank == 17U && __atomic_load_n(gate, __ATOMIC_SEQ_CST) == 0U) {
                    std::this_thread::yield();
                }
                __atomic_add_fetch(gate + 1, 1U, __ATOMIC_SEQ_CST);
            }
            RunTransportRank(context, tiling, timeout, rounds);
            _exit(0);
        }
        workers_.push_back(worker);
        setpgid(worker, worker);
    }

    void CheckTransportResources(const std::vector<Mc2Barrier::Context*>& contexts)
    {
        for (const auto* context : contexts) {
            for (uint32_t bank = 0; bank < 2; ++bank) {
                for (uint32_t slot = 0; slot < 5; ++slot) {
                    const auto* flags =
                        reinterpret_cast<const float*>(context->window + (bank * 5U + slot) * Mc2Barrier::SLOT_BYTES);
                    EXPECT_TRUE(std::all_of(flags, flags + 8, [](float value) { return value == 0.0F; }));
                }
            }
            EXPECT_EQ(*reinterpret_cast<uint64_t*>(context->window + Mc2Barrier::SequenceOffset(2, 3)),
                      Mc2BarrierUt::TRANSPORT_ROUNDS);
        }
        for (const auto* channel : transportChannels_) {
            for (uint32_t bank = 0; bank < 2; ++bank) {
                EXPECT_EQ(channel->writes[bank], Mc2BarrierUt::TRANSPORT_ROUNDS / 2);
                EXPECT_EQ(channel->drains[bank], Mc2BarrierUt::TRANSPORT_ROUNDS / 2);
            }
            EXPECT_EQ(channel->pending, nullptr);
            EXPECT_EQ(channel->locks,
                      channel->protocol == Mc2BarrierUt::Transport::ROCE ? 0U : Mc2BarrierUt::TRANSPORT_ROUNDS);
            EXPECT_EQ(channel->unlocks, channel->locks);
            EXPECT_EQ(channel->locked, 0U);
        }
        EXPECT_EQ(Mc2BarrierUt::g_commState->invalidCalls, 0U);
    }

    void WaitForCount(uint32_t* value, uint32_t expected)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (__atomic_load_n(value, __ATOMIC_SEQ_CST) < expected && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_EQ(__atomic_load_n(value, __ATOMIC_SEQ_CST), expected);
    }

    void LaunchDelayedRounds(Mc2Barrier::Context* context, uint8_t* tiling, uint32_t* control, uint32_t rounds,
                             int32_t* timeout)
    {
        const pid_t worker = fork();
        ASSERT_GE(worker, 0);
        if (worker == 0) {
            setpgid(0, 0);
            AscendC::SetKernelMode(KernelMode::AIV_MODE);
            for (uint32_t round = 0; round < rounds; ++round) {
                auto* gate = control + round * 3U;
                while (context->rank == round % context->worldSize && __atomic_load_n(gate, __ATOMIC_SEQ_CST) == 0U) {
                    std::this_thread::yield();
                }
                __atomic_add_fetch(gate + 1, 1U, __ATOMIC_SEQ_CST);
                ICPU_RUN_KF(distribute_barrier_extend, 2U + round % 3U, reinterpret_cast<GM_ADDR>(context), nullptr,
                            reinterpret_cast<GM_ADDR>(timeout), nullptr, nullptr, nullptr, tiling);
                __atomic_add_fetch(gate + 2, 1U, __ATOMIC_SEQ_CST);
            }
            _exit(0);
        }
        workers_.push_back(worker);
        setpgid(worker, worker);
    }

    void CheckEveryLateRank(uint32_t local, uint32_t rail, Mc2Barrier::Protocol protocol, int32_t timeoutUs = 0)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        const uint32_t world = local * rail;
        std::vector<uint64_t> windows(world);
        for (auto& window : windows) {
            window = reinterpret_cast<uint64_t>(Allocate(Mc2Barrier::WindowBytes(local, rail)));
        }
        auto* control = reinterpret_cast<uint32_t*>(Allocate(world * 3U * sizeof(uint32_t)));
        auto* timeout = timeoutUs == 0 ? nullptr : reinterpret_cast<int32_t*>(Allocate(sizeof(int32_t)));
        if (timeout != nullptr) {
            *timeout = timeoutUs;
        }
        for (uint32_t rank = 0; rank < world; ++rank) {
            context_ = InitTransportRank(rank, windows, protocol, false, local);
            auto* tiling = InitTiling(4);
            reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputTimeOut =
                timeout != nullptr;
            ASSERT_NO_FATAL_FAILURE(LaunchDelayedRounds(context_, tiling, control, world, timeout));
        }
        for (uint32_t round = 0; round < world; ++round) {
            SCOPED_TRACE(round);
            auto* gate = control + round * 3U;
            ASSERT_NO_FATAL_FAILURE(WaitForCount(gate + 1, world - 1U));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            ASSERT_EQ(__atomic_load_n(gate + 2, __ATOMIC_SEQ_CST), 0U);
            __atomic_store_n(gate, 1U, __ATOMIC_SEQ_CST);
            ASSERT_NO_FATAL_FAILURE(WaitForCount(gate + 2, world));
        }
        ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
        EXPECT_EQ(Mc2BarrierUt::g_commState->invalidCalls, 0U);
        EXPECT_EQ(Mc2BarrierUt::g_commState->trapCalls, 0U);
        for (const auto* channel : transportChannels_) {
            EXPECT_EQ(channel->locks, channel->protocol == Mc2BarrierUt::Transport::ROCE ? 0U : world);
            EXPECT_EQ(channel->unlocks, channel->locks);
            EXPECT_EQ(channel->locked, 0U);
        }
    }

    void CheckTimeoutDuringDrain(Mc2Barrier::Protocol protocol, bool elastic = false)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        Mc2BarrierUt::g_commState->failure = Mc2BarrierUt::Failure::STALL_DRAIN;
        std::vector<uint64_t> windows(6);
        for (auto& window : windows) {
            window = reinterpret_cast<uint64_t>(Allocate(Mc2Barrier::WindowBytes(2, 3)));
        }
        context_ = InitTransportRank(0, windows, protocol, false);
        auto* incoming =
            reinterpret_cast<uint64_t*>(context_->window + Mc2Barrier::BankBytes(2, 3) + Mc2Barrier::SLOT_BYTES);
        std::fill(incoming, incoming + 4, uint64_t{1});
        auto* tiling = InitTiling(2);
        reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputTimeOut = true;
        if (elastic) {
            elasticInfo_ = MakeElasticInfo(6, {1});
            reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo = true;
        }
        auto* timeout = reinterpret_cast<int32_t*>(Allocate(sizeof(int32_t)));
        *timeout = 1000000;
        ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(context_, tiling, timeout));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (__atomic_load_n(&Mc2BarrierUt::g_commState->trapCalls, __ATOMIC_SEQ_CST) == 0U &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        EXPECT_GT(__atomic_load_n(&Mc2BarrierUt::g_commState->drainEntered, __ATOMIC_SEQ_CST), 0U);
        EXPECT_EQ(__atomic_load_n(&Mc2BarrierUt::g_commState->trapCalls, __ATOMIC_SEQ_CST), 1U);
        EXPECT_EQ(Mc2BarrierUt::g_commState->trapCore, 0U);
    }

    void CheckConcurrentTransport(Mc2Barrier::Protocol protocol, bool mixed = false)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        std::vector<uint64_t> windows(6);
        for (auto& window : windows) {
            window = reinterpret_cast<uint64_t>(Allocate(Mc2Barrier::WindowBytes(2, 3)));
        }
        std::vector<Mc2Barrier::Context*> contexts;
        auto* timeout = reinterpret_cast<int32_t*>(Allocate(sizeof(int32_t)));
        *timeout = 5000000;
        for (uint32_t rank = 0; rank < windows.size(); ++rank) {
            context_ = InitTransportRank(rank, windows, protocol, mixed);
            contexts.push_back(context_);
            auto* tiling = InitTiling(2);
            reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputTimeOut = true;
            reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo =
                elasticInfo_ != nullptr;
            ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(context_, tiling, timeout));
        }
        ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
        CheckTransportResources(contexts);
    }

    uint8_t* MakeElasticInfo(uint32_t world, const std::vector<uint32_t>& removed)
    {
        auto* memory = Allocate((4U + 2U * world) * sizeof(int32_t));
        auto* info = reinterpret_cast<int32_t*>(memory);
        info[0] = 1;
        info[1] = world - removed.size();
        std::fill(info + 4, info + 4 + 2U * world, -1);
        uint32_t logical = 0;
        for (uint32_t rank = 0; rank < world; ++rank) {
            if (std::find(removed.begin(), removed.end(), rank) == removed.end()) {
                info[4U + rank] = logical;
                info[4U + world + logical++] = rank;
            }
        }
        return memory;
    }

    void InitLargeContext(uint32_t local, uint32_t rail)
    {
        context_ = new (Allocate(Mc2Barrier::ContextBytes(local, rail))) Mc2Barrier::Context;
        context_->worldSize = local * rail;
        context_->rank = local * rail - 1U;
        context_->localSize = local;
        context_->railSize = rail;
        context_->windowBytes = Mc2Barrier::WindowBytes(local, rail);
        context_->window = reinterpret_cast<uint64_t>(Allocate(context_->windowBytes));
        const auto permutation = [this](uint32_t i) { return (8191U * i + 137U) % context_->worldSize; };
        uint32_t self = 0;
        while (permutation(self) != context_->rank) {
            ++self;
        }
        context_->localSlot = self % local;
        context_->railSlot = self / local;
        peers_ = reinterpret_cast<Mc2Barrier::Peer*>(context_ + 1);
        const uint64_t remote = reinterpret_cast<uint64_t>(Allocate(context_->windowBytes));
        for (uint32_t i = 0; i < local + rail; ++i) {
            const uint32_t physical = i < local ? (self / local) * local + i : (i - local) * local + self % local;
            const uint32_t rank = permutation(physical);
            peers_[i] = {rank == context_->rank ? context_->window : remote, 0U, rank, Mc2Barrier::Protocol::UBMEM};
        }
    }

    void CheckLargeElasticRound(uint8_t* tiling, uint8_t* elastic, uint32_t cores)
    {
        const auto* mapping = reinterpret_cast<const int32_t*>(elastic) + 4;
        const uint32_t count = context_->localSize + context_->railSize;
        for (uint32_t bank = 0; bank < 2; ++bank) {
            for (uint32_t i = 0; i < count; ++i) {
                auto* flags = reinterpret_cast<float*>(
                    context_->window + bank * Mc2Barrier::BankBytes(context_->localSize, context_->railSize) +
                    i * Mc2Barrier::SLOT_BYTES);
                std::fill(flags, flags + 8, mapping[peers_[i].rank] >= 0 ? 1.0F : 0.0F);
            }
        }
        ICPU_RUN_KF(distribute_barrier_extend, cores, reinterpret_cast<GM_ADDR>(context_), nullptr, nullptr, elastic,
                    nullptr, nullptr, tiling);
        EXPECT_EQ(*reinterpret_cast<uint64_t*>(context_->window +
                                               Mc2Barrier::SequenceOffset(context_->localSize, context_->railSize)),
                  0U);
        for (uint32_t bank = 0; bank < 2; ++bank) {
            const uint32_t phases = context_->railSize > 1 ? count : context_->localSize;
            for (uint32_t i = 0; i < phases; ++i) {
                const auto* flags = reinterpret_cast<const float*>(
                    context_->window + bank * Mc2Barrier::BankBytes(context_->localSize, context_->railSize) +
                    i * Mc2Barrier::SLOT_BYTES);
                EXPECT_TRUE(std::all_of(flags, flags + 8, [](float value) { return value == 0.0F; }));
            }
        }
    }

    void ReverseElasticMapping(uint8_t* elastic)
    {
        auto* info = reinterpret_cast<int32_t*>(elastic);
        for (uint32_t rank = 0; rank < context_->worldSize; ++rank) {
            if (info[4U + rank] >= 0) {
                info[4U + rank] = info[1] - 1 - info[4U + rank];
                info[4U + context_->worldSize + info[4U + rank]] = rank;
            }
        }
    }

    bool IsElasticParticipant(const ElasticRound& round, uint32_t rank)
    {
        return round.info == nullptr || reinterpret_cast<const int32_t*>(round.info)[4U + rank] >= 0;
    }

    std::vector<ElasticRound> MakeElasticRounds(uint32_t world, const std::vector<uint32_t>& removed)
    {
        std::vector<ElasticRound> rounds{{nullptr, world, world - 1U}};
        auto* info = MakeElasticInfo(world, removed);
        for (uint32_t i = 0; i < 2U; ++i) {
            ElasticRound round{info, world - static_cast<uint32_t>(removed.size()), world - 1U - i};
            while (!IsElasticParticipant(round, round.lateRank)) {
                round.lateRank = (round.lateRank + world - 1U) % world;
            }
            rounds.push_back(round);
        }
        rounds.push_back({nullptr, world, world - 1U});
        return rounds;
    }

    void RunElasticRounds(Mc2Barrier::Context* context, uint8_t* tiling, uint32_t* control,
                          const std::vector<ElasticRound>& rounds)
    {
        for (uint32_t i = 0; i < rounds.size(); ++i) {
            const auto& round = rounds[i];
            if (!IsElasticParticipant(round, context->rank)) {
                continue;
            }
            auto* gate = control + 4U * i;
            while (__atomic_load_n(gate + 3, __ATOMIC_SEQ_CST) == 0U ||
                   (context->rank == round.lateRank && __atomic_load_n(gate, __ATOMIC_SEQ_CST) == 0U)) {
                std::this_thread::yield();
            }
            reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo =
                round.info != nullptr;
            __atomic_add_fetch(gate + 1, 1U, __ATOMIC_SEQ_CST);
            ICPU_RUN_KF(distribute_barrier_extend, 2U + i % 3U, reinterpret_cast<GM_ADDR>(context), nullptr, nullptr,
                        round.info, nullptr, nullptr, tiling);
            __atomic_add_fetch(gate + 2, 1U, __ATOMIC_SEQ_CST);
        }
    }

    void LaunchElasticRounds(Mc2Barrier::Context* context, uint8_t* tiling, uint32_t* control,
                             const std::vector<ElasticRound>& rounds)
    {
        const pid_t worker = fork();
        ASSERT_GE(worker, 0);
        if (worker == 0) {
            setpgid(0, 0);
            RunElasticRounds(context, tiling, control, rounds);
            _exit(0);
        }
        workers_.push_back(worker);
        setpgid(worker, worker);
    }

    void CheckElasticArrivals(uint32_t* control, const std::vector<ElasticRound>& rounds)
    {
        for (uint32_t i = 0; i < rounds.size(); ++i) {
            SCOPED_TRACE(i);
            auto* gate = control + 4U * i;
            __atomic_store_n(gate + 3, 1U, __ATOMIC_SEQ_CST);
            ASSERT_NO_FATAL_FAILURE(WaitForCount(gate + 1, rounds[i].participants - 1U));
            std::this_thread::sleep_for(std::chrono::milliseconds(i == 1U ? 200 : 20));
            ASSERT_EQ(__atomic_load_n(gate + 2, __ATOMIC_SEQ_CST), 0U);
            __atomic_store_n(gate, 1U, __ATOMIC_SEQ_CST);
            ASSERT_NO_FATAL_FAILURE(WaitForCount(gate + 2, rounds[i].participants));
        }
    }

    void CheckElasticResources(const std::vector<Mc2Barrier::Context*>& contexts)
    {
        for (const auto* context : contexts) {
            EXPECT_EQ(*reinterpret_cast<uint64_t*>(context->window +
                                                   Mc2Barrier::SequenceOffset(context->localSize, context->railSize)),
                      2U);
            for (uint32_t bank = 0; bank < 2; ++bank) {
                const uint64_t offset = bank * Mc2Barrier::BankBytes(context->localSize, context->railSize);
                for (uint32_t slot = 0; slot < context->localSize + context->railSize; ++slot) {
                    const auto* flags =
                        reinterpret_cast<const uint32_t*>(context->window + offset + slot * Mc2Barrier::SLOT_BYTES);
                    EXPECT_TRUE(std::all_of(flags, flags + 8, [](uint32_t value) { return value == 0U; }));
                }
            }
        }
        EXPECT_EQ(Mc2BarrierUt::g_commState->invalidCalls, 0U);
        EXPECT_EQ(Mc2BarrierUt::g_commState->trapCalls, 0U);
    }

    void CheckElasticTransport(uint32_t local, uint32_t rail, Mc2Barrier::Protocol protocol,
                               const std::vector<uint32_t>& removed, bool mixed = false)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        const uint32_t world = local * rail;
        std::vector<uint64_t> windows(world);
        for (auto& window : windows) {
            window = reinterpret_cast<uint64_t>(Allocate(Mc2Barrier::WindowBytes(local, rail)));
        }
        const auto rounds = MakeElasticRounds(world, removed);
        auto* control = reinterpret_cast<uint32_t*>(Allocate(rounds.size() * 4U * sizeof(uint32_t)));
        std::vector<Mc2Barrier::Context*> contexts;
        for (uint32_t rank = 0; rank < world; ++rank) {
            context_ = InitTransportRank(rank, windows, protocol, mixed, local);
            contexts.push_back(context_);
            ASSERT_NO_FATAL_FAILURE(LaunchElasticRounds(context_, InitTiling(4), control, rounds));
        }
        ASSERT_NO_FATAL_FAILURE(CheckElasticArrivals(control, rounds));
        ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
        CheckElasticResources(contexts);
    }

    void LaunchElasticReuse(Mc2Barrier::Context* context, uint8_t* tiling, uint8_t* elastic, uint8_t* next)
    {
        const pid_t worker = fork();
        ASSERT_GE(worker, 0);
        if (worker == 0) {
            setpgid(0, 0);
            AscendC::SetKernelMode(KernelMode::AIV_MODE);
            auto& info = reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo;
            if (context->rank != 3U) {
                info.isInputElasticInfo = true;
                ICPU_RUN_KF(distribute_barrier_extend, 2, reinterpret_cast<GM_ADDR>(context), nullptr, nullptr, elastic,
                            nullptr, nullptr, tiling);
                if (context->rank == 0U) {
                    __atomic_add_fetch(Mc2BarrierUt::g_drainGate + 2, 1U, __ATOMIC_SEQ_CST);
                }
            } else {
                while (__atomic_load_n(Mc2BarrierUt::g_drainGate, __ATOMIC_SEQ_CST) == 0U) {
                    std::this_thread::yield();
                }
            }
            if (context->rank != 3U || next == nullptr) {
                info.isInputElasticInfo = next != nullptr;
                ICPU_RUN_KF(distribute_barrier_extend, 2, reinterpret_cast<GM_ADDR>(context), nullptr, nullptr, next,
                            nullptr, nullptr, tiling);
            }
            _exit(0);
        }
        workers_.push_back(worker);
        setpgid(worker, worker);
    }

    void CheckElasticReuse(Mc2Barrier::Protocol protocol, bool restoreFull)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        std::vector<uint64_t> windows(4);
        for (auto& window : windows) {
            window = reinterpret_cast<uint64_t>(Allocate(Mc2Barrier::WindowBytes(2, 2)));
        }
        std::vector<Mc2Barrier::Context*> contexts;
        for (uint32_t rank = 0; rank < windows.size(); ++rank) {
            context_ = InitTransportRank(rank, windows, protocol, false);
            contexts.push_back(context_);
            ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(context_, InitTiling(2), nullptr, nullptr, 1));
        }
        ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
        auto* gate = reinterpret_cast<uint32_t*>(Allocate(3U * sizeof(uint32_t)));
        auto* elastic = MakeElasticInfo(4, {3});
        const auto* slowPeers = reinterpret_cast<const Mc2Barrier::Peer*>(contexts[2] + 1);
        Mc2BarrierUt::g_delayedChannel = reinterpret_cast<Mc2BarrierUt::Channel*>(slowPeers[2].handle);
        Mc2BarrierUt::g_drainGate = gate;
        for (auto* context : contexts) {
            context_ = context;
            ASSERT_NO_FATAL_FAILURE(
                LaunchElasticReuse(context, InitTiling(2), elastic, restoreFull ? nullptr : elastic));
        }
        ASSERT_NO_FATAL_FAILURE(WaitForCount(gate + 1, 1U));
        const auto* fastPeers = reinterpret_cast<const Mc2Barrier::Peer*>(contexts[0] + 1);
        const auto* fastChannel = reinterpret_cast<const Mc2BarrierUt::Channel*>(fastPeers[3].handle);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (__atomic_load_n(gate + 2, __ATOMIC_SEQ_CST) == 0U &&
               __atomic_load_n(&fastChannel->writes[1], __ATOMIC_SEQ_CST) < 2U &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_EQ(__atomic_load_n(gate + 2, __ATOMIC_SEQ_CST), 0U);
        ASSERT_EQ(__atomic_load_n(&fastChannel->writes[1], __ATOMIC_SEQ_CST), 2U);
        __atomic_store_n(gate, 1U, __ATOMIC_SEQ_CST);
        ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
        for (const auto* context : contexts) {
            EXPECT_EQ(*reinterpret_cast<uint64_t*>(context->window + Mc2Barrier::SequenceOffset(2, 2)),
                      restoreFull ? 2U : 1U);
        }
        EXPECT_EQ(Mc2BarrierUt::g_commState->invalidCalls, 0U);
        EXPECT_EQ(Mc2BarrierUt::g_commState->trapCalls, 0U);
    }

    void CheckTransportFailure(Mc2Barrier::Protocol protocol, Mc2BarrierUt::Failure failure)
    {
        Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
        Mc2BarrierUt::g_commState->failure = failure;
        InitContext(1, 2);
        context_->protocolMask = 1U << static_cast<uint32_t>(protocol);
        peers_[2].protocol = protocol;
        auto* channel = InitTransportChannel(*context_, peers_[2]);
        peers_[2].handle = reinterpret_cast<uint64_t>(channel);
        auto* incoming = reinterpret_cast<float*>(context_->window + 5U * Mc2Barrier::SLOT_BYTES);
        std::fill(incoming, incoming + 8, 1.0F);
        auto* tiling = InitTiling(2);
        reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputTimeOut = true;
        auto* timeout = reinterpret_cast<int32_t*>(Allocate(sizeof(int32_t)));
        *timeout = 500000;
        ICPU_RUN_KF(distribute_barrier_extend, 2, reinterpret_cast<GM_ADDR>(context_), nullptr,
                    reinterpret_cast<GM_ADDR>(timeout), nullptr, nullptr, nullptr, tiling);
        const bool initialized = failure != Mc2BarrierUt::Failure::INIT;
        const bool submitted = failure == Mc2BarrierUt::Failure::DRAIN || failure == Mc2BarrierUt::Failure::UNLOCK;
        EXPECT_EQ(Mc2BarrierUt::g_commState->initCalls[protocol == Mc2Barrier::Protocol::ROCE], 2U);
        EXPECT_EQ(channel->writes[1], initialized && failure != Mc2BarrierUt::Failure::LOCK ? 1U : 0U);
        EXPECT_EQ(channel->drains[1], submitted ? 1U : 0U);
        EXPECT_EQ(channel->writes[0] + channel->drains[0], 0U);
        EXPECT_EQ(Mc2BarrierUt::g_commState->invalidCalls, 0U);
        EXPECT_EQ(channel->locked, 0U);
        EXPECT_TRUE(std::all_of(incoming, incoming + 8, [](float value) { return value == 1.0F; }));
        EXPECT_EQ(*reinterpret_cast<uint32_t*>(context_->window + Mc2Barrier::SequenceOffset(1, 2)),
                  initialized ? 1U : 0U);
    }

    void WaitForWorkers()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        for (pid_t& worker : workers_) {
            SCOPED_TRACE(worker);
            int status = 0;
            pid_t result = 0;
            while (result == 0 && std::chrono::steady_clock::now() < deadline) {
                result = waitpid(worker, &status, WNOHANG);
                if (result == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            ASSERT_EQ(result, worker) << "Concurrent barrier test did not finish within 60 seconds";
            ASSERT_TRUE(WIFEXITED(status));
            ASSERT_EQ(WEXITSTATUS(status), 0);
            worker = 0;
        }
        workers_.clear();
    }

    Mc2Barrier::Context* context_ = nullptr;
    Mc2Barrier::Peer* peers_ = nullptr;
    uint8_t* elasticInfo_ = nullptr;
    std::vector<uint8_t*> allocations_;
    std::vector<pid_t> workers_;
    std::vector<Mc2BarrierUt::Channel*> transportChannels_;
    Mc2BarrierUt::Transport urmaTransport_ = Mc2BarrierUt::Transport::CTP;
};

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
TEST_F(DistributeBarrierExtendTest, UrmaTwoPhasesAndAlternatingBanks)
{
    CheckConcurrentTransport(Mc2Barrier::Protocol::URMA);
}

TEST_F(DistributeBarrierExtendTest, RoceTwoPhasesAndAlternatingBanks)
{
    elasticInfo_ = Allocate(16U * sizeof(int32_t));
    CheckConcurrentTransport(Mc2Barrier::Protocol::ROCE);
}

TEST_F(DistributeBarrierExtendTest, MixedRailTwoPhasesAndAlternatingBanks)
{
    elasticInfo_ = Allocate(16U * sizeof(int32_t));
    CheckConcurrentTransport(Mc2Barrier::Protocol::URMA, true);
}

TEST_F(DistributeBarrierExtendTest, EveryRankWaitsForDelayedArrivalOnUrma)
{
    CheckEveryLateRank(16, 2, Mc2Barrier::Protocol::URMA);
}
#endif

TEST_F(DistributeBarrierExtendTest, UbmemDelayedArrivalHonorsMicrosecondTimeout)
{
    CheckEveryLateRank(8, 1, Mc2Barrier::Protocol::UBMEM, 5000000);
}

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
TEST_F(DistributeBarrierExtendTest, CommunicationFailuresStopBeforeWait)
{
    for (auto protocol : {Mc2Barrier::Protocol::URMA, Mc2Barrier::Protocol::ROCE}) {
        SCOPED_TRACE(static_cast<uint32_t>(protocol));
        for (auto failure : {Mc2BarrierUt::Failure::INIT, Mc2BarrierUt::Failure::LOCK, Mc2BarrierUt::Failure::WRITE,
                             Mc2BarrierUt::Failure::DRAIN, Mc2BarrierUt::Failure::UNLOCK}) {
            if (protocol == Mc2Barrier::Protocol::ROCE &&
                (failure == Mc2BarrierUt::Failure::LOCK || failure == Mc2BarrierUt::Failure::UNLOCK)) {
                continue;
            }
            SCOPED_TRACE(static_cast<uint32_t>(failure));
            CheckTransportFailure(protocol, failure);
        }
    }
}
#endif

TEST_F(DistributeBarrierExtendTest, RejectsInvalidContext)
{
    InitContext(4, 1);
    auto* tiling = InitTiling(2);
    context_->magic = 0;
    CheckRejectedContext(tiling);
    context_->magic = Mc2Barrier::MAGIC;
    auto& info = reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo;
    --info.totalWinSize;
    CheckRejectedContext(tiling);
    ++info.totalWinSize;
    info.worldSize = 8;
    CheckRejectedContext(tiling);
    context_->worldSize = 8;
    CheckRejectedContext(tiling);
}

TEST_F(DistributeBarrierExtendTest, RejectsInvalidElasticMemberships)
{
    InitContext(4, 1);
    auto* tiling = InitTiling(2);
    reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo = true;
    auto* elastic = Allocate(12 * sizeof(int32_t));
    const int32_t memberships[][12] = {
        {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},     {1, 2, 0, 12, 0, 1, 2, -1, 0, 1, -1, -1},
        {1, 2, 0, 12, 0, 1, 1, -1, 0, 1, -1, -1},  {1, 2, 0, 12, 0, -1, -1, -1, 0, 1, -1, -1},
        {1, 2, 0, 12, 0, 1, -2, -1, 0, 1, -1, -1}, {1, 2, 0, 12, -1, 0, 1, -1, 1, 2, -1, -1}};
    for (const auto& members : memberships) {
        SCOPED_TRACE(::testing::PrintToString(members));
        std::memcpy(elastic, members, sizeof(members));
        CheckRejectedContext(tiling, elastic);
    }
}

TEST_F(DistributeBarrierExtendTest, Elastic204800RanksAndNoncontiguousPeers)
{
    InitLargeContext(16, 12800);
    auto* tiling = InitTiling(2);
    reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo = true;
    auto* elastic = MakeElasticInfo(context_->worldSize, {0, 17, 8193, 111111, 131071, 204798});
    for (uint32_t cores : {2U, 8U}) {
        CheckLargeElasticRound(tiling, elastic, cores);
        ReverseElasticMapping(elastic);
    }
}

TEST_F(DistributeBarrierExtendTest, ElasticMembershipChangesAtSameInputAddress)
{
    InitLargeContext(25, 41);
    auto* tiling = InitTiling(4);
    reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo = true;
    auto* elastic = MakeElasticInfo(context_->worldSize, {peers_[0].rank, peers_[25].rank});
    CheckLargeElasticRound(tiling, elastic, 4);
    auto* next = MakeElasticInfo(context_->worldSize, {peers_[1].rank, peers_[26].rank});
    std::memcpy(elastic, next, (4U + 2U * context_->worldSize) * sizeof(int32_t));
    ReverseElasticMapping(elastic);
    CheckLargeElasticRound(tiling, elastic, 4);
}

TEST_F(DistributeBarrierExtendTest, RejectsLargeCrossChunkInverseAndMembershipErrors)
{
    InitLargeContext(1024, 200);
    auto* tiling = InitTiling(2);
    reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo = true;
    auto* elastic = MakeElasticInfo(context_->worldSize, {0});
    ReverseElasticMapping(elastic);
    auto* info = reinterpret_cast<int32_t*>(elastic);
    for (uint32_t index : {1U, 4U + 204799U, 4U + 1024U, 4U + context_->worldSize + 1U}) {
        SCOPED_TRACE(index);
        const int32_t saved = info[index];
        info[index] = index == 1U ? 1 : -1;
        CheckRejectedContext(tiling, elastic);
        info[index] = saved;
    }
}

TEST_F(DistributeBarrierExtendTest, ElasticPreparationHonorsTimeout)
{
    InitLargeContext(1024, 200);
    auto* tiling = InitTiling(2);
    auto& info = reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo;
    info.isInputElasticInfo = true;
    info.isInputTimeOut = true;
    auto* elastic = MakeElasticInfo(context_->worldSize, {0});
    auto* timeout = reinterpret_cast<int32_t*>(Allocate(sizeof(int32_t)));
    *timeout = 1;
    ICPU_RUN_KF(distribute_barrier_extend, 1, reinterpret_cast<GM_ADDR>(context_), nullptr,
                reinterpret_cast<GM_ADDR>(timeout), elastic, nullptr, nullptr, tiling);
    const auto* window = reinterpret_cast<const uint8_t*>(context_->window);
    EXPECT_TRUE(std::all_of(window, window + context_->windowBytes, [](uint8_t value) { return value == 0U; }));
}

TEST_F(DistributeBarrierExtendTest, RejectsElasticWithoutGuaranteedRailConnectivity)
{
    InitContext(4, 2);
    auto* tiling = InitTiling(2);
    reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo.isInputElasticInfo = true;
    for (const auto& removed : {std::vector<uint32_t>{1, 2, 3, 4}, {1, 2, 3, 4, 5}}) {
        SCOPED_TRACE(::testing::PrintToString(removed));
        CheckRejectedContext(tiling, MakeElasticInfo(8, removed));
    }
}

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
TEST_F(DistributeBarrierExtendTest, MixedElasticDomainsKeepOneCommonRail)
{
    CheckElasticTransport(4, 3, Mc2Barrier::Protocol::URMA, {0, 4, 10}, true);
}
#endif

TEST_F(DistributeBarrierExtendTest, ElasticSingleDomainKeepsOneParticipant)
{
    CheckElasticTransport(4, 1, Mc2Barrier::Protocol::UBMEM, {0, 1, 2});
}

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
TEST_F(DistributeBarrierExtendTest, ElasticRailExitPrecedesNextCall)
{
    for (auto protocol : {Mc2Barrier::Protocol::URMA, Mc2Barrier::Protocol::ROCE}) {
        for (bool restoreFull : {false, true}) {
            SCOPED_TRACE(::testing::PrintToString(std::make_pair(protocol, restoreFull)));
            ASSERT_NO_FATAL_FAILURE(CheckElasticReuse(protocol, restoreFull));
        }
    }
}
#endif

#if ASC_DEVKIT_VERSION_NUM >= 90200000
TEST_F(DistributeBarrierExtendTest, RtpElasticRailExitPrecedesNextCall)
{
    urmaTransport_ = Mc2BarrierUt::Transport::RTP;
    for (bool restoreFull : {false, true}) {
        SCOPED_TRACE(restoreFull);
        ASSERT_NO_FATAL_FAILURE(CheckElasticReuse(Mc2Barrier::Protocol::URMA, restoreFull));
    }
}
#endif

TEST_F(DistributeBarrierExtendTest, LegacyContextRemainsCompatible)
{
    auto* context = new (Allocate(sizeof(Mc2Aclnn::Mc2MoeContext))) Mc2Aclnn::Mc2MoeContext{};
    for (uint32_t rank = 0; rank < 2; ++rank) {
        context->epHcclBuffer_[rank] = reinterpret_cast<uint64_t>(Allocate(2U * 1024U * 1024U));
    }
    auto* tiling = Allocate(sizeof(DistributeBarrierTilingData));
    auto& info = reinterpret_cast<DistributeBarrierTilingData*>(tiling)->distributeBarrierInfo;
    info.worldSize = 2;
    info.aivNum = 3;
    info.isMc2Context = true;
    for (uint32_t bank = 0; bank < 2; ++bank) {
        const uint64_t offset = bank * 512U * 1024U;
        auto* incoming = reinterpret_cast<float*>(context->epHcclBuffer_[0] + offset + 512U);
        std::fill(incoming, incoming + 8, 1.0F);
        ICPU_RUN_KF(distribute_barrier_extend, 3, reinterpret_cast<GM_ADDR>(context), nullptr, nullptr, nullptr,
                    nullptr, nullptr, tiling);
        EXPECT_TRUE(std::all_of(incoming, incoming + 8, [](float value) { return value == 0.0F; }));
        EXPECT_EQ(*reinterpret_cast<float*>(context->epHcclBuffer_[1] + offset), 1.0F);
        for (uint32_t core = 0; core < 3; ++core) {
            const uint64_t sequence = context->epHcclBuffer_[0] + 900U * 1024U + core * 32U * sizeof(uint32_t);
            EXPECT_EQ(*reinterpret_cast<uint32_t*>(sequence), 1U - bank);
        }
    }
}

#if ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
TEST_F(DistributeBarrierExtendTest, TimeoutMonitorRunsWhileUrmaDrainIsBlocked)
{
    CheckTimeoutDuringDrain(Mc2Barrier::Protocol::URMA);
}

TEST_F(DistributeBarrierExtendTest, ElasticTimeoutMonitorRunsWhileRoceDrainIsBlocked)
{
    CheckTimeoutDuringDrain(Mc2Barrier::Protocol::ROCE, true);
}

TEST_F(DistributeBarrierExtendTest, HistoricalRailFlagsCannotReleaseBeforeRank17)
{
    Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
    std::vector<uint64_t> windows(32);
    for (auto& window : windows) {
        window = reinterpret_cast<uint64_t>(Allocate(Mc2Barrier::WindowBytes(16, 2)));
    }
    std::vector<Mc2Barrier::Context*> contexts;
    std::vector<uint8_t*> tilings;
    for (uint32_t rank = 0; rank < windows.size(); ++rank) {
        context_ = InitTransportRank(rank, windows, Mc2Barrier::Protocol::URMA, false, 16);
        contexts.push_back(context_);
        tilings.push_back(InitTiling(2));
        ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(context_, tilings.back(), nullptr, nullptr, 2));
    }
    ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
    auto* stale =
        reinterpret_cast<Mc2BarrierUt::Channel*>(reinterpret_cast<Mc2Barrier::Peer*>(contexts[18] + 1)[16].handle);
    ASSERT_NE(stale, nullptr);
    std::memcpy(stale->destination[1], stale->payloads[1], Mc2BarrierUt::FLAG_BYTES);
    std::fill(std::begin(Mc2BarrierUt::g_commState->completed), std::end(Mc2BarrierUt::g_commState->completed), 0U);
    auto* gate = reinterpret_cast<uint32_t*>(Allocate(2 * sizeof(uint32_t)));
    for (uint32_t rank = 0; rank < windows.size(); ++rank) {
        ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(contexts[rank], tilings[rank], nullptr, gate, 2));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (__atomic_load_n(gate + 1, __ATOMIC_SEQ_CST) < 31U && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(__atomic_load_n(gate + 1, __ATOMIC_SEQ_CST), 31U);
    const auto observeUntil = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (__atomic_load_n(&Mc2BarrierUt::g_commState->completed[0], __ATOMIC_SEQ_CST) == 0U &&
           std::chrono::steady_clock::now() < observeUntil) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const auto completed = __atomic_load_n(&Mc2BarrierUt::g_commState->completed[0], __ATOMIC_SEQ_CST);
    ASSERT_EQ(completed, 0U) << "Historical rail flags released a rank while rank 17 was held";
    __atomic_store_n(gate, 1U, __ATOMIC_SEQ_CST);
    ASSERT_NO_FATAL_FAILURE(WaitForWorkers());
    EXPECT_EQ(Mc2BarrierUt::g_commState->completed[0], 32U);
    EXPECT_EQ(Mc2BarrierUt::g_commState->completed[1], 32U);
}
#endif

TEST_F(DistributeBarrierExtendTest, HistoricalLocalFlagsCannotReleaseBeforeCurrentSignals)
{
    InitContext(32, 1);
    auto* tiling = InitTiling(2);
    CheckRound(1, tiling);
    CheckRound(0, tiling);
    const uint64_t bank = Mc2Barrier::BankBytes(32, 1);
    auto* flags = reinterpret_cast<uint8_t*>(context_->window + bank);
    const auto* historical = reinterpret_cast<const uint8_t*>(peers_[27].window + bank);
    for (uint32_t slot = 0; slot < 32; ++slot) {
        std::memcpy(flags + slot * Mc2Barrier::SLOT_BYTES, historical, Mc2BarrierUt::FLAG_BYTES);
    }
    auto* sequence = reinterpret_cast<uint64_t*>(context_->window + Mc2Barrier::SequenceOffset(32, 1));
    const uint64_t previous = *sequence;
    CheckStaleSignals(tiling);
    EXPECT_EQ(*sequence, previous + 1U);
}

TEST_F(DistributeBarrierExtendTest, EpochExceedsUint32Range)
{
    for (uint32_t rail : {1U, 2U}) {
        InitContext(2, rail);
        *reinterpret_cast<uint64_t*>(context_->window + Mc2Barrier::SequenceOffset(2, rail)) = (uint64_t{1} << 32) + 2U;
        auto* tiling = InitTiling(2);
        CheckRound(1, tiling);
        CheckRound(0, tiling);
    }
}

TEST_F(DistributeBarrierExtendTest, RailEpochChecksEveryWord)
{
    constexpr uint32_t words = Mc2BarrierUt::FLAG_BYTES / sizeof(uint64_t);
    for (uint32_t word = 0; word < words; ++word) {
        SCOPED_TRACE(word);
        InitContext(1, 2);
        const uint64_t sequence = context_->window + Mc2Barrier::SequenceOffset(1, 2);
        *reinterpret_cast<uint64_t*>(sequence) = 2U;
        const uint64_t offset = Mc2Barrier::BankBytes(1, 2) + Mc2Barrier::PhaseOffset(1, 1) + Mc2Barrier::SLOT_BYTES;
        auto* incoming = reinterpret_cast<uint64_t*>(context_->window + offset);
        uint64_t stale[words] = {3U, 3U, 3U, 3U};
        stale[word] = 1U;
        stale[(word + 1U) % words] = 5U;
        std::copy(std::begin(stale), std::end(stale), incoming);
        CheckStaleSignals(InitTiling(2));
        EXPECT_EQ(*reinterpret_cast<uint64_t*>(sequence), 3U);
        EXPECT_TRUE(std::equal(std::begin(stale), std::end(stale), incoming));
    }
}

TEST_F(DistributeBarrierExtendTest, RailEpochCannotWrapIntoOldSignals)
{
    Mc2BarrierUt::g_commState = new (Allocate(sizeof(Mc2BarrierUt::CommState))) Mc2BarrierUt::CommState;
    InitContext(1, 2);
    auto* sequence = reinterpret_cast<uint64_t*>(context_->window + Mc2Barrier::SequenceOffset(1, 2));
    *sequence = ~uint64_t{0};
    ASSERT_NO_FATAL_FAILURE(LaunchTransportRank(context_, InitTiling(2), nullptr));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (__atomic_load_n(&Mc2BarrierUt::g_commState->trapCalls, __ATOMIC_SEQ_CST) == 0U &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_GT(Mc2BarrierUt::g_commState->trapCalls, 0U);
    EXPECT_EQ(*sequence, ~uint64_t{0});
}

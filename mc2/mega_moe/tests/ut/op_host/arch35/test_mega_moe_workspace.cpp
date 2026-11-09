/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "../../../../op_kernel/arch35_92/common/mega_moe_gm_layout.h"

namespace MegaMoeUT {
using namespace MegaMoeImpl;

static constexpr CclComputeLayoutParams MakeLayoutParams()
{
    CclComputeLayoutParams p{};
    p.bs = 257;
    p.h = 1024;
    p.hiddenDim = 512;
    p.maxOutputSize = 4096;
    p.moeExpertPerRank = 16;
    p.epWorldSize = 8;
    p.aicNum = 36;
    p.serverNum = 2;
    p.numMaxTokensPerRank = 257;
    p.topK = 6;
    p.mGroupsPerWave = 1;
    p.combineSyncSlotCountPerExpert = 16;
    p.reserveGmm1Output = true;
    p.reserveCumsum = true;
    return p;
}

TEST(MegaMoeWorkspace, LayoutsSupportConstantEvaluation)
{
    constexpr bool valid = []() {
        for (int64_t topology = TOPO_TYPE_MTE; topology <= TOPO_TYPE_URMA_ROCE; ++topology) {
            auto params = MakeLayoutParams();
            params.topoType = topology;
            params.sharedExpertNum = 1;
            params.sharedInputIndependent = true;
            params.topkWeightsPrefetch = true;
            params.dedupMode = topology == TOPO_TYPE_MTE ? 3 : DEDUP_MODE_SERVER_COMBINE;
            params.ring = CalcCclStageRingConfig(params);
            const WorkspaceLayout arch35(params);
            const WorkspaceLayout arch92(params, WorkspaceLayout::Arch92{GMM_MODE_A4W4_ND, GMM_MODE_A4W4_NZ});
            CclCommLayoutParams comm{};
            comm.topoType = topology;
            comm.h = params.h;
            comm.epWorldSize = params.epWorldSize;
            comm.moeExpertPerRank = params.moeExpertPerRank;
            comm.numMaxTokensPerRank = params.numMaxTokensPerRank;
            comm.topK = params.topK;
            comm.yDtypeSize = SIZE_BF_16;
            comm.elemsPerByte = 1;
            comm.topkWeightsPrefetch = true;
            comm.serverNum = params.serverNum;
            comm.serverCombineReduce = IsServerCombineDedupOn(params.dedupMode);
            const PeermemInfo window(comm, arch35.workspaceSize);
            if (arch35.workspaceSize <= 0 || arch92.workspaceSize <= 0 ||
                window.workspace.offsetBytes % MemoryLayoutDetail::ALIGN_512 != 0 ||
                window.workspace.sizeBytes != arch35.workspaceSize) {
                return false;
            }
        }
        return true;
    }();
    static_assert(valid, "GM layouts must support compile-time construction");
    EXPECT_TRUE(valid);
}

TEST(MegaMoeWorkspace, ModeCombinationsHaveDisjointBoundedRegions)
{
    for (int topo : {0, 1, 2}) {
        for (int mode : {0, 1, 2, 3, 4}) {
            for (bool prefetch : {false, true}) {
                auto p = MakeLayoutParams();
                p.topoType = topo;
                p.dedupMode = mode;
                p.topkWeightsPrefetch = prefetch;
                p.sharedExpertNum = 1;
                p.sharedInputIndependent = true;
                p.ring = CalcCclStageRingConfig(p);
                WorkspaceLayout w(p);
                // 连续清零区仅包含仍有消费者的 GM 标志，不再预留 GMM1 软同步区。
                const int64_t liveFlagBytes =
                    w.sync.activationReady.sizeBytes + w.sync.dispatchReady.sizeBytes + w.sync.countsReady.sizeBytes +
                    w.mte.sync.sharedActivationReady.sizeBytes + w.mte.sync.dedupReady.sizeBytes +
                    w.mte.sync.gmm2Ready.sizeBytes + w.mte.sync.sharedGmm2Tiles.sizeBytes +
                    w.mte.sync.aiv1TopkReady.sizeBytes + w.urma.sync.combineCounters.sizeBytes +
                    w.urma.sync.sharedGmm2Tiles.sizeBytes;
                EXPECT_EQ(w.reset.flags.sizeBytes, liveFlagBytes);
                std::vector<GmBuffer> regions{w.routed.gmm1Input.data,
                                              w.routed.gmm1Input.scale,
                                              w.routed.gmm1Output,
                                              w.routed.gmm2Input.data,
                                              w.routed.gmm2Input.scale,
                                              w.routed.gmm2Output,
                                              w.routing.recvCounts,
                                              w.routing.receivedTokenMetadata,
                                              w.routing.recvPrefixBackup,
                                              w.routing.tokenWeights,
                                              w.sync.activationReady,
                                              w.sync.dispatchReady,
                                              w.sync.countsReady,
                                              w.mte.sync.sharedActivationReady,
                                              w.mte.sync.aiv1TopkReady,
                                              w.mte.sync.gmm2Ready,
                                              w.mte.sync.sharedGmm2Tiles,
                                              w.mte.sync.dedupReady,
                                              w.mte.dedupRowDescriptors,
                                              w.urma.mask,
                                              w.urma.relaySendQueue,
                                              w.urma.remoteReadySnapshot,
                                              w.urma.sync.combineCounters,
                                              w.urma.sync.sharedGmm2Tiles,
                                              w.urma.serverCombine.tokenIds,
                                              w.urma.serverCombine.tokenBitmaps,
                                              w.urma.serverCombine.completedCounts,
                                              w.shared.gmm1Input.data,
                                              w.shared.gmm1Input.scale,
                                              w.shared.gmm1Output,
                                              w.shared.gmm2Input.data,
                                              w.shared.gmm2Input.scale,
                                              w.shared.gmm2Output};
                std::sort(regions.begin(), regions.end(),
                          [](GmBuffer a, GmBuffer b) { return a.offsetBytes < b.offsetBytes; });
                int64_t end = 0;
                for (const auto& region : regions) {
                    if (region.sizeBytes == 0) {
                        continue;
                    }
                    ASSERT_GE(region.offsetBytes, end);
                    end = region.offsetBytes + region.sizeBytes;
                    ASSERT_LE(end, w.workspaceSize);
                }
            }
        }
    }
}

TEST(MegaMoeWorkspace, ResetRangesReferenceAllocatedFlags)
{
    auto p = MakeLayoutParams();
    p.topkWeightsPrefetch = true;
    WorkspaceLayout w(p);
    EXPECT_EQ(w.reset.flags.offsetBytes, w.sync.activationReady.offsetBytes);
    EXPECT_EQ(w.reset.flags.sizeBytes % sizeof(int32_t), 0);
    p.topkWeightsPrefetch = false;
    EXPECT_EQ(WorkspaceLayout(p).reset.flags.sizeBytes, w.reset.flags.sizeBytes);
}

TEST(MegaMoeWorkspace, RegionPointersUseCallerBaseAndByteOffsets)
{
    alignas(512) uint8_t storage[1024]{};
    GmBuffer region{64, 128};
    EXPECT_EQ(region.get_ptr<int32_t>(storage), reinterpret_cast<int32_t*>(storage + 64));
    EXPECT_EQ(region.get_ptr(storage + 128), storage + 192);
    EXPECT_EQ(region.get_ptr(nullptr), nullptr);
    EXPECT_EQ(GmBuffer{}.get_ptr(storage), nullptr);
    EXPECT_EQ((GmBuffer{0, 0}).get_ptr(storage), storage);
}

TEST(MegaMoeWorkspace, SharedGmm1InputKeepsTopologyStorageContract)
{
    auto p = MakeLayoutParams();
    p.sharedExpertNum = 1;
    p.sharedInputIndependent = true;
    WorkspaceLayout mte(p);
    EXPECT_GT(mte.shared.gmm1Input.data.sizeBytes, 0);
    EXPECT_EQ(mte.shared.gmm1Input.scale.offsetBytes, -1);
    p.topoType = TOPO_TYPE_URMA;
    WorkspaceLayout urma(p);
    EXPECT_EQ(urma.shared.gmm1Input.data.offsetBytes, -1);
    EXPECT_EQ(urma.shared.gmm1Input.scale.offsetBytes, -1);
    p.topoType = TOPO_TYPE_URMA_ROCE;
    WorkspaceLayout ubx(p);
    EXPECT_EQ(ubx.shared.gmm1Input.data.offsetBytes, -1);
    EXPECT_EQ(ubx.shared.gmm1Input.scale.offsetBytes, -1);
    // 本次仅调整 Arch35 Layered，Arch92 的连续输入布局保留。
    WorkspaceLayout arch92(p, WorkspaceLayout::Arch92{GMM_MODE_A8W8_ND, GMM_MODE_A8W8_ND});
    EXPECT_GT(arch92.shared.gmm1Input.data.sizeBytes, 0);
    EXPECT_GT(arch92.shared.gmm1Input.scale.sizeBytes, 0);
    p.topoType = TOPO_TYPE_MTE;
    p.sharedInputIndependent = false;
    EXPECT_EQ(WorkspaceLayout(p).shared.gmm1Input.data.offsetBytes, -1);
}

TEST(MegaMoeWorkspace, UrmaSharedCompletionCountersAreResetPerTokenGroup)
{
    for (int topo : {TOPO_TYPE_MTE, TOPO_TYPE_URMA, TOPO_TYPE_URMA_ROCE}) {
        for (int bs : {0, 1, 255, 256, 257, 3405}) {
            for (int shared : {0, 1, 4}) {
                auto p = MakeLayoutParams();
                p.topoType = topo;
                p.bs = bs;
                p.sharedExpertNum = shared;
                WorkspaceLayout layout(p);
                const auto& counters = layout.urma.sync.sharedGmm2Tiles;
                const int64_t groups = (bs + 255) / 256;
                const int64_t expectedBytes = topo == TOPO_TYPE_MTE ? 0 : groups * shared * 64;
                EXPECT_EQ(counters.sizeBytes, expectedBytes);
                if (expectedBytes != 0) {
                    EXPECT_EQ(counters.offsetBytes % MemoryLayoutDetail::ALIGN_32, 0);
                    EXPECT_LE(layout.reset.flags.offsetBytes, counters.offsetBytes);
                    EXPECT_EQ(counters.offsetBytes + counters.sizeBytes,
                              layout.reset.flags.offsetBytes + layout.reset.flags.sizeBytes);
                }
                // Arch92 的同步布局保持不变，不分配 Arch35 URMA 的完成计数。
                WorkspaceLayout arch92(p, WorkspaceLayout::Arch92{GMM_MODE_A8W8_ND, GMM_MODE_A8W8_ND});
                EXPECT_EQ(arch92.urma.sync.sharedGmm2Tiles.offsetBytes, -1);
            }
        }
    }
}

TEST(MegaMoeWorkspace, RingReuseAndServerCombineRetainCapacityBoundaries)
{
    auto p = MakeLayoutParams();
    WorkspaceLayout full(p);
    p.ring = CalcCclStageRingConfig(p);
    WorkspaceLayout ring(p);
    EXPECT_LT(ring.routed.gmm1Input.data.sizeBytes, full.routed.gmm1Input.data.sizeBytes);
    EXPECT_LT(ring.workspaceSize, full.workspaceSize);
    p.topoType = TOPO_TYPE_URMA_ROCE;
    p.dedupMode = DEDUP_MODE_SERVER_COMBINE;
    WorkspaceLayout server(p);
    EXPECT_EQ(server.reset.serverCombine.offsetBytes, server.urma.serverCombine.completedCounts.offsetBytes);
    EXPECT_EQ(server.reset.serverCombine.sizeBytes, server.urma.serverCombine.completedCounts.sizeBytes);
    EXPECT_GT(server.urma.serverCombine.tokenIds.sizeBytes, 0);
    EXPECT_GT(server.urma.serverCombine.tokenBitmaps.sizeBytes, 0);
}

TEST(MegaMoeWorkspace, Arch92KeepsFullRowsAndPackedActivation)
{
    auto p = MakeLayoutParams();
    p.reserveGmm1Output = false;
    p.sharedExpertNum = 1;
    p.ring = CalcCclStageRingConfig(p);
    WorkspaceLayout layout(p, WorkspaceLayout::Arch92{GMM_MODE_A4W4_ND, GMM_MODE_A4W4_NZ});
    EXPECT_EQ(layout.routed.gmm1Input.data.sizeBytes, p.maxOutputSize * p.h / 2);
    EXPECT_EQ(layout.routed.gmm1Output.sizeBytes, p.maxOutputSize * p.hiddenDim * sizeof(uint16_t));
    EXPECT_EQ(layout.routed.gmm2Input.data.sizeBytes, p.maxOutputSize * p.hiddenDim / 4);
    EXPECT_EQ(layout.shared.gmm2Input.data.sizeBytes, 33280);
    EXPECT_EQ(layout.routing.recvPrefixBackup.offsetBytes, -1);
}

TEST(MegaMoeWorkspace, Arch92UrmaKeepsOptionalGmmOutputAndFp8Activation)
{
    auto p = MakeLayoutParams();
    p.topoType = TOPO_TYPE_URMA;
    WorkspaceLayout fp8(p, WorkspaceLayout::Arch92{GMM_MODE_A8W8_ND, GMM_MODE_A8W8_ND});
    EXPECT_EQ(fp8.routed.gmm1Output.offsetBytes, -1);
    EXPECT_GT(fp8.routing.recvPrefixBackup.sizeBytes, 0);
    WorkspaceLayout fp4(p, WorkspaceLayout::Arch92{GMM_MODE_A4W4_ND, GMM_MODE_A8W8_ND});
    EXPECT_EQ(fp4.routed.gmm2Input.data.sizeBytes, p.maxOutputSize * p.hiddenDim / 2);
}
} // namespace MegaMoeUT

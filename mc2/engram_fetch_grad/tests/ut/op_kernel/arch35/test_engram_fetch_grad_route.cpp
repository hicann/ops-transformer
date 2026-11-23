/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fetch_grad_route.cpp
 * \brief 分轨路由纯函数单测（route.h 自带 host 编译垫片，零 NPU 依赖）。
 *        示例几何：G2.0（ranksPerSupernode=128、world=204800）与
 *        32P 分卡床（ranksPerSupernode=16、world=32）。
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "../../../../op_kernel/arch35/railaligned/engram_fetch_grad_route.h"

using namespace Mc2Kernel;

static int g_failed = 0;
static int g_total = 0;

#define CHECK_EQ(actual, expected, ...) \
    do { \
        g_total++; \
        if ((actual) != (expected)) { \
            g_failed++; \
            printf("FAIL:%d: expect %u got %u (", __LINE__, static_cast<unsigned>(expected), \
                   static_cast<unsigned>(actual)); \
            printf(__VA_ARGS__); \
            printf(")\n"); \
        } \
    } while (0)

#define CHECK_TRUE(cond, ...) \
    do { \
        g_total++; \
        if (!(cond)) { \
            g_failed++; \
            printf("FAIL:%d: cond false (", __LINE__); \
            printf(__VA_ARGS__); \
            printf(")\n"); \
        } \
    } while (0)

// ═══ 坐标验算（rank = sn×ranksPerSupernode + L）═══
static void TestCoordinates()
{
    // G2.0：ranksPerSupernode=128、world=204800 → 1600 台超节点
    RailTopo topo = MakeRailTopo(128U, 204800U);
    CHECK_EQ(topo.supernodeCnt, 1600U, "sn=world/ranksPerSupernode");
    CHECK_EQ(topo.railCnt, 1599U, "rail=world/ranksPerSupernode-1");
    CHECK_EQ(topo.writerSlots, 1728U, "W=ranksPerSupernode+world/ranksPerSupernode");

    RankCoord me = DecodeRank(topo, 68200U);
    CHECK_EQ(me.supernode, 532U, "me.sn=68200/128");
    CHECK_EQ(me.local, 104U, "me.L=68200 mod 128");

    // 轨道对端：dst 超节点 970 的同 L 卡
    RankCoord dst = DecodeRank(topo, 970U * 128U + 9U);
    CHECK_EQ(dst.supernode, 970U, "dst.sn");
    CHECK_EQ(dst.local, 9U, "dst.L");
}

// ═══ 三桶分类：互斥完备 + 两条公式 ═══
static void TestClassifyAndFormulas()
{
    RailTopo topo = MakeRailTopo(16U, 32U); // 32P 分卡床：ranksPerSupernode=16、2 台超节点
    RankCoord me = DecodeRank(topo, 25U);
    CHECK_EQ(me.supernode, 1U, "25: sn=1");
    CHECK_EQ(me.local, 9U, "25: L=9");

    CHECK_EQ(ClassifyRow(topo, me, 25U), BUCKET_LOCAL, "self");
    CHECK_EQ(ClassifyRow(topo, me, 26U), BUCKET_DIRECT, "same supernode");
    CHECK_EQ(ClassifyRow(topo, me, 12U), BUCKET_RAIL, "other supernode rail");

    // 轨道腿公式：dst 超节点的同 L 卡（railPeer = sn_T×ranksPerSupernode + myL）
    CHECK_EQ(RailPeerOf(topo, 9U, 9U), 9U, "dst=9(rail peer itself)");
    CHECK_EQ(RailPeerOf(topo, 12U, 9U), 9U, "dst=12 -> rail peer 9");

    // G2.0 大几何
    RailTopo big = MakeRailTopo(128U, 204800U);
    RankCoord g = DecodeRank(big, 68200U); // sn=532, L=104
    CHECK_EQ(RailPeerOf(big, 137353U, 104U), 137448U, "G2.0: dst 137353 -> rail peer 137448(=1073*128+104)");
    CHECK_EQ(ClassifyRow(big, g, 137353U), BUCKET_RAIL, "G2.0 cross-sn rail");
    CHECK_EQ(ClassifyRow(big, g, 68205U), BUCKET_DIRECT, "G2.0 same-sn direct");
}

// ═══ FwdRoute：轨道区行恒直投 owner ═══
static void TestFwdRoute()
{
    RailTopo topo = MakeRailTopo(16U, 32U);
    RankCoord me = DecodeRank(topo, 25U);
    uint32_t ownerLocal = 0xFFFFFFFFU;
    HopKind hop = FwdRoute(topo, me, 12U, ownerLocal);
    CHECK_EQ(static_cast<uint32_t>(hop), static_cast<uint32_t>(HOP_OWNER), "rail row -> HOP_OWNER");
    CHECK_EQ(ownerLocal, 12U, "owner slot = dst local 12");

    hop = FwdRoute(topo, me, 9U, ownerLocal);
    CHECK_EQ(static_cast<uint32_t>(hop), static_cast<uint32_t>(HOP_OWNER), "any dst -> HOP_OWNER");
    CHECK_EQ(ownerLocal, 9U, "owner slot = dst local 9");
}

// ═══ barrier 写者槽：MySlotInPeer（关系对称）═══
static void TestWriterSlots()
{
    RailTopo topo = MakeRailTopo(16U, 32U);
    RankCoord me = DecodeRank(topo, 25U); // sn=1, L=9

    // 同超节点同伴 24：我在它的 owner 槽按本地号
    CHECK_EQ(MySlotInPeer(topo, me, 24U), 9U, "same-sn peer: slot=myL");
    // 轨道对端 9（sn=0）：我在它的 rail 槽按超节点号
    CHECK_EQ(MySlotInPeer(topo, me, 9U), 16U + 1U, "rail peer: slot=ranksPerSupernode+mySn");

    // G2.0
    RailTopo big = MakeRailTopo(128U, 204800U);
    RankCoord g = DecodeRank(big, 68200U); // sn=532, L=104
    CHECK_EQ(MySlotInPeer(big, g, 68205U), 104U, "G2.0 same-sn: slot=104");
    CHECK_EQ(MySlotInPeer(big, g, 124296U), 128U + 532U, "G2.0 rail: slot=ranksPerSupernode+532");
}

int main()
{
    TestCoordinates();
    TestClassifyAndFormulas();
    TestFwdRoute();
    TestWriterSlots();
    printf("%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}

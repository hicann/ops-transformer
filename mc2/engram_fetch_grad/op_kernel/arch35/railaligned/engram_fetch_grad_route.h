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
 * \file engram_fetch_grad_route.h
 * \brief 分轨（rail-aligned）模型路由纯函数——无 GM/UB 依赖，kernel / UT 共用。
 *        rank = sn×C + L（C = ranksPerSupernode 每超节点卡数，sn = 超节点号，L = 本地号）。
 *        发送规则（恒两跳）：dst 同超节点 → 直发 owner-区[L]；否则 → 轨道腿
 *        （dst 超节点的同 L 卡），到达后超节点内 CTP 直投。
 */

#ifndef ENGRAM_FETCH_GRAD_ROUTE_H
#define ENGRAM_FETCH_GRAD_ROUTE_H

// 本头被跨算子路径 include，不得依赖任何跨目录 include；布局常量在此定义，
// tiling_data.h 反向 include 本头派生 CTX_RAIL_ELEMS。

#ifndef __aicore__
#define ENGRAM_FETCH_GRAD_ROUTE_HOST_BUILD 1
#define __aicore__
#define ENGRAM_FETCH_GRAD_ROUTE_UNDEF_AICORE__
#endif

namespace Mc2Kernel {

struct EngramRailCtxHead {
    uint32_t localId;           // 本地号 L = rank mod C
    uint32_t supernodeId;       // 超节点号 sn = rank / C
    uint32_t supernodeCnt;      // 超节点数 = world / C（与 tiling 推导值双源校验）
    uint32_t ranksPerSupernode; // C = 每超节点卡数（与 attr 落 tilingData 的值双源校验）
};
static_assert(sizeof(EngramRailCtxHead) == 16U, "rail ctx head must stay 16B");
constexpr uint32_t MAX_QP_SIZE = 1024U;   // 同超节点本地号上限（node 表超节点段容量基数）
constexpr uint32_t MAX_SN_DOMAIN = 2048U; // 超节点数上限（node 表轨道段容量基数）
constexpr uint32_t RAIL_NODE_MAX = MAX_QP_SIZE + MAX_SN_DOMAIN;                 // node 表容量
constexpr uint32_t RAIL_CTX_WIN_OFF = sizeof(EngramRailCtxHead);                // nodeWinBase 偏移
constexpr uint32_t RAIL_CTX_CH_OFF = RAIL_CTX_WIN_OFF + RAIL_NODE_MAX * 8U;     // nodeChannel 偏移
constexpr uint32_t RAIL_CTX_TOTAL_BYTES = RAIL_CTX_CH_OFF + RAIL_NODE_MAX * 8U; // 49168B

// channel 建立在单一 world 通信域上（remoteRank = 全局 rank），建链侧换算落表：
//   node[p]（p < C）←→ rank = mySn×C + p（直连同伴）；node[p]（p ≥ C）←→ rank = (p−C)×C + myL（轨道对端）
#ifdef ENGRAM_FETCH_GRAD_ROUTE_HOST_BUILD
// host 垫片侧（普通地址空间）：route UT / host 侧建链复用
__aicore__ inline uint64_t NodeWinAt(const uint8_t* ctx, uint32_t p)
{
    return *reinterpret_cast<const uint64_t*>(ctx + RAIL_CTX_WIN_OFF + static_cast<uint64_t>(p) * 8U);
}
__aicore__ inline uint64_t NodeChannelAt(const uint8_t* ctx, uint32_t p)
{
    return *reinterpret_cast<const uint64_t*>(ctx + RAIL_CTX_CH_OFF + static_cast<uint64_t>(p) * 8U);
}
#else
// kernel 侧（__gm__ 空间）：经 __gm__ uint64_t* 视图读
__aicore__ inline uint64_t NodeWinAt(__gm__ uint8_t* ctx, uint32_t p)
{
    return *reinterpret_cast<__gm__ uint64_t*>(ctx + RAIL_CTX_WIN_OFF + static_cast<uint64_t>(p) * 8U);
}
__aicore__ inline uint64_t NodeChannelAt(__gm__ uint8_t* ctx, uint32_t p)
{
    return *reinterpret_cast<__gm__ uint64_t*>(ctx + RAIL_CTX_CH_OFF + static_cast<uint64_t>(p) * 8U);
}
#endif

/// @brief 目的节点索引：同超节点按对端 L；跨超节点按 C+对端超节点号
__aicore__ inline uint32_t NodeIdxOf(uint32_t mySn, uint32_t peerSn, uint32_t peerLocal, uint32_t ranksPerSupernode)
{
    return (peerSn == mySn) ? peerLocal : ranksPerSupernode + peerSn;
}

// ═══ 运行期拓扑（ranksPerSupernode 感知 → tiling 字段 → kernel；超节点数/world 全推导）═══
struct RailTopo {
    uint32_t ranksPerSupernode{128U}; // 每超节点卡数（感知）
    uint32_t world{204800U};          // 集群总卡数（输入张量形状反出）
    uint32_t supernodeCnt{1600U};     // 超节点数 = world÷C（推导；含自己）
    uint32_t railCnt{1599U};          // 轨道对端数 = world÷C − 1（推导）
    uint32_t writerSlots{1728U};      // 写者槽 W = C + world/C（同伴格 + 轨道格）
};

/// @brief 构造运行期拓扑：由感知数与 world 推导超节点数/轨道对端数/写者槽 W
__aicore__ inline RailTopo MakeRailTopo(uint32_t ranksPerSupernode, uint32_t world)
{
    RailTopo t;
    t.ranksPerSupernode = ranksPerSupernode;
    t.world = world;
    t.supernodeCnt = (ranksPerSupernode > 0U) ? world / ranksPerSupernode : 0U;
    t.railCnt = (t.supernodeCnt > 0U) ? t.supernodeCnt - 1U : 0U;
    t.writerSlots = ranksPerSupernode + t.supernodeCnt;
    return t;
}

struct RankCoord {
    uint32_t supernode; // sn = rank ÷ C
    uint32_t local;     // L = rank mod C（超节点内本地号）
};

/// @brief 全局 rank 分解：rank = sn * C + L
__aicore__ inline RankCoord DecodeRank(const RailTopo& topo, uint32_t rank)
{
    RankCoord c;
    c.supernode = rank / topo.ranksPerSupernode;
    c.local = rank % topo.ranksPerSupernode;
    return c;
}

enum BucketKind : uint32_t {
    BUCKET_LOCAL = 0,  // owner 是自己
    BUCKET_DIRECT = 1, // owner 同超节点
    BUCKET_RAIL = 2,   // owner 别的超节点
};

constexpr int32_t RAIL_BUCKET_KEY_STRIDE = 4096; // 桶键编码步距：kind×STRIDE + 子编号

/// @brief 三桶分类（互斥完备）：dst==self 本地；同超节点直发；否则轨道腿
__aicore__ inline BucketKind ClassifyRow(const RailTopo& topo, const RankCoord& me, uint32_t dstRank)
{
    if (dstRank == (me.supernode * topo.ranksPerSupernode + me.local)) {
        return BUCKET_LOCAL;
    }
    RankCoord d = DecodeRank(topo, dstRank);
    if (d.supernode == me.supernode) {
        return BUCKET_DIRECT;
    }
    return BUCKET_RAIL;
}

/// @brief 轨道对端卡号：dst 超节点的同本地号卡（railPeer = sn_T×C + myL）
__aicore__ inline uint32_t RailPeerOf(const RailTopo& topo, uint32_t dstRank, uint32_t myLocal)
{
    return (dstRank / topo.ranksPerSupernode) * topo.ranksPerSupernode + myLocal;
}

enum HopKind : uint32_t {
    HOP_OWNER = 0,
};

/// @brief F 组转发路由：轨道区到达的行 dst 必在本超节点，恒直投 owner 槽[L_T]
__aicore__ inline HopKind FwdRoute(const RailTopo& topo, const RankCoord& me, uint32_t dstRank, uint32_t& ownerLocal)
{
    (void)me;
    RankCoord d = DecodeRank(topo, dstRank);
    ownerLocal = d.local;
    return HOP_OWNER;
}

constexpr uint32_t EPOCH_SLOT_TAIL = 1U; // 窗头 = (W + 1) × 32B

constexpr uint32_t RECV_AREA_HEAD = 96U; // write/read/done 三计数头（各 32B 槽）

constexpr uint32_t ROW_HEAD_BYTES = 32U;
constexpr uint32_t CNT_WRITE_OFF = 0U; // 写者（S 核统一流）行计数
constexpr uint32_t CNT_READ_OFF = 32U; // 接收方已读行数（发送方远程读它当信用）
constexpr uint32_t CNT_DONE_OFF = 64U; // 总行数 + 1 偏置（0 行写 1；属主 S 核流完结后单写）
constexpr uint32_t ENTRY_BUF_BYTES = 64U * 1024U; // WinLocalCopy ping
constexpr uint32_t GRAD_BUF_BYTES = 64U * 1024U;  // WinLocalCopy pong
constexpr uint32_t IDX_BUF_BYTES = 4U * 1024U;    // 排序产物分片扫描
constexpr uint32_t ENTRY_BATCH_CAP = 1024U;       // GM 清零批容量（int32 个数）

/// @brief 本 rank 在对端写者枚举里的槽号（关系对称）：同超节点按本地号；轨道按超节点号
__aicore__ inline uint32_t MySlotInPeer(const RailTopo& topo, const RankCoord& me, uint32_t peerRank)
{
    RankCoord peer = DecodeRank(topo, peerRank);
    if (peer.supernode == me.supernode) {
        return me.local;
    }
    return topo.ranksPerSupernode + me.supernode;
}

} // namespace Mc2Kernel

#ifdef ENGRAM_FETCH_GRAD_ROUTE_UNDEF_AICORE__
#undef __aicore__
#undef ENGRAM_FETCH_GRAD_ROUTE_UNDEF_AICORE__
#endif

#endif // ENGRAM_FETCH_GRAD_ROUTE_H

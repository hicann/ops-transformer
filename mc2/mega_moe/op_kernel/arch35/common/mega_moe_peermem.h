/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_peermem.h
 * \brief peermem 跨卡对称窗口的布局与尺寸计算。
 *        窗口各区尺寸一律按全卡一致的 numMaxTokensPerRank 上界推导（可变 bs 下各卡真实 bs 可不同，
 *        但跨卡读写共用同一套偏移，布局必须逐字节一致）。
 *        本文件是 host 侧 tiling 校验与 device 侧地址装配的唯一事实源，两侧不得各写一份公式。
 */

#ifndef MEGA_MOE_PEERMEM_H
#define MEGA_MOE_PEERMEM_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#define HOST_DEVICE __forceinline__[aicore]
#else
#define GM_ADDR uint8_t*
// host 侧命名空间级自由函数需显式 inline 保 ODR；类内成员函数本就隐式 inline，加之无副作用。
#define HOST_DEVICE inline
// host 侧自洽：Ops::Base::CeilAlign/CeilDiv 的提供方，不再依赖包含方的 include 顺序。
#include "op_common/op_host/util/math_util.h"
#endif
#include "../mega_moe_tiling.h"
#include "mega_moe_constants.h"

namespace MegaMoeImpl {

// 原同步布局：前 48 KiB 是 rank 槽位，随后 12 KiB 是各 AIV 的同步计数区。
// URMA 的 rank 槽位超过预留区时动态扩展；MTE 保持原 60 KiB 固定布局。
constexpr int64_t PEERMEM_MIN_RANK_SYNC_SIZE = 1024 * 48LL;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
// MTE count 区固定预留 2048 个 512B 槽，每槽首个 int32 有效。
constexpr int64_t PEERMEM_MTE_COUNT_REGION_SIZE = 512LL * 2048LL;
constexpr int64_t PEERMEM_SYNC_COUNT_REGION_SIZE = PEERMEM_DATA_OFFSET - PEERMEM_MIN_RANK_SYNC_SIZE;
constexpr int64_t PEERMEM_SYNC_SLOT_SIZE = static_cast<int64_t>(INT_CACHELINE) * sizeof(int32_t);

// 通信拓扑：决定窗口数据区按 MTE 还是 URMA 布局推进。
constexpr int64_t TOPO_TYPE_MTE = 0U;
constexpr int64_t TOPO_TYPE_URMA = 1U;

HOST_DEVICE int64_t CalcUrmaRankSyncSize(int64_t epWorldSize)
{
    int64_t rankSyncSize = epWorldSize * PEERMEM_SYNC_SLOT_SIZE;
    return rankSyncSize > PEERMEM_MIN_RANK_SYNC_SIZE ? rankSyncSize : PEERMEM_MIN_RANK_SYNC_SIZE;
}

// ============================ 各区尺寸（host / device 共用） ============================

// URMA layered 路径仍使用 mask 槽；MTE WAVE 不再经过该布局。
HOST_DEVICE int64_t CalcDispatchMaskAlignSize(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t sendTotalNum = numMaxTokensPerRank * topK;
    int64_t alignedRouteCount = Ops::Base::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    return Ops::Base::CeilAlign(alignedRouteCount / static_cast<int64_t>(BITS_PER_BYTE), ALIGN_32);
}

// 所有 Rank 使用一致的 numMaxTokensPerRank，因此可据此统一选择 int16 或 int32 编码。
HOST_DEVICE bool UseInt16TopkIndex(int64_t numMaxTokensPerRank, int64_t topK)
{
    return topK > 0 && numMaxTokensPerRank >= 0 && numMaxTokensPerRank * topK <= MAX_INT16_TOPK_INDEX_COUNT;
}

HOST_DEVICE int64_t CalcTopkIndexTypeBytes(int64_t numMaxTokensPerRank, int64_t topK)
{
    return UseInt16TopkIndex(numMaxTokensPerRank, topK) ? static_cast<int64_t>(sizeof(int16_t)) :
                                                          static_cast<int64_t>(sizeof(int32_t));
}

// 每个 (localExpert, srcRank) 槽直接保存有序 topkIndex。
// 同一 token 的 topK expert id 不重复，因此单专家从一张源卡最多接收 numMaxTokensPerRank 个 index。
HOST_DEVICE int64_t CalcDispatchRouteIndexAlignSize(const MegaMoeTilingData* tilingData)
{
    return Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->numMaxTokensPerRank) *
                                    CalcTopkIndexTypeBytes(tilingData->numMaxTokensPerRank, tilingData->topK),
                                ALIGN_32);
}

// dedupMode 的值就是门控后实际开启的去重模式：
// 0 = dispatch 关、combine 关；1 = dispatch 关、combine 开；
// 2 = dispatch 开、combine 关；3 = dispatch 开、combine 开。
// 下面两个判断函数是唯一入口，host 和 kernel 都用它们，不要在别处自己写数值比较。
HOST_DEVICE constexpr bool IsDispatchDedupOn(int32_t dedupMode)
{
    return dedupMode == 2 || dedupMode == 3;
}

HOST_DEVICE constexpr bool IsCombineDedupOn(int32_t dedupMode)
{
    return dedupMode == 1 || dedupMode == 3;
}

// 单 token 的量化记录包含数据和 mx scale，按 32B 对齐。
// topkWeightsPrefetch 控制是否追加 topK 权重；调用方也可因 Combine 去重开启该记录段。
HOST_DEVICE int64_t CalcQuantTokenScaleBytes(int64_t h, uint32_t elemsPerByte, int64_t topK, bool topkWeightsPrefetch)
{
    uint32_t mxScaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(h), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataBytes =
        Ops::Base::CeilAlign(static_cast<uint32_t>(h) / elemsPerByte, static_cast<uint32_t>(ALIGN_256)) *
        static_cast<uint32_t>(sizeof(int8_t));
    uint32_t scaleBytes = mxScaleNum * static_cast<uint32_t>(sizeof(int8_t));
    uint32_t tokenScaleBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
    if (topkWeightsPrefetch) {
        uint32_t weightBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(topK) * static_cast<uint32_t>(sizeof(float)),
                                                    static_cast<uint32_t>(ALIGN_32));
        tokenScaleBytes = Ops::Base::CeilAlign(tokenScaleBytes + weightBytes, static_cast<uint32_t>(ALIGN_32));
    }
    return static_cast<int64_t>(tokenScaleBytes);
}

// 去重 rowDesc 每行的 int32 数：头 2 个 + 成员三元组 3*topK 个，向上对齐到 8 个 int32（32B），
// 使描述符批量搬运天然 32B 对齐。字段语义见 mega_moe_constants.h 的 DEDUP_ROW_* 注释。
HOST_DEVICE int64_t CalcDedupRowDescStrideInt32(int64_t topK)
{
    int64_t rawInt32 = static_cast<int64_t>(DEDUP_ROW_DESC_HEADER_INT32) + 3LL * topK;
    return Ops::Base::CeilAlign(rawInt32, static_cast<int64_t>(INT32_PER_256B));
}

// combine 接收区单 token 记录字节数：须与 kernel InitCombineBuffers 的 combine record 布局一致。
HOST_DEVICE int64_t CalcCombineTokenBytes(int64_t h, int64_t yDtypeSize, bool isQuantCombine)
{
    if (!isQuantCombine) {
        return h * yDtypeSize;
    }
    int64_t tokenStorageBytes = Ops::Base::CeilAlign(h, ALIGN_256);
    int64_t scaleCount = Ops::Base::CeilDiv(h, static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
    int64_t storedScaleBytes = Ops::Base::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
    return Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, ALIGN_32);
}

// GM 区域保存相对基址的偏移、字节数和初始化时计算的地址。
struct GmMemoryRegion {
    int64_t offset{-1};
    int64_t size{0};
    GM_ADDR ptr{nullptr};
};

// Host 和 kernel 使用同一布局；base 不包含 MTE 调用方预留的异常 dump 区。
struct PeermemInfo {
    GmMemoryRegion rankSync;
    GmMemoryRegion syncCount;
    GmMemoryRegion maskRecv;
    GmMemoryRegion expertCountRecv;
    GmMemoryRegion quantTokenScale;
    GmMemoryRegion dispatchReceive;
    GmMemoryRegion dispatchFlag;
    GmMemoryRegion combineSend;
    GM_ADDR cclDataPtr{nullptr}; // 通信区之后按 512B 对齐的 Workspace 基址。
    int64_t totalSize{0};

    HOST_DEVICE PeermemInfo() = default;

    HOST_DEVICE explicit PeermemInfo(const MegaMoeTilingData* tilingData, uint32_t elemsPerByte = 1,
                                     GM_ADDR windowBase = nullptr)
        : base_(windowBase)
    {
        if (tilingData->topoType == TOPO_TYPE_MTE) {
            InitializeMteLayout(*tilingData, elemsPerByte);
        } else {
            InitializeUrmaLayout(*tilingData, elemsPerByte);
        }
        cclDataPtr =
            base_ == nullptr ? nullptr : base_ + Ops::Base::CeilAlign(totalSize, static_cast<int64_t>(ALIGN_512));
    }

private:
    GM_ADDR base_{nullptr};

    HOST_DEVICE GmMemoryRegion Reserve(int64_t bytes)
    {
        GmMemoryRegion region{totalSize, bytes, base_ == nullptr ? nullptr : base_ + totalSize};
        totalSize += bytes;
        return region;
    }

    HOST_DEVICE void InitializeMteLayout(const MegaMoeTilingData& tilingData, uint32_t elemsPerByte)
    {
        const int64_t capacity = tilingData.numMaxTokensPerRank;

        // 同步区：固定预留跨卡同步槽和各核计数。
        rankSync = Reserve(PEERMEM_MIN_RANK_SYNC_SIZE);
        syncCount = Reserve(PEERMEM_SYNC_COUNT_REGION_SIZE);

        // Dispatch 计数与路由索引：计数独立存放，索引槽不再携带尾部 count。
        expertCountRecv = Reserve(PEERMEM_MTE_COUNT_REGION_SIZE);
        maskRecv =
            Reserve(Ops::Base::CeilAlign(static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize *
                                             CalcDispatchRouteIndexAlignSize(&tilingData),
                                         ALIGN_512));

        // Dispatch 量化记录：保存本卡 token 数据、scale 和可选的 topK 权重。
        const bool recordHasWeights = tilingData.topkWeightsPrefetch == 1 || IsCombineDedupOn(tilingData.dedupMode);
        const int64_t tokenScaleBytes =
            CalcQuantTokenScaleBytes(tilingData.h, elemsPerByte, tilingData.topK, recordHasWeights);
        quantTokenScale = Reserve(Ops::Base::CeilAlign(capacity * tokenScaleBytes, ALIGN_512));

        // Combine 接收区：保存回传结果，供 Unpermute 读取。
        // 输出 dtype 在容量检查前已限定为 BF16。
        constexpr int64_t outputElementBytes = 2;
        const int64_t tokenBytes =
            CalcCombineTokenBytes(tilingData.h, outputElementBytes, tilingData.combineQuantMode != COMBINE_NO_QUANT);
        combineSend = Reserve(Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData.numMaxTokensPerRank) * tilingData.topK * tokenBytes, ALIGN_512));
    }

    HOST_DEVICE void InitializeUrmaLayout(const MegaMoeTilingData& tilingData, uint32_t elemsPerByte)
    {
        const int64_t capacity = tilingData.numMaxTokensPerRank;
        uint32_t rankNumPerServer = tilingData.rankNumPerServer;
        if (rankNumPerServer == 0U || rankNumPerServer > tilingData.epWorldSize) {
            rankNumPerServer = tilingData.epWorldSize;
        }
        const int64_t serverNum = Ops::Base::CeilDiv(tilingData.epWorldSize, rankNumPerServer);

        // 同步区：按 Rank 数扩展同步槽，计数区之后补齐到 512 字节。
        rankSync = Reserve(CalcUrmaRankSyncSize(tilingData.epWorldSize));
        syncCount = Reserve(PEERMEM_SYNC_COUNT_REGION_SIZE);
        totalSize = Ops::Base::CeilAlign(totalSize, ALIGN_512);

        // Dispatch mask 与专家计数：分别记录路由集合和各源 Rank 的 token 数。
        const int64_t maskAlignSize = CalcDispatchMaskAlignSize(capacity, tilingData.topK);
        const int64_t maskSlotSize = maskAlignSize + ALIGN_32;
        maskRecv = Reserve(Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize * maskSlotSize, ALIGN_512));
        // URMA raw count 按 [localExpert][sourceRank] 连续排列。
        expertCountRecv =
            Reserve(Ops::Base::CeilAlign(static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize *
                                             static_cast<int64_t>(sizeof(int32_t)),
                                         ALIGN_512));

        // Dispatch 中继：按源 Server 保存量化记录及 ready 标志。
        const bool recordHasWeights = tilingData.topkWeightsPrefetch == 1 || IsCombineDedupOn(tilingData.dedupMode);
        const int64_t tokenScaleBytes =
            CalcQuantTokenScaleBytes(tilingData.h, elemsPerByte, tilingData.topK, recordHasWeights);
        const int64_t relayRecordBytes = Ops::Base::CeilAlign(tokenScaleBytes, ALIGN_512);
        dispatchReceive = Reserve(Ops::Base::CeilAlign(capacity * relayRecordBytes * serverNum, ALIGN_512));
        dispatchFlag =
            Reserve(Ops::Base::CeilAlign(serverNum * capacity * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512));

        // Combine 接收区：保存回传结果，供 Unpermute 读取。
        // 输出 dtype 在容量检查前已限定为 BF16。
        constexpr int64_t outputElementBytes = 2;
        const int64_t tokenBytes =
            CalcCombineTokenBytes(tilingData.h, outputElementBytes, tilingData.combineQuantMode != COMBINE_NO_QUANT);
        combineSend = Reserve(Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData.numMaxTokensPerRank) * tilingData.topK * tokenBytes, ALIGN_512));
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_PEERMEM_H

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
#define GM_ADDR uint8_t *
// host 侧命名空间级自由函数需显式 inline 保 ODR；类内成员函数本就隐式 inline，加之无副作用。
#define HOST_DEVICE inline
// host 侧自洽：Ops::Base::CeilAlign/CeilDiv 的提供方，不再依赖包含方的 include 顺序。
#include "op_common/op_host/util/math_util.h"
#endif
#include "../mega_moe_tiling.h"
#include "mega_moe_constants.h"

namespace MegaMoeImpl {

// 窗口首部的跨卡同步区（rankSyncInWorld）字节数，数据区从此偏移起。
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;

// 通信拓扑：决定窗口数据区按 MTE 还是 URMA 布局推进。
constexpr int64_t TOPO_TYPE_MTE = 0U;  // mte
constexpr int64_t TOPO_TYPE_URMA = 1U; // urma

// ============================ 各区尺寸（host / device 共用） ============================

// mask 位区字节数：每条路由 1 bit，位图按 32B 对齐。
HOST_DEVICE int64_t CalcDispatchMaskAlignSizeBy(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t sendTotalNum = numMaxTokensPerRank * topK;
    // 路由条目数按 256B 对齐后的 int32 元素数（位图每条目 1 bit，故下方 /8 得字节数）。
    int64_t alignedRouteCount = Ops::Base::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    return Ops::Base::CeilAlign(alignedRouteCount / 8, ALIGN_32);
}

HOST_DEVICE int64_t CalcDispatchMaskAlignSize(const MegaMoeTilingData *tilingData)
{
    return CalcDispatchMaskAlignSizeBy(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                       static_cast<int64_t>(tilingData->topK));
}

// mask 接收区总字节数：每个 (localExpert, srcRank) 一个槽，槽 = mask 位区 + 末尾 32B count 区。
HOST_DEVICE int64_t CalcMaskRecvSize(int64_t maskAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    int64_t maskSlotSize = maskAlignSize + ALIGN_32;
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * maskSlotSize, ALIGN_512);
}

// 独立 raw count 表按 [localExpert][sourceRank] 排列，供接收端一次搬入并做前缀和。
HOST_DEVICE int64_t CalcExpertCountRecvSize(int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);
}

// 单 token 的量化记录字节数 = 量化数据 + mx scale（prefetch 场景再拼 topk 权重），按 32B 对齐。
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

// 计算窗口总大小的入参（供 host 侧 tiling 校验 cclBufferSize 使用）。
struct PeermemSizeParams {
    int64_t numMaxTokensPerRank;
    int64_t topK;
    int64_t h;
    int64_t moeExpertPerRank;
    int64_t epWorldSize;
    int64_t yDtypeSize;
    uint32_t elemsPerByte;
    bool topkWeightsPrefetch;
    bool isQuantCombine;
};

/*
 * MTE 路径下 peermem 窗口所需的最小字节数，逐段与下方 PeermemInfo 构造函数的偏移推进同源：
 *   同步区(PEERMEM_DATA_OFFSET) + mask 接收区 + count 接收区 + 量化 token 接收区 + combine 接收区。
 * host 侧 tiling 用它校验用户传入的 cclBufferSize，不再各自手写一份布局公式。
 */
HOST_DEVICE int64_t CalcPeermemLeastSize(const PeermemSizeParams &params)
{
    int64_t maskAlignSize = CalcDispatchMaskAlignSizeBy(params.numMaxTokensPerRank, params.topK);
    int64_t maskRecvSize = CalcMaskRecvSize(maskAlignSize, params.moeExpertPerRank, params.epWorldSize);
    int64_t expertCountRecvSize = CalcExpertCountRecvSize(params.moeExpertPerRank, params.epWorldSize);
    int64_t tokenScaleBytes =
        CalcQuantTokenScaleBytes(params.h, params.elemsPerByte, params.topK, params.topkWeightsPrefetch);
    int64_t quantTokenScaleSize = Ops::Base::CeilAlign(params.numMaxTokensPerRank * tokenScaleBytes, ALIGN_512);
    int64_t combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    int64_t combineSendSize =
        Ops::Base::CeilAlign(params.numMaxTokensPerRank * params.topK * combineTokenBytes, ALIGN_512);
    return PEERMEM_DATA_OFFSET + maskRecvSize + expertCountRecvSize + quantTokenScaleSize + combineSendSize;
}

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// device 侧窗口各区基址；偏移推进顺序即上方 CalcPeermemLeastSize 的分段顺序。
struct PeermemInfo {
    GM_ADDR rankSyncInWorldPtr;
    GM_ADDR maskRecvPtr;
    GM_ADDR expertCountRecvPtr;
    GM_ADDR quantTokenScalePtr;
    GM_ADDR dispatchRecivePtr;
    GM_ADDR dispatchFlagPtr;
    GM_ADDR combineSendPtr;

    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData, uint32_t elemsPerByte = 1,
                                  uint32_t serverNum = 1)
    {
        rankSyncInWorldPtr = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        maskRecvPtr = base + offset;
        offset +=
            CalcMaskRecvSize(CalcDispatchMaskAlignSize(tilingData), static_cast<int64_t>(tilingData->moeExpertPerRank),
                             static_cast<int64_t>(tilingData->epWorldSize));

        expertCountRecvPtr = base + offset;
        offset += CalcExpertCountRecvSize(static_cast<int64_t>(tilingData->moeExpertPerRank),
                                          static_cast<int64_t>(tilingData->epWorldSize));

        int64_t tokenScaleBytes =
            CalcQuantTokenScaleBytes(static_cast<int64_t>(tilingData->h), elemsPerByte,
                                     static_cast<int64_t>(tilingData->topK), tilingData->topkWeightsPrefetch == 1);
        if (tilingData->topoType == TOPO_TYPE_MTE) {
            quantTokenScalePtr = base + offset;
            offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->numMaxTokensPerRank) * tokenScaleBytes,
                                           ALIGN_512);
        } else {
            dispatchRecivePtr = base + offset;
            int64_t relayRecordBytes = Ops::Base::CeilAlign(tokenScaleBytes, ALIGN_512);
            offset += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->bs) * relayRecordBytes * static_cast<int64_t>(serverNum), ALIGN_512);
            dispatchFlagPtr = base + offset;
            offset += Ops::Base::CeilAlign(
                static_cast<int64_t>(serverNum) * tilingData->bs * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);
        }
        combineSendPtr = base + offset;
    }
};
#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_PEERMEM_H

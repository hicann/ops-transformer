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
 * \file moe_distribute_check_win_size.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_CHECK_WIN_SIZE
#define MOE_DISTRIBUTE_CHECK_WIN_SIZE

#include <algorithm>
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_log_compat.h"

namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t FULL_MESH_DATA_ALIGN = 480UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t LAYERED_BS_UPPER_BOUND = 256UL;
constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL; // scale32B + 3*4expandIdx
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
} // namespace

namespace Mc2Tiling {
inline uint64_t AlignRoceBytes(uint64_t bytes, uint64_t alignment = 32U)
{
    return (bytes + alignment - 1U) / alignment * alignment;
}

inline uint64_t GetRoceAssistInfoSize(uint64_t expertRows, uint64_t epWorldSize, uint64_t tokensPerRank)
{
    const uint64_t tripleBytes = expertRows * 3U * sizeof(int32_t);
    const uint64_t serverInfoBytes = epWorldSize * (AlignRoceBytes(tokensPerRank, 16U) + 1U) * sizeof(uint16_t);
    return AlignRoceBytes(tripleBytes + serverInfoBytes, sizeof(int32_t)) / sizeof(int32_t);
}

struct RoceWindowLayout {
    uint64_t dispatchPacketBytes;
    uint64_t serverPacketBytes;
    uint64_t combineIntraBytes;
    uint64_t dispatchIntraBytes;
    uint64_t dispatchInterBytes;
};

template <typename TilingInfo>
inline RoceWindowLayout GetRoceWindowLayout(const TilingInfo& info, uint64_t serverNum)
{
    const uint64_t tokensPerRank = info.globalBs / info.epWorldSize;
    const uint64_t ranksPerServer = info.epWorldSize / serverNum;
    const uint64_t expertsPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    const uint64_t sharedSlots = info.sharedExpertRankNum > 0U ? info.sharedExpertNum : 0U;
    const uint64_t rawTokenBytes = AlignRoceBytes(static_cast<uint64_t>(info.h) * MAX_OUT_DTYPE_SIZE);
    const uint64_t combinePacketBytes =
        AlignRoceBytes(rawTokenBytes + sizeof(float), FULL_MESH_DATA_ALIGN) / FULL_MESH_DATA_ALIGN * WIN_ADDR_ALIGN;
    const uint64_t dispatchPacketBytes =
        AlignRoceBytes(rawTokenBytes + 2U * UB_ALIGN, FULL_MESH_DATA_ALIGN) / FULL_MESH_DATA_ALIGN * WIN_ADDR_ALIGN;
    const uint64_t countPacketBytes =
        AlignRoceBytes(expertsPerRank * ranksPerServer * sizeof(int32_t), FULL_MESH_DATA_ALIGN) / FULL_MESH_DATA_ALIGN *
        WIN_ADDR_ALIGN;
    // 框间 Dispatch 按 FP32 工作载荷上界预留，保证不依赖两算子各自的量化模式。
    const uint64_t tokenBytes = AlignRoceBytes(static_cast<uint64_t>(info.h) * sizeof(float), WIN_ADDR_ALIGN);
    const uint64_t expertIdsBytes = AlignRoceBytes(static_cast<uint64_t>(info.k) * sizeof(int32_t));
    const uint64_t expertCountsBytes = AlignRoceBytes(static_cast<uint64_t>(info.k) * sizeof(int16_t));
    const uint64_t expertScalesBytes = AlignRoceBytes(static_cast<uint64_t>(info.k) * sizeof(float));
    const uint64_t serverRecordBytes = tokenBytes + UB_ALIGN + expertIdsBytes + expertCountsBytes + expertScalesBytes;
    const uint64_t serverPacketBytes =
        AlignRoceBytes(serverRecordBytes, FULL_MESH_DATA_ALIGN) / FULL_MESH_DATA_ALIGN * WIN_ADDR_ALIGN;
    RoceWindowLayout layout;
    layout.dispatchPacketBytes = dispatchPacketBytes;
    layout.serverPacketBytes = serverPacketBytes;
    layout.combineIntraBytes = serverNum * tokensPerRank * (info.k + sharedSlots) * combinePacketBytes;
    layout.dispatchIntraBytes = info.epWorldSize * tokensPerRank * expertsPerRank * dispatchPacketBytes;
    layout.dispatchInterBytes =
        serverNum * (2U * WIN_ADDR_ALIGN + tokensPerRank * serverPacketBytes + 2U * countPacketBytes);
    return layout;
}

struct CheckWinSizeData {
    uint32_t localMoeExpertNum;
    uint32_t sharedExpertNum;
    uint64_t a;
    uint64_t h;
    uint64_t k;
    uint64_t bs;
    uint64_t epWorldSize;
    uint64_t globalBs;
    uint64_t moeExpertNum;
    uint64_t tpWorldSize;
    uint64_t totalWinSizeEp;
    bool isSetFullMeshV2;
    bool isLayered;
    bool isMc2Context;
};

inline uint64_t CalcMinWinSize(const CheckWinSizeData& winSizeData, uint64_t& tokenNeedSizeDispatch,
                               uint64_t& tokenNeedSizeCombine, uint64_t combineDataAlign = WIN_ADDR_ALIGN,
                               uint64_t dispatchTokenBytes = 0UL)
{
    // A3 Combine uses raw 512-byte-aligned data for both fullmesh variants.
    // A5 callers explicitly request the packed-data alignment.
    const uint64_t h = winSizeData.h;
    const uint64_t k = winSizeData.k;
    const uint64_t maxBs = winSizeData.globalBs / winSizeData.epWorldSize;
    tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE + combineDataAlign - 1UL) / combineDataAlign) * WIN_ADDR_ALIGN;
    // expertScale 复用三元组后的对齐空间，不增加窗口占用。
    const uint64_t defaultTokenLen =
        ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    const uint64_t tokenActualLen = std::max(defaultTokenLen, dispatchTokenBytes);
    const uint64_t tokenDataAlign = winSizeData.isSetFullMeshV2 ? FULL_MESH_DATA_ALIGN : WIN_ADDR_ALIGN;
    tokenNeedSizeDispatch = ((tokenActualLen + tokenDataAlign - 1UL) / tokenDataAlign) * WIN_ADDR_ALIGN;
    // 跨超win区计算，3代表token里拼的topK、expert_scale、实际有效token数量，32B对齐，64代表scale和flag， 404是400MB
    // win区和4MB RDMA状态区
    return winSizeData.isLayered ?
               (static_cast<uint64_t>(winSizeData.moeExpertNum) * maxBs *
                    (h * MAX_OUT_DTYPE_SIZE + (3 * (k + 7) / 8 * 8) * sizeof(uint32_t) + 64) +
                404 * MB_SIZE) :
               ((maxBs * tokenNeedSizeDispatch * winSizeData.epWorldSize * winSizeData.localMoeExpertNum) +
                (maxBs * tokenNeedSizeCombine * (k + winSizeData.sharedExpertNum))) *
                   DOUBLE_DATA_BUFFER;
}

inline ge::graphStatus CheckLayeredWinSizeParams(const gert::TilingContext* context, const char* nodeName,
                                                 const CheckWinSizeData& winSizeData)
{
    if (winSizeData.isLayered) {
        const std::string socVersion = mc2tiling::GetSocVersion(context);
        const uint64_t bs = winSizeData.bs;
        const uint64_t maxBs = winSizeData.globalBs / winSizeData.epWorldSize;
        OP_TILING_CHECK(
            (socVersion != "Ascend910_93") && (bs != maxBs),
            OP_LOGE_WITHOUT_REPORT(nodeName, "Layered cannot support variableBs on %s, bs is %lu, maxBs is %lu",
                                   socVersion.c_str(), bs, maxBs),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (maxBs > LAYERED_BS_UPPER_BOUND),
            OP_LOGE_WITHOUT_REPORT(nodeName, "maxBs is invalid for hierarchy, should be in range [1, %lu], but got %lu",
                                   LAYERED_BS_UPPER_BOUND, maxBs),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CalcMinWinSizeA3(const gert::TilingContext* context, const char* nodeName,
                                        CheckWinSizeData& winSizeData)
{
    OP_TILING_CHECK(CheckLayeredWinSizeParams(context, nodeName, winSizeData) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Invalid hierarchy window parameters."), return ge::GRAPH_FAILED);
    uint64_t tokenNeedSizeDispatch = 0;
    uint64_t tokenNeedSizeCombine = 0;
    winSizeData.totalWinSizeEp = CalcMinWinSize(winSizeData, tokenNeedSizeDispatch, tokenNeedSizeCombine);
    OP_LOGD(nodeName, "min required EP window size = %lu Bytes.", winSizeData.totalWinSizeEp);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckActualWinSize(const gert::TilingContext* context, const char* nodeName,
                                          const CheckWinSizeData& winSizeData, const uint64_t maxWindowSizeEp,
                                          const uint64_t hcclBufferSizeEp, uint64_t dispatchTokenBytes = 0UL)
{
    uint64_t h = winSizeData.h;
    uint64_t k = winSizeData.k;
    uint64_t epWorldSize = winSizeData.epWorldSize;
    uint64_t maxBs = winSizeData.globalBs / epWorldSize;
    uint64_t sharedExpertNum = winSizeData.sharedExpertNum;
    uint64_t tokenNeedSizeDispatch = 0;
    uint64_t tokenNeedSizeCombine = 0;
    const uint64_t combineDataAlign =
        (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201) ? WIN_ADDR_ALIGN : FULL_MESH_DATA_ALIGN;
    uint64_t actualSize =
        CalcMinWinSize(winSizeData, tokenNeedSizeDispatch, tokenNeedSizeCombine, combineDataAlign, dispatchTokenBytes);
    OP_TILING_CHECK(CheckLayeredWinSizeParams(context, nodeName, winSizeData) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Invalid hierarchy window parameters."), return ge::GRAPH_FAILED);
    if (winSizeData.isLayered) {
        // 校验buffersize
        OP_TILING_CHECK((actualSize > maxWindowSizeEp),
                        OP_LOGE_WITHOUT_REPORT(
                            nodeName,
                            "HCCL_BUFFSIZE_EP is too SMALL, maxBs = %lu, h = %lu,"
                            "NEEDED_HCCL_BUFFSIZE_HIERARCHY((moeExpertNum * maxBs * (h * MAX_OUT_DTYPE_SIZE + (3 * "
                            "(k + 7) / 8 * 8) *"
                            "sizeof(uint32_t) + 64) + 404 * 1024 * 1024)) = %luMB, HCCL_BUFFSIZE=%luMB.",
                            maxBs, h, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE),
                        return ge::GRAPH_FAILED);
    } else {
        if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510) {
            actualSize += epWorldSize * EP_RANK_OFFSET_STEP;
            OP_TILING_CHECK(
                (actualSize > maxWindowSizeEp),
                OP_LOGE_WITHOUT_REPORT(
                    nodeName,
                    "HCCL_BUFFSIZE_EP is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
                    " localMoeExpertNum = %u, sharedExpertNum = %lu, tokenNeedSizeDispatch = %lu,"
                    " tokenNeedSizeCombine = %lu,"
                    " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * epWorldSize * "
                    "localMoeExpertNum) +"
                    " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) + epWorldSize * 1024 = %luMB,"
                    " HCCL_BUFFSIZE=%luMB, actual CCL_BUFFSIZE=%luMB.",
                    maxBs, h, epWorldSize, winSizeData.localMoeExpertNum, sharedExpertNum, tokenNeedSizeDispatch,
                    tokenNeedSizeCombine, winSizeData.k, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / (MB_SIZE * 2),
                    hcclBufferSizeEp / MB_SIZE),
                return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK((actualSize > maxWindowSizeEp),
                            OP_LOGE_WITHOUT_REPORT(
                                nodeName,
                                "HCCL_BUFFSIZE_EP is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
                                " localMoeExpertNum = %u, sharedExpertNum = %lu, tokenNeedSizeDispatch = %lu,"
                                " tokenNeedSizeCombine = %lu,"
                                " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * epWorldSize * "
                                "localMoeExpertNum) +"
                                " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB,"
                                " HCCL_BUFFSIZE=%luMB, actual CCL_BUFFSIZE=%luMB.",
                                maxBs, h, epWorldSize, winSizeData.localMoeExpertNum, sharedExpertNum,
                                tokenNeedSizeDispatch, tokenNeedSizeCombine, winSizeData.k, actualSize / MB_SIZE + 1UL,
                                hcclBufferSizeEp / MB_SIZE, hcclBufferSizeEp / MB_SIZE),
                            return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckWinSize(const gert::TilingContext* context, const char* nodeName,
                                    CheckWinSizeData& winSizeData, uint64_t dispatchTokenBytes = 0UL)
{
    auto attrs = context->GetAttrs();
    uint64_t hcclBufferSizeEp = 0;
    uint64_t maxWindowSizeEp = 0;
    if (!winSizeData.isMc2Context) {
        OP_TILING_CHECK(mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSizeEp, maxWindowSizeEp,
                                                ATTR_GROUP_EP_INDEX, winSizeData.isLayered) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(nodeName, "Get EP WinSize failed"), return ge::GRAPH_FAILED);
    } else {
        auto attrs = context->GetAttrs();
        auto cclBuffSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(3)); // 3为V3算子中ccl_buffer_size的index
        if (cclBuffSizePtr == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "cclBuffSizePtr");
            return ge::GRAPH_FAILED;
        }
        OP_TILING_CHECK(
            *cclBuffSizePtr < static_cast<int64_t>(MB_SIZE),
            OP_LOGE_WITHOUT_REPORT(nodeName, "ccl_buffer_size is too small: %ld B < %lu B.", *cclBuffSizePtr, MB_SIZE),
            return ge::GRAPH_FAILED);
        maxWindowSizeEp = *cclBuffSizePtr - MB_SIZE;
        hcclBufferSizeEp = *cclBuffSizePtr;
    }
    OP_LOGD(nodeName, "windowSize = %lu", maxWindowSizeEp);
    OP_TILING_CHECK(CheckActualWinSize(context, nodeName, winSizeData, maxWindowSizeEp, hcclBufferSizeEp,
                                       dispatchTokenBytes) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check actual window size failed."),
                    return ge::GRAPH_FAILED);

    winSizeData.totalWinSizeEp = maxWindowSizeEp;
    if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510) {
        uint64_t rankOffsetSize = winSizeData.epWorldSize * EP_RANK_OFFSET_STEP;
        OP_TILING_CHECK(
            maxWindowSizeEp < rankOffsetSize,
            OP_LOGE_WITHOUT_REPORT(nodeName, "maxWindowSizeEp is too small: %lu B < %lu B (epWorldSize=%lu).",
                                   maxWindowSizeEp, rankOffsetSize, winSizeData.epWorldSize),
            return ge::GRAPH_FAILED);
        winSizeData.totalWinSizeEp = maxWindowSizeEp - rankOffsetSize;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace Mc2Tiling
#endif // MOE_DISTRIBUTE_CHECK_WIN_SIZE

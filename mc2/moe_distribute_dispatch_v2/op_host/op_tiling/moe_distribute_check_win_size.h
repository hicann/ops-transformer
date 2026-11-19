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
#include "../../op_kernel/moe_distribute_v2_constant.h"

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

// rsw_buffer挡位与dispatch/combine共用同一口径: combine单slot按最大输出dtype(2B)估算,
// 保证dispatch侧combine区预留与combine侧实际接收区严格一致, 避免两块区域交叠。
constexpr uint64_t RSW_TRAILER_BYTES = 128UL;
constexpr uint64_t RSW_BUFFER_TIER_LIST_BYTES[] = {16UL << 10,  32UL << 10,  64UL << 10,   128UL << 10,
                                                   256UL << 10, 512UL << 10, 1024UL << 10, 2048UL << 10};
constexpr uint32_t RSW_BUFFER_TIER_COUNT = sizeof(RSW_BUFFER_TIER_LIST_BYTES) / sizeof(RSW_BUFFER_TIER_LIST_BYTES[0]);
inline uint64_t SelectRswBufferTier(uint64_t commDataBytes)
{
    const uint64_t need = commDataBytes + RSW_TRAILER_BYTES;
    for (uint64_t tier : RSW_BUFFER_TIER_LIST_BYTES) {
        if (need <= tier) {
            return tier;
        }
    }
    return 0UL;
}

inline uint64_t GetRswBufferSize(uint64_t h)
{
    return SelectRswBufferTier(h * MAX_OUT_DTYPE_SIZE);
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
    uint32_t sharedExpertRankNum = 0;
    uint64_t rswBufferSize = 0;
    uint32_t rswBufferIdx = RSW_BUFFER_TIER_COUNT;
    uint32_t tokenNumPerSlot = 0;
};

inline uint64_t CalcMinWinSize(const CheckWinSizeData& winSizeData, uint64_t& tokenNeedSizeDispatch,
                               uint64_t& tokenNeedSizeCombine, uint64_t combineDataAlign = WIN_ADDR_ALIGN,
                               uint64_t dispatchTokenBytes = 0UL,
                               uint64_t dispatchMetadataSize = SCALE_EXPAND_IDX_BUFFER)
{
    // A3 Combine uses raw 512-byte-aligned data for both fullmesh variants.
    // A5 callers explicitly request the packed-data alignment.
    const uint64_t h = winSizeData.h;
    const uint64_t k = winSizeData.k;
    const uint64_t maxBs = winSizeData.globalBs / winSizeData.epWorldSize;
    tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE + combineDataAlign - 1UL) / combineDataAlign) * WIN_ADDR_ALIGN;
    // expertScale 复用三元组后的对齐空间，不增加窗口占用。
    const uint64_t defaultTokenLen =
        ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + dispatchMetadataSize;
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
    const auto npuArch = mc2tiling::GetNpuArch(context);
    const uint64_t combineDataAlign = (npuArch == Ops::Base::DAV_2201) ? WIN_ADDR_ALIGN : FULL_MESH_DATA_ALIGN;
    const uint64_t dispatchMetadataSize =
        (npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) ? UB_ALIGN : SCALE_EXPAND_IDX_BUFFER;
    uint64_t actualSize = CalcMinWinSize(winSizeData, tokenNeedSizeDispatch, tokenNeedSizeCombine, combineDataAlign,
                                         dispatchTokenBytes, dispatchMetadataSize);
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
        if (npuArch == Ops::Base::DAV_3510) {
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
                    tokenNeedSizeCombine, winSizeData.k, (actualSize + 2UL * MB_SIZE - 1UL) / MB_SIZE,
                    hcclBufferSizeEp / (MB_SIZE * 2), hcclBufferSizeEp / MB_SIZE),
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

inline uint32_t CalcSerialUbSizeForTokenCopy(const CheckWinSizeData& info, uint32_t aivNum, uint64_t ubRealSize)
{
    using mc2tiling::AlignUp;
    uint32_t moeExpertNumPerRank = info.localMoeExpertNum;
    uint32_t recvStatusNum = info.epWorldSize * moeExpertNumPerRank;
    uint32_t recvStatusPerCore = mc2tiling::MathCeil(recvStatusNum, aivNum);

    uint64_t fixedUbSize = Mc2Kernel::A5_CUMSUM_UB_RESERVE + 2U * UB_ALIGN;
    fixedUbSize += AlignUp<uint32_t>(recvStatusPerCore * UB_ALIGN, Mc2Kernel::SIZE_ALIGN_256);
    uint32_t gatherSize = std::max(AlignUp<uint32_t>(recvStatusPerCore * sizeof(float), UB_ALIGN),
                                   AlignUp<uint32_t>(recvStatusNum * sizeof(float), UB_ALIGN));
    fixedUbSize += std::max(gatherSize, AlignUp<uint32_t>(recvStatusNum * Mc2Kernel::STATE_OFFSET, UB_ALIGN));
    fixedUbSize += 2U * aivNum * UB_ALIGN;
    uint32_t expertPrefixNum = mc2tiling::MathCeil<uint32_t>(recvStatusPerCore, info.epWorldSize);
    fixedUbSize += std::max(AlignUp<uint32_t>(aivNum * sizeof(float), UB_ALIGN),
                            AlignUp<uint32_t>(expertPrefixNum * sizeof(int32_t), UB_ALIGN));
    fixedUbSize += 3U * UB_ALIGN;
    // 各 rank 使用同一套槽布局，按开启维测时的最大占用预留。
    fixedUbSize += AlignUp<uint32_t>(info.epWorldSize * sizeof(int64_t), UB_ALIGN);
    fixedUbSize += AlignUp<uint32_t>(recvStatusPerCore * sizeof(int32_t), UB_ALIGN);
    fixedUbSize += AlignUp<uint32_t>(moeExpertNumPerRank * sizeof(int64_t), UB_ALIGN);
    fixedUbSize += 4U * AlignUp<uint32_t>(recvStatusPerCore * sizeof(int32_t), UB_ALIGN);
    fixedUbSize += 2 * UB_ALIGN;
    return ubRealSize > fixedUbSize ? ubRealSize - fixedUbSize : 0U;
}

inline uint32_t CalcParallelUbSizeForTokenCopy(const CheckWinSizeData& info, uint32_t aivNum, uint64_t ubRealSize)
{
    using mc2tiling::AlignUp;
    constexpr uint32_t HALF_NUM = 2U;
    constexpr uint32_t A5_CUMSUM_MAX_CORE_NUM = 8U;
    uint32_t moeExpertNumPerRank = info.localMoeExpertNum;
    uint32_t recvStatusNum = info.epWorldSize * moeExpertNumPerRank;
    uint32_t totalExpertNum = info.sharedExpertRankNum + info.moeExpertNum;
    uint32_t aivUsedCumSum = totalExpertNum / 16; // 单核处理16个专家cnt发送
    aivUsedCumSum = (aivUsedCumSum == 0) ? 1 : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= (aivNum / HALF_NUM)) ? (aivNum / HALF_NUM) : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= A5_CUMSUM_MAX_CORE_NUM) ? A5_CUMSUM_MAX_CORE_NUM : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= recvStatusNum) ? recvStatusNum : aivUsedCumSum;
    aivUsedCumSum = std::max(aivUsedCumSum, 1U);

    uint32_t recvStatusPerCore = mc2tiling::MathCeil(recvStatusNum, aivNum);
    uint32_t recvCntPrefixNum = recvStatusPerCore + Mc2Kernel::UB_ALIGN_DATA_COUNT;
    uint32_t recvCntPrefixSize = AlignUp<uint32_t>(recvCntPrefixNum * sizeof(int32_t), UB_ALIGN);
    uint32_t expInfoSize = AlignUp<uint32_t>(recvStatusPerCore * sizeof(uint32_t), UB_ALIGN);
    uint32_t validExpertInfoExtraSize = AlignUp<uint32_t>(moeExpertNumPerRank * sizeof(int32_t), UB_ALIGN) +
                                        AlignUp<uint32_t>(moeExpertNumPerRank * sizeof(int64_t), UB_ALIGN);

    uint64_t fixedUbSize = Mc2Kernel::A5_CUMSUM_UB_RESERVE + UB_ALIGN + recvCntPrefixSize +
                           2 * aivUsedCumSum * UB_ALIGN + 3 * expInfoSize + validExpertInfoExtraSize;
    return ubRealSize > fixedUbSize ? ubRealSize - fixedUbSize : 0U;
}

inline uint64_t CalcRSWDispatchDataSectionSize(const gert::TilingContext* context, const char* nodeName,
                                               const CheckWinSizeData& winSizeData, uint64_t dispatchWinSize,
                                               uint32_t& rswBufferIdx, uint32_t& tokenNumPerSlot,
                                               uint32_t dispatchTokenBytes)
{
    constexpr uint32_t LARGE_PACKET_SIZE = 4U * 1024U;
    constexpr uint32_t SERIAL_MODE_BS_THRESHOLD = 32U;
    constexpr uint32_t SERIAL_MODE_EP_WORLD_SIZE_THRESHOLD = 16U;
    // 公共检查按 FP16/BF16 透传行计算，三元组与可选 expertScale 共用末尾的 32B。
    uint32_t tokenOutAlignSize =
        dispatchTokenBytes != 0U ?
            dispatchTokenBytes :
            mc2tiling::AlignUp<uint32_t>(winSizeData.h * MAX_OUT_DTYPE_SIZE, UB_ALIGN) + UB_ALIGN;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK(aivNum == 0U, OP_LOGE(nodeName, "AIV core count must be positive."), return 0U);
    uint64_t ubRealSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    uint32_t maxBs = winSizeData.globalBs / winSizeData.epWorldSize;
    uint32_t ubSizeForCopy = CalcParallelUbSizeForTokenCopy(winSizeData, aivNum, ubRealSize);
    if (maxBs < SERIAL_MODE_BS_THRESHOLD && winSizeData.epWorldSize < SERIAL_MODE_EP_WORLD_SIZE_THRESHOLD) {
        // 缩容时可能切换到并行收，槽布局取两种模式共同满足的容量。
        ubSizeForCopy = std::min(ubSizeForCopy, CalcSerialUbSizeForTokenCopy(winSizeData, aivNum, ubRealSize));
    }
    uint32_t counterSizeInUb = UB_ALIGN;
    uint32_t ubBatch =
        ubSizeForCopy > counterSizeInUb ?
            std::min(maxBs, (ubSizeForCopy - counterSizeInUb) / Mc2Kernel::BUFFER_NUM / tokenOutAlignSize) :
            0U;
    uint32_t tokenOffsetInWin = mc2tiling::AlignUp(tokenOutAlignSize, LARGE_PACKET_SIZE);
    uint32_t dispatchSectionNum = winSizeData.epWorldSize * winSizeData.localMoeExpertNum;
    uint64_t dispatchDataSectionSize = 0U;
    for (uint32_t idx = 0; ubBatch != 0 && idx < RSW_BUFFER_TIER_COUNT; ++idx) {
        uint32_t rswBufferSize = RSW_BUFFER_TIER_LIST_BYTES[idx];
        if (tokenOutAlignSize > rswBufferSize - RSW_TRAILER_BYTES) {
            continue;
        }
        uint32_t capacity = 1U + (rswBufferSize - RSW_TRAILER_BYTES - tokenOutAlignSize) / tokenOffsetInWin;
        uint32_t batch = std::min(ubBatch, capacity);
        uint64_t sectionSize = mc2tiling::MathCeil<uint64_t>(maxBs, batch) * rswBufferSize;
        // 每个半窗的 Dispatch 起点按绝对页地址对齐，最多额外占用 rswBufferSize - 1 字节。
        if (sectionSize * dispatchSectionNum + rswBufferSize - 1U <= dispatchWinSize) {
            dispatchDataSectionSize = sectionSize;
            rswBufferIdx = idx;
            tokenNumPerSlot = batch;
            if (batch == ubBatch) {
                break;
            }
        }
    }
    return dispatchDataSectionSize;
}

inline bool CheckRswWinSize(const gert::TilingContext* context, const char* nodeName,
                            const CheckWinSizeData& winSizeData, uint32_t& rswBufferIdx, uint32_t& tokenNumPerSlot,
                            uint32_t dispatchTokenBytes = 0U)
{
    rswBufferIdx = RSW_BUFFER_TIER_COUNT;
    tokenNumPerSlot = 0U;
    uint64_t maxBs = winSizeData.globalBs / winSizeData.epWorldSize;
    uint64_t tokenNeedSizeCombine = GetRswBufferSize(winSizeData.h);
    uint64_t combineDataRegionSize = maxBs * (winSizeData.k + winSizeData.sharedExpertNum) * tokenNeedSizeCombine;
    uint64_t halfWinSize = winSizeData.totalWinSizeEp / DOUBLE_DATA_BUFFER;
    if (tokenNeedSizeCombine == 0U || halfWinSize < MB_SIZE + combineDataRegionSize) {
        return false;
    }
    uint64_t dispatchWinSize = halfWinSize - MB_SIZE - combineDataRegionSize;
    return CalcRSWDispatchDataSectionSize(context, nodeName, winSizeData, dispatchWinSize, rswBufferIdx,
                                          tokenNumPerSlot, dispatchTokenBytes) != 0U;
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
    winSizeData.totalWinSizeEp = maxWindowSizeEp;
    winSizeData.rswBufferSize = 0;
    winSizeData.rswBufferIdx = RSW_BUFFER_TIER_COUNT;
    winSizeData.tokenNumPerSlot = 0;
    const auto npuArch = mc2tiling::GetNpuArch(context);
    if ((npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) && !winSizeData.isLayered) {
        if (CheckRswWinSize(context, nodeName, winSizeData, winSizeData.rswBufferIdx, winSizeData.tokenNumPerSlot)) {
            winSizeData.rswBufferSize = GetRswBufferSize(winSizeData.h);
            return ge::GRAPH_SUCCESS;
        }
        // arch92 双方一起回退。普通 Dispatch 每个 512B 块容纳 480B 有效载荷。
        winSizeData.isSetFullMeshV2 = true;
    }
    OP_TILING_CHECK(CheckActualWinSize(context, nodeName, winSizeData, maxWindowSizeEp, hcclBufferSizeEp,
                                       dispatchTokenBytes) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check actual window size failed."),
                    return ge::GRAPH_FAILED);

    if (npuArch == Ops::Base::DAV_3510) {
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

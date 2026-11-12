/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GM_LAYOUT_H
#define MEGA_MOE_GM_LAYOUT_H

#include <cstdint>

// Shared GM layout: host, kernel and Torch use the same byte offsets and capacities.
// No allocation, address binding, or synchronization is performed here.

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_CUBE__) || \
    defined(__DAV_310R6_VEC__) || defined(__DAV_920R1_CUBE__) || defined(__DAV_920R1_VEC__) || defined(__NPU_ARCH__)
#define HOST_DEVICE __forceinline__[aicore]
#else
#define HOST_DEVICE inline
#endif

namespace MegaMoeImpl {

enum class MegaMoeActMode : uint8_t {
    SWIGLU = 0U,
    SITU = 1U,
    SWIGLU_STEP = 2U,
    SWIGLU_OAI = 3U,
    GELU = 4U,
};

enum class MegaMoeActSubMode : uint8_t {
    DEFAULT = 0U,
    LINEAR = 1U,
};

enum class MegaMoeGmmMode : uint8_t {
    A8W8_ND = 0U,
    A8W4_NZ = 1U,
    A8W8_NZ = 2U,
    A4W4_ND = 3U,
    A4W4_NZ = 4U,
    A16W16_ND = 5U,
    A16W16_NZ = 6U,
};

constexpr uint8_t GMM_MODE_A8W8_ND = static_cast<uint8_t>(MegaMoeGmmMode::A8W8_ND);
constexpr uint8_t GMM_MODE_A8W4_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A8W4_NZ);
constexpr uint8_t GMM_MODE_A8W8_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A8W8_NZ);
constexpr uint8_t GMM_MODE_A4W4_ND = static_cast<uint8_t>(MegaMoeGmmMode::A4W4_ND);
constexpr uint8_t GMM_MODE_A4W4_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A4W4_NZ);
constexpr uint8_t GMM_MODE_A16W16_ND = static_cast<uint8_t>(MegaMoeGmmMode::A16W16_ND);
constexpr uint8_t GMM_MODE_A16W16_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A16W16_NZ);

// 本头只计算容量和偏移，供容量申请、tiling 校验和核内地址绑定共同使用。
namespace MemoryLayoutDetail {
constexpr int64_t ALIGN_32 = 32;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t INT_CACHELINE = 16;
constexpr int64_t INT32_PER_32B = ALIGN_32 / sizeof(int32_t);
constexpr int64_t BITS_PER_BYTE = 8;
constexpr uint32_t FP4_BITS = 4U;
constexpr uint32_t FP8_BITS = 8U;
constexpr uint32_t BF16_BITS = 16U;
constexpr int64_t MXFP_DIVISOR_SIZE = 64;
constexpr int64_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t L1_TILE_M_256 = 256;
constexpr int64_t ACTIVATION_N_HALF = 2;
constexpr int64_t DEDUP_PREPASS_MAX_CHUNKS = 128;
constexpr int64_t DEDUP_ROW_DESC_HEADER_INT32 = 2;
constexpr int64_t MAX_INT16_TOPK_INDEX_COUNT = 1LL << 15;

HOST_DEVICE constexpr int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

HOST_DEVICE constexpr int64_t CeilAlign(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}
} // namespace MemoryLayoutDetail

constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60 * 1024LL;
constexpr int64_t SIZE_INT_8 = 1;
constexpr int64_t SIZE_INT_32 = 4;
constexpr int64_t SIZE_BF_16 = 2;
constexpr int64_t INVALID_WORKSPACE_OFFSET = -1;
constexpr uint32_t MTE_RING_WAVE_COUNT = 3U;

// 原同步布局：前 48 KiB 是 rank 槽位，随后 12 KiB 是各 AIV 的同步计数区。
// URMA 的 rank 槽位超过预留区时动态扩展；MTE 保持原 60 KiB 固定布局。
constexpr int64_t PEERMEM_MIN_RANK_SYNC_SIZE = 1024 * 48LL;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
// MTE count 区固定预留 2048 个 512B 槽，每槽首个 int32 有效。
constexpr int64_t PEERMEM_MTE_COUNT_REGION_SIZE = 512LL * 2048LL;
constexpr int64_t PEERMEM_SYNC_COUNT_REGION_SIZE = PEERMEM_DATA_OFFSET - PEERMEM_MIN_RANK_SYNC_SIZE;
constexpr int64_t PEERMEM_SYNC_SLOT_SIZE = static_cast<int64_t>(MemoryLayoutDetail::INT_CACHELINE) * sizeof(int32_t);

// 通信拓扑：决定窗口数据区按 MTE 还是 URMA 布局推进。
constexpr int64_t TOPO_TYPE_MTE = 0U;
constexpr int64_t TOPO_TYPE_URMA = 1U;
constexpr int64_t TOPO_TYPE_URMA_ROCE = 2U;

HOST_DEVICE constexpr bool IsW4GmmMode(uint8_t gmmMode)
{
    return gmmMode == GMM_MODE_A8W4_NZ || gmmMode == GMM_MODE_A4W4_ND || gmmMode == GMM_MODE_A4W4_NZ;
}

HOST_DEVICE constexpr bool IsA4W4GmmMode(uint8_t gmmMode)
{
    return gmmMode == GMM_MODE_A4W4_ND || gmmMode == GMM_MODE_A4W4_NZ;
}

HOST_DEVICE constexpr bool IsA16W16GmmMode(uint8_t gmmMode)
{
    return gmmMode == GMM_MODE_A16W16_ND || gmmMode == GMM_MODE_A16W16_NZ;
}

HOST_DEVICE constexpr uint32_t GetActivationOutputDivisor(uint8_t actMode)
{
    return actMode == static_cast<uint8_t>(MegaMoeActMode::GELU) ? 1U : MemoryLayoutDetail::ACTIVATION_N_HALF;
}

HOST_DEVICE constexpr int64_t GetActivationOutputWidth(int64_t hiddenDim, uint8_t actMode)
{
    return hiddenDim / GetActivationOutputDivisor(actMode);
}

HOST_DEVICE constexpr bool IsLayeredTopology(int64_t topo)
{
    return topo == TOPO_TYPE_URMA || topo == TOPO_TYPE_URMA_ROCE;
}

HOST_DEVICE constexpr int64_t CalcUrmaRankSyncSize(int64_t epWorldSize)
{
    int64_t rankSyncSize = epWorldSize * PEERMEM_SYNC_SLOT_SIZE;
    return rankSyncSize > PEERMEM_MIN_RANK_SYNC_SIZE ? rankSyncSize : PEERMEM_MIN_RANK_SYNC_SIZE;
}

// ============================ 各区尺寸（host / device 共用） ============================

// URMA layered 路径仍使用 mask 槽；MTE WAVE 不再经过该布局。
HOST_DEVICE constexpr int64_t CalcDispatchMaskAlignSize(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t sendTotalNum = numMaxTokensPerRank * topK;
    int64_t alignedRouteCount = MemoryLayoutDetail::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)),
                                                              MemoryLayoutDetail::ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    return MemoryLayoutDetail::CeilAlign(alignedRouteCount / static_cast<int64_t>(MemoryLayoutDetail::BITS_PER_BYTE),
                                         MemoryLayoutDetail::ALIGN_32);
}

// 所有 Rank 使用一致的 numMaxTokensPerRank，因此可据此统一选择 int16 或 int32 编码。
HOST_DEVICE constexpr bool UseInt16TopkIndex(int64_t numMaxTokensPerRank, int64_t topK)
{
    return topK > 0 && numMaxTokensPerRank >= 0 &&
           numMaxTokensPerRank * topK <= MemoryLayoutDetail::MAX_INT16_TOPK_INDEX_COUNT;
}

HOST_DEVICE constexpr int64_t CalcTopkIndexTypeBytes(int64_t numMaxTokensPerRank, int64_t topK)
{
    return UseInt16TopkIndex(numMaxTokensPerRank, topK) ? static_cast<int64_t>(sizeof(int16_t)) :
                                                          static_cast<int64_t>(sizeof(int32_t));
}

// 保留 MTE 的原索引宽度阈值；分层通信固定使用 int32 路由索引。
HOST_DEVICE constexpr int64_t CalcTopkIndexTypeBytesForTopology(int64_t topoType, int64_t numMaxTokensPerRank,
                                                                int64_t topK)
{
    return topoType == TOPO_TYPE_MTE ? CalcTopkIndexTypeBytes(numMaxTokensPerRank, topK) :
                                       static_cast<int64_t>(sizeof(int32_t));
}

// 每个 (localExpert, srcRank) 槽直接保存有序 topkIndex。
// 同一 token 的 topK expert id 不重复，因此单专家从一张源卡最多接收 numMaxTokensPerRank 个 index。
// route index 接收区不携带槽尾 count；MTE count 使用独立的 512B 间隔表。
HOST_DEVICE constexpr int64_t CalcRouteIndexRecvSize(int64_t routeIndexAlignSize, int64_t moeExpertPerRank,
                                                     int64_t epWorldSize)
{
    return MemoryLayoutDetail::CeilAlign(moeExpertPerRank * epWorldSize * routeIndexAlignSize,
                                         MemoryLayoutDetail::ALIGN_512);
}

HOST_DEVICE constexpr int64_t CalcMaskRecvSize(int64_t maskAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    int64_t maskSlotSize = maskAlignSize + MemoryLayoutDetail::ALIGN_32;
    return MemoryLayoutDetail::CeilAlign(moeExpertPerRank * epWorldSize * maskSlotSize, MemoryLayoutDetail::ALIGN_512);
}

// URMA 的独立 raw count 表仍按 [localExpert][sourceRank] 连续排列；不使用 MTE slot 布局。
HOST_DEVICE constexpr int64_t CalcExpertCountRecvSize(int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return MemoryLayoutDetail::CeilAlign(moeExpertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)),
                                         MemoryLayoutDetail::ALIGN_512);
}

// dedupMode 的值就是门控后实际开启的去重模式：
// 0 = dispatch 关、combine 关；1 = dispatch 关、combine 开；
// 2 = dispatch 开、combine 关；3 = dispatch 开、combine 开；4 = URMA Server Combine。
// 去重判断由本头统一提供，host 和 kernel 不再各自比较模式数值。
constexpr int32_t DEDUP_MODE_SERVER_COMBINE = 4;
HOST_DEVICE constexpr bool IsDispatchDedupOn(int32_t dedupMode)
{
    return dedupMode == 2 || dedupMode == 3;
}

HOST_DEVICE constexpr bool IsCombineDedupOn(int32_t dedupMode)
{
    return dedupMode == 1 || dedupMode == 3;
}

HOST_DEVICE constexpr bool IsServerCombineDedupOn(int32_t dedupMode)
{
    return dedupMode == DEDUP_MODE_SERVER_COMBINE;
}

HOST_DEVICE constexpr int64_t CalcServerCombineReadyRowBytes(int64_t capacity)
{
    // 每个 token 一份 uint64 位图，末尾另存 count+1，整行按 32B 对齐。
    return MemoryLayoutDetail::CeilAlign((capacity + 1) * static_cast<int64_t>(sizeof(uint64_t)),
                                         MemoryLayoutDetail::ALIGN_32);
}

HOST_DEVICE constexpr int64_t CalcServerCombineTokenListStrideBytes(int64_t capacity)
{
    // 每个源 Server 紧凑保存 int32 count+1 和最多 capacity 个 token ID。
    return (capacity + 1) * static_cast<int64_t>(sizeof(int32_t));
}

// 去重 rowDesc 每行的 int32 数：头 2 个 + 成员三元组 3*topK 个，向上对齐到 8 个 int32（32B），
// 使描述符批量搬运天然 32B 对齐。字段语义见 mega_moe_constants.h 的 DEDUP_ROW_* 注释。
HOST_DEVICE constexpr int64_t CalcDedupRowDescStrideInt32(int64_t topK)
{
    int64_t rawInt32 = static_cast<int64_t>(MemoryLayoutDetail::DEDUP_ROW_DESC_HEADER_INT32) + 3LL * topK;
    return MemoryLayoutDetail::CeilAlign(rawInt32, static_cast<int64_t>(MemoryLayoutDetail::INT32_PER_32B));
}

// combine 接收区单 token 记录字节数：须与 kernel InitCombineBuffers 的 combine record 布局一致。
HOST_DEVICE constexpr int64_t CalcCombineTokenBytes(int64_t h, int64_t yDtypeSize, bool isQuantCombine)
{
    if (!isQuantCombine) {
        return h * yDtypeSize;
    }
    int64_t tokenStorageBytes = MemoryLayoutDetail::CeilAlign(h, MemoryLayoutDetail::ALIGN_256);
    int64_t scaleCount = MemoryLayoutDetail::CeilDiv(h, static_cast<int64_t>(MemoryLayoutDetail::MXFP_SCALE_GROUP_NUM));
    int64_t storedScaleBytes =
        MemoryLayoutDetail::CeilAlign(scaleCount, static_cast<int64_t>(MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE));
    return MemoryLayoutDetail::CeilAlign(tokenStorageBytes + storedScaleBytes, MemoryLayoutDetail::ALIGN_32);
}

// 位宽直接描述物理存储；当前 FP4/FP8 使用 MX scale，BF16 不使用。
struct TensorStorageSpec {
    uint32_t elementBits{MemoryLayoutDetail::FP8_BITS};

    HOST_DEVICE constexpr bool HasMxScale() const
    {
        return elementBits == MemoryLayoutDetail::FP4_BITS || elementBits == MemoryLayoutDetail::FP8_BITS;
    }
};

HOST_DEVICE constexpr int64_t CalcStorageDataBytes(int64_t elements, const TensorStorageSpec& storage)
{
    return MemoryLayoutDetail::CeilDiv(elements * storage.elementBits, MemoryLayoutDetail::BITS_PER_BYTE);
}

struct TokenScaleLayout {
    int64_t dataValidBytes;
    int64_t dataBytes;
    int64_t weightOffsetBytes;
    int64_t weightBytes;
    int64_t tokenScaleBytes;
};

// TokenScale 统一包含数据、可选 MX scale 和可选 FP32 topK 路由权重。
HOST_DEVICE constexpr TokenScaleLayout CalcTokenScaleLayout(int64_t h, const TensorStorageSpec& storage, int64_t topK,
                                                            bool carryTopkWeights)
{
    const int64_t dataValidBytes = CalcStorageDataBytes(h, storage);
    const int64_t dataBytes = MemoryLayoutDetail::CeilAlign(dataValidBytes, MemoryLayoutDetail::ALIGN_256);
    const int64_t scaleBytes =
        storage.HasMxScale() ?
            MemoryLayoutDetail::CeilAlign(MemoryLayoutDetail::CeilDiv(h, MemoryLayoutDetail::MXFP_SCALE_GROUP_NUM),
                                          MemoryLayoutDetail::ALIGN_32) :
            0;
    const int64_t weightOffsetBytes = dataBytes + scaleBytes;
    const int64_t weightBytes =
        carryTopkWeights ?
            MemoryLayoutDetail::CeilAlign(topK * static_cast<int64_t>(sizeof(float)), MemoryLayoutDetail::ALIGN_32) :
            0;
    return {dataValidBytes, dataBytes, weightOffsetBytes, weightBytes, weightOffsetBytes + weightBytes};
}

// 量化调用点沿用每字节元素数，记录各段布局与 BF16 共用同一公式。
HOST_DEVICE constexpr int64_t CalcQuantTokenScaleBytes(int64_t h, uint32_t elemsPerByte, int64_t topK,
                                                       bool topkWeightsPrefetch)
{
    const TensorStorageSpec storage{static_cast<uint32_t>(MemoryLayoutDetail::BITS_PER_BYTE) / elemsPerByte};
    return CalcTokenScaleLayout(h, storage, topK, topkWeightsPrefetch).tokenScaleBytes;
}

// 每个输入槽先保存完整 TokenScale，必要时在槽尾暂存待转换的 BF16 路由权重。
HOST_DEVICE constexpr int64_t CalcBf16InputStageBytes(const TokenScaleLayout& layout, int64_t topK,
                                                      bool bf16TopkWeights)
{
    const int64_t conversionBytes = layout.weightBytes != 0 && bf16TopkWeights ?
                                        MemoryLayoutDetail::CeilAlign(topK * SIZE_BF_16, MemoryLayoutDetail::ALIGN_32) :
                                        0;
    return MemoryLayoutDetail::CeilAlign(layout.tokenScaleBytes + conversionBytes, MemoryLayoutDetail::ALIGN_512);
}

// CCL 通信数据及协议状态的布局参数，不依赖 tiling 类型。
struct CclCommLayoutParams {
    int64_t numMaxTokensPerRank;
    int64_t topK;
    int64_t h;
    int64_t moeExpertPerRank;
    int64_t epWorldSize;
    int64_t yDtypeSize;
    uint32_t elemsPerByte;
    bool topkWeightsPrefetch;
    bool isQuantCombine;
    int64_t topoType;
    int64_t serverNum;
    bool serverCombineReduce{false};
    TensorStorageSpec dispatchStorage{};
};

// Offsets and capacities are bytes; the caller owns the supplied GM base.
struct GmBuffer {
    int64_t offsetBytes{-1};
    int64_t sizeBytes{0};
    template <typename T = uint8_t>
#if defined(__NPU_ARCH__) || defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_CUBE__) || \
    defined(__DAV_310R6_VEC__) || defined(__DAV_920R1_CUBE__) || defined(__DAV_920R1_VEC__)
    HOST_DEVICE __gm__ T* get_ptr(__gm__ uint8_t* base) const
    {
        return base != nullptr && offsetBytes >= 0 ? reinterpret_cast<__gm__ T*>(base + offsetBytes) : nullptr;
    }
#else
    HOST_DEVICE T* get_ptr(uint8_t* base) const
    {
        return base != nullptr && offsetBytes >= 0 ? reinterpret_cast<T*>(base + offsetBytes) : nullptr;
    }
#endif
};

struct QuantizedBuffers {
    GmBuffer data{};
    GmBuffer scale{};
};

struct GmmBuffers {
    QuantizedBuffers gmm1Input;
    GmBuffer gmm1Output{};
    QuantizedBuffers gmm2Input;
    GmBuffer gmm2Output{};
};

struct RoutingBuffers {
    GmBuffer recvCounts{};
    GmBuffer receivedTokenMetadata{};
    GmBuffer recvPrefixBackup{};
    GmBuffer tokenWeights{};
};

struct SyncBuffers {
    GmBuffer activationReady{};
    GmBuffer dispatchReady{};
    GmBuffer countsReady{};
};

struct MteSyncBuffers {
    GmBuffer sharedActivationReady{};
    GmBuffer aiv1TopkReady{};
    GmBuffer gmm2Ready{};
    GmBuffer sharedGmm2Tiles{};
    GmBuffer dedupReady{};
};

struct MteBuffers {
    GmBuffer dedupRowDescriptors{};
    MteSyncBuffers sync;
};

struct UrmaSyncBuffers {
    GmBuffer combineCounters{};
    GmBuffer sharedGmm2Tiles{};
};

struct ServerCombineBuffers {
    GmBuffer tokenIds{};
    GmBuffer tokenBitmaps{};
    GmBuffer completedCounts{};
};

struct UrmaBuffers {
    ServerCombineBuffers serverCombine;
    GmBuffer mask{};
    GmBuffer relaySendQueue{};
    GmBuffer remoteReadySnapshot{};
    UrmaSyncBuffers sync;
};

// These ranges reference existing sync storage; they do not allocate extra buffers.
struct WorkspaceResetRanges {
    GmBuffer serverCombine;
    GmBuffer flags;
};

// CCL 通信布局的尺寸与偏移，容量申请、主机校验和核内地址绑定共用。
struct CclCommLayout {
    GmBuffer rankSync;
    GmBuffer syncCount;
    GmBuffer maskRecv;
    GmBuffer expertCountRecv;
    GmBuffer quantTokenScale;
    GmBuffer dispatchReceive;
    GmBuffer dispatchFlag;
    GmBuffer sourceReady;
    GmBuffer contributionReady;
    GmBuffer combineSend;
    GmBuffer contribution;
    GmBuffer combineReceive;
    int64_t totalSize{0};

    HOST_DEVICE constexpr GmBuffer AppendRegion(int64_t bytes)
    {
        GmBuffer region{totalSize, bytes};
        totalSize += bytes;
        return region;
    }
};

HOST_DEVICE constexpr CclCommLayout CalcMteCclCommLayout(const CclCommLayoutParams& params)
{
    CclCommLayout sizes{};
    sizes.rankSync = sizes.AppendRegion(PEERMEM_MIN_RANK_SYNC_SIZE);
    sizes.syncCount = sizes.AppendRegion(PEERMEM_SYNC_COUNT_REGION_SIZE);
    const int64_t routeIndexAlignSize = MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * CalcTopkIndexTypeBytes(params.numMaxTokensPerRank, params.topK),
        MemoryLayoutDetail::ALIGN_32);
    const int64_t maskRecvSize =
        CalcRouteIndexRecvSize(routeIndexAlignSize, params.moeExpertPerRank, params.epWorldSize);
    sizes.expertCountRecv = sizes.AppendRegion(PEERMEM_MTE_COUNT_REGION_SIZE);
    sizes.maskRecv = sizes.AppendRegion(maskRecvSize);
    const int64_t tokenScaleBytes =
        CalcTokenScaleLayout(params.h, params.dispatchStorage, params.topK, params.topkWeightsPrefetch).tokenScaleBytes;
    const int64_t dispatchRecordAreaSize =
        MemoryLayoutDetail::CeilAlign(params.numMaxTokensPerRank * tokenScaleBytes, MemoryLayoutDetail::ALIGN_512);
    sizes.quantTokenScale = sizes.AppendRegion(dispatchRecordAreaSize);
    const int64_t combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    const int64_t combineSendSize = MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * params.topK * combineTokenBytes, MemoryLayoutDetail::ALIGN_512);
    sizes.combineSend = sizes.AppendRegion(combineSendSize);
    return sizes;
}

HOST_DEVICE constexpr CclCommLayout CalcUrmaCclCommLayout(const CclCommLayoutParams& params)
{
    CclCommLayout sizes{};
    sizes.rankSync = sizes.AppendRegion(CalcUrmaRankSyncSize(params.epWorldSize));
    sizes.syncCount = sizes.AppendRegion(PEERMEM_SYNC_COUNT_REGION_SIZE);
    sizes.totalSize = MemoryLayoutDetail::CeilAlign(sizes.totalSize, MemoryLayoutDetail::ALIGN_512);
    const int64_t maskRecvSize = CalcMaskRecvSize(CalcDispatchMaskAlignSize(params.numMaxTokensPerRank, params.topK),
                                                  params.moeExpertPerRank, params.epWorldSize);
    sizes.maskRecv = sizes.AppendRegion(maskRecvSize);
    sizes.expertCountRecv = sizes.AppendRegion(CalcExpertCountRecvSize(params.moeExpertPerRank, params.epWorldSize));
    const int64_t tokenScaleBytes = CalcQuantTokenScaleBytes(params.h, params.elemsPerByte, params.topK,
                                                             params.topkWeightsPrefetch || params.serverCombineReduce);
    const int64_t relayRecordBytes = MemoryLayoutDetail::CeilAlign(tokenScaleBytes, MemoryLayoutDetail::ALIGN_512);
    const int64_t relayDataSize = MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * relayRecordBytes * params.serverNum, MemoryLayoutDetail::ALIGN_512);
    const int64_t readyRowBytes = params.serverCombineReduce ?
                                      CalcServerCombineReadyRowBytes(params.numMaxTokensPerRank) :
                                      params.numMaxTokensPerRank * static_cast<int64_t>(sizeof(uint64_t));
    const int64_t relayFlagSize =
        MemoryLayoutDetail::CeilAlign(params.serverNum * readyRowBytes, MemoryLayoutDetail::ALIGN_512);
    sizes.dispatchReceive = sizes.AppendRegion(relayDataSize);
    sizes.dispatchFlag = sizes.AppendRegion(relayFlagSize);
    const int64_t combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    if (params.serverCombineReduce) {
        // 三个状态区连续分配，供 Layered 入口统一清零。
        sizes.sourceReady = sizes.AppendRegion(
            MemoryLayoutDetail::CeilAlign(params.serverNum * readyRowBytes, MemoryLayoutDetail::ALIGN_512));
        sizes.contributionReady = sizes.AppendRegion(MemoryLayoutDetail::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * params.topK * static_cast<int64_t>(sizeof(uint64_t)),
            MemoryLayoutDetail::ALIGN_512));
        const int64_t tokenBytes = params.h * SIZE_BF_16;
        sizes.contribution = sizes.AppendRegion(MemoryLayoutDetail::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * params.topK * tokenBytes, MemoryLayoutDetail::ALIGN_512));
        sizes.combineReceive = sizes.AppendRegion(MemoryLayoutDetail::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * tokenBytes, MemoryLayoutDetail::ALIGN_512));
    } else {
        const int64_t combineSendSize = MemoryLayoutDetail::CeilAlign(
            params.numMaxTokensPerRank * params.topK * combineTokenBytes, MemoryLayoutDetail::ALIGN_512);
        sizes.combineSend = sizes.AppendRegion(combineSendSize);
    }
    return sizes;
}

HOST_DEVICE constexpr CclCommLayout CalcCclCommLayout(const CclCommLayoutParams& params)
{
    return params.topoType == TOPO_TYPE_MTE ? CalcMteCclCommLayout(params) : CalcUrmaCclCommLayout(params);
}

// Communication regions are relative to the caller's window; workspace is its local-compute suffix.
struct PeermemInfo : CclCommLayout {
    GmBuffer workspace{};
    HOST_DEVICE constexpr PeermemInfo() = default;
    HOST_DEVICE constexpr explicit PeermemInfo(const CclCommLayoutParams& params, int64_t workspaceBytes = 0)
        : CclCommLayout(CalcCclCommLayout(params)),
          workspace{MemoryLayoutDetail::CeilAlign(totalSize, MemoryLayoutDetail::ALIGN_512), workspaceBytes}
    {}
};

// 各阶段的循环配置同时用于容量规划与核内寻址，不再嵌套判断总开关。
struct CclStageRingConfig {
    bool dispatch{false};
    bool gmm1{false};
    bool activation{false};
    bool gmm2{false};
    uint64_t rows{0};
};

// CCL 本地计算数据及辅助状态的布局参数；未知特征由容量申请侧取安全上界。
struct CclComputeLayoutParams {
    CclStageRingConfig ring{};
    int64_t topoType{TOPO_TYPE_MTE};
    int64_t maxOutputSize{0};
    int64_t h{0};
    int64_t hiddenDim{0};
    uint32_t activationOutputDivisor{MemoryLayoutDetail::ACTIVATION_N_HALF};
    int64_t bs{0};               // 本卡真实 token 数。
    int64_t moeExpertPerRank{0}; // 本卡参与 topK 路由的专家数，与 weight1 的专家数一致。
    int64_t epWorldSize{0};
    int64_t aicNum{0};
    int64_t sharedExpertNum{0}; // 独立 dense 路径的共享专家数，不进入路由 expert id 空间。
    int64_t mGroupsPerWave{0};  // 每个 Wave 消费的 256 行 M 分组预算。
    int64_t combineSyncSlotCountPerExpert{0}; // 每专家预留的 GMM2 到 Combine 同步槽数。
    int64_t serverNum{1};
    int64_t numMaxTokensPerRank{0}; // 全卡一致的单卡 token 数上界，用于通信槽和 mask 的几何。
    int64_t topK{0};
    int32_t dedupMode{0};
    bool combineQuant{false};
    bool topkWeightsPrefetch{false};
    bool dispatchFp4{false};
    bool reserveGmm1Output{false};
    bool reserveCumsum{false};
    bool sharedInputIndependent{false};
    uint32_t sharedInputElementsPerByte{1};
    TensorStorageSpec dispatchInputStorage{};
    TensorStorageSpec activationOutputStorage{};
    TensorStorageSpec sharedInputStorage{};
    TensorStorageSpec sharedActivationStorage{};
};

HOST_DEVICE constexpr int64_t CalcDispatchRouteIndexAlignSize(const CclComputeLayoutParams& params)
{
    return MemoryLayoutDetail::CeilAlign(
        params.numMaxTokensPerRank * CalcTopkIndexTypeBytes(params.numMaxTokensPerRank, params.topK),
        MemoryLayoutDetail::ALIGN_32);
}

// GMM1 每核至少覆盖四个逻辑 N tile，GMM2 每核至少覆盖一个。
HOST_DEVICE constexpr int64_t CalcMemoryMGroupsPerWave(int64_t hiddenDim, int64_t h, int64_t aicNum)
{
    if (hiddenDim == 0 || h == 0 || aicNum == 0) {
        return 1;
    }
    const int64_t gmm1Groups = MemoryLayoutDetail::CeilDiv(
        aicNum * 4, MemoryLayoutDetail::CeilDiv(hiddenDim, MemoryLayoutDetail::L1_TILE_M_256));
    const int64_t gmm2Groups =
        MemoryLayoutDetail::CeilDiv(aicNum, MemoryLayoutDetail::CeilDiv(h, MemoryLayoutDetail::L1_TILE_M_256));
    return gmm1Groups > gmm2Groups ? gmm1Groups : gmm2Groups;
}

// CCL 本地计算区从通信布局尾部开始，MTE 另保留头部异常记录区域。
HOST_DEVICE constexpr int64_t CalcCclComputeOffset(const CclCommLayoutParams& params)
{
    const int64_t exceptionBytes = params.topoType == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
    return MemoryLayoutDetail::CeilAlign(exceptionBytes + CalcCclCommLayout(params).totalSize,
                                         MemoryLayoutDetail::ALIGN_512);
}

HOST_DEVICE constexpr int64_t CalcMemoryRingRows(int64_t mGroupsPerWave)
{
    return static_cast<int64_t>(MTE_RING_WAVE_COUNT) * mGroupsPerWave * MemoryLayoutDetail::L1_TILE_M_256;
}

// 准入规则仅在此处维护；布局与核内寻址直接使用对应阶段的配置。
HOST_DEVICE constexpr CclStageRingConfig CalcCclStageRingConfig(const CclComputeLayoutParams& params)
{
    CclStageRingConfig ring{};
    ring.rows = CalcMemoryRingRows(params.mGroupsPerWave);
    const bool canReuse = params.topoType == TOPO_TYPE_MTE && !params.combineQuant && params.mGroupsPerWave > 0 &&
                          params.maxOutputSize > static_cast<int64_t>(ring.rows);
    // Combine 去重按完整专家消费，专家内缺少逐 tile 进度约束，Dispatch 输入也需全量保存。
    ring.dispatch = canReuse && !IsDispatchDedupOn(params.dedupMode) && !IsCombineDedupOn(params.dedupMode);
    ring.gmm1 = canReuse;
    ring.activation = canReuse;
    ring.gmm2 = canReuse && !IsCombineDedupOn(params.dedupMode);
    return ring;
}

HOST_DEVICE constexpr int64_t CalcSharedActivationFlagElementsPerExpert(int64_t tokenNum)
{
    return MemoryLayoutDetail::CeilDiv(tokenNum, MemoryLayoutDetail::L1_TILE_M_256) * MemoryLayoutDetail::INT_CACHELINE;
}

// Regions are relative to the workspace base. Architecture-specific differences enter through constructors.
struct WorkspaceLayout {
    // Logical grouping does not determine physical allocation order.
    GmmBuffers routed;
    GmmBuffers shared;
    RoutingBuffers routing;
    SyncBuffers sync;
    MteBuffers mte;
    UrmaBuffers urma;
    WorkspaceResetRanges reset;

    int64_t workspaceSize{0};
    HOST_DEVICE constexpr WorkspaceLayout() = default;

    // Arch92 supplies its GMM modes; all other geometry comes from the shared params.
    struct Arch92 {
        uint8_t moeGmmMode;
        uint8_t sharedGmmMode;
    };

    HOST_DEVICE constexpr explicit WorkspaceLayout(const CclComputeLayoutParams& params)
    {
        Initialize(params, Architecture::Arch35);
    }

    HOST_DEVICE constexpr WorkspaceLayout(const CclComputeLayoutParams& params, Arch92 modes)
    {
        auto archParams = params;
        archParams.ring = {};
        archParams.dispatchFp4 = IsA4W4GmmMode(modes.moeGmmMode);
        archParams.reserveGmm1Output =
            params.topoType == TOPO_TYPE_MTE || modes.moeGmmMode == GMM_MODE_A8W4_NZ || params.topkWeightsPrefetch;
        archParams.reserveCumsum = false;
        archParams.sharedInputElementsPerByte = IsA4W4GmmMode(modes.sharedGmmMode) ? 2U : 1U;
        archParams.dispatchInputStorage = {archParams.dispatchFp4 ? MemoryLayoutDetail::FP4_BITS :
                                                                    MemoryLayoutDetail::FP8_BITS};
        archParams.sharedInputStorage = {IsA4W4GmmMode(modes.sharedGmmMode) ? MemoryLayoutDetail::FP4_BITS :
                                                                              MemoryLayoutDetail::FP8_BITS};
        archParams.activationOutputStorage = {params.topoType == TOPO_TYPE_MTE ?
                                                  archParams.dispatchInputStorage.elementBits :
                                                  MemoryLayoutDetail::FP8_BITS};
        archParams.sharedActivationStorage = {params.topoType == TOPO_TYPE_MTE ?
                                                  archParams.sharedInputStorage.elementBits :
                                                  MemoryLayoutDetail::FP8_BITS};
        Initialize(archParams, Architecture::Arch92);
    }

private:
    enum class Architecture {
        Arch35,
        Arch92
    };

    // Append exactly the supplied bytes; alignment is explicit at the call site.
    HOST_DEVICE constexpr void Allocate(GmBuffer& buffer, int64_t sizeBytes)
    {
        buffer.offsetBytes = workspaceSize;
        buffer.sizeBytes = sizeBytes;
        workspaceSize += sizeBytes;
    }

    HOST_DEVICE constexpr void Initialize(const CclComputeLayoutParams& params, Architecture arch)
    {
        workspaceSize = 0;
        // MTE 与 URMA 分别保持既有分区顺序。
        if (params.topoType == TOPO_TYPE_MTE) {
            InitializeMte(params, arch);
        } else {
            InitializeUrma(params, params.serverNum, arch);
        }
    }

    HOST_DEVICE constexpr void InitializeMte(const CclComputeLayoutParams& params, Architecture arch)
    {
        InitializeDispatchBuffers(params);
        // tkw=0 且 combine 去重：dispatch 把 LAST 行整 token 权重段落 revWeights 区、merge 读回乘权，
        // MTE 同样依赖本区。漏分配则 dispatchRevWeightsPtr 为 nullptr，首 launch 即 MTE invalid GM。
        // tkw=1 与去重关闭态在 MTE 下不使用本区，保持既有布局不变。
        if (!params.topkWeightsPrefetch && IsCombineDedupOn(params.dedupMode)) {
            InitializeDispatchRevWeights(params);
        }
        // 每个 AIC 的计数跨度，以 int32_t 元素计。
        int64_t expertRecvTokenCountStride = static_cast<int64_t>(MemoryLayoutDetail::CeilAlign(
            params.moeExpertPerRank, static_cast<uint32_t>(MemoryLayoutDetail::INT_CACHELINE)));
        int64_t expertRecvTokenCountBytes =
            expertRecvTokenCountStride * static_cast<int64_t>(sizeof(int32_t)) * params.aicNum;
        InitializeActivationBuffers(params);
        InitializeRoutingMetadata(params, expertRecvTokenCountBytes);
        InitializeMteSyncFlags(params);
        const bool needsReloadCumsum = params.reserveCumsum;
        InitializeGmmIntermediateBuffers(params, needsReloadCumsum, true);
        InitializeDedupRowDesc(params);
        if (params.sharedExpertNum > 0) {
            InitializeSharedExpertGmmOutputs(params);
            if (params.sharedInputIndependent) {
                InitializeMteSharedExpertInput(params);
            }
            InitializeSharedExpertActivationBuffers(params);
        }
    }

    HOST_DEVICE constexpr void InitializeUrma(const CclComputeLayoutParams& params, int64_t serverNum,
                                              Architecture arch)
    {
        InitializeDispatchBuffers(params);
        InitializeDispatchRevWeights(params);
        const int64_t expertRecvTokenCountStride =
            MemoryLayoutDetail::CeilAlign(params.moeExpertPerRank, MemoryLayoutDetail::INT_CACHELINE);
        int64_t expertRecvTokenCountBytes = expertRecvTokenCountStride * SIZE_INT_32 * params.aicNum;
        InitializeActivationBuffers(params);
        InitializeRoutingMetadata(params, expertRecvTokenCountBytes);
        InitializeUrmaSyncFlags(params, arch);
        // URMA 所有 GMM 模式均保留 dispatch 累积计数。
        InitializeGmmIntermediateBuffers(params, true, false);
        InitializeDedupRowDesc(params);
        InitializeUrmaDispatchBuffers(params, serverNum);
        if (arch == Architecture::Arch35 && IsServerCombineDedupOn(params.dedupMode)) {
            InitializeServerCombineBuffers(params, serverNum);
        }
        if (params.sharedExpertNum > 0) {
            InitializeSharedExpertGmmOutputs(params);
            // Arch35 Layered 直接读取本卡交织量化记录；保留 Arch92 的原布局。
            if (arch == Architecture::Arch92) {
                InitializeUrmaSharedExpertInput(params);
            }
            InitializeSharedExpertActivationBuffers(params);
        }
    }

    HOST_DEVICE constexpr void InitializeServerCombineBuffers(const CclComputeLayoutParams& params, int64_t serverNum)
    {
        const int64_t capacity = params.numMaxTokensPerRank;
        // 每个 Server 的完成计数独占 64B，仅本段在入口统一清零。
        Allocate(urma.serverCombine.completedCounts,
                 MemoryLayoutDetail::CeilAlign(serverNum * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32,
                                               MemoryLayoutDetail::ALIGN_512));
        reset.serverCombine = GmBuffer{urma.serverCombine.completedCounts.offsetBytes,
                                       workspaceSize - urma.serverCombine.completedCounts.offsetBytes};
        Allocate(urma.serverCombine.tokenIds,
                 MemoryLayoutDetail::CeilAlign(serverNum * CalcServerCombineTokenListStrideBytes(capacity),
                                               MemoryLayoutDetail::ALIGN_512));
        Allocate(urma.serverCombine.tokenBitmaps,
                 MemoryLayoutDetail::CeilAlign(serverNum * capacity * static_cast<int64_t>(sizeof(uint32_t)),
                                               MemoryLayoutDetail::ALIGN_512));
    }

    HOST_DEVICE constexpr void InitializeDispatchBuffers(const CclComputeLayoutParams& params)
    {
        const int64_t dataRows = params.ring.dispatch ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        // 接收矩阵使用有效数据行宽，BF16 不包含通信 TokenScale 记录的填充及权重段。
        const int64_t dispatchDataBytesPerToken = CalcStorageDataBytes(params.h, params.dispatchInputStorage);
        Allocate(routed.gmm1Input.data,
                 MemoryLayoutDetail::CeilAlign(dataRows * dispatchDataBytesPerToken, MemoryLayoutDetail::ALIGN_512));

        if (params.dispatchInputStorage.HasMxScale()) {
            int64_t dispatchScaleElementsPerToken =
                MemoryLayoutDetail::CeilDiv(static_cast<int64_t>(params.h),
                                            static_cast<int64_t>(MemoryLayoutDetail::MXFP_DIVISOR_SIZE)) *
                MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE;
            Allocate(routed.gmm1Input.scale,
                     MemoryLayoutDetail::CeilAlign(SIZE_INT_8 * dataRows * dispatchScaleElementsPerToken,
                                                   MemoryLayoutDetail::ALIGN_512));
        }
    }

    HOST_DEVICE constexpr void InitializeDispatchRevWeights(const CclComputeLayoutParams& params)
    {
        // dispatch 接收侧权重区，两种场景共用、MTE 与 URMA 两条初始化路径都必须调用：
        // tkw=1 的 topk 权重在激活量化阶段同步计算；tkw=0 且 combine 去重——dispatch 侧把 LAST 行
        // 整 token 的 topK 权重段落在这里，merge 一次连续读取后按成员 topkIndex 乘权。
        const bool dedupCombineWeights = !params.topkWeightsPrefetch && IsCombineDedupOn(params.dedupMode);
        if (params.topkWeightsPrefetch || dedupCombineWeights) {
            uint32_t weightAlignBytes =
                MemoryLayoutDetail::CeilAlign(static_cast<uint32_t>(params.topK * sizeof(float)),
                                              static_cast<uint32_t>(MemoryLayoutDetail::ALIGN_32));
            Allocate(routing.tokenWeights, MemoryLayoutDetail::CeilAlign(static_cast<int64_t>(params.maxOutputSize) *
                                                                             static_cast<int64_t>(weightAlignBytes),
                                                                         MemoryLayoutDetail::ALIGN_512));
        }
    }

    HOST_DEVICE constexpr void InitializeActivationBuffers(const CclComputeLayoutParams& params)
    {
        const int64_t dataRows = params.ring.activation ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        Allocate(routed.gmm2Input.data,
                 MemoryLayoutDetail::CeilAlign(
                     dataRows * CalcStorageDataBytes(params.hiddenDim / params.activationOutputDivisor,
                                                     params.activationOutputStorage),
                     MemoryLayoutDetail::ALIGN_512));

        if (params.activationOutputStorage.HasMxScale()) {
            Allocate(routed.gmm2Input.scale,
                     MemoryLayoutDetail::CeilAlign(
                         SIZE_INT_8 * dataRows *
                             MemoryLayoutDetail::CeilDiv(
                                 static_cast<int64_t>(params.hiddenDim / params.activationOutputDivisor),
                                 static_cast<int64_t>(MemoryLayoutDetail::MXFP_DIVISOR_SIZE)) *
                             MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE,
                         MemoryLayoutDetail::ALIGN_512));
        }
    }

    HOST_DEVICE constexpr void InitializeRoutingMetadata(const CclComputeLayoutParams& params,
                                                         int64_t expertRecvTokenCountBytes)
    {
        Allocate(routing.recvCounts,
                 MemoryLayoutDetail::CeilAlign(expertRecvTokenCountBytes, MemoryLayoutDetail::ALIGN_512));

        Allocate(routing.receivedTokenMetadata,
                 MemoryLayoutDetail::CeilAlign(params.maxOutputSize * MemoryLayoutDetail::ALIGN_32,
                                               MemoryLayoutDetail::ALIGN_512));
    }

    HOST_DEVICE constexpr void InitializeMteSyncFlags(const CclComputeLayoutParams& params)
    {
        // 标量通知区连续分配，保证统一清零范围。
        int64_t flagRegionBeginOffset = workspaceSize;
        int64_t moeExpertCount = static_cast<int64_t>(params.moeExpertPerRank);
        int64_t wavesPerExpert = MemoryLayoutDetail::CeilDiv(static_cast<int64_t>(params.maxOutputSize),
                                                             static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
        Allocate(sync.activationReady,
                 SIZE_INT_32 * moeExpertCount * wavesPerExpert * MemoryLayoutDetail::INT_CACHELINE);
        if (params.sharedExpertNum > 0) {
            Allocate(mte.sync.sharedActivationReady,
                     SIZE_INT_32 * CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(params.bs)) *
                         static_cast<int64_t>(params.sharedExpertNum));
        }
        InitializeRoutingFlags(params);
        // 去重 prepass 完成标记（flag 连续区内，随 ResetSyncStatus 每 launch 自动清零）：
        // 槽 0 = 每核 done 计数（AIV1 完成全部认领项后 +1；combine 消费 desc 前等其到齐）；
        // 槽 1+item = 该工作项 ready（dispatch 每批只等自己覆盖的 (srcRank,chunk) 项）。
        // 按 worldSize*MemoryLayoutDetail::DEDUP_PREPASS_MAX_CHUNKS 上界分配，每槽独占一个 cache line。
        if (params.dedupMode != 0) {
            Allocate(mte.sync.dedupReady,
                     SIZE_INT_32 * MemoryLayoutDetail::INT_CACHELINE *
                         (1 + static_cast<int64_t>(params.epWorldSize) * MemoryLayoutDetail::DEDUP_PREPASS_MAX_CHUNKS));
        }
        // 量化 Combine 在完整专家结束后等待每个 AIC 的完成标记；combine 去重下非量化路径的
        // 合并驱动行也要等组内成员所在专家的 GMM2 全部落地，同样需要该 per-expert 标记区。
        if (params.combineQuant || IsCombineDedupOn(params.dedupMode)) {
            Allocate(mte.sync.gmm2Ready,
                     SIZE_INT_32 * moeExpertCount * params.aicNum * MemoryLayoutDetail::INT_CACHELINE);
        }
        if (params.sharedExpertNum > 0) {
            int64_t tokenGroupCount = MemoryLayoutDetail::CeilDiv(
                static_cast<int64_t>(params.bs), static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
            Allocate(mte.sync.sharedGmm2Tiles, SIZE_INT_32 * tokenGroupCount *
                                                   static_cast<int64_t>(params.sharedExpertNum) *
                                                   MemoryLayoutDetail::INT_CACHELINE);

            Allocate(mte.sync.aiv1TopkReady,
                     static_cast<int64_t>(params.aicNum) * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32);
        }
        reset.flags = GmBuffer{sync.activationReady.offsetBytes,
                               ((workspaceSize - flagRegionBeginOffset) / SIZE_INT_32) * SIZE_INT_32};
    }

    HOST_DEVICE constexpr void InitializeUrmaSyncFlags(const CclComputeLayoutParams& params, Architecture arch)
    {
        // 标量通知区连续分配，保证统一清零范围。
        int64_t flagRegionBeginOffset = workspaceSize;
        int64_t moeExpertCount = static_cast<int64_t>(params.moeExpertPerRank);
        // Layered token wave 为专家内每个 256 行分组保留激活完成标志。
        const int64_t maxGroupsPerExpert =
            MemoryLayoutDetail::CeilDiv(params.maxOutputSize, MemoryLayoutDetail::L1_TILE_M_256);
        Allocate(sync.activationReady,
                 SIZE_INT_32 * moeExpertCount * maxGroupsPerExpert * MemoryLayoutDetail::INT_CACHELINE);
        InitializeRoutingFlags(params);
        Allocate(urma.sync.combineCounters, static_cast<int64_t>(params.combineSyncSlotCountPerExpert) *
                                                moeExpertCount * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32);
        if (params.sharedExpertNum > 0 && arch == Architecture::Arch35) {
            const int64_t tokenGroupCount = MemoryLayoutDetail::CeilDiv(
                static_cast<int64_t>(params.bs), static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
            Allocate(urma.sync.sharedGmm2Tiles,
                     SIZE_INT_32 * tokenGroupCount * params.sharedExpertNum * MemoryLayoutDetail::INT_CACHELINE);
        }
        reset.flags = GmBuffer{sync.activationReady.offsetBytes,
                               ((workspaceSize - flagRegionBeginOffset) / SIZE_INT_32) * SIZE_INT_32};
    }

    HOST_DEVICE constexpr void InitializeRoutingFlags(const CclComputeLayoutParams& params)
    {
        int64_t moeExpertCount = static_cast<int64_t>(params.moeExpertPerRank);
        int64_t maxWavesPerExpert = MemoryLayoutDetail::CeilDiv(
            static_cast<int64_t>(params.maxOutputSize), static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(MemoryLayoutDetail::INT_CACHELINE);
        Allocate(sync.dispatchReady, SIZE_INT_32 * moeExpertCount * waveFlagSlotsPerExpert);

        // 每(expert, aiCore)单独占一个cache_line
        Allocate(sync.countsReady, SIZE_INT_32 * MemoryLayoutDetail::INT_CACHELINE * moeExpertCount * params.aicNum);
    }

    HOST_DEVICE constexpr void InitializeGmmIntermediateBuffers(const CclComputeLayoutParams& params,
                                                                bool reserveCumsum, bool alignGmm2Output)
    {
        const int64_t dataRows = params.ring.gmm1 ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        if (reserveCumsum) {
            // cumsumInfo：逐核备份 cumsum 状态，每核 moeExpertPerRank × epWorldSize 个 int32。
            Allocate(routing.recvPrefixBackup,
                     MemoryLayoutDetail::CeilAlign(
                         static_cast<int64_t>(SIZE_INT_32 * params.moeExpertPerRank * params.epWorldSize),
                         MemoryLayoutDetail::ALIGN_32) *
                         params.aicNum);
        }
        if (params.reserveGmm1Output) {
            // 支持复用的 MTE 路径按循环容量分配，其余路径保留全量 GMM1 结果。
            Allocate(routed.gmm1Output, SIZE_BF_16 * dataRows * params.hiddenDim);
        }
        // Combine 去重使用全量逻辑行，其余可复用路径使用循环物理行；结果始终只有一份。
        const int64_t gmm2Rows = params.ring.gmm2 ? static_cast<int64_t>(params.ring.rows) : params.maxOutputSize;
        int64_t outputBytes = SIZE_BF_16 * gmm2Rows * params.h;
        if (alignGmm2Output) {
            workspaceSize =
                MemoryLayoutDetail::CeilAlign(workspaceSize, static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
            outputBytes =
                MemoryLayoutDetail::CeilAlign(outputBytes, static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
        }
        Allocate(routed.gmm2Output, outputBytes);
    }

    // 去重 rowDesc 表：每个接收行一条定长描述符。无需清零——prepass 在 dispatch 读取前
    // 覆盖写全部有效行，超出实际接收计数的行不会被读取。
    HOST_DEVICE constexpr void InitializeDedupRowDesc(const CclComputeLayoutParams& params)
    {
        if (params.topoType == TOPO_TYPE_MTE && params.dedupMode != 0) {
            int64_t rowDescStrideInt32 = CalcDedupRowDescStrideInt32(static_cast<int64_t>(params.topK));
            Allocate(mte.dedupRowDescriptors,
                     MemoryLayoutDetail::CeilAlign(
                         SIZE_INT_32 * rowDescStrideInt32 * static_cast<int64_t>(params.maxOutputSize),
                         MemoryLayoutDetail::ALIGN_512));
        }
    }

    HOST_DEVICE constexpr void InitializeUrmaDispatchBuffers(const CclComputeLayoutParams& params, int64_t serverNum)
    {
        // 远端 mask 槽的 count 位于全卡共用的容量尾部，本地发送暂存必须使用相同槽宽。
        const int64_t maskAlignSize = CalcDispatchMaskAlignSize(params.numMaxTokensPerRank, params.topK);
        int64_t maskSlotSize = maskAlignSize + static_cast<int64_t>(MemoryLayoutDetail::ALIGN_32); // mask + 32B count

        Allocate(urma.mask, MemoryLayoutDetail::CeilAlign(
                                static_cast<int64_t>(params.moeExpertPerRank) * params.epWorldSize * maskSlotSize,
                                static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512)));

        int64_t serverWorkspaceBytes = static_cast<int64_t>(MemoryLayoutDetail::ALIGN_32) +
                                       static_cast<int64_t>(params.bs) * MemoryLayoutDetail::ALIGN_32;
        Allocate(urma.relaySendQueue,
                 MemoryLayoutDetail::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                               static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512)));

        // 每个逻辑核只缓存当前 256-token 窗口；稀疏 token 按命中的窗口分块读取。
        int64_t flagSnapshotBytes = static_cast<int64_t>(MemoryLayoutDetail::L1_TILE_M_256) * sizeof(uint64_t);
        int64_t relayFlagSnapshotBytesPerBlock =
            MemoryLayoutDetail::CeilAlign(flagSnapshotBytes, static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512));
        Allocate(urma.remoteReadySnapshot, static_cast<int64_t>(params.aicNum) * relayFlagSnapshotBytesPerBlock);
    }

    HOST_DEVICE constexpr void InitializeSharedExpertGmmOutputs(const CclComputeLayoutParams& params)
    {
        // sharedExpertResult：共享专家 GMM2 输出 [sharedExpertNum × bs × h]。
        Allocate(shared.gmm2Output,
                 MemoryLayoutDetail::CeilAlign(SIZE_BF_16 * params.bs * params.sharedExpertNum * params.h,
                                               MemoryLayoutDetail::ALIGN_512));
        // BF16 的共享 GMM1 直接写 UB，其余路径保留 GM 中间结果区。
        if (params.sharedActivationStorage.elementBits != SIZE_BF_16 * MemoryLayoutDetail::BITS_PER_BYTE) {
            Allocate(shared.gmm1Output,
                     MemoryLayoutDetail::CeilAlign(SIZE_BF_16 * params.bs * params.sharedExpertNum * params.hiddenDim,
                                                   MemoryLayoutDetail::ALIGN_512));
        }
    }

    HOST_DEVICE constexpr void InitializeMteSharedExpertInput(const CclComputeLayoutParams& params)
    {
        // MTE 共享专家使用逐 token 交织数据 [Align256(data) | Align32(scale)]，仅独立量化时分配。
        int64_t recordBytes = CalcTokenScaleLayout(params.h, params.sharedInputStorage, 0, false).tokenScaleBytes;
        Allocate(shared.gmm1Input.data,
                 MemoryLayoutDetail::CeilAlign(static_cast<int64_t>(params.bs) * recordBytes,
                                               static_cast<int64_t>(MemoryLayoutDetail::ALIGN_512)));
    }

    HOST_DEVICE constexpr void InitializeUrmaSharedExpertInput(const CclComputeLayoutParams& params)
    {
        // 保留 Arch92 的连续 data/scale 输入布局。
        Allocate(shared.gmm1Input.data,
                 MemoryLayoutDetail::CeilAlign(SIZE_INT_8 * params.bs * params.h, MemoryLayoutDetail::ALIGN_512));
        Allocate(shared.gmm1Input.scale,
                 MemoryLayoutDetail::CeilAlign(
                     SIZE_INT_8 * params.bs *
                         MemoryLayoutDetail::CeilDiv(static_cast<uint32_t>(params.h),
                                                     static_cast<uint32_t>(MemoryLayoutDetail::MXFP_SCALE_GROUP_NUM)) *
                         MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE,
                     MemoryLayoutDetail::ALIGN_512));
    }

    HOST_DEVICE constexpr void InitializeSharedExpertActivationBuffers(const CclComputeLayoutParams& params)
    {
        // 共享专家激活输出宽度随激活类型变化：门控激活减半，GELU 保留完整宽度。
        Allocate(shared.gmm2Input.data, MemoryLayoutDetail::CeilAlign(
                                            params.bs * params.sharedExpertNum *
                                                CalcStorageDataBytes(params.hiddenDim / params.activationOutputDivisor,
                                                                     params.sharedActivationStorage),
                                            MemoryLayoutDetail::ALIGN_512));
        // 每个 MXFP scale group 保存两个分量；BF16 激活输出不分配 scale。
        if (params.sharedActivationStorage.HasMxScale()) {
            Allocate(shared.gmm2Input.scale,
                     MemoryLayoutDetail::CeilAlign(
                         SIZE_INT_8 * params.bs * params.sharedExpertNum *
                             MemoryLayoutDetail::CeilDiv(
                                 static_cast<int64_t>(params.hiddenDim / params.activationOutputDivisor),
                                 static_cast<int64_t>(MemoryLayoutDetail::MXFP_DIVISOR_SIZE)) *
                             MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE,
                         MemoryLayoutDetail::ALIGN_512));
        }
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GM_LAYOUT_H

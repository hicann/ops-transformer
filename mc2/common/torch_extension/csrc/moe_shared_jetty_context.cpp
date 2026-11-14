/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "hccl_common.h"
#include "moe_shared_jetty_adapter.h"
#include "moe_shared_jetty_context.h"
#include "torch_npu/csrc/aten/common/from_blob.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Mc2Api {
namespace {
constexpr uint32_t SHARED_JETTY_NUM = Mc2Kernel::MOE_SHARED_JETTY_NUM;
constexpr size_t SHARED_JETTY_MAX_TAG_SIZE = 255;
constexpr CommEngine COMM_ENGINE = CommEngine::COMM_ENGINE_AIV;
constexpr uint64_t BUFFER_ALIGNMENT = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t HUGE1G_SIZE = 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t GLOBAL_LANE_ROOT_RANK = 0;

struct SharedBufferEntry {
    HcclComm comm = nullptr;
    void* deviceBuffer = nullptr;
    aclrtDrvMemHandle physicalMemHandle = nullptr;
    HcclMemHandle memHandle = nullptr;
    uint64_t size = 0;
};

std::mutex gSharedBufferMutex;
std::unordered_map<std::string, SharedBufferEntry> gSharedBuffers;
std::mutex gSharedContextMutexMapMutex;
std::unordered_map<std::string, std::shared_ptr<std::mutex>> gSharedContextMutexes;

std::shared_ptr<std::mutex> GetSharedContextMutex(const std::string& contextTag)
{
    std::lock_guard<std::mutex> lock(gSharedContextMutexMapMutex);
    auto [entry, inserted] = gSharedContextMutexes.try_emplace(contextTag);
    if (inserted) {
        entry->second = std::make_shared<std::mutex>();
    }
    return entry->second;
}

} // namespace

class MoeSharedJettyContextBuilder {
private:
    struct SharedLinkGroup {
        EndpointAttrDieId dieId = 0;
        EndpointAttrBwCoeff bwCoeff = 1;
        EndpointAttrBwCoeff localBwCoeff = 1;
        std::vector<HcclChannelDesc> channelDescs;
    };

    struct GlobalLane {
        EndpointAttrDieId rootDieId = 0;
        EndpointAttrBwCoeff bwCoeff = 1;
        std::vector<EndpointDesc> rankEndpoints;
        EndpointAttrDieId localDieId = 0;
        EndpointAttrBwCoeff localBwCoeff = 0;
    };

    struct RootLaneMetadata {
        EndpointDesc endpoint = {};
        EndpointAttrBwCoeff bwCoeff = 0;
        EndpointAttrDieId dieId = 0;
        uint32_t valid = 0;
    };

    struct SelectedEndpointMetadata {
        EndpointDesc endpoint = {};
        EndpointAttrBwCoeff bwCoeff = 0;
        EndpointAttrDieId dieId = 0;
        uint32_t valid = 0;
    };

    struct LayerCandidateMetadata {
        uint32_t layer = 0;
        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        uint32_t ready = 0;
        uint64_t rootEndpointCount = 0;
    };

    struct UnsupportedProtocolMetadata {
        uint32_t detected = 0;
        uint32_t layer = 0;
        uint32_t peer = 0;
    };

    HcclComm comm_ = nullptr;
    uint32_t rankSize_ = 0;
    Mc2Kernel::MoeSharedJettyContext context_;
    uint64_t requiredBufferSize_ = 0;
    std::vector<SharedLinkGroup> sharedLinkGroups_;
    void* deviceBuffer_ = nullptr;
    aclrtDrvMemHandle physicalMemHandle_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    bool bufferRetained_ = false;
    bool bufferEntryReserved_ = false;
    std::string reservedContextTag_;

    static void CheckHccl(HcclResult ret, const char* operation)
    {
        TORCH_CHECK(ret == HCCL_SUCCESS, operation, " failed, ret=", ret);
    }

    static void LogCleanupFailure(aclError ret, const char* operation)
    {
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGE("%s failed during MoE buffer cleanup, ret=%d", operation, ret);
        }
    }

    void ReleaseBuffer()
    {
        if (deviceBuffer_ == nullptr) {
            return;
        }
        if (physicalMemHandle_ != nullptr) {
            LogCleanupFailure(aclrtUnmapMem(deviceBuffer_), "aclrtUnmapMem");
            LogCleanupFailure(aclrtFreePhysical(physicalMemHandle_), "aclrtFreePhysical");
            physicalMemHandle_ = nullptr;
        }
        LogCleanupFailure(aclrtReleaseMemAddress(deviceBuffer_), "aclrtReleaseMemAddress");
        deviceBuffer_ = nullptr;
        memHandle_ = nullptr;
    }

    bool AllocateHugePageBuffer(uint64_t size, aclrtPhysicalMemProp& property, uint32_t deviceId)
    {
        void* address = nullptr;
        auto ret = aclrtReserveMemAddress(&address, size, 0, nullptr, 0);
        if (ret != ACL_SUCCESS) {
            return false;
        }
        aclrtDrvMemHandle handle = nullptr;
        ret = aclrtMallocPhysical(&handle, size, &property, 0);
        if (ret != ACL_SUCCESS) {
            LogCleanupFailure(aclrtReleaseMemAddress(address), "aclrtReleaseMemAddress");
            return false;
        }
        ret = aclrtMapMem(address, size, 0, handle, 0);
        if (ret != ACL_SUCCESS) {
            LogCleanupFailure(aclrtFreePhysical(handle), "aclrtFreePhysical");
            LogCleanupFailure(aclrtReleaseMemAddress(address), "aclrtReleaseMemAddress");
            return false;
        }
        aclrtMemAccessDesc access = {};
        access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        access.location.id = deviceId;
        access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        ret = aclrtMemSetAccess(address, size, &access, 1);
        if (ret != ACL_SUCCESS) {
            LogCleanupFailure(aclrtUnmapMem(address), "aclrtUnmapMem");
            LogCleanupFailure(aclrtFreePhysical(handle), "aclrtFreePhysical");
            LogCleanupFailure(aclrtReleaseMemAddress(address), "aclrtReleaseMemAddress");
            return false;
        }
        deviceBuffer_ = address;
        physicalMemHandle_ = handle;
        return true;
    }

    void AllocateDeviceBuffer()
    {
        int32_t deviceId = 0;
        auto ret = aclrtGetDevice(&deviceId);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtGetDevice failed, ret=", ret);
        TORCH_CHECK(requiredBufferSize_ <= std::numeric_limits<uint64_t>::max() - (HUGE1G_SIZE - 1),
                    "MoE communication buffer size overflows 1GB alignment: ", requiredBufferSize_);
        const uint64_t allocBytes = ((requiredBufferSize_ + HUGE1G_SIZE - 1) / HUGE1G_SIZE) * HUGE1G_SIZE;
        aclrtPhysicalMemProp property = {};
        property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        property.location.id = static_cast<uint32_t>(deviceId);
        property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        property.memAttr = ACL_MEM_HUGE1G;
        TORCH_CHECK(AllocateHugePageBuffer(allocBytes, property, static_cast<uint32_t>(deviceId)),
                    "Allocate MoE communication buffer with 1 GiB huge pages failed, requested=", requiredBufferSize_,
                    ", aligned=", allocBytes);
        ret = aclrtMemset(deviceBuffer_, requiredBufferSize_, 0, requiredBufferSize_);
        TORCH_CHECK(ret == ACL_SUCCESS, "Clear MoE communication buffer failed, ret=", ret);
    }

    void ReserveBufferEntry(const std::string& contextTag)
    {
        reservedContextTag_ = contextTag;
        std::lock_guard<std::mutex> lock(gSharedBufferMutex);
        bool inserted = gSharedBuffers.try_emplace(contextTag).second;
        TORCH_CHECK(inserted, "MoE shared Jetty buffer already exists but its engine context is missing for '",
                    contextTag,
                    "'; previous initialization is incomplete and cannot be retried safely, restart the "
                    "process");
        bufferEntryReserved_ = true;
    }

    void RetainBuffer(const std::string& contextTag)
    {
        std::lock_guard<std::mutex> lock(gSharedBufferMutex);
        auto entry = gSharedBuffers.find(contextTag);
        // HcclCommMemReg has no matching unregister API. Once registration succeeds, never release the buffer even
        // if an internal invariant fails below, because HCCL may still reference its address.
        bufferRetained_ = true;
        TORCH_CHECK(bufferEntryReserved_ && reservedContextTag_ == contextTag && entry != gSharedBuffers.end(),
                    "MoE shared Jetty buffer reservation is missing for context '", contextTag, "'");
        entry->second = {comm_, deviceBuffer_, physicalMemHandle_, memHandle_, requiredBufferSize_};
    }

    void ReleaseBufferEntryReservation()
    {
        if (!bufferEntryReserved_ || bufferRetained_) {
            return;
        }
        std::lock_guard<std::mutex> lock(gSharedBufferMutex);
        auto entry = gSharedBuffers.find(reservedContextTag_);
        if (entry != gSharedBuffers.end() && entry->second.deviceBuffer == nullptr) {
            gSharedBuffers.erase(entry);
        }
        bufferEntryReserved_ = false;
    }

    void ReuseBuffer(const std::string& contextTag)
    {
        std::lock_guard<std::mutex> lock(gSharedBufferMutex);
        auto entry = gSharedBuffers.find(contextTag);
        TORCH_CHECK(entry != gSharedBuffers.end(), "MoE shared Jetty buffer is missing for context '", contextTag,
                    "'; use a new communication group");
        TORCH_CHECK(entry->second.comm == comm_, "MoE shared Jetty context does not match the current communicator");
        TORCH_CHECK(requiredBufferSize_ <= entry->second.size, "MoE communication buffer exceeds the existing buffer,",
                    " requested=", requiredBufferSize_, ", allocated=", entry->second.size,
                    "; use a new communication group initialized with the larger shape");
        deviceBuffer_ = entry->second.deviceBuffer;
        physicalMemHandle_ = entry->second.physicalMemHandle;
        memHandle_ = entry->second.memHandle;
        bufferRetained_ = true;
    }

    void GetRankInfo()
    {
        CheckHccl(HcclGetRankIdFunc(comm_, &context_.epRankId), "HcclGetRankId");
        CheckHccl(HcclGetRankSizeFunc(comm_, &rankSize_), "HcclGetRankSize");
        TORCH_CHECK(rankSize_ >= Mc2Kernel::MOE_SHARED_JETTY_MIN_RANK_SIZE &&
                        rankSize_ <= Mc2Kernel::MOE_SHARED_JETTY_MAX_RANK_SIZE,
                    "MoE shared Jetty rank size must be in [", Mc2Kernel::MOE_SHARED_JETTY_MIN_RANK_SIZE, ", ",
                    Mc2Kernel::MOE_SHARED_JETTY_MAX_RANK_SIZE, "], got ", rankSize_);
        TORCH_CHECK(context_.epRankId < rankSize_, "Invalid shared Jetty rank ID: ", context_.epRankId);
        context_.epRankSize = rankSize_;
        context_.sharedJettyNum = SHARED_JETTY_NUM;
    }

    static bool IsSameEndpoint(const EndpointDesc& lhs, const EndpointDesc& rhs)
    {
        return lhs.protocol == rhs.protocol && lhs.commAddr.type == rhs.commAddr.type &&
               std::memcmp(lhs.commAddr.raws, rhs.commAddr.raws, sizeof(lhs.commAddr.raws)) == 0 &&
               lhs.loc.locType == rhs.loc.locType && std::memcmp(lhs.loc.raws, rhs.loc.raws, sizeof(lhs.loc.raws)) == 0;
    }

    static bool IsEndpointLess(const EndpointDesc& lhs, const EndpointDesc& rhs)
    {
        if (lhs.protocol != rhs.protocol) {
            return lhs.protocol < rhs.protocol;
        }
        if (lhs.commAddr.type != rhs.commAddr.type) {
            return lhs.commAddr.type < rhs.commAddr.type;
        }
        int compare = std::memcmp(lhs.commAddr.raws, rhs.commAddr.raws, sizeof(lhs.commAddr.raws));
        if (compare != 0) {
            return compare < 0;
        }
        if (lhs.loc.locType != rhs.loc.locType) {
            return lhs.loc.locType < rhs.loc.locType;
        }
        return std::memcmp(lhs.loc.raws, rhs.loc.raws, sizeof(lhs.loc.raws)) < 0;
    }

    std::vector<CommLink> GetProtocolLinks(uint32_t layer, uint32_t srcRank, uint32_t dstRank, CommProtocol protocol)
    {
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        CheckHccl(HcclRankGraphGetLinksFunc(comm_, layer, srcRank, dstRank, &links, &linkNum), "HcclRankGraphGetLinks");
        TORCH_CHECK(linkNum == 0 || links != nullptr, "Null link list for layer ", layer, ", source rank ", srcRank,
                    ", destination rank ", dstRank);
        std::vector<CommLink> candidates;
        for (uint32_t i = 0; i < linkNum; ++i) {
            if (links[i].linkAttr.linkProtocol == protocol) {
                candidates.push_back(links[i]);
            }
        }
        return candidates;
    }

    template <typename T>
    bool TryGetLocalEndpointInfo(const EndpointDesc& endpoint, EndpointAttr attr, T& value)
    {
        const HcclResult ret =
            HcclRankGraphGetEndpointInfoFunc(comm_, context_.epRankId, &endpoint, attr, sizeof(value), &value);
        if (ret != HCCL_SUCCESS) {
            ASCEND_LOGW("Query local Endpoint attribute %d failed on rank %u, ret=%d", static_cast<int>(attr),
                        context_.epRankId, ret);
            return false;
        }
        return true;
    }

    template <typename T>
    std::vector<T> AllGatherMetadata(const std::vector<T>& localMetadata)
    {
        TORCH_CHECK(!localMetadata.empty(), "Cannot all-gather empty shared Jetty metadata");
        TORCH_CHECK(localMetadata.size() <= std::numeric_limits<size_t>::max() / sizeof(T),
                    "Shared Jetty metadata size overflows");
        const size_t sendBytes = localMetadata.size() * sizeof(T);
        TORCH_CHECK(rankSize_ <= std::numeric_limits<size_t>::max() / sendBytes,
                    "Shared Jetty gathered metadata size overflows");
        const size_t recvBytes = sendBytes * rankSize_;
        void* deviceSend = nullptr;
        void* deviceRecv = nullptr;
        auto cleanup = [&]() {
            if (deviceRecv != nullptr) {
                LogCleanupFailure(aclrtFree(deviceRecv), "aclrtFree(metadata recv)");
                deviceRecv = nullptr;
            }
            if (deviceSend != nullptr) {
                LogCleanupFailure(aclrtFree(deviceSend), "aclrtFree(metadata send)");
                deviceSend = nullptr;
            }
        };
        try {
            auto aclRet = aclrtMalloc(&deviceSend, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "Allocate shared Jetty metadata send buffer failed, ret=", aclRet);
            aclRet = aclrtMalloc(&deviceRecv, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "Allocate shared Jetty metadata receive buffer failed, ret=", aclRet);
            aclRet = aclrtMemcpy(deviceSend, sendBytes, localMetadata.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "Copy shared Jetty metadata to device failed, ret=", aclRet);
            aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
            CheckHccl(HcclAllGatherFunc(deviceSend, deviceRecv, sendBytes, HCCL_DATA_TYPE_UINT8, comm_, stream),
                      "HcclAllGather(shared Jetty metadata)");
            aclRet = aclrtSynchronizeStream(stream);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "Synchronize shared Jetty metadata all-gather failed, ret=", aclRet);
            std::vector<T> gathered(localMetadata.size() * rankSize_);
            aclRet = aclrtMemcpy(gathered.data(), recvBytes, deviceRecv, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "Copy shared Jetty metadata to host failed, ret=", aclRet);
            cleanup();
            return gathered;
        } catch (...) {
            cleanup();
            throw;
        }
    }

    SelectedEndpointMetadata SelectLocalEndpoint(const EndpointDesc& rootEndpoint, EndpointAttrBwCoeff rootBwCoeff,
                                                 const std::vector<std::vector<CommLink>>& localPeerLinks)
    {
        SelectedEndpointMetadata result;
        if (context_.epRankId == GLOBAL_LANE_ROOT_RANK) {
            result.endpoint = rootEndpoint;
            result.bwCoeff = rootBwCoeff;
            result.valid = rootBwCoeff != 0;
            return result;
        }
        const CommLink* sameBwLink = nullptr;
        const CommLink* fallbackLink = nullptr;
        EndpointAttrBwCoeff fallbackBwCoeff = 0;
        // Query only links originating at this rank. A remote rank's outgoing links may not be present locally.
        for (const auto& link : localPeerLinks[GLOBAL_LANE_ROOT_RANK]) {
            if (!IsSameEndpoint(rootEndpoint, link.dstEndpointDesc)) {
                continue;
            }
            EndpointAttrBwCoeff localBwCoeff = 0;
            const bool gotBw = TryGetLocalEndpointInfo(link.srcEndpointDesc, ENDPOINT_ATTR_BW_COEFF, localBwCoeff);
            if (!gotBw || localBwCoeff == 0) {
                continue;
            }
            if (localBwCoeff == rootBwCoeff) {
                if (sameBwLink == nullptr || IsEndpointLess(link.srcEndpointDesc, sameBwLink->srcEndpointDesc)) {
                    sameBwLink = &link;
                }
            } else if (fallbackLink == nullptr || IsEndpointLess(link.srcEndpointDesc, fallbackLink->srcEndpointDesc)) {
                fallbackLink = &link;
                fallbackBwCoeff = localBwCoeff;
            }
        }
        const CommLink* selected = sameBwLink != nullptr ? sameBwLink : fallbackLink;
        if (selected == nullptr) {
            return result;
        }
        result.endpoint = selected->srcEndpointDesc;
        result.bwCoeff = sameBwLink != nullptr ? rootBwCoeff : fallbackBwCoeff;
        result.valid = 1;
        if (sameBwLink == nullptr) {
            ASCEND_LOGW("No equal-bandwidth Endpoint for global lane root coefficient %u at rank %u; falling back "
                        "to coefficient %u",
                        rootBwCoeff, context_.epRankId, result.bwCoeff);
        }
        return result;
    }

    SharedLinkGroup MakeSharedLinkGroup(const GlobalLane& lane,
                                        const std::vector<std::vector<CommLink>>& localPeerLinks, CommProtocol protocol)
    {
        SharedLinkGroup group;
        const auto& localEndpoint = lane.rankEndpoints[context_.epRankId];
        group.bwCoeff = lane.bwCoeff;
        // Attributes and reciprocal links were already validated collectively for this lane.
        group.dieId = lane.localDieId;
        group.localBwCoeff = lane.localBwCoeff;
        TORCH_CHECK(group.localBwCoeff > 0, "Zero-bandwidth Endpoint selected for shared Jetty lane");
        group.channelDescs.resize(rankSize_ - 1);
        CheckHccl(HcclChannelDescInit(group.channelDescs.data(), static_cast<uint32_t>(group.channelDescs.size())),
                  "HcclChannelDescInit");
        for (uint32_t peer = 0, index = 0; peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            const auto& links = localPeerLinks[peer];
            auto link = std::find_if(links.begin(), links.end(), [&](const CommLink& value) {
                return IsSameEndpoint(localEndpoint, value.srcEndpointDesc) &&
                       IsSameEndpoint(lane.rankEndpoints[peer], value.dstEndpointDesc);
            });
            TORCH_CHECK(link != links.end(), "Global shared Jetty lane is not reciprocal between rank ",
                        context_.epRankId, " and rank ", peer, "; root die=", lane.rootDieId);
            auto& desc = group.channelDescs[index++];
            desc.channelProtocol = protocol;
            desc.remoteRank = peer;
            desc.localEndpoint = link->srcEndpointDesc;
            desc.remoteEndpoint = link->dstEndpointDesc;
        }
        return group;
    }

    bool LayerCoversAllRanks(uint32_t layer)
    {
        uint32_t* ranks = nullptr;
        uint32_t rankNum = 0;
        CheckHccl(HcclRankGraphGetRanksByLayerFunc(comm_, layer, &ranks, &rankNum), "HcclRankGraphGetRanksByLayer");
        TORCH_CHECK(rankNum == 0 || ranks != nullptr, "Null rank list for layer ", layer);
        std::vector<bool> covered(rankSize_, false);
        for (uint32_t i = 0; i < rankNum; ++i) {
            TORCH_CHECK(ranks[i] < rankSize_, "Invalid rank in layer ", layer, ": ", ranks[i]);
            covered[ranks[i]] = true;
        }
        return std::all_of(covered.begin(), covered.end(), [](bool value) { return value; });
    }

    bool SelectSharedLinks(uint32_t layer, CommProtocol protocol, bool layerPresent)
    {
        bool localReady = layerPresent && LayerCoversAllRanks(layer);
        std::vector<std::vector<CommLink>> localPeerLinks(rankSize_);
        for (uint32_t peer = 0; localReady && peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            localPeerLinks[peer] = GetProtocolLinks(layer, context_.epRankId, peer, protocol);
            localReady = !localPeerLinks[peer].empty();
        }

        // Only rank 0 enumerates root candidates. Other ranks cannot rely on querying rank 0 -> rank 1 links
        // across servers; they receive the root's descriptors and match their own local -> root links instead.
        std::vector<EndpointDesc> rootEndpoints;
        if (localReady && context_.epRankId == GLOBAL_LANE_ROOT_RANK) {
            for (const auto& candidate : localPeerLinks[1]) {
                const bool endpointSeen =
                    std::any_of(rootEndpoints.begin(), rootEndpoints.end(),
                                [&](const auto& value) { return IsSameEndpoint(value, candidate.srcEndpointDesc); });
                if (!endpointSeen) {
                    rootEndpoints.push_back(candidate.srcEndpointDesc);
                }
            }
            std::sort(rootEndpoints.begin(), rootEndpoints.end(), IsEndpointLess);
        }
        // Missing local layers or links must not skip a collective. All ranks either reject this layer together
        // or proceed with the same root endpoint count and metadata sizes.
        const LayerCandidateMetadata candidate{layer, protocol, static_cast<uint32_t>(localReady),
                                               rootEndpoints.size()};
        const auto gatheredCandidates = AllGatherMetadata(std::vector<LayerCandidateMetadata>{candidate});
        bool globallyReady = true;
        for (const auto& entry : gatheredCandidates) {
            TORCH_CHECK(entry.layer == layer && entry.protocol == protocol,
                        "Inconsistent shared Jetty layer/protocol selection across ranks");
            globallyReady = globallyReady && entry.ready != 0;
        }
        if (!globallyReady) {
            return false;
        }
        const uint64_t rootCount = gatheredCandidates[GLOBAL_LANE_ROOT_RANK].rootEndpointCount;
        if (rootCount == 0) {
            return false;
        }
        TORCH_CHECK(rootCount <= std::numeric_limits<size_t>::max(), "Shared Jetty root candidate count overflows");

        // GetEndpointInfo only owns an Endpoint-to-interface map for the calling process's rank. Rank 0 publishes
        // its local lane attributes first; every rank then selects and publishes only its own local Endpoint.
        const size_t laneCount = static_cast<size_t>(rootCount);
        std::vector<RootLaneMetadata> localRootMetadata(laneCount);
        if (context_.epRankId == GLOBAL_LANE_ROOT_RANK) {
            for (size_t laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
                auto& metadata = localRootMetadata[laneIndex];
                metadata.endpoint = rootEndpoints[laneIndex];
                const bool gotBw =
                    TryGetLocalEndpointInfo(rootEndpoints[laneIndex], ENDPOINT_ATTR_BW_COEFF, metadata.bwCoeff);
                const bool gotDie =
                    TryGetLocalEndpointInfo(rootEndpoints[laneIndex], ENDPOINT_ATTR_DIE_ID, metadata.dieId);
                metadata.valid = gotBw && gotDie && metadata.bwCoeff != 0;
            }
        }
        const auto gatheredRootMetadata = AllGatherMetadata(localRootMetadata);
        const RootLaneMetadata* rootMetadata = gatheredRootMetadata.data() + GLOBAL_LANE_ROOT_RANK * laneCount;

        std::vector<SelectedEndpointMetadata> localSelections(laneCount);
        for (size_t laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            if (rootMetadata[laneIndex].valid != 0) {
                localSelections[laneIndex] = SelectLocalEndpoint(rootMetadata[laneIndex].endpoint,
                                                                 rootMetadata[laneIndex].bwCoeff, localPeerLinks);
                auto& selection = localSelections[laneIndex];
                if (selection.valid != 0) {
                    selection.valid =
                        TryGetLocalEndpointInfo(selection.endpoint, ENDPOINT_ATTR_DIE_ID, selection.dieId);
                }
            }
        }
        const auto gatheredSelections = AllGatherMetadata(localSelections);

        // A selected endpoint must reach every selected peer endpoint on this exact layer. Publish failures
        // before constructing groups, so a missing edge makes every rank reject the same lane.
        std::vector<uint32_t> localReciprocal(laneCount, 1U);
        for (size_t laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            const auto& local = localSelections[laneIndex];
            for (uint32_t peer = 0; peer < rankSize_; ++peer) {
                if (peer == context_.epRankId) {
                    continue;
                }
                const auto& remote = gatheredSelections[static_cast<size_t>(peer) * laneCount + laneIndex];
                const auto& links = localPeerLinks[peer];
                if (local.valid == 0 || remote.valid == 0 ||
                    std::none_of(links.begin(), links.end(), [&](const CommLink& link) {
                        return IsSameEndpoint(local.endpoint, link.srcEndpointDesc) &&
                               IsSameEndpoint(remote.endpoint, link.dstEndpointDesc);
                    })) {
                    localReciprocal[laneIndex] = 0U;
                    break;
                }
            }
        }
        const auto gatheredReciprocal = AllGatherMetadata(localReciprocal);

        std::vector<GlobalLane> lanes;
        for (size_t laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            if (rootMetadata[laneIndex].valid == 0) {
                continue;
            }
            GlobalLane lane;
            lane.rootDieId = rootMetadata[laneIndex].dieId;
            lane.bwCoeff = rootMetadata[laneIndex].bwCoeff;
            lane.rankEndpoints.resize(rankSize_);
            bool laneValid = true;
            for (uint32_t rank = 0; rank < rankSize_; ++rank) {
                const auto& selection = gatheredSelections[static_cast<size_t>(rank) * laneCount + laneIndex];
                if (selection.valid == 0 || selection.bwCoeff == 0 ||
                    gatheredReciprocal[static_cast<size_t>(rank) * laneCount + laneIndex] == 0U) {
                    laneValid = false;
                    break;
                }
                lane.rankEndpoints[rank] = selection.endpoint;
                lane.bwCoeff = std::min(lane.bwCoeff, selection.bwCoeff);
            }
            if (!laneValid) {
                continue;
            }
            const auto& local = localSelections[laneIndex];
            lane.localDieId = local.dieId;
            lane.localBwCoeff = local.bwCoeff;
            auto sameRootDie = std::find_if(lanes.begin(), lanes.end(),
                                            [&](const GlobalLane& value) { return value.rootDieId == lane.rootDieId; });
            if (sameRootDie == lanes.end()) {
                lanes.push_back(std::move(lane));
                continue;
            }
            const auto& currentEndpoint = sameRootDie->rankEndpoints[GLOBAL_LANE_ROOT_RANK];
            const auto& candidateEndpoint = lane.rankEndpoints[GLOBAL_LANE_ROOT_RANK];
            if (lane.bwCoeff > sameRootDie->bwCoeff ||
                (lane.bwCoeff == sameRootDie->bwCoeff && IsEndpointLess(candidateEndpoint, currentEndpoint))) {
                *sameRootDie = std::move(lane);
            }
        }
        if (lanes.empty()) {
            return false;
        }
        std::sort(lanes.begin(), lanes.end(), [](const GlobalLane& lhs, const GlobalLane& rhs) {
            if (lhs.rootDieId != rhs.rootDieId) {
                return lhs.rootDieId < rhs.rootDieId;
            }
            return IsEndpointLess(lhs.rankEndpoints[GLOBAL_LANE_ROOT_RANK], rhs.rankEndpoints[GLOBAL_LANE_ROOT_RANK]);
        });
        TORCH_CHECK(lanes.size() <= context_.sharedJettyNum, "Usable shared Jetty lane count ", lanes.size(),
                    " exceeds Jetty count ", context_.sharedJettyNum);
        sharedLinkGroups_.clear();
        for (const auto& lane : lanes) {
            sharedLinkGroups_.push_back(MakeSharedLinkGroup(lane, localPeerLinks, protocol));
        }
        for (uint32_t laneIndex = 0; laneIndex < sharedLinkGroups_.size(); ++laneIndex) {
            const auto& group = sharedLinkGroups_[laneIndex];
            ASCEND_LOGI("Selected global shared Jetty lane %u on layer %u: local die %u, local bandwidth "
                        "coefficient %u, global bandwidth coefficient %u",
                        laneIndex, layer, group.dieId, group.localBwCoeff, group.bwCoeff);
        }
        return true;
    }

    void CheckSupportedProtocols(const std::vector<uint32_t>& layers)
    {
        UnsupportedProtocolMetadata unsupported{};
        for (uint32_t layer : layers) {
            uint32_t* ranks = nullptr;
            uint32_t rankNum = 0;
            CheckHccl(HcclRankGraphGetRanksByLayerFunc(comm_, layer, &ranks, &rankNum), "HcclRankGraphGetRanksByLayer");
            TORCH_CHECK(rankNum == 0 || ranks != nullptr, "Null rank list for layer ", layer);
            if (rankNum == 0) {
                continue;
            }
            // RankGraph arrays are borrowed; retain the rank IDs before querying links.
            const std::vector<uint32_t> peers(ranks, ranks + rankNum);
            for (uint32_t peer : peers) {
                TORCH_CHECK(peer < rankSize_, "Invalid rank in layer ", layer, ": ", peer);
                if (peer == context_.epRankId) {
                    continue;
                }
                if (unsupported.detected == 0 &&
                    !GetProtocolLinks(layer, context_.epRankId, peer, CommProtocol::COMM_PROTOCOL_UB_RTP).empty()) {
                    unsupported = {1U, layer, peer};
                }
            }
        }
        const auto gathered = AllGatherMetadata(std::vector<UnsupportedProtocolMetadata>{unsupported});
        for (uint32_t rank = 0; rank < rankSize_; ++rank) {
            const auto& entry = gathered[rank];
            TORCH_CHECK(entry.detected == 0, "comm_alg='clos_only' does not support UBG (UB_RTP) links; rank=", rank,
                        ", peer=", entry.peer, ", layer=", entry.layer);
        }
    }

    void SelectSharedLayer()
    {
        uint32_t* layers = nullptr;
        uint32_t layerNum = 0;
        CheckHccl(HcclRankGraphGetLayersFunc(comm_, &layers, &layerNum), "HcclRankGraphGetLayers");
        TORCH_CHECK(layerNum == 0 || layers != nullptr, "Null HCCL network layer list for shared Jetty");
        // Some deployed HCOMM versions reuse or invalidate the borrowed layer array on the next RankGraph query.
        // SelectSharedLinks performs several such queries, so retain the values before entering the loop.
        std::vector<uint32_t> localLayers;
        if (layerNum != 0) {
            localLayers.assign(layers, layers + layerNum);
        }
        std::sort(localLayers.begin(), localLayers.end());
        localLayers.erase(std::unique(localLayers.begin(), localLayers.end()), localLayers.end());
        CheckSupportedProtocols(localLayers);

        // Use rank 0's ordered layer IDs as the common schedule. A layer absent on rank 0 cannot serve all ranks.
        const auto counts = AllGatherMetadata(std::vector<uint64_t>{localLayers.size()});
        const uint64_t rootCount = counts[GLOBAL_LANE_ROOT_RANK];
        TORCH_CHECK(rootCount > 0 && rootCount <= std::numeric_limits<size_t>::max(),
                    "No usable network layer list on shared Jetty root rank");
        std::vector<uint32_t> rootLayers(static_cast<size_t>(rootCount), 0U);
        if (context_.epRankId == GLOBAL_LANE_ROOT_RANK) {
            rootLayers = localLayers;
        }
        const auto gatheredLayers = AllGatherMetadata(rootLayers);
        const auto rootBegin = gatheredLayers.begin() + GLOBAL_LANE_ROOT_RANK * rootLayers.size();
        rootLayers.assign(rootBegin, rootBegin + rootLayers.size());
        for (const uint32_t layer : rootLayers) {
            const bool present = std::binary_search(localLayers.begin(), localLayers.end(), layer);
            if (SelectSharedLinks(layer, CommProtocol::COMM_PROTOCOL_UBC_CTP, present)) {
                return;
            }
        }
        TORCH_CHECK(false, "No UBC layer has a shared Jetty endpoint that covers all peers");
    }

    void RegisterDeviceBuffer(const std::string& contextTag, const std::string& bufferTag)
    {
        AllocateDeviceBuffer();
        CommMem memory = {};
        memory.type = COMM_MEM_TYPE_DEVICE;
        memory.addr = deviceBuffer_;
        memory.size = requiredBufferSize_;
        CheckHccl(HcclCommMemRegFunc(comm_, bufferTag.c_str(), &memory, &memHandle_), "HcclCommMemReg");
        RetainBuffer(contextTag);
        TORCH_CHECK(memHandle_ != nullptr, "HcclCommMemReg returned a null memory handle");
        for (auto& group : sharedLinkGroups_) {
            for (auto& desc : group.channelDescs) {
                desc.memHandles = &memHandle_;
                desc.memHandleNum = 1;
            }
        }
    }

    std::vector<ChannelHandle> AcquireSharedChannels(const std::string& queueTag,
                                                     const std::vector<HcclChannelDesc>& channelDescs)
    {
        std::vector<ChannelHandle> channels(channelDescs.size());
        HcclChannelConfig config = nullptr;
        CheckHccl(HcclChannelConfigCreateFunc(&config), "HcclChannelConfigCreate");
        const char* operation = "HcclChannelConfigSetInt";
        auto ret = HcclChannelConfigSetIntFunc(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
        if (ret == HCCL_SUCCESS) {
            operation = "HcclChannelConfigSetStr";
            ret = HcclChannelConfigSetStrFunc(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, queueTag.c_str());
        }
        if (ret == HCCL_SUCCESS) {
            operation = "HcclChannelAcquireWithConfig";
            ret = HcclChannelAcquireWithConfigFunc(comm_, COMM_ENGINE, channelDescs.data(),
                                                   static_cast<uint32_t>(channelDescs.size()), config, channels.data());
        }
        auto destroyRet = HcclChannelConfigDestroyFunc(config);
        TORCH_CHECK(ret == HCCL_SUCCESS, operation, " failed, ret=", ret, ", config cleanup ret=", destroyRet);
        CheckHccl(destroyRet, "HcclChannelConfigDestroy");
        return channels;
    }

    void PopulateRemoteBuffers(const std::vector<ChannelHandle>& channels, const std::string& bufferTag)
    {
        context_.epHcclBuffer[context_.epRankId] = reinterpret_cast<uint64_t>(deviceBuffer_);
        for (uint32_t peer = 0, index = 0; peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            uint32_t count = 0;
            CommMem* memories = nullptr;
            char** tags = nullptr;
            CheckHccl(HcclChannelGetRemoteMemsFunc(comm_, channels[index++], &count, &memories, &tags),
                      "HcclChannelGetRemoteMems");
            bool found = false;
            for (uint32_t i = 0; i < count && memories != nullptr && tags != nullptr; ++i) {
                if (tags[i] != nullptr && bufferTag == tags[i]) {
                    TORCH_CHECK(memories[i].addr != nullptr && memories[i].size >= requiredBufferSize_,
                                "Invalid remote MoE buffer for peer ", peer);
                    context_.epHcclBuffer[peer] = reinterpret_cast<uint64_t>(memories[i].addr);
                    found = true;
                    break;
                }
            }
            TORCH_CHECK(found, "Registered MoE buffer not found for peer ", peer);
        }
    }

    std::vector<uint32_t> BuildJettyGroupSchedule() const
    {
        const uint32_t groupNum = static_cast<uint32_t>(sharedLinkGroups_.size());
        TORCH_CHECK(groupNum > 0 && groupNum <= context_.sharedJettyNum, "Invalid shared Jetty lane count: ", groupNum);
        uint64_t totalWeight = 0;
        for (const auto& group : sharedLinkGroups_) {
            TORCH_CHECK(group.bwCoeff > 0, "Zero-bandwidth shared Jetty lane selected");
            totalWeight += group.bwCoeff;
        }
        TORCH_CHECK(totalWeight > 0, "Invalid zero shared Jetty lane weight");

        std::vector<uint32_t> counts(groupNum, 0);
        std::vector<uint64_t> remainders(groupNum, 0);
        uint32_t allocated = 0;
        for (uint32_t i = 0; i < groupNum; ++i) {
            const uint64_t product = static_cast<uint64_t>(context_.sharedJettyNum) * sharedLinkGroups_[i].bwCoeff;
            counts[i] = static_cast<uint32_t>(product / totalWeight);
            remainders[i] = product % totalWeight;
            allocated += counts[i];
        }
        std::vector<uint32_t> remainderOrder(groupNum);
        for (uint32_t i = 0; i < groupNum; ++i) {
            remainderOrder[i] = i;
        }
        std::sort(remainderOrder.begin(), remainderOrder.end(), [&](uint32_t lhs, uint32_t rhs) {
            return remainders[lhs] != remainders[rhs] ? remainders[lhs] > remainders[rhs] : lhs < rhs;
        });
        for (uint32_t i = 0; allocated < context_.sharedJettyNum; ++i, ++allocated) {
            ++counts[remainderOrder[i % groupNum]];
        }

        // Even a very narrow lane must receive one Jetty so every physical Endpoint remains active.
        for (uint32_t empty = 0; empty < groupNum; ++empty) {
            if (counts[empty] != 0) {
                continue;
            }
            uint32_t donor = groupNum;
            for (uint32_t i = 0; i < groupNum; ++i) {
                if (counts[i] > 1 &&
                    (donor == groupNum || counts[i] > counts[donor] || (counts[i] == counts[donor] && i < donor))) {
                    donor = i;
                }
            }
            TORCH_CHECK(donor < groupNum, "Cannot assign at least one Jetty to every global lane");
            --counts[donor];
            ++counts[empty];
        }

        std::vector<uint32_t> schedule;
        schedule.reserve(context_.sharedJettyNum);
        for (uint32_t group = 0; group < groupNum; ++group) {
            schedule.insert(schedule.end(), counts[group], group);
            ASCEND_LOGI("Assigned %u shared Jettys to global lane %u with bandwidth coefficient %u", counts[group],
                        group, sharedLinkGroups_[group].bwCoeff);
        }
        TORCH_CHECK(schedule.size() == context_.sharedJettyNum,
                    "Invalid shared Jetty schedule size: ", schedule.size());
        return schedule;
    }

    void CreateResources(const std::string& contextTag, const std::string& bufferTag)
    {
        SelectSharedLayer();
        RegisterDeviceBuffer(contextTag, bufferTag);
        // BW_COEFF is the physical-port count behind the Endpoint on current HCOMM implementations. All ranks use
        // the same global coefficient and largest-remainder schedule, so a 2:6 topology receives exactly 8:24 of
        // the 32 Jettys instead of making an independent local choice.
        const auto jettyGroups = BuildJettyGroupSchedule();
        std::vector<ChannelHandle> channels;
        for (uint32_t jetty = 0; jetty < context_.sharedJettyNum; ++jetty) {
            const std::string queueTag = contextTag + "_jetty_" + std::to_string(jetty);
            const uint32_t groupIndex = jettyGroups[jetty];
            const auto& channelDescs = sharedLinkGroups_[groupIndex].channelDescs;
            if (jetty == 0) {
                channels = AcquireSharedChannels(queueTag, channelDescs);
            }
            CheckHccl(CreateMoeMultiChannelHandle(comm_, queueTag.c_str(), channelDescs.data(),
                                                  static_cast<uint32_t>(channelDescs.size()),
                                                  &context_.multiChannelHandles[jetty]),
                      "MakeMultiChannelHandle");
            TORCH_CHECK(context_.multiChannelHandles[jetty] != 0, "Null shared Jetty handle, index=", jetty);
            ASCEND_LOGI("Mapped shared Jetty %u to die %u", jetty, sharedLinkGroups_[groupIndex].dieId);
        }
        PopulateRemoteBuffers(channels, bufferTag);
    }

    void* GetOrCreateContext(const std::string& contextTag, const std::string& bufferTag)
    {
        // HcclEngineCtxGet does not make the following get/create sequence atomic. Serialize initialization of the
        // same context tag while allowing independent communication groups to initialize concurrently.
        auto contextMutex = GetSharedContextMutex(contextTag);
        std::lock_guard<std::mutex> contextLock(*contextMutex);
        void* ctx = nullptr;
        uint64_t size = 0;
        auto ret = HcclEngineCtxGetFunc(comm_, contextTag.c_str(), COMM_ENGINE, &ctx, &size);
        if (ret == HCCL_SUCCESS) {
            TORCH_CHECK(ctx != nullptr && size == sizeof(context_), "Invalid shared Jetty context cache");
            ReuseBuffer(contextTag);
            return ctx;
        }
        ReserveBufferEntry(contextTag);
        CreateResources(contextTag, bufferTag);
        CheckHccl(HcclEngineCtxCreateFunc(comm_, contextTag.c_str(), COMM_ENGINE, sizeof(context_), &ctx),
                  "HcclEngineCtxCreate");
        TORCH_CHECK(ctx != nullptr, "HcclEngineCtxCreate returned a null context");
        ret = HcclEngineCtxCopyFunc(comm_, COMM_ENGINE, contextTag.c_str(), &context_, sizeof(context_), 0);
        if (ret != HCCL_SUCCESS) {
            auto destroyRet = HcclEngineCtxDestroyFunc(comm_, contextTag.c_str(), COMM_ENGINE);
            TORCH_CHECK(false, "Copy shared Jetty context failed, ret=", ret, ", cleanup ret=", destroyRet);
        }
        return ctx;
    }

public:
    void ReleaseResources()
    {
        if (!bufferRetained_) {
            ReleaseBuffer();
            ReleaseBufferEntryReservation();
        }
    }

    at::Tensor Build(const std::string& groupName, uint64_t requiredBufferSize)
    {
        TORCH_CHECK(!groupName.empty(), "HCCL group name must not be empty");
        // std::string::c_str() appends a terminator after size(), but a string can still contain embedded NULs.
        // Reject them because the HCCL C API would stop at the first NUL and use a truncated group name.
        TORCH_CHECK(groupName.find('\0') == std::string::npos, "HCCL group name must not contain embedded NULs");
        const std::string contextTag = groupName + "_moe_dispatch_combine_shared_jetty_v2_j" +
                                       std::to_string(SHARED_JETTY_NUM) + "_sq" +
                                       std::to_string(Mc2Kernel::MOE_SHARED_JETTY_SQ_DEPTH) + "_cq" +
                                       std::to_string(Mc2Kernel::MOE_SHARED_JETTY_CQ_DEPTH);
        const std::string bufferTag = contextTag + "_buffer";
        const std::string lastQueueTag = contextTag + "_jetty_" + std::to_string(SHARED_JETTY_NUM - 1);
        TORCH_CHECK(lastQueueTag.size() <= SHARED_JETTY_MAX_TAG_SIZE && bufferTag.size() <= SHARED_JETTY_MAX_TAG_SIZE,
                    "Shared Jetty resource tag is too long");
        TORCH_CHECK(requiredBufferSize > 0 && requiredBufferSize % BUFFER_ALIGNMENT == 0,
                    "MoE communication buffer must be positive and 2MB aligned, got ", requiredBufferSize);
        InitHcclSharedJettyFunctions();
        CheckHccl(HcomGetCommHandleByGroupFunc(groupName.c_str(), &comm_), "HcomGetCommHandleByGroup");
        TORCH_CHECK(comm_ != nullptr, "Null HCCL communicator");
        context_ = {};
        GetRankInfo();
        TORCH_CHECK(context_.epRankSize >= MOE_SHARED_JETTY_MIN_RANK_SIZE, "shared Jetty requires at least ",
                    MOE_SHARED_JETTY_MIN_RANK_SIZE, " ranks, but HCCL communicator has ", context_.epRankSize);
        requiredBufferSize_ = requiredBufferSize;
        void* ctx = GetOrCreateContext(contextTag, bufferTag);
        const int64_t elements = sizeof(context_) / sizeof(int32_t);
        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        return at_npu::native::from_blob(ctx, {elements}, options);
    }
};

at::Tensor BuildMoeSharedJettyContext(const std::string& groupName, uint64_t requiredBufferSize)
{
    MoeSharedJettyContextBuilder builder;
    try {
        auto context = builder.Build(groupName, requiredBufferSize);
        builder.ReleaseResources();
        return context;
    } catch (...) {
        builder.ReleaseResources();
        throw;
    }
}
} // namespace Mc2Api

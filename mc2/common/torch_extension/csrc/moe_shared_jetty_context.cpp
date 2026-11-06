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
    HcclComm comm_ = nullptr;
    uint32_t rankSize_ = 0;
    Mc2Kernel::MoeSharedJettyContext context_;
    uint64_t requiredBufferSize_ = 0;
    std::vector<HcclChannelDesc> channelDescs_;
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

    bool SelectSharedLinks(uint32_t layer, CommProtocol protocol)
    {
        if (!LayerCoversAllRanks(layer)) {
            return false;
        }
        std::vector<std::vector<CommLink>> peerLinks;
        for (uint32_t peer = 0; peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            CommLink* links = nullptr;
            uint32_t linkNum = 0;
            CheckHccl(HcclRankGraphGetLinksFunc(comm_, layer, context_.epRankId, peer, &links, &linkNum),
                      "HcclRankGraphGetLinks");
            TORCH_CHECK(linkNum == 0 || links != nullptr, "Null link list for layer ", layer, ", peer ", peer);
            std::vector<CommLink> candidates;
            for (uint32_t i = 0; i < linkNum; ++i) {
                if (links[i].linkAttr.linkProtocol == protocol) {
                    candidates.push_back(links[i]);
                }
            }
            if (candidates.empty()) {
                return false;
            }
            peerLinks.push_back(std::move(candidates));
        }
        for (const auto& candidate : peerLinks.front()) {
            std::vector<HcclChannelDesc> descs(peerLinks.size());
            CheckHccl(HcclChannelDescInit(descs.data(), static_cast<uint32_t>(descs.size())), "HcclChannelDescInit");
            bool matched = true;
            for (uint32_t peer = 0, index = 0; peer < rankSize_; ++peer) {
                if (peer == context_.epRankId) {
                    continue;
                }
                const auto& links = peerLinks[index];
                auto link = std::find_if(links.begin(), links.end(), [&](const CommLink& value) {
                    return IsSameEndpoint(candidate.srcEndpointDesc, value.srcEndpointDesc);
                });
                if (link == links.end()) {
                    matched = false;
                    break;
                }
                auto& desc = descs[index++];
                desc.channelProtocol = protocol;
                desc.remoteRank = peer;
                desc.localEndpoint = link->srcEndpointDesc;
                desc.remoteEndpoint = link->dstEndpointDesc;
            }
            if (matched) {
                channelDescs_ = std::move(descs);
                ASCEND_LOGI("Selected shared Jetty layer %u", layer);
                return true;
            }
        }
        return false;
    }

    void SelectSharedLayer()
    {
        uint32_t* layers = nullptr;
        uint32_t layerNum = 0;
        CheckHccl(HcclRankGraphGetLayersFunc(comm_, &layers, &layerNum), "HcclRankGraphGetLayers");
        TORCH_CHECK(layerNum > 0 && layers != nullptr, "No HCCL network layers for shared Jetty");
        for (uint32_t i = 0; i < layerNum; ++i) {
            if (SelectSharedLinks(layers[i], CommProtocol::COMM_PROTOCOL_UBC_CTP)) {
                return;
            }
        }
        TORCH_CHECK(false, "No UBC layer supports one shared Jetty endpoint to all peers");
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
        for (auto& desc : channelDescs_) {
            desc.memHandles = &memHandle_;
            desc.memHandleNum = 1;
        }
    }

    std::vector<ChannelHandle> AcquireSharedChannels(const std::string& queueTag)
    {
        std::vector<ChannelHandle> channels(channelDescs_.size());
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
            ret =
                HcclChannelAcquireWithConfigFunc(comm_, COMM_ENGINE, channelDescs_.data(),
                                                 static_cast<uint32_t>(channelDescs_.size()), config, channels.data());
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

    void CreateResources(const std::string& contextTag, const std::string& bufferTag)
    {
        SelectSharedLayer();
        RegisterDeviceBuffer(contextTag, bufferTag);
        std::vector<ChannelHandle> channels;
        for (uint32_t jetty = 0; jetty < context_.sharedJettyNum; ++jetty) {
            const std::string queueTag = contextTag + "_jetty_" + std::to_string(jetty);
            if (jetty == 0) {
                channels = AcquireSharedChannels(queueTag);
            }
            CheckHccl(CreateMoeMultiChannelHandle(comm_, queueTag.c_str(), channelDescs_.data(),
                                                  static_cast<uint32_t>(channelDescs_.size()),
                                                  &context_.multiChannelHandles[jetty]),
                      "MakeMultiChannelHandle");
            TORCH_CHECK(context_.multiChannelHandles[jetty] != 0, "Null shared Jetty handle, index=", jetty);
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
        TORCH_CHECK(context_.epRankSize >= MOE_CLOS_ONLY_MIN_RANK_SIZE, "clos_only requires at least ",
                    MOE_CLOS_ONLY_MIN_RANK_SIZE, " ranks, but HCCL communicator has ", context_.epRankSize);
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

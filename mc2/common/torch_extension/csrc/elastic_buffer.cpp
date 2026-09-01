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
 * \file elastic_buffer.cpp
 * \brief
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

// CANN ACL Runtime API
#include "acl/acl.h"

// HCCL types
#include "hccl/hccl_types.h"

// HCCL common utilities
#include "hccl_common.h"

// ACLNN common utilities
#include "aclnn_common.h"

// torch_npu stream utilities
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace Mc2Api {

// Constants
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint32_t HCCL_MIN_RANK_SIZE = 2;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr int COMM_PROTOCOL_UBC_CTP_VALUE = 4;
constexpr int COMM_PROTOCOL_UBC_TP_VALUE = 5;
constexpr int COMM_PROTOCOL_UBG_VALUE = 9;
constexpr int64_t NETWORK_DIRECT = 0;
constexpr int64_t NETWORK_HYBRID = 1;
constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;
constexpr int DIM_TWO = 2;
constexpr uint32_t MOE_CHANNEL_HANDLE_NUM = 72U;
constexpr uint32_t MOE_CHANNEL_NOTIFY_NUM = 3U;
constexpr int64_t SEND_COUNTS_ALIGN_FACTOR = 8;

// RAII guard for multi-step host buffer allocation
struct HostBufferGuard {
    void *hostPtr = nullptr;
    bool registered = false;

    ~HostBufferGuard()
    {
        if (registered && hostPtr) {
            aclrtHostUnregister(hostPtr);
        }
        if (hostPtr) {
            aclrtFreeHost(hostPtr);
        }
    }

    void Release()
    {
        hostPtr = nullptr;
        registered = false;
    }
};

// Helper functions
static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    TORCH_CHECK(y > 0, "CeilDiv divisor must be positive, got ", y);
    TORCH_CHECK(x <= INT64_MAX - y + 1, "CeilDiv overflow: x=", x, " y=", y);
    return (x + y - 1) / y;
}

static inline int64_t AlignTo(int64_t x, int64_t y)
{
    TORCH_CHECK(y > 0, "AlignTo divisor must be positive, got ", y);
    TORCH_CHECK(x <= INT64_MAX - y + 1, "AlignTo overflow: x=", x, " y=", y);
    return CeilDiv(x, y) * y;
}

static inline void NpuStreamWait(aclrtStream waitStream, aclrtStream recordStream)
{
    if (waitStream == recordStream) {
        return;
    }
    aclrtEvent event = nullptr;
    aclError ret = aclrtCreateEvent(&event);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtCreateEvent failed, ret: ", ret);
    ret = aclrtRecordEvent(event, recordStream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtRecordEvent failed, ret: ", ret);
    ret = aclrtStreamWaitEvent(waitStream, event);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtStreamWaitEvent failed, ret: ", ret);
    ret = aclrtDestroyEvent(event);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtDestroyEvent failed, ret: ", ret);
}

// CommContext structure for HCCL communication
struct EngramCommContext {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE * 2] = {};
    uint32_t channelsPerRank = 1;
};

struct MoeCommContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE] = {};
    ChannelHandle hcommHandle[HCCL_MAX_RANK_SIZE] = {};
    uint32_t channelsPerRank = 1;
};

struct EngramContextResources {
    HcclComm hcclComm = nullptr;
    HcclMemHandle memHandle = nullptr;
    void *hostBufPtr = nullptr;
    void *deviceBufPtr = nullptr;
    int64_t commBufferSize = 0;
    EngramCommContext context;
    at::Tensor contextTensor;
};

template <typename ContextT>
static at::Tensor CreateCommContextTensor(const ContextT &context)
{
    int64_t numElements = (sizeof(ContextT) + sizeof(int32_t) - 1) / sizeof(int32_t);
    at::Tensor tensor = at::empty({numElements}, at::TensorOptions()
                                                     .dtype(at::kInt)
                                                     .device(c10::DeviceType::PrivateUse1)
                                                     .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor hostContext = at::empty({numElements}, at::TensorOptions().dtype(at::kInt));
    errno_t memRet = memcpy_s(hostContext.data_ptr<int32_t>(), hostContext.nbytes(), &context, sizeof(ContextT));
    TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet);
    tensor.copy_(hostContext);
    return tensor;
}

class HcclContextBuilderBase {
protected:
    static void AcquireHcclHandle(const std::string &groupName, HcclComm &hcclComm)
    {
        auto hcclRet = HcomGetCommHandleByGroupFunc(groupName.c_str(), &hcclComm);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", groupName.c_str(), ", ret: ", hcclRet);
    }

    static void CheckContextTag(const std::string &contextTag)
    {
        TORCH_CHECK(contextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ", contextTag.size());
    }

    static void CreateEngineContext(const HcclComm &commHandle, const std::string &contextTag, const CommEngine &engine,
                                    uint64_t contextSize, void *&ctx)
    {
        auto hcclRet = HcclEngineCtxCreateFunc(commHandle, contextTag.c_str(), engine, contextSize, &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);
    }

    static void CopyContextToDevice(const HcclComm &commHandle, const std::string &contextTag, const CommEngine &engine,
                                    const void *context, uint64_t contextSize)
    {
        auto hcclRet =
            HcclEngineCtxCopyFunc(commHandle, engine, contextTag.c_str(), const_cast<void *>(context), contextSize, 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
    }

    static void GetRankInfo(const HcclComm &commHandle, uint32_t &rankId, uint32_t &rankSize)
    {
        auto hcclRet = HcclGetRankIdFunc(commHandle, &rankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

        hcclRet = HcclGetRankSizeFunc(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
    }

    static void AcquireChannels(const HcclComm &commHandle, const CommEngine &engine,
                                std::vector<HcclChannelDesc> &descs, ChannelHandle *channels)
    {
        auto hcclRet =
            HcclChannelAcquireFunc(commHandle, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", hcclRet);
    }

    static void GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
    {
        auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", hcclRet);
    }
};

class EngramContextBuilder : public HcclContextBuilderBase {
public:
    EngramContextResources Build(const std::string &groupName, int64_t numCpuBytes, bool withGrad)
    {
        withGrad_ = withGrad;
        EngramContextResources resources;
        AcquireHcclHandle(groupName, resources.hcclComm);

        std::string contextTag = groupName + "engram_embedding";
        CheckContextTag(contextTag);

        HostBufferGuard guard;
        CreateContext(resources, contextTag, numCpuBytes, guard);
        resources.contextTensor = CreateCommContextTensor(resources.context);
        guard.Release();
        return resources;
    }

private:
    bool withGrad_ = false;

    static void ValidateRankSize(uint32_t rankSize)
    {
        TORCH_CHECK(rankSize >= HCCL_MIN_RANK_SIZE, "rankSize must be at least HCCL_MIN_RANK_SIZE, got ", rankSize,
                    ", min ", HCCL_MIN_RANK_SIZE);
        TORCH_CHECK(rankSize <= HCCL_MAX_RANK_SIZE, "rankSize exceeds HCCL_MAX_RANK_SIZE, got ", rankSize, ", max ",
                    HCCL_MAX_RANK_SIZE);
    }

    static void AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag,
                                          int64_t numCpuBytes, EngramContextResources &resources,
                                          HostBufferGuard &guard)
    {
        aclError ar = aclrtMallocHost(&guard.hostPtr, static_cast<uint64_t>(numCpuBytes));
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes, " B) failed, ret=", ar);

        ar = aclrtHostRegisterV2(guard.hostPtr, static_cast<uint64_t>(numCpuBytes), ACL_HOST_REG_MAPPED);
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", numCpuBytes, " B) failed, ret=", ar);
        guard.registered = true;

        void *devPtr = nullptr;
        ar = aclrtHostGetDevicePointer(guard.hostPtr, &devPtr, 0);
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar);

        CommMem mem;
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = devPtr;
        mem.size = static_cast<uint64_t>(numCpuBytes);

        auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &resources.memHandle);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", numCpuBytes,
                    ") failed, ret=", hcclRet);

        resources.hostBufPtr = guard.hostPtr;
        resources.deviceBufPtr = devPtr;
    }

    static void BuildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                                  uint32_t channelsPerRank, HcclMemHandle &memHandle,
                                  std::vector<HcclChannelDesc> &channelDesc)
    {
        channelDesc.clear();
        uint32_t totalChannels = (rankDim - 1) * channelsPerRank;
        channelDesc.reserve(totalChannels);

        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(commHandle, netLayers, netLayerNum);

        HcclResult r;
        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            bool found = false;
            for (uint32_t li = 0; li < netLayerNum && !found; ++li) {
                CommLink *linkList = nullptr;
                uint32_t listSize = 0;
                r = HcclRankGraphGetLinksFunc(commHandle, netLayers[li], srcRankId, peer, &linkList, &listSize);
                if (r != HCCL_SUCCESS)
                    continue;
                for (uint32_t i = 0; i < listSize && !found; ++i) {
                    const int p = static_cast<int>(linkList[i].linkAttr.linkProtocol);
                    if ((p != COMM_PROTOCOL_UBC_CTP_VALUE) && (p != COMM_PROTOCOL_UBC_TP_VALUE) &&
                        (p != COMM_PROTOCOL_UBG_VALUE)) {
                        continue;
                    }
                    for (uint32_t ch = 0; ch < channelsPerRank; ++ch) {
                        HcclChannelDesc desc;
                        HcclResult initRet = HcclChannelDescInit(&desc, 1);
                        TORCH_CHECK(initRet == HCCL_SUCCESS, "HcclChannelDescInit failed, ret=", initRet);
                        desc.remoteRank = peer;
                        desc.channelProtocol = linkList[i].linkAttr.linkProtocol;
                        desc.localEndpoint.protocol = linkList[i].srcEndpointDesc.protocol;
                        desc.localEndpoint.commAddr = linkList[i].srcEndpointDesc.commAddr;
                        desc.localEndpoint.loc = linkList[i].srcEndpointDesc.loc;
                        desc.remoteEndpoint.protocol = linkList[i].dstEndpointDesc.protocol;
                        desc.remoteEndpoint.commAddr = linkList[i].dstEndpointDesc.commAddr;
                        desc.remoteEndpoint.loc = linkList[i].dstEndpointDesc.loc;
                        desc.notifyNum = 3;
                        desc.memHandles = &memHandle;
                        desc.memHandleNum = 1;
                        channelDesc.push_back(desc);
                    }
                    found = true;
                }
            }
            TORCH_CHECK(found, "No UBC_CTP/UBC_TP/UBG link found for srcRankID ", srcRankId, ", dstRankID ", peer);
        }
    }

    static void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                                   uint32_t channelsPerRank, HcclMemHandle &memHandle, ChannelHandle *channels)
    {
        std::vector<HcclChannelDesc> descs;
        ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};
        BuildChannelDescs(commHandle, srcRankId, rankDim, channelsPerRank, memHandle, descs);
        AcquireChannels(commHandle, CommEngine::COMM_ENGINE_AIV, descs, channelBuf);
        uint32_t descIdx = 0;
        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId)
                continue;
            for (uint32_t ch = 0; ch < channelsPerRank; ++ch) {
                channels[peer * channelsPerRank + ch] = channelBuf[descIdx];
                descIdx++;
            }
        }
    }

    static void GetHcclCommResource(const HcclComm &commHandle, EngramContextResources &resources,
                                    const std::string &targetTag)
    {
        uint32_t rankId = resources.context.rankId;
        constexpr uint32_t handleArraySize = 72U;
        resources.context.channelsPerRank = static_cast<uint32_t>(CeilDiv(handleArraySize, resources.context.rankSize));
        if (resources.context.channelsPerRank == 0U) {
            resources.context.channelsPerRank = 1U;
        }
        uint32_t channelsPerRank = resources.context.channelsPerRank;

        ChannelHandle handlesByRank[HCCL_MAX_RANK_SIZE] = {};
        GetHcclCommChannel(commHandle, resources.context.rankSize, rankId, channelsPerRank, resources.memHandle,
                           handlesByRank);

        for (uint32_t peer = 0; peer < resources.context.rankSize; ++peer) {
            if (peer == rankId)
                continue;
            for (uint32_t ch = 0; ch < channelsPerRank; ++ch) {
                resources.context.hcommHandle[peer * channelsPerRank + ch] = handlesByRank[peer * channelsPerRank + ch];
            }
        }

        resources.context.virtualAddrList[rankId] = reinterpret_cast<uint64_t>(resources.deviceBufPtr);

        for (uint32_t i = 0; i < resources.context.rankSize; ++i) {
            if (i == rankId)
                continue;
            uint32_t memNum = 0;
            CommMem *remoteMems = nullptr;
            char **memTags = nullptr;
            auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, resources.context.hcommHandle[i * channelsPerRank],
                                                        &memNum, &remoteMems, &memTags);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", i, ") failed, ret=", hcclRet);
            bool hasTargetMem = false;
            for (uint32_t j = 0; j < memNum; j++) {
                if (memTags == nullptr || remoteMems == nullptr) {
                    break;
                }
                if (memTags[j] != nullptr && targetTag == memTags[j]) {
                    uint64_t targetMemAddr = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                    resources.context.virtualAddrList[i] = targetMemAddr;
                    ASCEND_LOGI("Get Target Mem(%s) Success, Mem id is %d, Addr is %lu", targetTag.c_str(), j,
                                targetMemAddr);
                    hasTargetMem = true;
                    break;
                }
            }
            TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
        }
    }

    static void QueryHcclBufferResource(const HcclComm &commHandle, EngramContextResources &resources)
    {
        uint32_t rankId = resources.context.rankId;
        uint32_t rankSize = resources.context.rankSize;
        constexpr uint32_t aivNum = 72U;
        uint32_t numSendCores = aivNum / 2U;
        if (numSendCores == 0U) {
            numSendCores = 1U;
        }
        uint32_t groupSize = numSendCores / rankSize;
        if (groupSize == 0U) {
            groupSize = 1U;
        }
        uint32_t channelsPerPeer = groupSize * 2U;
        resources.context.channelsPerRank = channelsPerPeer;

        std::vector<ChannelHandle> handlesByRank(rankSize * channelsPerPeer);
        GetHcclCommChannel(commHandle, rankSize, rankId, channelsPerPeer, resources.memHandle, handlesByRank.data());
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (peer == rankId)
                continue;
            for (uint32_t ch = 0; ch < channelsPerPeer; ++ch) {
                resources.context.hcommHandle[peer * channelsPerPeer + ch] = handlesByRank[peer * channelsPerPeer + ch];
            }
        }

        void *localBuffer = nullptr;
        uint64_t localBufferSize = 0;
        auto hcclRet = HcclGetHcclBufferFunc(commHandle, &localBuffer, &localBufferSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclGetHcclBuffer failed, ret=", hcclRet);
        TORCH_CHECK(localBuffer != nullptr && localBufferSize > 0,
                    "HCCL default buffer is null or empty, size=", localBufferSize);

        resources.commBufferSize = static_cast<int64_t>(localBufferSize);
        resources.context.virtualAddrList[rankId] = reinterpret_cast<uint64_t>(localBuffer);

        for (uint32_t i = 0; i < resources.context.rankSize; ++i) {
            if (i == rankId) {
                continue;
            }
            void *remoteBuffer = nullptr;
            uint64_t remoteBufSize = 0;
            hcclRet = HcclChannelGetHcclBufferFunc(commHandle, resources.context.hcommHandle[i * channelsPerPeer],
                                                   &remoteBuffer, &remoteBufSize);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetHcclBuffer(peer=", i, ") failed, ret=", hcclRet);
            TORCH_CHECK(remoteBuffer != nullptr, "HCCL remote buffer is null for peer=", i);
            resources.context.virtualAddrList[i] = reinterpret_cast<uint64_t>(remoteBuffer);
        }
    }

    void CreateContext(EngramContextResources &resources, const std::string &contextTag, int64_t numCpuBytes,
                       HostBufferGuard &guard)
    {
        uint64_t contextSize = sizeof(EngramCommContext);
        void *ctx = nullptr;
        CreateEngineContext(resources.hcclComm, contextTag, CommEngine::COMM_ENGINE_AIV, contextSize, ctx);

        GetRankInfo(resources.hcclComm, resources.context.rankId, resources.context.rankSize);
        ValidateRankSize(resources.context.rankSize);

        if (numCpuBytes == 0) {
            return;
        }

        std::string memBufferTag = contextTag + "_buffer";
        AllocateAndRegisterBuffer(resources.hcclComm, memBufferTag, numCpuBytes, resources, guard);

        if (withGrad_) {
            QueryHcclBufferResource(resources.hcclComm, resources);
        } else {
            GetHcclCommResource(resources.hcclComm, resources, memBufferTag);
        }

        CopyContextToDevice(resources.hcclComm, contextTag, CommEngine::COMM_ENGINE_AIV, &resources.context,
                            contextSize);
    }
};

class MoeContextBuilder : public HcclContextBuilderBase {
public:
    at::Tensor Build(const std::string &groupName, int64_t &cclBufferSize, uint32_t &rankSizePerServer)
    {
        InitHcclEngineCtxFunctions();

        HcclComm hcclComm = nullptr;
        AcquireHcclHandle(groupName, hcclComm);

        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        GetCommProtocol(hcclComm, protocol);

        MoeCommContext context;
        rankNumPerServer_ = rankNumPerUbDomain_;
        TORCH_CHECK(rankNumPerServer_ > 0, "rank_num_per_server must be positive after resolving MoE topology");
        context.rankSizePerServer = rankNumPerServer_;
        rankSizePerServer = rankNumPerServer_;

        void *ctx = nullptr;
        BuildContext(hcclComm, groupName, "moe_dispatch_combine_multi_channel", protocol, context, cclBufferSize, ctx);
        TORCH_CHECK(ctx != nullptr, "Create MoE context tensor failed: ctx is nullptr");
        int64_t numElements = (sizeof(MoeCommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        // HCCL owns ctx; the tensor only provides a non-owning view of the cached device context.
        return at_npu::native::from_blob(ctx, {numElements}, options);
    }

private:
    void BuildContext(const HcclComm &commHandle, const std::string &groupName, const std::string &opName,
                      const CommProtocol &protocol, MoeCommContext &context, int64_t &cclBufferSize, void *&ctx)
    {
        std::string contextTag = groupName + opName;
        CheckContextTag(contextTag);
        CommEngine engine = CommEngine::COMM_ENGINE_AIV;
        uint64_t hcclBufferSize = 0;

        GetOrCreateContext(commHandle, contextTag, engine, protocol, ctx, hcclBufferSize, context);
        cclBufferSize = static_cast<int64_t>(hcclBufferSize);
    }

    void CreateContext(const HcclComm &commHandle, const std::string &contextTag, const CommEngine &engine,
                       const CommProtocol &protocol, void *&ctx, MoeCommContext *context, uint64_t &hcclBufferSize)
    {
        uint64_t contextSize = sizeof(MoeCommContext);
        CreateEngineContext(commHandle, contextTag, engine, contextSize, ctx);

        uint32_t rankSize = 0;
        GetRankInfo(commHandle, context->epRankId, rankSize);
        GetHcclCommResource(commHandle, engine, protocol, *context, rankSize, hcclBufferSize);

        CopyContextToDevice(commHandle, contextTag, engine, context, contextSize);
    }

    void GetOrCreateContext(const HcclComm &commHandle, const std::string &contextTag, const CommEngine &engine,
                            const CommProtocol &protocol, void *&ctx, uint64_t &hcclBufferSize, MoeCommContext &context)
    {
        uint64_t ctxSize = 0;
        auto hcclRet = HcclEngineCtxGetFunc(commHandle, contextTag.c_str(), engine, &ctx, &ctxSize);
        if (hcclRet != HCCL_SUCCESS) {
            CreateContext(commHandle, contextTag, engine, protocol, ctx, &context, hcclBufferSize);
        } else {
            GetHcclBufferSize(commHandle, hcclBufferSize);
        }
    }

    void GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol)
    {
        uint32_t layerNum = 0;
        uint32_t *layerList = nullptr;
        GetNetLayers(commHandle, layerList, layerNum);

        if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
            GetRankSizePerServer(commHandle, rankNumPerUbDomain_);
            return;
        }

        CheckProtocolSupport(commHandle, layerList, layerNum, protocol);
    }

    void CheckProtocolSupport(const HcclComm &commHandle, const uint32_t *layerList, uint32_t layerNum,
                              CommProtocol &protocol)
    {
        uint32_t srcRankId = 0;
        uint32_t rankSize = 0;
        GetRankInfo(commHandle, srcRankId, rankSize);

        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            uint32_t *rankIdLists = nullptr;
            uint32_t rankNumInLayer = 0;
            auto hcclRet =
                HcclRankGraphGetRanksByLayerFunc(commHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);

            bool allSupportProtocol = true;
            for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
                if (rankIdLists[rankId] == srcRankId || layerMap_.find(rankIdLists[rankId]) != layerMap_.end()) {
                    continue;
                }
                CommLink *linksList = nullptr;
                uint32_t netLinkNum = 0;
                hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                                    &linksList, &netLinkNum);
                TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                            "Get HCCL links failed when checking protocol support, ret: ", hcclRet);
                TORCH_CHECK(netLinkNum > 0, "No available HCCL links found");
                if (!CheckLinks(netLinkNum, linksList, protocol)) {
                    allSupportProtocol = false;
                    break;
                }
                layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
            }
            if (!allSupportProtocol) {
                break;
            }
            rankNumPerUbDomain_ = rankNumInLayer;
        }

        if (rankNumPerUbDomain_ != 0 && rankNumPerUbDomain_ < rankSize) {
            TORCH_CHECK(rankSize % rankNumPerUbDomain_ == 0,
                        "rankNumPerUbDomain_ must be less than rankSize and divisible, rankNumPerUbDomain_: ",
                        rankNumPerUbDomain_, ", rankSize: ", rankSize);
            CheckIsCrossSuperNode(commHandle, layerList, layerNum, protocol, srcRankId);
        }
    }

    void CheckIsCrossSuperNode(const HcclComm &commHandle, const uint32_t *layerList, uint32_t layerNum,
                               CommProtocol &protocol, uint32_t srcRankId)
    {
        protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        layerMap_.clear();

        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            uint32_t *rankIdLists = nullptr;
            uint32_t rankNumInLayer = 0;
            auto hcclRet =
                HcclRankGraphGetRanksByLayerFunc(commHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);

            for (uint32_t rankIdx = 0; rankIdx < rankNumInLayer; ++rankIdx) {
                if (rankIdLists[rankIdx] == srcRankId || layerMap_.find(rankIdLists[rankIdx]) != layerMap_.end()) {
                    continue;
                }
                CommLink *linksList = nullptr;
                uint32_t netLinkNum = 0;
                hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[layerIndex], srcRankId, rankIdLists[rankIdx],
                                                    &linksList, &netLinkNum);
                TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                            "Get HCCL links failed when checking protocol support, ret: ", hcclRet);
                TORCH_CHECK(netLinkNum > 0, "No available HCCL links found");
                if (!CheckLinks(netLinkNum, linksList, protocol)) {
                    return;
                }
                layerMap_[rankIdLists[rankIdx]] = layerList[layerIndex];
            }
        }
    }

    static bool CheckLinks(uint32_t netLinkNum, CommLink *linksList, const CommProtocol &protocol)
    {
        for (uint32_t i = 0; i < netLinkNum; ++i) {
            if (linksList[i].linkAttr.linkProtocol == protocol) {
                return true;
            }
        }
        return false;
    }

    static void GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId, uint32_t dstRankId,
                                const CommProtocol &protocol, CommLink *&links)
    {
        CommLink *linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Communication link failed, ret: ", hcclRet);
        TORCH_CHECK(netLinkNum > 0, "The Net Link Is nullptr. srcRankId is ", srcRankId, ", dstRankId is ", dstRankId,
                    ", layerId is ", netLayerId);
        uint32_t index = 0;
        for (; index < netLinkNum; ++index) {
            if (linksList[index].linkAttr.linkProtocol == protocol) {
                links = &linksList[index];
                break;
            }
        }
        TORCH_CHECK(index < netLinkNum, "No matching communication protocol found in HCCL links, protocol is ",
                    static_cast<int>(protocol));
    }

    void InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId, uint32_t channelsPerRank,
                         const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc)
    {
        uint32_t channelNum = static_cast<uint32_t>(channelDesc.size());
        auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);

        uint32_t netLayerNum = 0;
        uint32_t *netLayerList = nullptr;
        GetNetLayers(commHandle, netLayerList, netLayerNum);
        TORCH_CHECK(netLayerNum > 0, "Get HCCL net layers failed, netLayerNum is ", netLayerNum);

        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            uint32_t peerIndex = (peer > srcRankId) ? (peer - 1) : peer;
            uint32_t layerId = netLayerNum == 1 ? netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap_[peer];
            CommLink *links = nullptr;
            GetHcclCommLink(commHandle, layerId, srcRankId, peer, protocol, links);
            for (uint32_t channel = 0; channel < channelsPerRank; ++channel) {
                // Handles are compact by remote rank because the local rank does not need an HCOMM channel.
                uint32_t channelId = peerIndex * channelsPerRank + channel;
                channelDesc[channelId].channelProtocol = protocol;
                channelDesc[channelId].remoteRank = peer;
                channelDesc[channelId].notifyNum = MOE_CHANNEL_NOTIFY_NUM;
                channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
                channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
            }
        }
    }

    void GetHcclCommChannel(const HcclComm &commHandle, const CommEngine &engine, uint32_t rankDim, uint32_t srcRankId,
                            const CommProtocol &protocol, MoeCommContext &context)
    {
        TORCH_CHECK(rankDim >= HCCL_MIN_RANK_SIZE && rankDim <= HCCL_MAX_RANK_SIZE, "Invalid HCCL rank size ", rankDim);
        uint32_t remoteRankNum = rankDim - 1;
        context.channelsPerRank = static_cast<uint32_t>(CeilDiv(MOE_CHANNEL_HANDLE_NUM, rankDim));
        TORCH_CHECK(context.channelsPerRank > 0, "No HCCL channel capacity for rank size ", rankDim);
        TORCH_CHECK(context.channelsPerRank <= HCCL_MAX_RANK_SIZE / rankDim,
                    "HCCL channel handles exceed capacity, rank size ", rankDim, ", channels per rank ",
                    context.channelsPerRank);
        uint32_t channelNum = remoteRankNum * context.channelsPerRank;
        std::vector<HcclChannelDesc> channelDesc(channelNum);
        ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};

        InitHcclChannel(commHandle, rankDim, srcRankId, context.channelsPerRank, protocol, channelDesc);
        AcquireChannels(commHandle, engine, channelDesc, channelBuf);

        uint32_t channelIndex = 0;
        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId) {
                continue;
            }
            for (uint32_t channel = 0; channel < context.channelsPerRank; ++channel) {
                context.hcommHandle[peer * context.channelsPerRank + channel] = channelBuf[channelIndex++];
            }
        }
    }

    void GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                             MoeCommContext &context, uint32_t rankSize, uint64_t &hcclBufferSize)
    {
        uint32_t rankId = context.epRankId;
        GetHcclCommChannel(commHandle, engine, rankSize, rankId, protocol, context);

        for (uint32_t i = 0; i < rankSize; ++i) {
            void *tempBuffer = nullptr;
            uint64_t bufferSize = 0;
            HcclResult hcclRet;

            if (i == rankId) {
                hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBufferSize);
            } else {
                uint32_t channelIndex = i * context.channelsPerRank;
                hcclRet = HcclChannelGetHcclBufferFunc(commHandle, context.hcommHandle[channelIndex], &tempBuffer,
                                                       &bufferSize);
            }

            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL buffer failed, src: ", rankId, ", dst: ", i,
                        ", ret: ", hcclRet);
            context.epHcclBuffer[i] = reinterpret_cast<uint64_t>(tempBuffer);
        }
    }

    static void GetHcclBufferSize(const HcclComm &commHandle, uint64_t &hcclBufferSize)
    {
        void *tempBuffer = nullptr;
        auto hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBufferSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Buffer Size failed, ret: ", hcclRet);
    }

    static void GetRankSizePerServer(const HcclComm &commHandle, uint32_t &rankSizePerServer)
    {
        uint32_t *netLayerList = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(commHandle, netLayerList, netLayerNum);

        uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
        auto hcclRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, netLayers, &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL rank size per server failed, ret: ", hcclRet);
    }

    std::unordered_map<uint32_t, uint32_t> layerMap_;
    uint32_t rankNumPerUbDomain_ = 0;
    uint32_t rankNumPerServer_ = 2;
};

// ElasticBuffer class - unified interface for Engram storage and MoE EP kernels
class ElasticBuffer {
public:
    ElasticBuffer(const std::string &groupName, int64_t numCpuBytes, int64_t numMaxTokensPerRank = 0,
                  bool withGrad = false, bool explicitlyDestroy = false);
    ~ElasticBuffer();

    void EngramWrite(const at::Tensor &storage);
    void EngramBarrier(bool useCommStream = true, bool withCpuSync = false);
    void Destroy();

    int64_t GetHostBufPtr() const
    {
        return reinterpret_cast<int64_t>(engramHostBufPtr_);
    }
    const at::Tensor &GetContextTensor() const
    {
        return engramContextTensor_;
    }
    const at::Tensor &GetLocalStorageAddrTensor() const
    {
        return localStorageAddrTensor_;
    }
    int64_t GetCommBufferSize() const
    {
        return commBufferSize_;
    }
    uint32_t GetRankSize() const
    {
        return engramCommContext_.rankSize;
    }

    static at::Tensor EngramFetch(const at::Tensor &context, const at::Tensor &indices, int64_t hiddenSize,
                                  int64_t numEntries, int64_t dtypeEnum);
    using EngramFetchTrainOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    static EngramFetchTrainOutput EngramFetchTrain(const at::Tensor &context, const at::Tensor &indices,
                                                   int64_t hiddenSize, int64_t numEntries, int64_t dtypeEnum,
                                                   const at::Tensor &localStorageAddr, int64_t numMaxTokensPerRank,
                                                   int64_t commBufferSize, int64_t rankSize);
    static at::Tensor EngramFetchWait(const at::Tensor &context, const at::Tensor &fetched);

    std::tuple<at::Tensor, at::Tensor> EngramFetchGrad(const at::Tensor &gradFetched, const at::Tensor &perm,
                                                       const at::Tensor &sendCounts, const at::Tensor &recvCounts,
                                                       const at::Tensor &recvLocalEntry, const at::Tensor &numRecv);

    // Stateless static method for training backward (graph-mode compatible).
    // Outputs: gradUnique [maxR, H], uniqueLocalEntry [maxR], numUnique [1] (NOT narrowed).
    // Caller is responsible for narrowing by numUnique.item() outside the graph.
    using EngramFetchGradOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
    static EngramFetchGradOutput EngramFetchGrad(const at::Tensor &context, const at::Tensor &gradFetched,
                                                 const at::Tensor &perm, const at::Tensor &sendCounts,
                                                 const at::Tensor &recvCounts, const at::Tensor &recvLocalEntry,
                                                 const at::Tensor &numRecv, int64_t numEntries, int64_t commBufferSize,
                                                 int64_t numMaxTokensPerRank, int64_t rankSize);

    bool IsWithGrad() const
    {
        return withGrad_;
    }

    static int64_t GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize,
                                            at::ScalarType dtype = at::kBFloat16);

    using DispatchTensorList = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    using DispatchEpilogueTensorList =
        std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>>;
    using CombineEpilogueTensorList = std::tuple<at::Tensor, c10::optional<at::Tensor>>;

    DispatchTensorList MoeEpDispatch(const at::Tensor &x, const at::Tensor &topkIdx,
                                     const c10::optional<at::Tensor> &topkWeights,
                                     const c10::optional<at::Tensor> &scales,
                                     const c10::optional<at::Tensor> &cachedDstSlotIdx,
                                     const c10::optional<at::Tensor> &cachedRouteCount,
                                     const c10::optional<at::Tensor> &cachedRouteDstScaleout,
                                     const c10::optional<at::Tensor> &cachedRouteScaleoutSlot, int64_t epWorldSize,
                                     int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank,
                                     int64_t expertAlignment, bool doCpuSync, int64_t hostPinnedCounterAddr);
    DispatchEpilogueTensorList MoeEpDispatchEpilogue(
        const at::Tensor &dstBufferSlotIdx, const at::Tensor &numRecvPerRank, const at::Tensor &numRecvPerExpert,
        const c10::optional<at::Tensor> &cachedRecvSrcMetadata, int64_t epWorldSize, int64_t epRankId,
        int64_t numExperts, int64_t numMaxTokensPerRank, at::Tensor &recvX, at::Tensor &recvSrcMetadata,
        const c10::optional<at::Tensor> &recvTopkWeightsOpt, const c10::optional<at::Tensor> &recvScalesOpt);
    void MoeEpCombine(const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &recvSrcMetadata,
                      const at::Tensor &numRecvTokensPerExpert, const c10::optional<at::Tensor> &topkWeights,
                      int64_t epWorldSize, int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank);
    CombineEpilogueTensorList MoeEpCombineEpilogue(const at::Tensor &topkIdx,
                                                   const c10::optional<at::Tensor> &topkWeights, int64_t epWorldSize,
                                                   int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank,
                                                   at::Tensor &combinedX,
                                                   const c10::optional<at::Tensor> &combinedTopkWeightsOpt);

private:
    void EnsureEngramContext();
    void EnsureMoeContext();
    int64_t ResolveRankNumPerServer(int64_t epWorldSize) const;
    int64_t ResolveTopoType(int64_t epWorldSize, int64_t rankNumPerServer) const;

    std::string groupName_;
    int64_t engramNumCpuBytes_;
    int64_t numMaxTokensPerRank_ = 0;
    bool explicitlyDestroy_ = false;
    bool withGrad_ = false;

    void *engramHostBufPtr_ = nullptr;
    void *engramDeviceBufPtr_ = nullptr;
    HcclMemHandle engramMemHandle_ = nullptr;
    int64_t commBufferSize_ = 0; // HCCL 默认 buffer 大小（从 HcclGetHcclBufferFunc 查询，训练 a2a 收发缓冲）
    HcclComm engramHcclComm_ = nullptr;
    EngramCommContext engramCommContext_;
    at::Tensor engramContextTensor_;    // Cached Engram context tensor
    at::Tensor localStorageAddrTensor_; // int64 scalar tensor, stores deviceBufPtr_ address
    bool engramContextInitialized_ = false;

    at::Tensor moeContextTensor_;
    int64_t moeCclBufferSize_ = 0;
    uint32_t moeRankSizePerServer_ = 2;
    bool moeContextInitialized_ = false;

    int64_t engramHiddenSize_ = 0;
    int64_t engramNumEntries_ = 0;
    at::ScalarType engramDtype_ = at::kBFloat16;

    bool destroyed_ = false;
    bool engramWriteCalled_ = false;
    aclrtStream commStream_ = nullptr;
};

// Constructor

ElasticBuffer::ElasticBuffer(const std::string &groupName, int64_t numCpuBytes, int64_t numMaxTokensPerRank,
                             bool withGrad, bool explicitlyDestroy)
    : groupName_(groupName),
      engramNumCpuBytes_(numCpuBytes),
      destroyed_(false),
      engramWriteCalled_(false),
      numMaxTokensPerRank_(numMaxTokensPerRank),
      explicitlyDestroy_(explicitlyDestroy),
      withGrad_(withGrad)
{
    InitHcclEngineCtxFunctions();
    InitHcclFunctions();
}

// Destructor - automatic resource cleanup only when explicitlyDestroy is false
ElasticBuffer::~ElasticBuffer()
{
    if (explicitlyDestroy_) {
        if (!destroyed_) {
            ASCEND_LOGI("ElasticBuffer is destroyed without explicit destroy() call, "
                        "resource leak may occur when explicitly_destroy is set to true.");
        }
        return;
    }
    try {
        Destroy();
    } catch (const std::exception &e) {
        ASCEND_LOGE("ElasticBuffer destructor cleanup failed: %s", e.what());
    }
}

void ElasticBuffer::EnsureEngramContext()
{
    TORCH_CHECK(!destroyed_, "ElasticBuffer cannot be used after destroy, please create a new ElasticBuffer instance");
    if (engramContextInitialized_) {
        return;
    }
    commStream_ = c10_npu::getNPUStreamFromPool().stream(false);
    TORCH_CHECK(commStream_ != nullptr, "Failed to get NPU stream from pool for comm stream");
    EngramContextBuilder builder;
    EngramContextResources resources = builder.Build(groupName_, engramNumCpuBytes_, withGrad_);
    engramHcclComm_ = resources.hcclComm;
    engramMemHandle_ = resources.memHandle;
    engramHostBufPtr_ = resources.hostBufPtr;
    engramDeviceBufPtr_ = resources.deviceBufPtr;
    engramCommContext_ = resources.context;
    engramContextTensor_ = resources.contextTensor;
    commBufferSize_ = resources.commBufferSize;
    int64_t addrValue = reinterpret_cast<int64_t>(engramDeviceBufPtr_);
    auto hostAddrTensor = at::full({1}, addrValue, at::TensorOptions().dtype(at::kLong));
    localStorageAddrTensor_ = hostAddrTensor.to(c10::DeviceType::PrivateUse1);
    engramContextInitialized_ = true;
}

void ElasticBuffer::EnsureMoeContext()
{
    TORCH_CHECK(!destroyed_, "ElasticBuffer cannot be used after destroy, please create a new ElasticBuffer instance");
    if (moeContextInitialized_) {
        return;
    }
    MoeContextBuilder builder;
    moeContextTensor_ = builder.Build(groupName_, moeCclBufferSize_, moeRankSizePerServer_);
    moeContextInitialized_ = true;
}

int64_t ElasticBuffer::ResolveRankNumPerServer(int64_t epWorldSize) const
{
    int64_t rankNumPerServer = static_cast<int64_t>(moeRankSizePerServer_);
    TORCH_CHECK(rankNumPerServer > 0, "rank_num_per_server must be positive, got ", rankNumPerServer);
    TORCH_CHECK(epWorldSize % rankNumPerServer == 0, "ep_world_size must be divisible by rank_num_per_server, got ",
                epWorldSize, " and ", rankNumPerServer);
    return rankNumPerServer;
}

int64_t ElasticBuffer::ResolveTopoType(int64_t epWorldSize, int64_t rankNumPerServer) const
{
    return (epWorldSize / rankNumPerServer > 1) ? NETWORK_HYBRID : NETWORK_DIRECT;
}

// EngramWrite - write data with automatic barrier
void ElasticBuffer::EngramWrite(const at::Tensor &storage)
{
    TORCH_CHECK(!destroyed_, "engram_write cannot be called after destroy, "
                             "please create a new ElasticBuffer instance");
    EnsureEngramContext();

    TORCH_CHECK(storage.nbytes() <= static_cast<size_t>(engramNumCpuBytes_), "storage size ", storage.nbytes(),
                " exceeds buffer capacity ", engramNumCpuBytes_);

    constexpr int64_t int32Max = static_cast<int64_t>(INT32_MAX);
    TORCH_CHECK(storage.size(0) * static_cast<int64_t>(engramCommContext_.rankSize) <= int32Max,
                "num_entries * rank_size must not exceed INT32_MAX, got num_entries=", storage.size(0),
                ", rank_size=", engramCommContext_.rankSize,
                ", product=", storage.size(0) * static_cast<int64_t>(engramCommContext_.rankSize));

    EngramBarrier(false, true);

    engramHiddenSize_ = storage.size(1);
    engramNumEntries_ = storage.size(0);
    engramDtype_ = storage.scalar_type();

    if (engramNumEntries_ > 0) {
        constexpr size_t MEMCPY_MAX_BYTES = 0x7fffffff;
        size_t totalBytes = storage.nbytes();
        size_t remaining = totalBytes;
        uint8_t *dst = static_cast<uint8_t *>(engramHostBufPtr_);
        const uint8_t *src = static_cast<const uint8_t *>(storage.data_ptr());
        while (remaining > 0) {
            size_t chunkSize = std::min(remaining, MEMCPY_MAX_BYTES);
            errno_t memRet = memcpy_s(dst, chunkSize, src, chunkSize);
            TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet, ", offset=", totalBytes - remaining,
                        ", chunkSize=", chunkSize);
            dst += chunkSize;
            src += chunkSize;
            remaining -= chunkSize;
        }
    }

    EngramBarrier(false, true);
    engramWriteCalled_ = true;
}

// EngramFetch - stateless static method for torch CustomOp registration (graph-mode compatible).
at::Tensor ElasticBuffer::EngramFetch(const at::Tensor &context, const at::Tensor &indices, int64_t hiddenSize,
                                      int64_t numEntries, int64_t dtypeEnum)
{
    auto dtype = static_cast<at::ScalarType>(dtypeEnum);
    int64_t numTokens = indices.size(0);
    auto fetched = at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(dtype).device(indices.device()));
    if (numTokens == 0) {
        return fetched;
    }
    aclTensor *nullTensor = nullptr;
    int64_t zero = 0;
    ACLNN_CMD(aclnnEngramFetch, context, indices, nullTensor, fetched, nullTensor, nullTensor, nullTensor, nullTensor,
              nullTensor, hiddenSize, numEntries, zero, zero, zero);
    return fetched;
}

// EngramFetchTrain - stateless static method for training forward (graph-mode compatible).
// Outputs: fetched + save-for-backward ctx tensors (perm, sendCounts, recvCounts, recvLocalEntry, numRecv).
ElasticBuffer::EngramFetchTrainOutput ElasticBuffer::EngramFetchTrain(
    const at::Tensor &context, const at::Tensor &indices, int64_t hiddenSize, int64_t numEntries, int64_t dtypeEnum,
    const at::Tensor &localStorageAddr, int64_t numMaxTokensPerRank, int64_t commBufferSize, int64_t rankSize)
{
    auto dtype = static_cast<at::ScalarType>(dtypeEnum);
    int64_t numTokens = indices.size(0);
    auto fetched = at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(dtype).device(indices.device()));
    auto intOpts = at::TensorOptions().dtype(at::kInt).device(indices.device());
    TORCH_CHECK(rankSize > 0 && numMaxTokensPerRank <= INT64_MAX / rankSize,
                "numMaxTokensPerRank * rankSize overflow, got numMaxTokensPerRank=", numMaxTokensPerRank,
                ", rankSize=", rankSize);
    int64_t maxR = numMaxTokensPerRank * rankSize;
    at::Tensor perm = at::empty({numTokens}, intOpts);
    at::Tensor sendCounts = at::empty({rankSize * SEND_COUNTS_ALIGN_FACTOR}, intOpts); // 32b对齐
    at::Tensor recvCounts = at::empty({rankSize}, intOpts);
    at::Tensor recvLocalEntry = at::empty({maxR}, intOpts);
    at::Tensor numRecv = at::empty({1}, intOpts);

    if (numTokens > 0) {
        constexpr int64_t withGrad = 1;
        ACLNN_CMD(aclnnEngramFetch, context, indices, localStorageAddr, fetched, perm, sendCounts, recvCounts,
                  recvLocalEntry, numRecv, hiddenSize, numEntries, numMaxTokensPerRank, commBufferSize, withGrad);
    }
    return std::make_tuple(fetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv);
}

// EngramFetchWait - stateless static method for torch CustomOp registration.
at::Tensor ElasticBuffer::EngramFetchWait(const at::Tensor &context, const at::Tensor &fetched)
{
    if (fetched.size(0) == 0) {
        return fetched;
    }
    ACLNN_CMD(aclnnEngramFetchWait, context, fetched);
    return fetched;
}

// EngramFetchGrad - training backward: produce gradUnique, uniqueLocalEntry (sparse index).。
std::tuple<at::Tensor, at::Tensor> ElasticBuffer::EngramFetchGrad(const at::Tensor &gradFetched, const at::Tensor &perm,
                                                                  const at::Tensor &sendCounts,
                                                                  const at::Tensor &recvCounts,
                                                                  const at::Tensor &recvLocalEntry,
                                                                  const at::Tensor &numRecv)
{
    EnsureEngramContext();
    TORCH_CHECK(!destroyed_, "engram_fetch_grad cannot be called after destroy");
    TORCH_CHECK(withGrad_, "engram_fetch_grad requires with_grad=True");
    TORCH_CHECK(engramWriteCalled_, "engram_fetch_grad must be called after at least one engram_write");

    int64_t hidden = gradFetched.size(1);
    uint32_t rankSize = engramCommContext_.rankSize;
    int64_t rankSizeI64 = static_cast<int64_t>(rankSize);
    TORCH_CHECK(rankSizeI64 > 0 && numMaxTokensPerRank_ <= INT64_MAX / rankSizeI64,
                "numMaxTokensPerRank_ * rankSize overflow, got numMaxTokensPerRank_=", numMaxTokensPerRank_,
                ", rankSize=", rankSize);
    int64_t maxR = numMaxTokensPerRank_ * rankSizeI64;

    auto gradUnique =
        at::empty({maxR, hidden}, at::TensorOptions().dtype(gradFetched.dtype()).device(gradFetched.device()));
    auto uniqueLocalEntry = at::empty({maxR}, at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));
    auto numUnique = at::empty({1}, at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));

    ACLNN_CMD(aclnnEngramFetchGrad, engramContextTensor_, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry,
              numRecv, gradUnique, uniqueLocalEntry, numUnique, engramNumEntries_, commBufferSize_);

    int64_t actualK = static_cast<int64_t>(numUnique.item<int32_t>());
    if (actualK < maxR) {
        gradUnique = gradUnique.narrow(0, 0, actualK);
        uniqueLocalEntry = uniqueLocalEntry.narrow(0, 0, actualK);
    }
    return {gradUnique, uniqueLocalEntry};
}

// EngramFetchGrad - stateless static method for training backward (graph-mode compatible).
ElasticBuffer::EngramFetchGradOutput ElasticBuffer::EngramFetchGrad(
    const at::Tensor &context, const at::Tensor &gradFetched, const at::Tensor &perm, const at::Tensor &sendCounts,
    const at::Tensor &recvCounts, const at::Tensor &recvLocalEntry, const at::Tensor &numRecv, int64_t numEntries,
    int64_t commBufferSize, int64_t numMaxTokensPerRank, int64_t rankSize)
{
    int64_t hidden = gradFetched.size(1);
    TORCH_CHECK(rankSize > 0 && numMaxTokensPerRank <= INT64_MAX / rankSize,
                "numMaxTokensPerRank * rankSize overflow, got numMaxTokensPerRank=", numMaxTokensPerRank,
                ", rankSize=", rankSize);
    int64_t maxR = numMaxTokensPerRank * rankSize;

    auto gradUnique =
        at::empty({maxR, hidden}, at::TensorOptions().dtype(gradFetched.dtype()).device(gradFetched.device()));
    auto uniqueLocalEntry = at::empty({maxR}, at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));
    auto numUnique = at::empty({1}, at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));

    ACLNN_CMD(aclnnEngramFetchGrad, context, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv,
              gradUnique, uniqueLocalEntry, numUnique, numEntries, commBufferSize);

    return std::make_tuple(gradUnique, uniqueLocalEntry, numUnique);
}

// EngramBarrier - cross-rank synchronization
void ElasticBuffer::EngramBarrier(bool useCommStream, bool withCpuSync)
{
    TORCH_CHECK(!destroyed_,
                "engram_barrier cannot be called after destroy, please create a new ElasticBuffer instance");
    EnsureEngramContext();
    TORCH_CHECK(engramHcclComm_ != nullptr, "HCCL comm not initialized");

    aclrtStream computeStream = c10_npu::getCurrentNPUStream().stream(false);
    aclrtStream stream = useCommStream ? commStream_ : computeStream;

    if (useCommStream) {
        NpuStreamWait(commStream_, computeStream);
    }

    if (withCpuSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }

    HcclResult ret = HcclBarrierFunc(engramHcclComm_, stream);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed, ret: ", ret);

    if (withCpuSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }

    if (useCommStream) {
        NpuStreamWait(computeStream, commStream_);
    }
}

// Destroy - explicit resource cleanup
void ElasticBuffer::Destroy()
{
    if (destroyed_) {
        return;
    }

    try {
        EngramBarrier(true, true);
    } catch (const std::exception &e) {
        ASCEND_LOGE("EngramBarrier in Destroy failed: %s", e.what());
    }

    // HCCL 默认 buffer 由框架管理，无需手动释放
    commBufferSize_ = 0;

    if (engramHostBufPtr_ != nullptr) {
        aclError ret = aclrtHostUnregister(engramHostBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed, ret: ", ret);
        ret = aclrtFreeHost(engramHostBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed, ret: ", ret);
        engramHostBufPtr_ = nullptr;
        engramDeviceBufPtr_ = nullptr;
    }
    engramContextInitialized_ = false;
    moeContextInitialized_ = false;
    engramContextTensor_ = at::Tensor();
    localStorageAddrTensor_ = at::Tensor();
    moeContextTensor_ = at::Tensor();

    destroyed_ = true;
}

// GetEngramStorageSizeHint - calculate recommended CPU buffer size (static method)
int64_t ElasticBuffer::GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize, at::ScalarType dtype)
{
    int64_t dtypeSize = at::elementSize(dtype);
    TORCH_CHECK(hiddenSize <= INT64_MAX / dtypeSize, "hiddenSize * dtypeSize overflow");
    int64_t hiddenSizeBytes = hiddenSize * dtypeSize;
    int64_t numSfPacks = (dtypeSize <= 1) ? CeilDiv(hiddenSize, 32) : 0;
    TORCH_CHECK(hiddenSizeBytes <= INT64_MAX - numSfPacks * 4, "numBytesPerEntry addition overflow");
    int64_t numBytesPerEntry = AlignTo(hiddenSizeBytes + numSfPacks * 4, 32);
    TORCH_CHECK(numBytesPerEntry > 0 && numEntries <= INT64_MAX / numBytesPerEntry,
                "numBytesPerEntry * numEntries overflow");
    int64_t numCpuBytes = AlignTo(numBytesPerEntry * numEntries, BUFFER_ALIGNMENT);
    return numCpuBytes;
}

} // namespace Mc2Api

namespace OpApi {
namespace {

#define ACL_CHECK(expr) \
    do { \
        aclError _s = (expr); \
        if (_s != ACL_SUCCESS) { \
            throw std::runtime_error("ACL error: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                                     " code=" + std::to_string(_s)); \
        } \
    } while (0)

} // namespace

class HostPinnedCounter {
public:
    HostPinnedCounter()
    {
        ACL_CHECK(aclrtMallocHost(&hostPtr_, 4 * sizeof(int64_t)));
        ACL_CHECK(aclrtHostRegisterV2(hostPtr_, 4 * sizeof(int64_t), ACL_HOST_REG_MAPPED));
        ACL_CHECK(aclrtHostGetDevicePointer(hostPtr_, &devPtr_, 0));
        Reset();
    }

    ~HostPinnedCounter()
    {
        if (hostPtr_ != nullptr) {
            aclrtHostUnregister(hostPtr_);
            aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            devPtr_ = nullptr;
        }
    }

    void Reset()
    {
        *reinterpret_cast<volatile int64_t *>(hostPtr_) = -1;
    }

    int64_t SpinWait()
    {
        while (true) {
            int64_t v = *reinterpret_cast<volatile int64_t *>(hostPtr_);
            if (v >= 0) {
                return v;
            }
        }
    }

    uintptr_t DevicePtr() const
    {
        return reinterpret_cast<uintptr_t>(devPtr_);
    }

    uintptr_t HostPtr() const
    {
        return reinterpret_cast<uintptr_t>(hostPtr_);
    }

private:
    void *hostPtr_ = nullptr;
    void *devPtr_ = nullptr;
};

} // namespace OpApi

Mc2Api::ElasticBuffer::DispatchTensorList Mc2Api::ElasticBuffer::MoeEpDispatch(
    const at::Tensor &x, const at::Tensor &topkIdx, const c10::optional<at::Tensor> &topkWeights,
    const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &cachedDstSlotIdx,
    const c10::optional<at::Tensor> &cachedRouteCount, const c10::optional<at::Tensor> &cachedRouteDstScaleout,
    const c10::optional<at::Tensor> &cachedRouteScaleoutSlot, int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
    int64_t numMaxTokensPerRank, int64_t expertAlignment, bool doCpuSync, int64_t hostPinnedCounterAddr)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x dims must be 2, but got ", x.dim());
    TORCH_CHECK(topkIdx.dim() == DIM_TWO, "topk_idx dims must be 2, but got ", topkIdx.dim());
    EnsureMoeContext();
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    bool anyCached = cachedDstSlotIdx.has_value();
    TORCH_CHECK(!(anyCached && doCpuSync), "cached mode is incompatible with do_cpu_sync=True");
    bool anyCachedRoute =
        cachedRouteCount.has_value() || cachedRouteDstScaleout.has_value() || cachedRouteScaleoutSlot.has_value();
    bool allCachedRoute =
        cachedRouteCount.has_value() && cachedRouteDstScaleout.has_value() && cachedRouteScaleoutSlot.has_value();
    TORCH_CHECK(!anyCachedRoute || allCachedRoute, "cached route tensors must be all present or all absent");
    bool hybridCached = anyCached && topoType == NETWORK_HYBRID;
    TORCH_CHECK(!hybridCached || allCachedRoute, "hybrid cached dispatch requires all cached route tensors");

    auto xSize = x.sizes();
    int64_t numTokens = xSize[0];
    int64_t topK = topkIdx.size(1);
    int64_t numLocalExperts = numExperts / epWorldSize;
    int64_t routeCapacity = topK;

    at::Tensor numRecvPerRank = at::empty({epWorldSize}, x.options().dtype(at::kInt));
    at::Tensor numRecvPerExpert = at::empty({numLocalExperts}, x.options().dtype(at::kLong));
    at::Tensor dstSlot = at::empty({numTokens, topK}, x.options().dtype(at::kInt));
    at::Tensor routeCount = at::zeros({numTokens}, x.options().dtype(at::kInt));
    at::Tensor routeDstScaleout = at::full({numTokens, routeCapacity}, -1, x.options().dtype(at::kInt));
    at::Tensor routeScaleoutSlot = at::full({numTokens, routeCapacity}, -1, x.options().dtype(at::kInt));

    at::Tensor topkWeightsTensor = topkWeights.has_value() ? *topkWeights : at::Tensor();
    at::Tensor cachedSlotTensor = cachedDstSlotIdx.has_value() ? *cachedDstSlotIdx : at::Tensor();
    at::Tensor cachedRouteCountTensor = cachedRouteCount.has_value() ? *cachedRouteCount : at::Tensor();
    at::Tensor cachedRouteDstScaleoutTensor =
        cachedRouteDstScaleout.has_value() ? *cachedRouteDstScaleout : at::Tensor();
    at::Tensor cachedRouteScaleoutSlotTensor =
        cachedRouteScaleoutSlot.has_value() ? *cachedRouteScaleoutSlot : at::Tensor();

    at::Tensor scalesTensor = scales.has_value() ? *scales : at::Tensor();
    aclDataType scalesDtype = (scales.has_value() && scalesTensor.scalar_type() == at::kByte) ?
                                  aclDataType::ACL_FLOAT8_E8M0 :
                                  ConvertToAclDataType(scales.has_value() ? scalesTensor.scalar_type() : at::kFloat);
    TensorWrapper scalesWrapper = TensorWrapper{scalesTensor, scalesDtype};

    ACLNN_CMD(aclnnMoeEpDispatch, moeContextTensor_, x, topkIdx, topkWeightsTensor, scalesWrapper, cachedSlotTensor,
              cachedRouteCountTensor, cachedRouteDstScaleoutTensor, cachedRouteScaleoutSlotTensor, epWorldSize,
              epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_, expertAlignment, doCpuSync,
              hostPinnedCounterAddr, topoType, rankNumPerServer, numRecvPerRank, numRecvPerExpert, dstSlot, routeCount,
              routeDstScaleout, routeScaleoutSlot);

    return std::make_tuple(numRecvPerRank, numRecvPerExpert, dstSlot, routeCount, routeDstScaleout, routeScaleoutSlot);
}

Mc2Api::ElasticBuffer::DispatchEpilogueTensorList Mc2Api::ElasticBuffer::MoeEpDispatchEpilogue(
    const at::Tensor &dstBufferSlotIdx, const at::Tensor &numRecvPerRank, const at::Tensor &numRecvPerExpert,
    const c10::optional<at::Tensor> &cachedRecvSrcMetadata, int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
    int64_t numMaxTokensPerRank, at::Tensor &recvX, at::Tensor &recvSrcMetadata,
    const c10::optional<at::Tensor> &recvTopkWeightsOpt, const c10::optional<at::Tensor> &recvScalesOpt)
{
    EnsureMoeContext();
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    at::Tensor cachedRecvSrcMetadataTensor = cachedRecvSrcMetadata.has_value() ? *cachedRecvSrcMetadata : at::Tensor();

    aclDataType recvScalesDtype = aclDataType::ACL_FLOAT;
    at::Tensor recvScalesTensor =
        recvScalesOpt.has_value() ? *recvScalesOpt : at::empty({1}, recvX.options().dtype(at::kFloat));
    if (recvScalesOpt.has_value() && recvScalesTensor.scalar_type() == at::kByte) {
        recvScalesDtype = aclDataType::ACL_FLOAT8_E8M0;
    }

    TensorWrapper recvScalesWrapper = TensorWrapper{recvScalesTensor, recvScalesDtype};

    at::Tensor recvTopkWeightsTensor =
        recvTopkWeightsOpt.has_value() ? *recvTopkWeightsOpt : at::empty({1}, recvX.options().dtype(at::kFloat));
    bool hasTopkWeights = recvTopkWeightsOpt.has_value();

    ACLNN_CMD(aclnnMoeEpDispatchEpilogue, moeContextTensor_, dstBufferSlotIdx, numRecvPerRank, numRecvPerExpert,
              cachedRecvSrcMetadataTensor, epWorldSize, epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_,
              hasTopkWeights, topoType, rankNumPerServer, recvX, recvSrcMetadata, recvTopkWeightsTensor,
              recvScalesWrapper);

    c10::optional<at::Tensor> recvTopkWeightsOutput;
    if (recvTopkWeightsOpt.has_value()) {
        recvTopkWeightsOutput = *recvTopkWeightsOpt;
    }
    c10::optional<at::Tensor> recvScalesOutput;
    if (recvScalesOpt.has_value()) {
        recvScalesOutput = *recvScalesOpt;
    }
    return std::make_tuple(recvX, recvSrcMetadata, recvTopkWeightsOutput, recvScalesOutput);
}

void Mc2Api::ElasticBuffer::MoeEpCombine(const at::Tensor &x, const at::Tensor &topkIdx,
                                         const at::Tensor &recvSrcMetadata, const at::Tensor &numRecvTokensPerExpert,
                                         const c10::optional<at::Tensor> &topkWeights, int64_t epWorldSize,
                                         int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x dims must be 2, but got ", x.dim());
    TORCH_CHECK(topkIdx.dim() == DIM_TWO, "topk_idx dims must be 2, but got ", topkIdx.dim());
    EnsureMoeContext();
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    c10::optional<at::Tensor> topkWeightsOpt = topkWeights;

    ACLNN_CMD(aclnnMoeEpCombine, moeContextTensor_, x, topkIdx, recvSrcMetadata, numRecvTokensPerExpert, topkWeightsOpt,
              epWorldSize, epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_, topoType, rankNumPerServer);
}

Mc2Api::ElasticBuffer::CombineEpilogueTensorList Mc2Api::ElasticBuffer::MoeEpCombineEpilogue(
    const at::Tensor &topkIdx, const c10::optional<at::Tensor> &topkWeights, int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, at::Tensor &combinedX,
    const c10::optional<at::Tensor> &combinedTopkWeightsOpt)
{
    EnsureMoeContext();
    int64_t rankNumPerServer = ResolveRankNumPerServer(epWorldSize);
    int64_t topoType = ResolveTopoType(epWorldSize, rankNumPerServer);

    bool hasTopkWeights = topkWeights.has_value();
    at::Tensor combinedTopkWeightsTensor = combinedTopkWeightsOpt.has_value() ?
                                               *combinedTopkWeightsOpt :
                                               at::empty({1}, combinedX.options().dtype(at::kFloat));

    ACLNN_CMD(aclnnMoeEpCombineEpilogue, moeContextTensor_, topkIdx, epWorldSize, epRankId, numExperts,
              numMaxTokensPerRank, moeCclBufferSize_, hasTopkWeights, topoType, rankNumPerServer, combinedX,
              combinedTopkWeightsTensor);

    c10::optional<at::Tensor> combinedTopkWeightsOutput;
    if (hasTopkWeights) {
        combinedTopkWeightsOutput = *combinedTopkWeightsOpt;
    }
    return std::make_tuple(combinedX, combinedTopkWeightsOutput);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<OpApi::HostPinnedCounter>(m, "HostPinnedCounter")
        .def(pybind11::init<>())
        .def("spin_wait", &OpApi::HostPinnedCounter::SpinWait)
        .def("reset", &OpApi::HostPinnedCounter::Reset)
        .def("device_ptr", &OpApi::HostPinnedCounter::DevicePtr)
        .def("host_ptr", &OpApi::HostPinnedCounter::HostPtr);

    pybind11::class_<Mc2Api::ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<const std::string &, int64_t, int64_t, bool, bool>(), pybind11::arg("groupName"),
             pybind11::arg("numCpuBytes"), pybind11::arg("numMaxTokensPerRank") = 0, pybind11::arg("withGrad") = false,
             pybind11::arg("explicitlyDestroy") = false)
        .def("engram_write", &Mc2Api::ElasticBuffer::EngramWrite, pybind11::arg("storage").noconvert())
        .def_static("engram_fetch",
                    static_cast<at::Tensor (*)(const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t)>(
                        &Mc2Api::ElasticBuffer::EngramFetch),
                    pybind11::arg("context"), pybind11::arg("indices"), pybind11::arg("hidden_size"),
                    pybind11::arg("num_entries"), pybind11::arg("dtype"))
        .def_static("engram_fetch_train", &Mc2Api::ElasticBuffer::EngramFetchTrain, pybind11::arg("context"),
                    pybind11::arg("indices"), pybind11::arg("hidden_size"), pybind11::arg("num_entries"),
                    pybind11::arg("dtype"), pybind11::arg("local_storage_addr"),
                    pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("comm_buffer_size"),
                    pybind11::arg("rank_size"))
        .def_static("engram_fetch_wait", &Mc2Api::ElasticBuffer::EngramFetchWait, pybind11::arg("context"),
                    pybind11::arg("fetched"))
        .def("engram_fetch_grad",
             static_cast<std::tuple<at::Tensor, at::Tensor> (Mc2Api::ElasticBuffer::*)(
                 const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                 const at::Tensor &)>(&Mc2Api::ElasticBuffer::EngramFetchGrad),
             pybind11::arg("gradFetched").noconvert(), pybind11::arg("perm").noconvert(),
             pybind11::arg("sendCounts").noconvert(), pybind11::arg("recvCounts").noconvert(),
             pybind11::arg("recvLocalEntry").noconvert(), pybind11::arg("numRecv").noconvert())
        .def_static(
            "engram_fetch_grad_op",
            static_cast<Mc2Api::ElasticBuffer::EngramFetchGradOutput (*)(
                const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t, int64_t)>(
                &Mc2Api::ElasticBuffer::EngramFetchGrad),
            pybind11::arg("context"), pybind11::arg("gradFetched").noconvert(), pybind11::arg("perm").noconvert(),
            pybind11::arg("sendCounts").noconvert(), pybind11::arg("recvCounts").noconvert(),
            pybind11::arg("recvLocalEntry").noconvert(), pybind11::arg("numRecv").noconvert(),
            pybind11::arg("numEntries"), pybind11::arg("commBufferSize"), pybind11::arg("numMaxTokensPerRank"),
            pybind11::arg("rankSize"))
        .def("engram_barrier", &Mc2Api::ElasticBuffer::EngramBarrier, pybind11::arg("useCommStream") = true,
             pybind11::arg("withCpuSync") = false)
        .def("destroy", &Mc2Api::ElasticBuffer::Destroy)
        .def("get_host_buf_ptr", &Mc2Api::ElasticBuffer::GetHostBufPtr)
        .def("get_context_tensor", &Mc2Api::ElasticBuffer::GetContextTensor)
        .def("get_local_storage_addr", &Mc2Api::ElasticBuffer::GetLocalStorageAddrTensor)
        .def("get_comm_buffer_size", &Mc2Api::ElasticBuffer::GetCommBufferSize)
        .def("get_rank_size", &Mc2Api::ElasticBuffer::GetRankSize)
        .def("moe_ep_dispatch", &Mc2Api::ElasticBuffer::MoeEpDispatch)
        .def("moe_ep_dispatch_epilogue", &Mc2Api::ElasticBuffer::MoeEpDispatchEpilogue)
        .def("moe_ep_combine", &Mc2Api::ElasticBuffer::MoeEpCombine)
        .def("moe_ep_combine_epilogue", &Mc2Api::ElasticBuffer::MoeEpCombineEpilogue)
        .def_static("get_engram_storage_size_hint", &Mc2Api::ElasticBuffer::GetEngramStorageSizeHint,
                    pybind11::arg("numEntries"), pybind11::arg("hiddenSize"), pybind11::arg("dtype") = at::kBFloat16);
}

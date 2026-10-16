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
 * \file mc2_context.cpp
 * \brief MC2 Context management implementation
 */

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include "mc2_context.h"
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
constexpr uint32_t GROUP_RANK_NUM = 4U;

bool MatchesInstanceEndpoint(const EndpointDesc& linkEndpoint, const EndpointDesc& instanceEndpoint)
{
    if (linkEndpoint.protocol != instanceEndpoint.protocol ||
        linkEndpoint.loc.locType != instanceEndpoint.loc.locType ||
        linkEndpoint.commAddr.type != instanceEndpoint.commAddr.type) {
        return false;
    }
    // 实例查询不填设备位置ID，仅比较有效地址字段
    const auto& lhs = linkEndpoint.commAddr;
    const auto& rhs = instanceEndpoint.commAddr;
    switch (lhs.type) {
        case COMM_ADDR_TYPE_EID:
            return std::memcmp(lhs.eid, rhs.eid, sizeof(lhs.eid)) == 0;
        case COMM_ADDR_TYPE_IP_V4:
            return std::memcmp(&lhs.addr, &rhs.addr, sizeof(lhs.addr)) == 0;
        case COMM_ADDR_TYPE_IP_V6:
            return std::memcmp(&lhs.addr6, &rhs.addr6, sizeof(lhs.addr6)) == 0;
        case COMM_ADDR_TYPE_ID:
            return lhs.id == rhs.id;
        default:
            return false;
    }
}

} // namespace

namespace Mc2Aclnn {
struct Mc2Context::TopologyInstance {
    uint32_t layer;
    CommTopo type;
    std::vector<uint32_t> ranks;
    std::vector<EndpointDesc> endpoints;
};

const std::string Mc2Context::GetLibPath()
{
    const char* ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#endif
    std::string fullPath = std::string(ascendPath) + hcclPathPostfix;
    OP_LOGI("Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T Mc2Context::GetHcclLibFunc(void* handle, const std::string& funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

Mc2Context::Mc2Context()
{
    OP_LOGI("Init Mc2Context Success!");
}

Mc2Context::~Mc2Context()
{
    if (hcclLibHandle_ != nullptr) {
        dlclose(hcclLibHandle_);
        hcclLibHandle_ = nullptr;
    }
}

aclnnStatus Mc2Context::LoadHcclSymbols()
{
    OP_LOGD("Start to load HCCL library and symbols");
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to get HCCL library path");
        return ACLNN_ERR_INNER;
    }

    hcclLibHandle_ = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (hcclLibHandle_ == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "dlopen HCCL library failed: %s, error: %s", libPath.c_str(), dlerror());
        return ACLNN_ERR_INNER;
    }

    struct SymbolInfo {
        void** ptr;
        const char* name;
    };

    SymbolInfo symbols[] = {
        {reinterpret_cast<void**>(&HcomGetCommHandleByGroup), "HcomGetCommHandleByGroup"},
        {reinterpret_cast<void**>(&HcclRankGraphGetLinks), "HcclRankGraphGetLinks"},
        {reinterpret_cast<void**>(&HcclRankGraphGetLayers), "HcclRankGraphGetLayers"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRankSizeByLayer), "HcclRankGraphGetRankSizeByLayer"},
        {reinterpret_cast<void**>(&HcclChannelAcquire), "HcclChannelAcquire"},
        {reinterpret_cast<void**>(&HcclGetHcclBuffer), "HcclGetHcclBuffer"},
        {reinterpret_cast<void**>(&HcclChannelGetHcclBuffer), "HcclChannelGetHcclBuffer"},
        {reinterpret_cast<void**>(&HcclEngineCtxCreate), "HcclEngineCtxCreate"},
        {reinterpret_cast<void**>(&HcclEngineCtxGet), "HcclEngineCtxGet"},
        {reinterpret_cast<void**>(&HcclEngineCtxCopy), "HcclEngineCtxCopy"},
        {reinterpret_cast<void**>(&HcclGetRankId), "HcclGetRankId"},
        {reinterpret_cast<void**>(&HcclGetRankSize), "HcclGetRankSize"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRanksByLayer), "HcclRankGraphGetRanksByLayer"},
    };

    for (auto& sym : symbols) {
        *(sym.ptr) = GetHcclLibFunc<void*>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Failed to load %s symbol", sym.name);
            dlclose(hcclLibHandle_);
            hcclLibHandle_ = nullptr;
            return ACLNN_ERR_INNER;
        }
    }
    OP_LOGD("All HCCL symbols loaded successfully");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::LoadTopologySymbols()
{
    struct SymbolInfo {
        void** ptr;
        const char* name;
    };
    SymbolInfo topologySymbols[] = {
        {reinterpret_cast<void**>(&HcclRankGraphGetTopoTypeByLayer), "HcclRankGraphGetTopoTypeByLayer"},
        {reinterpret_cast<void**>(&HcclRankGraphGetTopoInstsByLayer), "HcclRankGraphGetTopoInstsByLayer"},
        {reinterpret_cast<void**>(&HcclRankGraphGetTopoType), "HcclRankGraphGetTopoType"},
        {reinterpret_cast<void**>(&HcclRankGraphGetRanksByTopoInst), "HcclRankGraphGetRanksByTopoInst"},
        {reinterpret_cast<void**>(&HcclRankGraphGetEndpointNum), "HcclRankGraphGetEndpointNum"},
        {reinterpret_cast<void**>(&HcclRankGraphGetEndpointDesc), "HcclRankGraphGetEndpointDesc"},
    };
    for (auto& sym : topologySymbols) {
        *(sym.ptr) = GetHcclLibFunc<void*>(hcclLibHandle_, sym.name);
        if (*(sym.ptr) == nullptr) {
            return ACLNN_ERR_INNER;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommHandle(const char* groupEp, HcclComm& hcclHandle)
{
    OP_LOGI("Start to get HCCL communication handle, groupEp: %s", groupEp);
    auto ret = HcomGetCommHandleByGroup(groupEp, &hcclHandle);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL handle failed, groupEp: %s", groupEp);
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommLink(const HcclComm& hcclHandle, uint32_t netLayerId, uint32_t srcRankId,
                                        uint32_t dstRankId, const CommProtocol& protocol, CommLink*& links)
{
    OP_LOGD("Start to get HCCL communication link");
    CommLink* linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinks(hcclHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL Communication link failed");
        return ACLNN_ERR_INNER;
    }
    if (netLinkNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                "The Net Link Is nullptr. srcRankId is %u, dstRankId is %u, layerId is %u", srcRankId,
                                dstRankId, netLayerId);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }
    if (index == netLinkNum) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "No matching communication protocol found in HCCL links protocol is %d",
                                protocol);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL communication link success, protocol is: %d", protocol);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetNetLayers(const HcclComm& hcclHandle, uint32_t*& netLayerList, uint32_t& netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers)
{
    auto hcclRet = HcclRankGraphGetRankSizeByLayer(hcclHandle, netLayers, &rankSizePerServer_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL rank size per server failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get HCCL rank size per server success, rankSizePerServer_ is: %u", rankSizePerServer_);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::InitHcclChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                        const CommProtocol& protocol, std::vector<HcclChannelDesc>& channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "HCCL channel init failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("HCCL channel init success");

    uint32_t netLayerNum = 0;
    uint32_t layerId = 0;
    uint32_t* netLayerList = nullptr;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    if (ret != ACLNN_SUCCESS || netLayerNum == 0) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL net layers failed netLayerNum is: %u", netLayerNum);
        return ret;
    }

    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) {
            continue;
        }
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        CommLink* links = nullptr;
        layerId = netLayerNum == 1 ?
                      netLayerList[HCCL_COMM_LAYERS_UB_MEM] :
                      layerMap[dstRank]; // 如果只有一层通信，直接使用该层；如果有多层通信，使用之前记录的通信层
        ret = GetHcclCommLink(hcclHandle, layerId, srcRankId, dstRank, protocol, links);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                           const CommProtocol& protocol, const CommEngine& engine,
                                           std::vector<ChannelHandle>& channels, Mc2MoeContext* mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication channel");
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);

    uint32_t* netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    auto ret = GetNetLayers(hcclHandle, netLayerList, netLayerNum);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER]; // 获取本server卡数
    ret = GetRankSizePerServer(hcclHandle, netLayers);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = InitHcclChannel(hcclHandle, rankDim, srcRankId, protocol, channelDesc);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // COMM_PROTOCOL_UBC_CTP协议 使用 mc2ContextStruct->hcommHandle_
    auto useChannel = mc2ContextStruct->hcommHandle_;
    if (CommProtocol::COMM_PROTOCOL_UBC_CTP != protocol) {
        useChannel = channels.data();
    }
    ret = HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, useChannel);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Acquire HCCL channel failed");
        return ret;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                            const CommProtocol& protocol, Mc2MoeContext* mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication resource");
    uint32_t rankId = mc2ContextStruct->epRankId;
    std::vector<ChannelHandle> channels;

    aclnnStatus ret = GetHcclCommChannel(hcclHandle, epRankSize_, rankId, protocol, engine, channels, mc2ContextStruct);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    // 将rankSizePerServer_写入通信资源结构体中，后续DPU直驱需要使用
    mc2ContextStruct->rankSizePerServer = rankSizePerServer_;
    OP_LOGD("Get HCCL communication channel success, channel num is: %u", channels.size());

    for (uint32_t i = 0; i < epRankSize_; ++i) {
        void* tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            auto channelHandle = channels[idx];
            if (CommProtocol::COMM_PROTOCOL_UBC_CTP == protocol) {
                channelHandle = mc2ContextStruct->hcommHandle_[idx];
            }
            hcclRet = HcclChannelGetHcclBuffer(hcclHandle, channelHandle, &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS || tempBuffer == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ACLNN_ERR_INNER;
        }

        mc2ContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(tempBuffer) + i * EP_RANK_OFFSET_STEP;
    }
    OP_LOGD("Get HCCL CommResource success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupLinks(const HcclComm& hcclHandle, const TopologyInstance& instance, uint32_t rank,
                                        std::vector<bool>& mesh, std::vector<bool>& clos) const
{
    const bool isMesh = instance.type == COMM_TOPO_1DMESH;
    if (isMesh) {
        if (instance.ranks.size() != GROUP_RANK_NUM) {
            return ACLNN_SUCCESS;
        }
        const uint32_t groupStart = rank / GROUP_RANK_NUM * GROUP_RANK_NUM;
        for (uint32_t i = 0; i < GROUP_RANK_NUM; ++i) {
            if (std::find(instance.ranks.begin(), instance.ranks.end(), groupStart + i) == instance.ranks.end()) {
                return ACLNN_SUCCESS;
            }
        }
    }
    auto& matchedLinks = isMesh ? mesh : clos;
    for (uint32_t peer : instance.ranks) {
        const bool isSameGroup = peer / GROUP_RANK_NUM == rank / GROUP_RANK_NUM;
        if (peer == rank || (!isMesh && isSameGroup)) {
            continue;
        }
        CommLink* links = nullptr;
        uint32_t count = 0U;
        HcclResult hcclRet = HcclRankGraphGetLinks(hcclHandle, instance.layer, rank, peer, &links, &count);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology links failed, rank=%u, peer=%u, layer=%u, ret=%d", rank, peer, instance.layer,
                    hcclRet);
            return ACLNN_ERR_INNER;
        }
        for (uint32_t i = 0U; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UB_MEM) {
                continue;
            }
            const bool endpointMatched =
                instance.endpoints.empty() ||
                std::any_of(instance.endpoints.begin(), instance.endpoints.end(), [&](const EndpointDesc& endpoint) {
                    return MatchesInstanceEndpoint(links[i].srcEndpointDesc, endpoint);
                });
            if (!endpointMatched) {
                continue;
            }
            matchedLinks[peer] = true;
            break;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupInstances(const HcclComm& hcclHandle, uint32_t layer, uint32_t rank,
                                            std::vector<bool>& mesh, std::vector<bool>& clos) const
{
    uint32_t* list = nullptr;
    uint32_t count = 0U;
    HcclResult hcclRet = HcclRankGraphGetTopoInstsByLayer(hcclHandle, layer, &list, &count);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology instances failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    const std::vector<uint32_t> instances(list, list + count);
    for (uint32_t instance : instances) {
        CommTopo type = COMM_TOPO_RESERVED;
        hcclRet = HcclRankGraphGetTopoType(hcclHandle, layer, instance, &type);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology instance type failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        OP_LOGD("Group send topology: rank=%u, layer=%u, instance=%u, type=%d", rank, layer, instance, type);
        if (type != COMM_TOPO_1DMESH && type != COMM_TOPO_CLOS) {
            continue;
        }
        hcclRet = HcclRankGraphGetRanksByTopoInst(hcclHandle, layer, instance, &list, &count);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology instance ranks failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        TopologyInstance topology{layer, type, std::vector<uint32_t>(list, list + count), {}};
        hcclRet = HcclRankGraphGetEndpointNum(hcclHandle, layer, instance, &count);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology endpoint count failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        if (count == 0U) {
            continue;
        }
        topology.endpoints.resize(count);
        hcclRet = HcclRankGraphGetEndpointDesc(hcclHandle, layer, instance, &count, topology.endpoints.data());
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology endpoints failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        topology.endpoints.resize(count);
        // CUSTOM层按实例端点筛选，避免混用Mesh和Clos链路
        if (topology.endpoints.empty()) {
            continue;
        }
        auto ret = CheckGroupLinks(hcclHandle, topology, rank, mesh, clos);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckGroupSendEnabled(const HcclComm& hcclHandle, uint32_t rankSize, uint32_t rank,
                                              bool& enableGroupSend) const
{
    if (rankSize < GROUP_RANK_NUM || rankSize % GROUP_RANK_NUM != 0U || rank >= rankSize) {
        OP_LOGI("Group send disabled: unsupported rank layout, rank=%u, rankSize=%u, groupSize=%u", rank, rankSize,
                GROUP_RANK_NUM);
        return ACLNN_ERR_INNER;
    }
    uint32_t* list = nullptr;
    uint32_t count = 0U;
    HcclResult hcclRet = HcclRankGraphGetLayers(hcclHandle, &list, &count);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGI("Get topology layers failed, ret=%d", hcclRet);
        return ACLNN_ERR_INNER;
    }
    // 库内列表可能被后续查询覆盖
    const std::vector<uint32_t> layers(list, list + count);
    std::vector<bool> mesh(rankSize, false);
    std::vector<bool> clos(rankSize, false);
    for (uint32_t layer : layers) {
        CommTopo type = COMM_TOPO_RESERVED;
        hcclRet = HcclRankGraphGetTopoTypeByLayer(hcclHandle, layer, &type);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGI("Get topology layer type failed, ret=%d", hcclRet);
            return ACLNN_ERR_INNER;
        }
        aclnnStatus ret = ACLNN_SUCCESS;
        OP_LOGD("Group send topology: rank=%u, layer=%u, type=%d", rank, layer, type);
        if (type == COMM_TOPO_CUSTOM) {
            ret = CheckGroupInstances(hcclHandle, layer, rank, mesh, clos);
        } else if (type == COMM_TOPO_1DMESH || type == COMM_TOPO_CLOS) {
            hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layer, &list, &count);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGI("Get topology layer ranks failed, ret=%d", hcclRet);
                return ACLNN_ERR_INNER;
            }
            TopologyInstance topology{layer, type, std::vector<uint32_t>(list, list + count), {}};
            ret = CheckGroupLinks(hcclHandle, topology, rank, mesh, clos);
        }
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    for (uint32_t peer = 0U; peer < rankSize; ++peer) {
        const bool isSameGroup = peer / GROUP_RANK_NUM == rank / GROUP_RANK_NUM;
        const bool hasRequiredLink = isSameGroup ? mesh[peer] : clos[peer];
        if (peer != rank && !hasRequiredLink) {
            OP_LOGI("Group send disabled: missing UB_MEM link, rank=%u, peer=%u, sameGroup=%d, mesh=%d, clos=%d", rank,
                    peer, isSameGroup, static_cast<bool>(mesh[peer]), static_cast<bool>(clos[peer]));
            return ACLNN_ERR_INNER;
        }
    }
    enableGroupSend = true;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreatMc2Context(const HcclComm& hcclHandle, const std::string& mc2ContextTag,
                                        const CommEngine& engine, const CommProtocol& protocol,
                                        Mc2MoeContext* mc2ContextStruct, void*& ctx, uint64_t& hcclBuffSize)
{
    OP_LOGD("Start to create HCCL context");
    uint64_t ctxSize = sizeof(Mc2MoeContext);
    auto hcclRet = HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create HCCL context memory failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Create HCCL context success, ctx: %p", ctx);

    hcclRet = HcclGetRankId(hcclHandle, &mc2ContextStruct->epRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank ID failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get rank ID success, rankId is: %u", mc2ContextStruct->epRankId);

    hcclRet = HcclGetRankSize(hcclHandle, &epRankSize_);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank size failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get rank size success, rankSize is: %u", epRankSize_);

    if (epRankSize_ > HCCL_MAX_RANK_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Rank size [%u] exceeds max rank size [%u]", epRankSize_,
                                HCCL_MAX_RANK_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret = GetHcclCommResource(hcclHandle, engine, protocol, mc2ContextStruct);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL communication resource failed");
        return ret;
    }

    mc2ContextStruct->channelsPerRank = 0;
    mc2ContextStruct->enableGroupSend = false;
#if __has_include("version/hcomm_version.h")
    if (protocol == COMM_PROTOCOL_UB_MEM) {
        ret = LoadTopologySymbols();
        if (ret == ACLNN_SUCCESS) {
            OP_LOGI("LoadTopologySymbols success");
            ret = CheckGroupSendEnabled(hcclHandle, epRankSize_, mc2ContextStruct->epRankId,
                                        mc2ContextStruct->enableGroupSend);
            if (ret == ACLNN_SUCCESS) {
                OP_LOGI("CheckGroupSendEnabled success");
            } else {
                OP_LOGI("CheckGroupSendEnabled failed, ret=%d", ret);
            }
        } else {
            OP_LOGI("LoadTopologySymbols failed, ret=%d", ret);
        }
    }
#endif

    hcclRet = HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct, ctxSize,
                                KOPY_DEFAULT_CTX_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Copy context from host to device failed");
        return ACLNN_ERR_INNER;
    }
    hcclBuffSize = hcclBuffSize_;
    OP_LOGD("Copy context from host to device success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CreatMc2ContextTensor(void* ctx, aclTensor*& mc2Context)
{
    OP_LOGD("Start to create Mc2Context Tensor");
    if (ctx == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create Mc2Context Tensor failed ctx is nullptr.");
        return ACLNN_ERR_INNER;
    }

    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shape[1] = {static_cast<int64_t>(mc2ContextLength / sizeof(uint32_t))};
    int64_t strides[1] = {1};

    mc2Context = aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, ctx);
    if (mc2Context == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("CreatMc2ContextTensor Success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetHcclBufferSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize)
{
    void* tempBuffer = nullptr;
    auto hcclRet = HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL Buffer Size failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckLinks(uint32_t& netLinkNum, CommLink* linksList)
{
    bool isFoundUbMemProtocol = false;
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            isFoundUbMemProtocol = true;
            break;
        }
    }
    return isFoundUbMemProtocol ? ACLNN_SUCCESS : ACLNN_ERR_INNER;
}

aclnnStatus Mc2Context::CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum)
{
    uint32_t srcRankId = 0;
    uint32_t dstRankId = 0;
    uint32_t netLinkNum = 0;
    uint32_t rankNumInLayer = 0;
    uint32_t* rankIdLists = nullptr;
    CommLink* linksList = nullptr;

    auto hcclRet = HcclGetRankId(hcclHandle, &srcRankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "CheckProtocolSupport Get rank ID failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CheckProtocolSupport Get rank ID success, rankId is: %u", srcRankId);

    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        OP_LOGD("CheckProtocolSupport Check layer %u", layerList[layerIndex]);
        hcclRet = HcclRankGraphGetRanksByLayer(hcclHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank IDs by layer failed");
            return ACLNN_ERR_INNER;
        }
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId ||
                layerMap.find(rankIdLists[rankId]) != layerMap.end()) { // 本卡或者已经校验过的卡跳过
                continue;
            }
            hcclRet = HcclRankGraphGetLinks(hcclHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                            &linksList, &netLinkNum);
            if (hcclRet != HCCL_SUCCESS) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL links failed");
                return ACLNN_ERR_INNER;
            }
            if (netLinkNum == 0) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                        "No available HCCL links found, srcRankID %u, dstRankID %u layer is %u",
                                        srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return ACLNN_ERR_INNER;
            }
            if (CheckLinks(netLinkNum, linksList) != ACLNN_SUCCESS) {
                OP_LOGE_LIBOPAPI_REPORT("Mc2Context",
                                        "No HCCL links support UB_MEM srcRankID %u, dstRankID %u layer is %u",
                                        srcRankId, rankIdLists[rankId], layerList[layerIndex]);
                return ACLNN_ERR_INNER;
            }
            layerMap[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol)
{
    OP_LOGD("Start to get HCCL communication protocol");
    uint32_t layerNum = 0;
    uint32_t* layerList = nullptr;
    auto ret = HcclRankGraphGetLayers(hcclHandle, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get HCCL layers failed");
        return ACLNN_ERR_INNER;
    }

    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        OP_LOGI("HCCL communication layerNum is %u,so set protocol to UB_MEM", layerNum);
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return ACLNN_SUCCESS;
    }

    OP_LOGD("start CheckProtocolSupport, layerNum is %u", layerNum);

    auto aclnnRet = CheckProtocolSupport(hcclHandle, layerList, layerNum);
    if (aclnnRet != ACLNN_SUCCESS) {
        return aclnnRet;
    }

    OP_LOGD("CheckProtocolSupport success!");
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::ValidateContextTag(const std::string& mc2ContextTag)
{
    if (mc2ContextTag.size() > MAX_CONTEXT_TAG_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Mc2ContextTag is too long, max size is %u, but current size is %u",
                                MAX_CONTEXT_TAG_SIZE, mc2ContextTag.size());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::CheckContextCache(const HcclComm& hcclHandle, const std::string& mc2ContextTag,
                                          const CommEngine& engine, void*& ctx, uint64_t& hcclBuffSize)
{
    uint64_t ctxSize = 0;
    auto hcclRet = HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
    if (hcclRet != HCCL_SUCCESS) { // 没找到缓存，创建context
        hcclBuffSize = 0;
        OP_LOGI("Context cache not found, need to create");
        return ACLNN_SUCCESS;
    }
    auto aclnnRet = GetHcclBufferSize(hcclHandle, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetMc2ContextTensor(const char* groupEp, const char* opName, uint64_t& hcclBuffSize,
                                            aclTensor*& mc2Context, CommProtocol protocol)
{
    OP_LOGI("Start to get Mc2MoeContext Tensor");
    Mc2Context instance;

    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    void* ctx = nullptr;
    std::string mc2ContextTag = std::string(groupEp) + std::string(opName);
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    hcclBuffSize = 0; // Default to 0, will be updated in CheckContextCache

    aclnnRet = instance.ValidateContextTag(mc2ContextTag);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = instance.CheckContextCache(hcclHandle, mc2ContextTag, engine, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    if (hcclBuffSize != 0) { // Cache not found, need to create context
        aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        OP_LOGI("Found context cache, tag=%s, reuse group send setting from context creation", mc2ContextTag.c_str());
        return ACLNN_SUCCESS;
    }

    // COMM_PROTOCOL_UBC_CTP协议直接从算子op_api中传入，不通过 GetCommProtocol 获取和校验
    if (CommProtocol::COMM_PROTOCOL_UBC_CTP != protocol) {
        aclnnRet = instance.GetCommProtocol(hcclHandle, protocol);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    }

    Mc2MoeContext mc2ContextStruct;
    aclnnRet =
        instance.CreatMc2Context(hcclHandle, mc2ContextTag, engine, protocol, &mc2ContextStruct, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = instance.CreatMc2ContextTensor(ctx, mc2Context);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    OP_LOGI("Get Mc2MoeContext Tensor Success");
    return ACLNN_SUCCESS;
}

aclnnStatus Mc2Context::GetMc2RankSize(const char* groupEp, uint32_t& rankSize)
{
    OP_LOGI("Start to get Mc2RankSize Tensor");
    Mc2Context instance;

    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(groupEp, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    auto hcclRet = instance.HcclGetRankSize(hcclHandle, &rankSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("Mc2Context", "Get rank size failed");
        return ACLNN_ERR_INNER;
    }

    OP_LOGI("Get rank size success, rankSize is: %u", rankSize);
    return ACLNN_SUCCESS;
}

// 模板函数显式实例化
template void* Mc2Context::GetHcclLibFunc<void*>(void* handle, const std::string& funcName);

} // namespace Mc2Aclnn
#endif

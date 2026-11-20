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
 * \file mc2_context.h
 * \brief MC2 Context management for HCCL communication
 */

#ifndef MC2_CONTEXT_H
#define MC2_CONTEXT_H

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#define HCOMM_BARRIER_RTP_MIN_VERSION 90200000
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM HCCL_CHANNEL_SUPPORT_VERSION
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "hccl/hccl_rank_graph.h"
#include "aclnn/aclnn_base.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "mc2_moe_context.h"

namespace Mc2Aclnn {

class Mc2Context {
public:
    static aclnnStatus GetMc2ContextTensor(const char* groupEp, const char* opName, uint64_t& hcclBuffSize,
                                           aclTensor*& mc2Context,
                                           CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_MEM,
                                           uint32_t* rankSizePerServer = nullptr);
    static aclnnStatus GetMc2RankSize(const char* groupEp, uint32_t& rankSize);
    static aclnnStatus GetBarrierContextTensor(const char* groupEp, int64_t worldSize, aclTensor*& mc2Context);
    static bool SupportsBarrierStaticGraph(const char* groupEp, int64_t worldSize);

private:
    struct BarrierContext {
        struct State {
            void* context;
            uint32_t localSize;
            uint32_t railSize;
        };
        Mc2Context& owner;
        HcclComm comm;
        uint32_t rank = 0;
        uint32_t selfCard = Mc2Barrier::NO_RANK;
        bool allowMixedProtocols = true;
        std::unordered_map<uint32_t, CommLink> links{};
        std::vector<uint32_t> local{};
        std::vector<uint32_t> rail{};

        aclnnStatus Init(int64_t worldSize);
        aclnnStatus GetContext(const std::string& tag, void*& ctx, uint64_t& bytes);
        aclnnStatus CreateContext(const std::string& tag, void*& ctx, uint64_t& bytes);
        aclnnStatus CacheContext(const std::string& tag, const Mc2Barrier::Context& context, State& cached, void*& ctx,
                                 uint64_t bytes);
        aclnnStatus SelectLink(uint32_t layer, uint32_t src, uint32_t dst, CommLink& link, bool& found);
        aclnnStatus DiscoverLayer(uint32_t layer);
        aclnnStatus CheckReverseLinks(const std::vector<uint32_t>& layers);
        aclnnStatus Discover();
        bool SupportsStaticGraph(int64_t worldSize);
        aclnnStatus CheckTransportSupport(uint32_t railSize);
        aclnnStatus RegisterWindow(const std::string& tag, Mc2Barrier::Context& context, HcclMemHandle& memHandle);
        aclnnStatus ResolvePeerWindow(Mc2Barrier::Peer& peer, const std::string& tag, uint64_t required);
        aclnnStatus DescribeChannels(Mc2Barrier::Context& context, HcclMemHandle& memHandle,
                                     std::vector<HcclChannelDesc>& descs);
        aclnnStatus AcquireBatch(const std::string& tag, const std::vector<HcclChannelDesc>& descs,
                                 const std::vector<Mc2Barrier::Peer*>& peers, uint64_t required);
        aclnnStatus AcquireGroup(Mc2Barrier::Peer* entries, uint32_t count, uint32_t selfSlot,
                                 const HcclChannelDesc* descs, const std::string& tag, uint64_t required);
        aclnnStatus BuildChannels(Mc2Barrier::Context& context, const std::string& tag, HcclMemHandle memHandle);
        static bool IsUsableLink(const CommLink& link);
        static uint32_t Priority(CommProtocol protocol);
        void InitPeers(Mc2Barrier::Context& context);
    };
    struct TopologyInstance;
    struct RankLinkInfo {
        CommProtocol protocol;
        uint32_t layer;
    };
    struct LayerRanks {
        uint32_t layer;
        std::vector<uint32_t> ranks;
    };
    struct RegisteredBuffer {
        void* addr;
        uint64_t size;
        HcclMemHandle handle;
    };
    struct RegisteredWindows {
        RegisteredBuffer ub;
        RegisteredBuffer roce;
    };

    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void* handle, const std::string& funcName);
    Mc2Context();
    ~Mc2Context();
    aclnnStatus OpenHcclLibrary();
    aclnnStatus LoadProtocolSymbols(CommProtocol protocol);
    aclnnStatus LoadHcclSymbols(CommProtocol protocol);
    aclnnStatus LoadCommonHcclSymbols();
    aclnnStatus LoadBarrierSymbols();
    aclnnStatus LoadTopologySymbols();
    aclnnStatus GetCommHandle(const char* groupEp, HcclComm& hcclHandle);
    aclnnStatus GetHcclCommLink(const HcclComm& hcclHandle, uint32_t netLayerId, uint32_t srcRankId, uint32_t dstRankId,
                                const CommProtocol& protocol, CommLink*& links);
    aclnnStatus GetNetLayers(const HcclComm& hcclHandle, uint32_t*& netLayerList, uint32_t& netLayerNum);
    aclnnStatus InitRegisteredChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                      const CommProtocol& protocol, uint32_t channelsPerPeer,
                                      std::vector<HcclChannelDesc>& channelDesc);
    aclnnStatus GetRegisteredCommChannels(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                          const CommEngine& engine, const CommProtocol& protocol,
                                          std::vector<ChannelHandle>& channels);
    aclnnStatus InitRegisteredBuffer(const HcclComm& hcclHandle, const char* tag, RegisteredBuffer& buffer);
    aclnnStatus GetConfiguredWindowSize(uint64_t& windowSize) const;
    aclnnStatus CheckRegisteredWindowSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize);
    aclnnStatus InitRegisteredWindows(const HcclComm& hcclHandle);
    aclnnStatus GetRegisteredRemoteBuffer(const HcclComm& hcclHandle, ChannelHandle channel, const char* tag,
                                          void*& buffer, uint64_t& bufferSize);
    aclnnStatus CheckHcclBuffer(uint32_t srcRankId, uint32_t peer, const void* buffer, uint64_t bufferSize) const;
    aclnnStatus SetHcclBufferAddress(uint32_t srcRankId, uint32_t peer, void* buffer, uint64_t bufferSize,
                                     uint64_t& bufferAddr);
    aclnnStatus SetRoceBufferAddress(uint32_t srcRankId, uint32_t peer, void* buffer, uint64_t bufferSize,
                                     uint64_t& bufferAddr);
    aclnnStatus GetHcclUbMemResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                     Mc2MoeContext* mc2ContextStruct);
    aclnnStatus GetHcclRoceResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                    Mc2MoeContext* mc2ContextStruct);
    aclnnStatus GetRegisteredCommResource(const HcclComm& hcclHandle, const CommEngine& engine,
                                          Mc2MoeContext* mc2ContextStruct);
    aclnnStatus CreateContextCache(const HcclComm& hcclHandle, const std::string& tag, void*& ctx,
                                   void**& readyContext);
    aclnnStatus GetRankId(const HcclComm& hcclHandle, uint32_t& rankId) const;
    aclnnStatus GetRankSize(const HcclComm& hcclHandle, uint32_t& rankSize) const;
    aclnnStatus GetRankSizePerServer(const HcclComm& hcclHandle, uint32_t netLayers);
    aclnnStatus InitHcclChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                const CommProtocol& protocol, std::vector<HcclChannelDesc>& channelDesc);
    aclnnStatus GetHcclCommChannel(const HcclComm& hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                                   const CommProtocol& protocol, const CommEngine& engine,
                                   std::vector<ChannelHandle>& channels, Mc2MoeContext* mc2ContextStruct);
    aclnnStatus GetHcclCommResource(const HcclComm& hcclHandle, const CommEngine& engine, const CommProtocol& protocol,
                                    Mc2MoeContext* mc2ContextStruct);
    aclnnStatus CheckGroupLinks(const HcclComm& hcclHandle, const TopologyInstance& instance, uint32_t rank,
                                std::vector<bool>& mesh, std::vector<bool>& clos) const;
    aclnnStatus GetTopologyEndpoints(const HcclComm& hcclHandle, uint32_t layer, uint32_t instance,
                                     std::vector<EndpointDesc>& endpoints) const;
    aclnnStatus CheckGroupInstances(const HcclComm& hcclHandle, uint32_t layer, uint32_t rank, std::vector<bool>& mesh,
                                    std::vector<bool>& clos) const;
    aclnnStatus CheckGroupLayer(const HcclComm& hcclHandle, uint32_t layer, uint32_t rank, std::vector<bool>& mesh,
                                std::vector<bool>& clos) const;
    aclnnStatus CheckGroupSendEnabled(const HcclComm& hcclHandle, uint32_t rankSize, uint32_t rank,
                                      bool& enableGroupSend) const;
    void InitGroupSend(const HcclComm& hcclHandle, CommProtocol protocol, Mc2MoeContext* mc2ContextStruct);
    aclnnStatus CreatMc2Context(const HcclComm& hcclHandle, const std::string& mc2ContextTag, const CommEngine& engine,
                                CommProtocol protocol, Mc2MoeContext* mc2ContextStruct, void*& ctx,
                                uint64_t& hcclBuffSize);
    aclnnStatus CreatMc2ContextTensor(void* ctx, aclTensor*& mc2Context, uint64_t ctxSize);
    aclnnStatus GetLayerRanks(const HcclComm& hcclHandle, uint32_t layerId, LayerRanks& layer) const;
    static bool HasLinkWithProtocol(const CommLink* linksList, uint32_t netLinkNum, const CommProtocol& protocol);
    bool SupportsProtocol(const HcclComm& hcclHandle, uint32_t layerId, uint32_t srcRankId, uint32_t dstRankId,
                          const CommProtocol& protocol) const;
    bool SupportsDomainProtocolWithAllRanks(const HcclComm& hcclHandle, const LayerRanks& layer,
                                            const CommProtocol& domainProtocol, uint32_t srcRankId) const;
    void RecordDomainLayerRanks(const CommProtocol& protocol, const LayerRanks& layer, uint32_t srcRankId);
    bool FindUbDomain(const HcclComm& hcclHandle, const uint32_t* layerList, uint32_t layerNum,
                      const CommProtocol& domainProtocol, uint32_t srcRankId, LayerRanks& ubDomain);
    bool CheckCrossUbDomainProtocols(const HcclComm& hcclHandle, const uint32_t* layerList, uint32_t layerNum,
                                     uint32_t srcRankId);
    aclnnStatus ValidateRankLinkMap(uint32_t rankSize, uint32_t srcRankId) const;
    aclnnStatus CheckRoceProtocolSupport(const HcclComm& hcclHandle, const uint32_t* layerList, uint32_t layerNum,
                                         const CommProtocol& protocol);
    aclnnStatus BuildRankLinkMap(const HcclComm& hcclHandle);
    aclnnStatus ValidateContextTag(const std::string& mc2ContextTag);
    aclnnStatus GetHcclBufferSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize);
    aclnnStatus CheckContextCache(const HcclComm& hcclHandle, const std::string& mc2ContextTag,
                                  const CommEngine& engine, CommProtocol protocol, void*& ctx, uint64_t& hcclBuffSize);
    aclnnStatus CheckLinks(uint32_t& netLinkNum, CommLink* linksList);
    aclnnStatus CheckProtocolSupport(const HcclComm& hcclHandle, uint32_t*& layerList, uint32_t& layerNum);
    aclnnStatus GetCommProtocol(const HcclComm& hcclHandle, CommProtocol& protocol);
    aclnnStatus PrepareContextTopology(const HcclComm& hcclHandle, CommProtocol& protocol);

    void* hcclLibHandle_ = nullptr;
    RegisteredWindows* registeredWindows_ = nullptr; // Owned by the communication domain, shared across contexts.
    std::unordered_map<uint32_t, RankLinkInfo> rankLinkMap_;
    std::unordered_map<uint32_t, uint32_t> layerMap;

    HcclResult (*HcomGetCommHandleByGroup)(const char*, HcclComm*) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t**, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t*) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void**, uint64_t*) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, const HcclChannelDesc*, uint32_t, ChannelHandle*) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void**, uint64_t*) = nullptr;
    HcclResult (*HcclCommMemReg)(HcclComm, const char*, const CommMem*, HcclMemHandle*) = nullptr;
    HcclResult (*HcclChannelGetRemoteMems)(HcclComm, ChannelHandle, uint32_t*, CommMem**, char***) = nullptr;
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char*, CommEngine, uint64_t, void**) = nullptr;
    HcclResult (*HcclEngineCtxGet)(HcclComm, const char*, CommEngine, void**, uint64_t*) = nullptr;
    HcclResult (*HcclEngineCtxCopy)(HcclComm, CommEngine, const char*, const void*, uint64_t, uint64_t) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t*) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t**, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetTopoTypeByLayer)(HcclComm, uint32_t, CommTopo*) = nullptr;
    HcclResult (*HcclRankGraphGetTopoInstsByLayer)(HcclComm, uint32_t, uint32_t**, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetTopoType)(HcclComm, uint32_t, uint32_t, CommTopo*) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByTopoInst)(HcclComm, uint32_t, uint32_t, uint32_t**, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetEndpointNum)(HcclComm, uint32_t, uint32_t, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetEndpointDesc)(HcclComm, uint32_t, uint32_t, uint32_t*, EndpointDesc*) = nullptr;
    uint64_t hcclBuffSize_ = 0;   // Size of each registered window: 2 * HCCL_BUFFSIZE * 1024 * 1024 bytes.
    uint64_t roceBufferSize_ = 0; // Registered RoCE size, identical on every peer.
    uint32_t epRankSize_ = 0;
    uint32_t rankSizePerServer_ = 0;
};
} // namespace Mc2Aclnn

#endif
#endif // MC2_CONTEXT_H

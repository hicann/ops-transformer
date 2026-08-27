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
 * \file mc2_moe_graph_context.h
 * \brief MC2 MoeContext builder for graph fusion pass (A5), shares the same context
 *        semantics with the aclnn single-op path (mc2/common/op_api/mc2_context.cpp).
 */

#ifndef MC2_MOE_GRAPH_CONTEXT_H
#define MC2_MOE_GRAPH_CONTEXT_H

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM HCCL_CHANNEL_SUPPORT_VERSION
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hccl/hccl_rank_graph.h"
#include "graph/graph.h"

namespace ops {
class Mc2MoeGraphContext {
public:
    // 按 groupEp 获取 Mc2MoeContext 序列化数据与 HCCL buffer 大小
    static bool GetContextData(const std::string &groupEp, std::vector<int32_t> &contextData, int64_t &hcclBuffSize);

    // 与单算子路径一致（tag = groupEp + opName）：获取或创建 HCCL 持有的 ctx 地址
    static bool GetContextDeviceAddr(const std::string &groupEp, const std::string &opName, void *&ctx,
                                     uint64_t &ctxSize);

    // 创建引用 HCCL ctx 的 ConstPlaceHolder 节点（GE 零拷贝引用，不分配不拷贝）
    static ge::graphStatus CreateContextPlaceHolderNode(ge::Graph &graph, const std::string &groupEp, void *ctxAddr,
                                                        uint64_t ctxSize, ge::GNode &placeholderNode);

    // 按 groupEp 查找已存在的 ConstPlaceHolder 节点
    static ge::GNode FindContextPlaceHolderNode(ge::Graph &graph, const std::string &groupEp);

private:
    static constexpr const char *kContextConstNamePrefix = "mc2_moe_context_";
    Mc2MoeGraphContext() = default;
    ~Mc2MoeGraphContext();
    static Mc2MoeGraphContext &GetInstance();

    bool BuildContextData(const std::string &groupEp, std::vector<int32_t> &contextData, int64_t &hcclBuffSize);
    bool LoadHcclSymbols();
    bool GetCommHandle(const char *groupEp, HcclComm &hcclHandle);
    bool GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayerId, uint32_t srcRankId, uint32_t dstRankId,
                         const CommProtocol &protocol, CommLink *&links);
    bool GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum);
    bool GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers);
    bool InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId, const CommProtocol &protocol,
                         std::vector<HcclChannelDesc> &channelDesc);
    bool GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                            const CommProtocol &protocol, const CommEngine &engine,
                            std::vector<ChannelHandle> &channels);
    bool CheckLinks(uint32_t &netLinkNum, CommLink *linksList);
    bool CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum);
    bool GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void *handle, const std::string &funcName);

    void *hcclLibHandle_ = nullptr;
    uint64_t hcclBuffSize_ = 0;
    uint32_t epRankSize_ = 0;
    uint32_t rankSizePerServer_ = 0;
    std::unordered_map<uint32_t, uint32_t> layerMap; // 记录本卡与其他卡的通信层数，key为其他卡的rankId，value为通信层数

    HcclResult (*HcomGetCommHandleByGroup)(const char *, HcclComm *) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t *) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void **, uint64_t *) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void **, uint64_t *) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t **, uint32_t *) = nullptr;
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char *, CommEngine, uint64_t, void **) = nullptr;
    HcclResult (*HcclEngineCtxGet)(HcclComm, const char *, CommEngine, void **, uint64_t *) = nullptr;
    HcclResult (*HcclEngineCtxCopy)(HcclComm, CommEngine, const char *, void *, uint64_t, uint64_t) = nullptr;

    std::unordered_map<std::string, std::pair<std::vector<int32_t>, int64_t>> contextCache_;
};
} // namespace ops
#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#endif // MC2_MOE_GRAPH_CONTEXT_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * 将图中的 MoeDistributeDispatchV2 节点 1:1 替换为 MoeDistributeDispatchV3（仅 A5，comm_alg != "ccu"）。
 * V3 相比 V2：头部新增 context 输入（Mc2MoeContext 序列化数据，ConstPlaceHolder 节点承载，
 * 引用 HCCL 持有的 ctx 地址），属性删除 group_ep/group_tp，新增 ccl_buffer_size。与 Combine 侧
 * pass 共用 Mc2MoeGraphContext，同一 groupEp 得到的 context 内容一致，保证 dispatch/combine 收发同一块 HCCL window
 * buffer。
 *
 * 输入/输出按原型索引 1:1 映射（V2 port i → V3 port i+1），可选输入槽位按
 * GetInDataNodesAndPortIndexs 是否有生产者判定连边，未连接槽位只保留 desc 占位——
 * 与 torchair "空输入 + 无效描述占位" 构图语义一致，不依赖任何按名查索引接口。
 */
#include "moe_distribute_dispatch_v2_to_v3_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <array>
#include <map>
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "op_graph/mc2_moe_graph_context.h"
#include "graph/operator_factory.h"
#include "acl/acl_rt.h" // 运行时判断 cann 版本

namespace ops {
namespace {
const std::string FUSION_PASS_NAME = "MoeDistributeDispatchV2FusionPass";
const std::string FUSED_OP_TYPE2 = "MoeDistributeDispatchV2";
const std::string FUSED_OP_TYPE3 = "MoeDistributeDispatchV3";
constexpr size_t kV2TotalInputs = 7;  // x, expert_ids + 5 个可选输入
constexpr size_t kV2TotalOutputs = 7; // expand_x, dynamic_scales, assist_info_for_combine,
                                      // expert_token_nums, ep_recv_count, tp_recv_count, expand_scales
constexpr size_t kV2RequiredInputs = 2;

// V2 原型输入名（def 顺序），用于 compact 构图（老 torchair 可选输入为 None 时不占位）下按名解析本地索引
const std::vector<std::string> V2_IR_INPUT_NAMES = {
    "x", "expert_ids", "scales", "x_active_mask", "expert_scales", "elastic_info", "performance_info"};

const std::vector<std::string> REQUIRED_INT_ATTRS = {"ep_world_size", "ep_rank_id", "moe_expert_num"};
const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "tp_world_size",          "tp_rank_id",      "expert_shard_type", "shared_expert_num",
    "shared_expert_rank_num", "quant_mode",      "global_bs",         "expert_token_nums_type",
    "zero_expert_num",        "copy_expert_num", "const_expert_num",  "y_dtype"};
const std::vector<std::string> OPTIONAL_STR_ATTRS = {"comm_alg"};

// 运行时按 ge_compiler 版本决定注册 stage：9.0.0 之前的老 toolkit 没有 kCompatibleInherited 枚举，
ge::CustomPassStage GetDispatchV2FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}

// 构建 原型槽位 -> 节点本地索引 的映射（-1 表示该输入未连接）。
// 按原型输入名解析本地索引：全槽位构图（新 torchair 占位）与 compact 构图（老 torchair 不占位）统一处理；
// GNode 没有可靠的槽位总数接口（GetInputsSize 返回的是有效 desc 数），不做形态区分。
// 必选输入解析失败说明名字表异常，安全回退。
bool BuildPortMapping(ge::GNode &moeNode, std::array<int32_t, kV2TotalInputs> &mapping)
{
    for (size_t irSlot = 0; irSlot < kV2TotalInputs; ++irSlot) {
        int32_t localIdx = -1;
        if (moeNode.GetInputIndexByName(ge::AscendString(V2_IR_INPUT_NAMES[irSlot].c_str()), localIdx) ==
                ge::GRAPH_SUCCESS &&
            localIdx >= 0) {
            mapping[irSlot] = localIdx;
        } else {
            mapping[irSlot] = -1; // 未连接的可选输入
        }
    }
    // 必选输入必须全部解析成功，否则节点形态异常，安全回退
    for (size_t i = 0; i < kV2RequiredInputs; ++i) {
        if (mapping[i] < 0) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "required input %s not found on node, skip fusion.",
                      V2_IR_INPUT_NAMES[i].c_str());
            return false;
        }
    }
    return true;
}
} // namespace

ge::graphStatus MoeDistributeDispatchV2FusionPass::AddAttr(ge::GNode &moeNode, ge::GNode &fusionNode,
                                                           int64_t hcclBuffSize)
{
    ge::AscendString nameStr;
    moeNode.GetName(nameStr);
    int64_t attrInt = 0;
    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (moeNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: get required attr %s failed.", nameStr.GetString(),
                      attrName.c_str());
            return ge::GRAPH_FAILED;
        }
        fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
    }
    // V3 新增必选属性 ccl_buffer_size
    fusionNode.SetAttr(ge::AscendString("ccl_buffer_size"), hcclBuffSize);
    // 可选属性存在才拷贝；group_ep/group_tp 在 V3 已删除，不拷贝
    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (moeNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
        }
    }
    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (moeNode.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), strVal);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::CreateFusionNode(ge::Graph &graph, ge::GNode &moeNode,
                                                                    int64_t hcclBuffSize,
                                                                    const ge::TensorDesc &contextDesc,
                                                                    ge::GNode &fusionNode)
{
    ge::AscendString nameStr;
    moeNode.GetName(nameStr);
    auto fusionOp = ge::OperatorFactory::CreateOperator(std::string(nameStr.GetString()) + "_V3", FUSED_OP_TYPE3);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: %s op not found in factory.", nameStr.GetString(),
                  FUSED_OP_TYPE3.c_str());
        return ge::GRAPH_FAILED;
    }
    // port 0 为新增的 context 输入
    fusionOp.UpdateInputDesc(0U, contextDesc);
    // 输入 desc 按映射拷贝：仅实际连接的槽位（mapping[irSlot] >= 0），V3 落位 = 原型槽位 + 1；
    // 未连接槽位保留 factory 注册的占位 desc，与 torchair "空输入 + 无效描述占位" 语义一致
    std::array<int32_t, kV2TotalInputs> mapping{};
    if (!BuildPortMapping(moeNode, mapping)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t irSlot = 0; irSlot < kV2TotalInputs; ++irSlot) {
        if (mapping[irSlot] < 0) {
            continue;
        }
        ge::TensorDesc srcDesc;
        if (moeNode.GetInputDesc(mapping[irSlot], srcDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        fusionOp.UpdateInputDesc(static_cast<uint32_t>(irSlot + 1), srcDesc);
    }
    // 输出 desc 索引不变
    for (size_t v2Port = 0; v2Port < kV2TotalOutputs; ++v2Port) {
        ge::TensorDesc outDesc;
        if (moeNode.GetOutputDesc(static_cast<int32_t>(v2Port), outDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(v2Port), outDesc);
        }
    }
    fusionNode = graph.AddNodeByOp(fusionOp);
    return AddAttr(moeNode, fusionNode, hcclBuffSize);
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::AddEdge(ge::Graph &graph, ge::GNode &moeNode, ge::GNode &fusionNode,
                                                           ge::GNode &contextNode)
{
    // 输入边：按映射取本地锚点，连到 V3 的 原型槽位+1
    std::array<int32_t, kV2TotalInputs> mapping{};
    if (!BuildPortMapping(moeNode, mapping)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t irSlot = 0; irSlot < kV2TotalInputs; ++irSlot) {
        if (mapping[irSlot] < 0) {
            continue;
        }
        auto srcPair = moeNode.GetInDataNodesAndPortIndexs(mapping[irSlot]);
        if (srcPair.first == nullptr) {
            continue;
        }
        if (graph.AddDataEdge(*srcPair.first, srcPair.second, fusionNode, static_cast<int32_t>(irSlot) + 1) !=
            ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add input edge for ir slot %zu failed.", irSlot);
            return ge::GRAPH_FAILED;
        }
    }
    // context 边：ConstPlaceHolder 节点 → V3 port 0
    if (graph.AddDataEdge(contextNode, 0, fusionNode, 0) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add context edge failed.");
        return ge::GRAPH_FAILED;
    }
    // 迁移控制边到 V3（MC2 节点的流序控制依赖），否则 RemoveNode 时 IsolateNode 会把控制边旁路拼出环
    ge::AscendString moeName;
    moeNode.GetName(moeName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "node %s in-ctrl %zu, out-ctrl %zu.", moeName.GetString(),
              moeNode.GetInControlNodes().size(), moeNode.GetOutControlNodes().size());
    for (auto &inCtrlNode : moeNode.GetInControlNodes()) {
        ge::AscendString ctrlName;
        inCtrlNode->GetName(ctrlName);
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "migrate in-control edge: %s -> %s.", ctrlName.GetString(),
                  moeName.GetString());
        if (graph.AddControlEdge(*inCtrlNode, fusionNode) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "migrate in-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    for (auto &outCtrlNode : moeNode.GetOutControlNodes()) {
        ge::AscendString ctrlName;
        outCtrlNode->GetName(ctrlName);
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "migrate out-control edge: %s -> %s.", moeName.GetString(),
                  ctrlName.GetString());
        if (graph.AddControlEdge(fusionNode, *outCtrlNode) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "migrate out-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    // 收集输出消费者，先摘除 V2 的输出数据边（避免 IsolateNode 旁路拼接出垃圾边）
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(kV2TotalOutputs);
    for (size_t i = 0; i < kV2TotalOutputs; ++i) {
        outputConsumers[i] = moeNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        for (auto &outPair : outputConsumers[i]) {
            if (graph.RemoveEdge(moeNode, static_cast<int32_t>(i), *outPair.first, outPair.second) !=
                ge::GRAPH_SUCCESS) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "remove output edge for port %zu failed.", i);
                return ge::GRAPH_FAILED;
            }
        }
    }
    // 摘除 V2 的控制边（src_port/dst_port 均为 -1 表示控制锚点）
    for (auto &inCtrlNode : moeNode.GetInControlNodes()) {
        if (graph.RemoveEdge(*inCtrlNode, -1, moeNode, -1) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "remove in-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    for (auto &outCtrlNode : moeNode.GetOutControlNodes()) {
        if (graph.RemoveEdge(moeNode, -1, *outCtrlNode, -1) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "remove out-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    // 删除 V2 节点后，把输出数据边连到 V3（输出索引不变）
    graph.RemoveNode(moeNode);
    for (size_t i = 0; i < kV2TotalOutputs; ++i) {
        for (auto &outPair : outputConsumers[i]) {
            if (graph.AddDataEdge(fusionNode, static_cast<int32_t>(i), *outPair.first, outPair.second) !=
                ge::GRAPH_SUCCESS) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add output edge for port %zu failed.", i);
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::FusionNode(ge::Graph &graph, ge::GNode &moeNode,
                                                              ge::GNode &contextNode, int64_t hcclBuffSize,
                                                              const ge::TensorDesc &contextDesc)
{
    ge::GNode fusionNode;
    if (CreateFusionNode(graph, moeNode, hcclBuffSize, contextDesc, fusionNode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AddEdge(graph, moeNode, fusionNode, contextNode);
}

ge::graphStatus MoeDistributeDispatchV2FusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &passContext)
{
    (void)passContext;
    if (graph == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::AscendString graphName;
    (void)graph->GetName(graphName);
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter fusion pass %s, graph %s", FUSION_PASS_NAME.c_str(),
              graphName.GetString());
    // 9.0.0 版本前运行降级空跑
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "skip when cann version not compatible.");
        return ge::GRAPH_SUCCESS;
    }
    // 仅 A5 支持 MoeDistributeDispatchV3
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "target platform is not A5, skip.");
        return ge::GRAPH_SUCCESS;
    }
    // 按 groupEp 缓存 context ConstPlaceHolder 节点与 buffer 大小，同图同通信域的多个 V2 节点复用
    std::map<std::string, std::pair<ge::GNode, int64_t>> contextNodes;
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if ((gNode.GetType(nodeType) != ge::GRAPH_SUCCESS) || (nodeType != FUSED_OP_TYPE2.c_str())) {
            continue;
        }
        // ccu 通信算法不支持 mc2Context，保持 V2 不变
        ge::AscendString commAlg;
        if (gNode.GetAttr(ge::AscendString("comm_alg"), commAlg) == ge::GRAPH_SUCCESS &&
            commAlg.GetString() != nullptr && std::string(commAlg.GetString()) == "ccu") {
            continue;
        }
        ge::AscendString groupEp;
        if (gNode.GetAttr(ge::AscendString("group_ep"), groupEp) != ge::GRAPH_SUCCESS ||
            groupEp.GetString() == nullptr) {
            continue;
        }
        // 槽位映射预检：全槽位走索引、compact 按名解析，失败（必选输入缺失）安全回退
        {
            std::array<int32_t, kV2TotalInputs> mapping{};
            if (!BuildPortMapping(gNode, mapping)) {
                continue;
            }
        }
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        std::string groupEpStr = groupEp.GetString();
        if (contextNodes.find(groupEpStr) == contextNodes.end()) {
            std::vector<int32_t> contextData;
            int64_t hcclBuffSize = 0;
            if (!ops::Mc2MoeGraphContext::GetContextData(groupEpStr, contextData, hcclBuffSize)) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "get mc2 context data failed, V2 node kept as-is.");
                continue;
            }
            // 与单算子路径共用 HCCL engine-ctx 注册表（tag = groupEp + "moe_distribute_v2"）：
            // 复用 HCCL 持有的固定 device buffer，避免每图拷贝一份 context 进 weight 池
            void *ctxAddr = nullptr;
            uint64_t ctxSize = 0;
            if (!ops::Mc2MoeGraphContext::GetContextDeviceAddr(groupEpStr, "moe_distribute_v2", ctxAddr, ctxSize)) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "get mc2 context device addr failed, V2 node kept as-is.");
                continue;
            }
            // 先按名查重：另一个 pass（dispatch/combine 互为对端）可能已为本通信域创建过 context 节点
            ge::GNode contextNode = ops::Mc2MoeGraphContext::FindContextPlaceHolderNode(*graph, groupEpStr);
            ge::AscendString tmpType;
            if (contextNode.GetType(tmpType) != ge::GRAPH_SUCCESS) {
                if (ops::Mc2MoeGraphContext::CreateContextPlaceHolderNode(*graph, groupEpStr, ctxAddr, ctxSize,
                                                                          contextNode) != ge::GRAPH_SUCCESS) {
                    continue;
                }
            }
            contextNodes.emplace(groupEpStr, std::make_pair(contextNode, hcclBuffSize));
        }
        ge::GNode contextNode = contextNodes[groupEpStr].first;
        int64_t hcclBuffSize = contextNodes[groupEpStr].second;
        ge::TensorDesc contextDesc;
        if (contextNode.GetOutputDesc(0, contextDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        if (FusionNode(*graph, gNode, contextNode, hcclBuffSize, contextDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: V2->V3 fusion failed.", nameStr.GetString());
            continue;
        }
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "replace %s with %s success.", nameStr.GetString(), FUSED_OP_TYPE3.c_str());
#else
        // 老 HCCL 版本不支持 mc2 context，保持 V2 不变
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "hccl version not support mc2 context, V2 node kept as-is.");
        continue;
#endif
    }
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(MoeDistributeDispatchV2FusionPass).Stage(GetDispatchV2FusionPassStage());
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

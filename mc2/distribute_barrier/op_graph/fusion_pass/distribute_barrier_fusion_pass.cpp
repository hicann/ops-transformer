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
 * 将图中的 DistributeBarrier 节点 1:1 替换为 DistributeBarrierExtend（仅 A5）。
 * Extend 相比原算子：头部新增 context 输入（Mc2MoeContext 序列化数据，ConstPlaceHolder 节点承载，
 * 引用 HCCL 持有的 ctx 地址），属性（group/world_size）不变。与 D&C 的 V2->V3 pass 共用 Mc2MoeGraphContext，
 * 同一通信域共享同一个 context ConstPlaceHolder 节点（mc2_moe_context_<group>）。
 *
 * 输入按原型名解析本地索引（兼容全槽位与 compact 两种构图），V2 port i → Extend port i+1；
 * 未连接槽位只保留占位 desc、不连边。控制边迁移到 Extend 并先摘除原节点输出/控制边，
 * 避免 RemoveNode 时 IsolateNode 旁路拼接出环。
 */
#include "distribute_barrier_fusion_pass.h"

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
const std::string FUSION_PASS_NAME = "DistributeBarrierFusionPass";
const std::string FUSED_OP_TYPE2 = "DistributeBarrier";
const std::string FUSED_OP_TYPE_EXTEND = "DistributeBarrierExtend";
constexpr size_t kBarrierTotalInputs = 3;  // x_ref, time_out, elastic_info
constexpr size_t kBarrierTotalOutputs = 1; // x_ref
constexpr size_t kBarrierRequiredInputs = 1;

// 原型输入名（def 顺序），用于 compact 构图下按名解析本地索引
const std::vector<std::string> BARRIER_IR_INPUT_NAMES = {"x_ref", "time_out", "elastic_info"};

// 运行时按 ge_compiler 版本决定注册 stage：9.0.0 之前的老 toolkit 没有 kCompatibleInherited 枚举，
ge::CustomPassStage GetDistributeBarrierFusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}

// 构建 原型槽位 -> 节点本地索引 的映射（-1 表示该输入未连接），必选输入解析失败则安全回退
bool BuildPortMapping(ge::GNode &barrierNode, std::array<int32_t, kBarrierTotalInputs> &mapping)
{
    for (size_t irSlot = 0; irSlot < kBarrierTotalInputs; ++irSlot) {
        int32_t localIdx = -1;
        if (barrierNode.GetInputIndexByName(ge::AscendString(BARRIER_IR_INPUT_NAMES[irSlot].c_str()), localIdx) ==
                ge::GRAPH_SUCCESS &&
            localIdx >= 0) {
            mapping[irSlot] = localIdx;
        } else {
            mapping[irSlot] = -1; // 未连接的可选输入
        }
    }
    for (size_t i = 0; i < kBarrierRequiredInputs; ++i) {
        if (mapping[i] < 0) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "required input %s not found on node, skip fusion.",
                      BARRIER_IR_INPUT_NAMES[i].c_str());
            return false;
        }
    }
    return true;
}
} // namespace

ge::graphStatus DistributeBarrierFusionPass::AddAttr(ge::GNode &barrierNode, ge::GNode &fusionNode)
{
    ge::AscendString nameStr;
    barrierNode.GetName(nameStr);
    // group（必选 string）与 world_size（必选 int）在 Extend 中保持不变，原样拷贝
    ge::AscendString group;
    if (barrierNode.GetAttr(ge::AscendString("group"), group) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: get required attr group failed.", nameStr.GetString());
        return ge::GRAPH_FAILED;
    }
    fusionNode.SetAttr(ge::AscendString("group"), group);
    int64_t worldSize = 0;
    if (barrierNode.GetAttr(ge::AscendString("world_size"), worldSize) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: get required attr world_size failed.", nameStr.GetString());
        return ge::GRAPH_FAILED;
    }
    fusionNode.SetAttr(ge::AscendString("world_size"), worldSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DistributeBarrierFusionPass::CreateFusionNode(ge::Graph &graph, ge::GNode &barrierNode,
                                                              const ge::TensorDesc &contextDesc, ge::GNode &fusionNode)
{
    ge::AscendString nameStr;
    barrierNode.GetName(nameStr);
    auto fusionOp =
        ge::OperatorFactory::CreateOperator(std::string(nameStr.GetString()) + "_Extend", FUSED_OP_TYPE_EXTEND);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: %s op not found in factory.", nameStr.GetString(),
                  FUSED_OP_TYPE_EXTEND.c_str());
        return ge::GRAPH_FAILED;
    }
    // port 0 为新增的 context 输入
    fusionOp.UpdateInputDesc(0U, contextDesc);
    // 输入 desc 按映射拷贝：仅实际连接的槽位，Extend 落位 = 原型槽位 + 1
    std::array<int32_t, kBarrierTotalInputs> mapping{};
    if (!BuildPortMapping(barrierNode, mapping)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t irSlot = 0; irSlot < kBarrierTotalInputs; ++irSlot) {
        if (mapping[irSlot] < 0) {
            continue;
        }
        ge::TensorDesc srcDesc;
        if (barrierNode.GetInputDesc(mapping[irSlot], srcDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        fusionOp.UpdateInputDesc(static_cast<uint32_t>(irSlot + 1), srcDesc);
    }
    // 输出 desc 索引不变
    for (size_t outPort = 0; outPort < kBarrierTotalOutputs; ++outPort) {
        ge::TensorDesc outDesc;
        if (barrierNode.GetOutputDesc(static_cast<int32_t>(outPort), outDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(outPort), outDesc);
        }
    }
    fusionNode = graph.AddNodeByOp(fusionOp);
    return AddAttr(barrierNode, fusionNode);
}

ge::graphStatus DistributeBarrierFusionPass::AddEdge(ge::Graph &graph, ge::GNode &barrierNode, ge::GNode &fusionNode,
                                                     ge::GNode &contextNode)
{
    // 输入边：按映射取本地锚点，连到 Extend 的 原型槽位+1
    std::array<int32_t, kBarrierTotalInputs> mapping{};
    if (!BuildPortMapping(barrierNode, mapping)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t irSlot = 0; irSlot < kBarrierTotalInputs; ++irSlot) {
        if (mapping[irSlot] < 0) {
            continue;
        }
        auto srcPair = barrierNode.GetInDataNodesAndPortIndexs(mapping[irSlot]);
        if (srcPair.first == nullptr) {
            continue;
        }
        if (graph.AddDataEdge(*srcPair.first, srcPair.second, fusionNode, static_cast<int32_t>(irSlot) + 1) !=
            ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add input edge for ir slot %zu failed.", irSlot);
            return ge::GRAPH_FAILED;
        }
    }
    // context 边：ConstPlaceHolder 节点 → Extend port 0
    if (graph.AddDataEdge(contextNode, 0, fusionNode, 0) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "add context edge failed.");
        return ge::GRAPH_FAILED;
    }
    // 迁移控制边到 Extend，否则 RemoveNode 时 IsolateNode 会把控制边旁路拼出环
    for (auto &inCtrlNode : barrierNode.GetInControlNodes()) {
        if (graph.AddControlEdge(*inCtrlNode, fusionNode) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "migrate in-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    for (auto &outCtrlNode : barrierNode.GetOutControlNodes()) {
        if (graph.AddControlEdge(fusionNode, *outCtrlNode) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "migrate out-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    // 收集输出消费者，先摘除原节点的输出数据边（避免 IsolateNode 旁路拼接出垃圾边）
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(kBarrierTotalOutputs);
    for (size_t i = 0; i < kBarrierTotalOutputs; ++i) {
        outputConsumers[i] = barrierNode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
        for (auto &outPair : outputConsumers[i]) {
            if (graph.RemoveEdge(barrierNode, static_cast<int32_t>(i), *outPair.first, outPair.second) !=
                ge::GRAPH_SUCCESS) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "remove output edge for port %zu failed.", i);
                return ge::GRAPH_FAILED;
            }
        }
    }
    // 摘除原节点的控制边（src_port/dst_port 均为 -1 表示控制锚点）
    for (auto &inCtrlNode : barrierNode.GetInControlNodes()) {
        if (graph.RemoveEdge(*inCtrlNode, -1, barrierNode, -1) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "remove in-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    for (auto &outCtrlNode : barrierNode.GetOutControlNodes()) {
        if (graph.RemoveEdge(barrierNode, -1, *outCtrlNode, -1) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "remove out-control edge failed.");
            return ge::GRAPH_FAILED;
        }
    }
    // 删除原节点后，把输出数据边连到 Extend（输出索引不变）
    graph.RemoveNode(barrierNode);
    for (size_t i = 0; i < kBarrierTotalOutputs; ++i) {
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

ge::graphStatus DistributeBarrierFusionPass::FusionNode(ge::Graph &graph, ge::GNode &barrierNode,
                                                        ge::GNode &contextNode, const ge::TensorDesc &contextDesc)
{
    ge::GNode fusionNode;
    if (CreateFusionNode(graph, barrierNode, contextDesc, fusionNode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AddEdge(graph, barrierNode, fusionNode, contextNode);
}

ge::graphStatus DistributeBarrierFusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &passContext)
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
    // 仅 A5 支持 DistributeBarrierExtend
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "target platform is not A5, skip.");
        return ge::GRAPH_SUCCESS;
    }
    // 按 group 缓存 context ConstPlaceHolder 节点，同图同通信域的多个 barrier 节点复用
    std::map<std::string, ge::GNode> contextNodes;
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if ((gNode.GetType(nodeType) != ge::GRAPH_SUCCESS) || (nodeType != FUSED_OP_TYPE2.c_str())) {
            continue;
        }
        ge::AscendString group;
        if (gNode.GetAttr(ge::AscendString("group"), group) != ge::GRAPH_SUCCESS || group.GetString() == nullptr) {
            continue;
        }
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        std::string groupStr = group.GetString();
        if (contextNodes.find(groupStr) == contextNodes.end()) {
            std::vector<int32_t> contextData;
            int64_t hcclBuffSize = 0;
            if (!ops::Mc2MoeGraphContext::GetContextData(groupStr, contextData, hcclBuffSize)) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "get mc2 context data failed, barrier node kept as-is.");
                continue;
            }
            // 与单算子路径共用 HCCL engine-ctx 注册表（tag = groupEp + "distribute_barrier_extend"）：
            // 复用 HCCL 持有的固定 device buffer，避免每图拷贝一份 context 进 weight 池
            void *ctxAddr = nullptr;
            uint64_t ctxSize = 0;
            if (!ops::Mc2MoeGraphContext::GetContextDeviceAddr(groupStr, "distribute_barrier_extend", ctxAddr,
                                                               ctxSize)) {
                OPS_LOG_E(FUSION_PASS_NAME.c_str(), "get mc2 context device addr failed, barrier node kept as-is.");
                continue;
            }
            // 先按名查重：D&C 的 pass 可能已为本通信域创建过 context 节点
            ge::GNode contextNode = ops::Mc2MoeGraphContext::FindContextPlaceHolderNode(*graph, groupStr);
            ge::AscendString tmpType;
            if (contextNode.GetType(tmpType) != ge::GRAPH_SUCCESS) {
                if (ops::Mc2MoeGraphContext::CreateContextPlaceHolderNode(*graph, groupStr, ctxAddr, ctxSize,
                                                                          contextNode) != ge::GRAPH_SUCCESS) {
                    continue;
                }
            }
            contextNodes.emplace(groupStr, contextNode);
        }
        ge::GNode contextNode = contextNodes[groupStr];
        ge::TensorDesc contextDesc;
        if (contextNode.GetOutputDesc(0, contextDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        if (FusionNode(*graph, gNode, contextNode, contextDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "node %s: barrier to extend fusion failed.", nameStr.GetString());
            continue;
        }
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "replace %s with %s success.", nameStr.GetString(),
                  FUSED_OP_TYPE_EXTEND.c_str());
#else
        // 老 HCCL 版本不支持 mc2 context，保持原 barrier 节点不变
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "hccl version not support mc2 context, barrier node kept as-is.");
        continue;
#endif
    }
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(DistributeBarrierFusionPass).Stage(GetDistributeBarrierFusionPassStage());
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION

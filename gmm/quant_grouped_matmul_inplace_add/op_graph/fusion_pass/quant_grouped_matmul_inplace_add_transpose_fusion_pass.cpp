/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_grouped_matmul_inplace_add_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= QGMM_INPLACE_ADD_GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <cstdint>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "acl/acl_rt.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "securec.h"
#include "version/ge-compiler_version.h"

namespace ge {
namespace fusion {
// GE 9.1 exposes the inspector implementation at link time, but older public
// headers do not declare it.  Keep the declaration weak and local to preserve
// compatibility with both GE versions.
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge

namespace ops {
namespace {
constexpr int32_t kCompatibleInheritedSupportVersion = 90100000;

bool IsTargetVersion()
{
    int32_t version = 0;
    return aclsysGetVersionNum("ge-compiler", &version) == ACL_SUCCESS && version >= kCompatibleInheritedSupportVersion;
}

constexpr char kPassName[] = "QuantGroupedMatmulInplaceAddTransposeFusionPass";
constexpr char kTargetType[] = "QuantGroupedMatmulInplaceAdd";
constexpr char kTransposeType[] = "Transpose";
constexpr char kTransposeDType[] = "TransposeD";
constexpr char kReshapeType[] = "Reshape";
constexpr char kBitcastType[] = "Bitcast";
constexpr char kConstType[] = "Const";
constexpr int32_t kX1Index = 0;
constexpr int32_t kX2Index = 1;
constexpr int32_t kScale1Index = 5;
constexpr int32_t kPermIndex = 1;
const std::vector<int64_t> kX1Perm = {1, 0};
const std::vector<int64_t> kScale1Perm = {1, 0, 2};

struct InputPattern {
    ge::GNodePtr bitcast = nullptr;
    ge::GNodePtr transposeLike = nullptr;
    ge::GNodePtr source = nullptr;
    ge::GNodePtr oldSource = nullptr;
    int32_t sourcePort = 0;
};

bool IsType(const ge::GNodePtr& node, const char* expected)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString type;
    return node->GetType(type) == ge::GRAPH_SUCCESS && type == expected;
}

bool IsType(const ge::GNode& node, const char* expected)
{
    ge::AscendString type;
    return node.GetType(type) == ge::GRAPH_SUCCESS && type == expected;
}

ge::GNodePtr GetInputNode(const ge::GNode& node, int32_t inputIndex, int32_t* outputPort = nullptr)
{
    auto [inputNode, inputPort] = node.GetInDataNodesAndPortIndexs(inputIndex);
    if (outputPort != nullptr) {
        *outputPort = inputPort;
    }
    return inputNode;
}

bool BuildInputPattern(ge::GNode& target, int32_t inputIndex, InputPattern& pattern)
{
    auto immediateInput = GetInputNode(target, inputIndex);
    if (IsType(immediateInput, kBitcastType)) {
        pattern.bitcast = immediateInput;
        pattern.transposeLike = GetInputNode(*immediateInput, 0);
    } else {
        pattern.transposeLike = immediateInput;
    }
    if (pattern.transposeLike == nullptr) {
        return false;
    }
    pattern.source = GetInputNode(*pattern.transposeLike, 0, &pattern.sourcePort);
    pattern.oldSource = pattern.transposeLike;
    return pattern.source != nullptr;
}

bool GetTransposePerm(const ge::GNodePtr& node, std::vector<int64_t>& perm)
{
    if (IsType(node, kTransposeDType)) {
        return node->GetAttr("perm", perm) == ge::GRAPH_SUCCESS;
    }
    if (!IsType(node, kTransposeType)) {
        return false;
    }
    ge::Tensor value;
    ge::TensorDesc desc;
    if (node->GetInputConstData(kPermIndex, value) != ge::GRAPH_SUCCESS ||
        node->GetInputDesc(kPermIndex, desc) != ge::GRAPH_SUCCESS || value.GetData() == nullptr) {
        return false;
    }
    const auto count = desc.GetDataType() == ge::DT_INT32 ?
                           value.GetSize() / sizeof(int32_t) :
                           (desc.GetDataType() == ge::DT_INT64 ? value.GetSize() / sizeof(int64_t) : 0U);
    for (std::size_t i = 0; i < count; ++i) {
        int64_t element = 0;
        if (desc.GetDataType() == ge::DT_INT32) {
            int32_t element32 = 0;
            if (memcpy_s(&element32, sizeof(element32), value.GetData() + i * sizeof(element32), sizeof(element32)) !=
                EOK) {
                return false;
            }
            element = element32;
        } else if (memcpy_s(&element, sizeof(element), value.GetData() + i * sizeof(element), sizeof(element)) != EOK) {
            return false;
        }
        perm.emplace_back(element);
    }
    return count != 0U;
}

bool IsUnknownDim(int64_t dim)
{
    return dim < 0;
}

bool IsSameOrUnknown(int64_t expected, int64_t actual)
{
    // Dynamic GE descriptors may leave either side unresolved.  A known
    // dimension mismatch must still reject the candidate; an unknown side is
    // accepted because the concrete reshape shape is resolved at runtime.
    return expected == actual || IsUnknownDim(expected) || IsUnknownDim(actual);
}

bool IsEquivalentReshape(const ge::GNodePtr& node, const std::vector<int64_t>& perm)
{
    if (!IsType(node, kReshapeType)) {
        return false;
    }
    ge::TensorDesc inputDesc;
    ge::TensorDesc outputDesc;
    if (node->GetInputDesc(0, inputDesc) != ge::GRAPH_SUCCESS ||
        node->GetOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const auto inputDims = inputDesc.GetShape().GetDims();
    const auto outputDims = outputDesc.GetShape().GetDims();
    if (inputDims.size() != perm.size()) {
        return false;
    }
    // Only a reshape with a statically known unit axis can be proven to be a
    // transpose-equivalent view.  This keeps ordinary dynamic reshapes out of
    // the fusion path while allowing the unit-axis case after view_optimize.
    if (inputDims.size() < 2U || (inputDims[0] != 1 && inputDims[1] != 1)) {
        return false;
    }
    // In a dynamic graph GE can expose an empty output shape before shape
    // inference.  The input unit axis and the expected rank are sufficient to
    // identify the transpose-generated view; the output shape is resolved by
    // the runtime graph later.
    if (outputDims.empty()) {
        return true;
    }
    if (outputDims.size() != perm.size()) {
        return false;
    }
    std::vector<int64_t> expected(inputDims.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0 || static_cast<std::size_t>(perm[i]) >= inputDims.size()) {
            return false;
        }
        expected[i] = inputDims[static_cast<std::size_t>(perm[i])];
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!IsSameOrUnknown(expected[i], outputDims[i])) {
            return false;
        }
    }
    return true;
}

bool CheckPattern(const InputPattern& pattern, const std::vector<int64_t>& expectedPerm)
{
    if (pattern.bitcast != nullptr && pattern.bitcast->GetOutDataNodesAndPortIndexs(0).size() != 1U) {
        return false;
    }
    if (pattern.transposeLike->GetOutDataNodesAndPortIndexs(0).size() != 1U) {
        return false;
    }
    if (IsType(pattern.transposeLike, kReshapeType)) {
        return IsEquivalentReshape(pattern.transposeLike, expectedPerm);
    }
    std::vector<int64_t> actualPerm;
    return GetTransposePerm(pattern.transposeLike, actualPerm) && actualPerm == expectedPerm;
}

bool CheckTarget(ge::GNode& target)
{
    ge::TensorDesc x1Desc;
    ge::TensorDesc x2Desc;
    ge::TensorDesc outputDesc;
    if (target.GetInputDesc(kX1Index, x1Desc) != ge::GRAPH_SUCCESS ||
        target.GetInputDesc(kX2Index, x2Desc) != ge::GRAPH_SUCCESS ||
        target.GetOutputDesc(0, outputDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const std::vector<ge::DataType> supported = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
    return std::find(supported.begin(), supported.end(), x1Desc.GetDataType()) != supported.end() &&
           std::find(supported.begin(), supported.end(), x2Desc.GetDataType()) != supported.end() &&
           outputDesc.GetDataType() == ge::DT_FLOAT && x1Desc.GetShape().GetDimNum() >= 2U &&
           x2Desc.GetShape().GetDimNum() >= 2U;
}

bool IsMxMode(ge::GNode& target)
{
    ge::TensorDesc scale1Desc;
    return target.GetInputDesc(kScale1Index, scale1Desc) == ge::GRAPH_SUCCESS &&
           scale1Desc.GetDataType() == ge::DT_FLOAT8_E8M0;
}

bool UpdateBitcast(ge::GNode& bitcast, const ge::TensorDesc& sourceDesc, ge::TensorDesc& targetDesc)
{
    ge::TensorDesc bitcastOutput;
    if (bitcast.GetOutputDesc(0, bitcastOutput) != ge::GRAPH_SUCCESS ||
        bitcast.UpdateInputDesc(0, sourceDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    bitcastOutput.SetShape(sourceDesc.GetShape());
    bitcastOutput.SetOriginShape(sourceDesc.GetOriginShape());
    bitcastOutput.SetFormat(sourceDesc.GetFormat());
    bitcastOutput.SetOriginFormat(sourceDesc.GetOriginFormat());
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    if (sourceDesc.GetShapeRange(shapeRange) == ge::GRAPH_SUCCESS) {
        (void)bitcastOutput.SetShapeRange(shapeRange);
    }
    if (bitcast.UpdateOutputDesc(0, bitcastOutput) != ge::GRAPH_SUCCESS) {
        return false;
    }
    targetDesc = bitcastOutput;
    return true;
}

bool RelinkInput(ge::GraphPtr& graph, ge::GNode& target, int32_t inputIndex, const InputPattern& pattern)
{
    ge::GNode& destination = pattern.bitcast == nullptr ? target : *pattern.bitcast;
    const int32_t destinationPort = pattern.bitcast == nullptr ? inputIndex : 0;
    if (graph->RemoveEdge(*pattern.oldSource, 0, destination, destinationPort) != ge::GRAPH_SUCCESS ||
        graph->AddDataEdge(*pattern.source, pattern.sourcePort, destination, destinationPort) != ge::GRAPH_SUCCESS) {
        return false;
    }
    ge::TensorDesc sourceDesc;
    if (pattern.source->GetOutputDesc(pattern.sourcePort, sourceDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    ge::TensorDesc targetDesc = sourceDesc;
    if (pattern.bitcast != nullptr && !UpdateBitcast(*pattern.bitcast, sourceDesc, targetDesc)) {
        return false;
    }
    return target.UpdateInputDesc(inputIndex, targetDesc) == ge::GRAPH_SUCCESS;
}

bool RemoveNodeIfUnused(ge::GraphPtr& graph, const ge::GNodePtr& node)
{
    if (node == nullptr || !node->GetOutDataNodesAndPortIndexs(0).empty()) {
        return true;
    }
    std::vector<ge::GNodePtr> constNodes;
    for (std::size_t i = 0; i < node->GetInputsSize(); ++i) {
        int32_t sourcePort = 0;
        auto source = GetInputNode(*node, static_cast<int32_t>(i), &sourcePort);
        if (source == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*source, sourcePort, *node, static_cast<int32_t>(i)) != ge::GRAPH_SUCCESS) {
            return false;
        }
        if (IsType(source, kConstType)) {
            constNodes.emplace_back(source);
        }
    }
    if (graph->RemoveNode(*node) != ge::GRAPH_SUCCESS) {
        return false;
    }
    for (const auto& constNode : constNodes) {
        if (!RemoveNodeIfUnused(graph, constNode)) {
            return false;
        }
    }
    return true;
}

void ReportFusion(const std::vector<ge::GNode>& nodesBeforeFuse, const ge::GNode& target,
                  ge::CustomPassContext& passContext)
{
    const auto reportFuse = ge::fusion::GraphFuseInspectorUtils::ReportFuse;
    if (reportFuse == nullptr) {
        OP_LOGD(kPassName, "Skip reporting fusion result because GraphFuseInspector is unavailable.");
        return;
    }
    if (reportFuse(nodesBeforeFuse, {target}, passContext) != ge::SUCCESS) {
        OP_LOGW(kPassName, "Failed to report fusion result.");
    }
}

void AddNodeToReport(const ge::GNodePtr& node, std::vector<ge::GNode>& nodesBeforeFuse)
{
    if (node != nullptr) {
        nodesBeforeFuse.emplace_back(*node);
    }
}

ge::Status FuseNode(ge::GraphPtr& graph, ge::GNode& target, ge::CustomPassContext& passContext)
{
    if (!CheckTarget(target)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    InputPattern x1Pattern;
    if (!BuildInputPattern(target, kX1Index, x1Pattern) || !CheckPattern(x1Pattern, kX1Perm)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    InputPattern scale1Pattern;
    const bool isMx = IsMxMode(target);
    if (isMx &&
        (!BuildInputPattern(target, kScale1Index, scale1Pattern) || !CheckPattern(scale1Pattern, kScale1Perm))) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!RelinkInput(graph, target, kX1Index, x1Pattern) ||
        (isMx && !RelinkInput(graph, target, kScale1Index, scale1Pattern))) {
        return ge::GRAPH_FAILED;
    }
    std::vector<ge::GNode> nodesBeforeFuse;
    nodesBeforeFuse.emplace_back(target);
    AddNodeToReport(x1Pattern.transposeLike, nodesBeforeFuse);
    if (isMx) {
        AddNodeToReport(scale1Pattern.transposeLike, nodesBeforeFuse);
    }
    ReportFusion(nodesBeforeFuse, target, passContext);
    if (!RemoveNodeIfUnused(graph, x1Pattern.transposeLike) ||
        (isMx && !RemoveNodeIfUnused(graph, scale1Pattern.transposeLike))) {
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

ge::Status RunFusion(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OP_LOGD(kPassName, "Enter QuantGroupedMatmulInplaceAdd transpose fusion pass.");
    if (graph == nullptr || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    std::vector<ge::GNode> targets;
    for (auto& node : graph->GetDirectNode()) {
        if (IsType(node, kTargetType)) {
            targets.emplace_back(node);
        }
    }
    bool changed = false;
    for (auto& target : targets) {
        const auto status = FuseNode(graph, target, passContext);
        if (status == ge::SUCCESS) {
            OP_LOGD(kPassName, "QuantGroupedMatmulInplaceAdd transpose fusion succeeded.");
            changed = true;
        } else if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}
} // namespace

ge::Status QuantGroupedMatmulInplaceAddTransposeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    if (!IsTargetVersion()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    return RunFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(QuantGroupedMatmulInplaceAddTransposeFusionPass)
    .Stage(IsTargetVersion() ? ge::CustomPassStage::kCompatibleInherited : ge::CustomPassStage::kAfterInferShape);
#endif
} // namespace ops
#endif

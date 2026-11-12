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
 * \file grouped_matmul_swiglu_quant_transpose_fusion_pass.cpp
 * \brief Delete transpose-like nodes before GroupedMatmulSwigluQuantV2 weight inputs and set transpose_weight.
 */

#include "grouped_matmul_swiglu_quant_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "acl/acl_rt.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "securec.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace ge::fusion;

namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    // GE versions with ReportMatch account for structural matches separately from ReportFuse.
    static Status ReportMatch(const std::vector<GNode>& matchedNodes, CustomPassContext& ctx) __attribute__((weak));
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

constexpr char PASS_NAME[] = "GroupedMatmulSwigluQuantTransposeFusionPass";
constexpr char OP_TYPE_GMMSWIGLU[] = "GroupedMatmulSwigluQuantV2";
constexpr char OP_TYPE_TRANSPOSE[] = "Transpose";
constexpr char OP_TYPE_TRANSPOSED[] = "TransposeD";
constexpr char OP_TYPE_RESHAPE[] = "Reshape";
constexpr char OP_TYPE_BITCAST[] = "Bitcast";
constexpr char OP_TYPE_SHAPE[] = "Shape";
constexpr char ATTR_TRANSPOSE_WEIGHT[] = "transpose_weight";

constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_INDEX = 3;
constexpr int64_t WEIGHT_SCALE_INDEX = 4;
constexpr int64_t MINI_SHAPE_LEN = 2;
constexpr int64_t WEIGHT_MINI_SHAPE_LEN = 3;
constexpr size_t INDEX_THREE = 2;
constexpr int64_t DOUBLE_SHAPE = 2;
constexpr int64_t PERM_INPUT_INDEX = 1;
const std::vector<int64_t> PERM_WEIGHT = {0, 2, 1};
const std::vector<int64_t> PERM_WEIGHT_RESHAPE = {1, 0};
const std::vector<int64_t> PERM_WEIGHT_SCALE = {0, 2, 1, 3};
const std::vector<int64_t> PERM_WEIGHT_SCALE_RESHAPE = {1, 0, 2};

struct TransposeChain {
    GNodePtr transLikeNode = nullptr;
    GNodePtr sourceNode = nullptr;
    int64_t sourceOutputPort = 0;
    int64_t inputIndex = WEIGHT_INDEX;
    GNodePtr bitcastNode = nullptr;
    GNodePtr tailReshapeNode = nullptr;
    bool relinkToBitcast = false;
    bool specialFp4Path = false;
    std::vector<GNodePtr> nodesToRemove;
};

bool IsType(const GNodePtr& node, const char* type)
{
    if (node == nullptr) {
        return false;
    }
    AscendString nodeType;
    return node->GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsType(const GNode& node, const char* type)
{
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type;
}

bool IsTransposeType(const GNodePtr& node)
{
    return IsType(node, OP_TYPE_TRANSPOSE) || IsType(node, OP_TYPE_TRANSPOSED);
}

bool IsTransposeLikeType(const GNodePtr& node)
{
    return IsTransposeType(node) || IsType(node, OP_TYPE_RESHAPE);
}

GNodePtr GetInputNode(const GNode& dstNode, int64_t dstInputPort, int64_t* srcOutputPort = nullptr)
{
    auto [srcNode, resolvedSrcOutputPort] = dstNode.GetInDataNodesAndPortIndexs(dstInputPort);
    if (srcOutputPort != nullptr) {
        *srcOutputPort = resolvedSrcOutputPort;
    }
    return srcNode;
}

std::size_t GetOutputConsumerNum(const GNodePtr& node)
{
    if (node == nullptr) {
        return 0;
    }
    std::size_t count = 0;
    for (const auto& consumer : node->GetOutDataNodesAndPortIndexs(0)) {
        if (!IsType(consumer.first, OP_TYPE_SHAPE)) {
            ++count;
        }
    }
    return count;
}

bool IsUnknownDim(int64_t dim)
{
    return dim < 0;
}

bool IsSameOrUnknown(int64_t lhs, int64_t rhs)
{
    // During dynamic graph construction GE may resolve only one side of a
    // reshape descriptor.  A known mismatch is still rejected; an unknown
    // side is accepted because the concrete shape is resolved at runtime.
    return lhs == rhs || IsUnknownDim(lhs) || IsUnknownDim(rhs);
}

bool GetTransposePerm(const GNodePtr& transposeNode, std::vector<int64_t>& perm)
{
    if (IsType(transposeNode, OP_TYPE_TRANSPOSED)) {
        return transposeNode->GetAttr("perm", perm) == GRAPH_SUCCESS;
    }
    if (!IsType(transposeNode, OP_TYPE_TRANSPOSE)) {
        return false;
    }
    Tensor permTensor;
    TensorDesc permDesc;
    if (transposeNode->GetInputConstData(PERM_INPUT_INDEX, permTensor) != GRAPH_SUCCESS ||
        transposeNode->GetInputDesc(PERM_INPUT_INDEX, permDesc) != GRAPH_SUCCESS || permTensor.GetData() == nullptr) {
        return false;
    }
    const auto* data = permTensor.GetData();
    if (permDesc.GetDataType() == DT_INT32) {
        const auto size = permTensor.GetSize() / sizeof(int32_t);
        for (std::size_t i = 0; i < size; ++i) {
            int32_t value = 0;
            if (memcpy_s(&value, sizeof(value), data + i * sizeof(value), sizeof(value)) != EOK) {
                return false;
            }
            perm.emplace_back(value);
        }
        return true;
    }
    if (permDesc.GetDataType() == DT_INT64) {
        const auto size = permTensor.GetSize() / sizeof(int64_t);
        for (std::size_t i = 0; i < size; ++i) {
            int64_t value = 0;
            if (memcpy_s(&value, sizeof(value), data + i * sizeof(value), sizeof(value)) != EOK) {
                return false;
            }
            perm.emplace_back(value);
        }
        return true;
    }
    return false;
}

std::vector<int64_t> GetExpectedPerm(int64_t inputIndex, bool reshapePattern)
{
    if (inputIndex == WEIGHT_SCALE_INDEX) {
        return reshapePattern ? PERM_WEIGHT_SCALE_RESHAPE : PERM_WEIGHT_SCALE;
    }
    return reshapePattern ? PERM_WEIGHT_RESHAPE : PERM_WEIGHT;
}

bool CheckTransposeNode(const GNodePtr& transposeNode, int64_t inputIndex, bool reshapePattern)
{
    std::vector<int64_t> perm;
    return GetOutputConsumerNum(transposeNode) == 1 && GetTransposePerm(transposeNode, perm) &&
           perm == GetExpectedPerm(inputIndex, reshapePattern);
}

bool GetShapeDims(const GNodePtr& node, bool input, std::vector<int64_t>& dims)
{
    TensorDesc desc;
    const auto status = input ? node->GetInputDesc(0, desc) : node->GetOutputDesc(0, desc);
    if (status != GRAPH_SUCCESS) {
        return false;
    }
    dims = desc.GetShape().GetDims();
    return true;
}

bool IsExpectedSwap(const std::vector<int64_t>& inputDims, const std::vector<int64_t>& outputDims, int64_t inputIndex,
                    bool requireUnitAxis)
{
    if (inputDims.size() < 3) {
        return false;
    }
    std::size_t firstAxis = inputDims.size() - 2;
    std::size_t secondAxis = inputDims.size() - 1;
    if (inputIndex == WEIGHT_SCALE_INDEX) {
        if (inputDims.size() != 4) {
            return false;
        }
        firstAxis = 1;
        secondAxis = 2;
    }
    if (requireUnitAxis && inputDims[firstAxis] != 1 && inputDims[secondAxis] != 1) {
        return false;
    }
    // A dynamic Reshape may not have an output shape descriptor yet.  The
    // known unit axis and rank are sufficient to identify the view-generated
    // transpose; ordinary Reshape inputs without that proof remain rejected.
    if (outputDims.empty()) {
        return requireUnitAxis;
    }
    if (inputDims.size() != outputDims.size()) {
        return false;
    }
    auto expectedDims = inputDims;
    std::swap(expectedDims[firstAxis], expectedDims[secondAxis]);
    for (std::size_t i = 0; i < expectedDims.size(); ++i) {
        if (!IsSameOrUnknown(expectedDims[i], outputDims[i])) {
            return false;
        }
    }
    return true;
}

bool CheckDirectReshape(const GNodePtr& reshapeNode, int64_t inputIndex)
{
    TensorDesc inputDesc;
    TensorDesc outputDesc;
    if (reshapeNode->GetInputDesc(0, inputDesc) != GRAPH_SUCCESS ||
        reshapeNode->GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS ||
        inputDesc.GetDataType() != outputDesc.GetDataType()) {
        return false;
    }
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
    return GetOutputConsumerNum(reshapeNode) == 1 && GetShapeDims(reshapeNode, true, inputDims) &&
           GetShapeDims(reshapeNode, false, outputDims) && IsExpectedSwap(inputDims, outputDims, inputIndex, true);
}

bool IsLegalDataType(DataType dtype, const std::vector<DataType>& legalDtypes)
{
    return std::find(legalDtypes.begin(), legalDtypes.end(), dtype) != legalDtypes.end();
}

bool IsSpecialFp4Path(const GNodePtr& node, int64_t inputIndex)
{
    if (inputIndex != WEIGHT_INDEX || !IsType(node, OP_TYPE_RESHAPE)) {
        return false;
    }
    TensorDesc outputDesc;
    if (node->GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    auto dtype = outputDesc.GetDataType();
    return dtype == DT_FLOAT4_E2M1 || dtype == DT_HIFLOAT8;
}

bool CheckGroupedMatmulSwigluQuantV2(const GNode& nodeGmmSwiglu)
{
    return IsType(nodeGmmSwiglu, OP_TYPE_GMMSWIGLU);
}

bool CheckNodeDtype(const GNode& nodeGmmSwiglu)
{
    // SwiGLU is followed by quantization: FP8/FP4/HIFLOAT8 outputs are
    // intentional. This fusion whitelist is not a dtype-combination checker.
    static const std::vector<DataType> legalFusionDtypes = {DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_INT8,
                                                            DT_HIFLOAT8};

    TensorDesc xDesc;
    TensorDesc weightDesc;
    TensorDesc outputDesc;
    if (nodeGmmSwiglu.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS ||
        nodeGmmSwiglu.GetInputDesc(WEIGHT_INDEX, weightDesc) != GRAPH_SUCCESS ||
        nodeGmmSwiglu.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get GroupedMatmulSwigluQuantV2 input or output desc.");
        return false;
    }
    if (!IsLegalDataType(xDesc.GetDataType(), legalFusionDtypes)) {
        OP_LOGW(PASS_NAME, "X dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(xDesc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(weightDesc.GetDataType(), legalFusionDtypes)) {
        OP_LOGW(PASS_NAME, "Weight dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(weightDesc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(outputDesc.GetDataType(), legalFusionDtypes)) {
        OP_LOGW(PASS_NAME, "Output dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(outputDesc.GetDataType()).c_str());
        return false;
    }
    return true;
}

bool CheckNodeShape(const GNode& nodeGmmSwiglu)
{
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (nodeGmmSwiglu.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS ||
        nodeGmmSwiglu.GetInputDesc(WEIGHT_INDEX, weightDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get shape desc before fusion.");
        return false;
    }
    const auto xRank = xDesc.GetOriginShape().GetDimNum();
    const auto weightRank = weightDesc.GetOriginShape().GetDimNum();
    if ((xRank != 0 && xRank < MINI_SHAPE_LEN) || (weightRank != 0 && weightRank < WEIGHT_MINI_SHAPE_LEN)) {
        OP_LOGW(PASS_NAME, "X dim num or weight dim num does not support transpose fusion.");
        return false;
    }
    return true;
}

bool IsMxWeightScale(const GNode& nodeGmmSwiglu)
{
    TensorDesc weightScaleDesc;
    return nodeGmmSwiglu.GetInputDesc(WEIGHT_SCALE_INDEX, weightScaleDesc) == GRAPH_SUCCESS &&
           weightScaleDesc.GetDataType() == DT_FLOAT8_E8M0;
}

bool FillChainFromSource(TransposeChain& chain, const GNodePtr& transLikeNode)
{
    int64_t srcOutputPort = 0;
    auto sourceNode = GetInputNode(*transLikeNode, 0, &srcOutputPort);
    if (sourceNode == nullptr) {
        OP_LOGD(PASS_NAME, "Failed to get source node before transpose-like node.");
        return false;
    }
    chain.transLikeNode = transLikeNode;
    chain.sourceNode = sourceNode;
    chain.sourceOutputPort = srcOutputPort;
    return true;
}

bool TryResolveTwoReshapeChain(const GNode& nodeGmmSwiglu, int64_t inputIndex, TransposeChain& chain)
{
    auto bottomNode = GetInputNode(nodeGmmSwiglu, inputIndex);
    if (bottomNode == nullptr) {
        return false;
    }
    if (IsSpecialFp4Path(bottomNode, inputIndex)) {
        chain.specialFp4Path = true;
        chain.tailReshapeNode = bottomNode;
        bottomNode = GetInputNode(*bottomNode, 0);
    }
    if (IsType(bottomNode, OP_TYPE_BITCAST)) {
        chain.relinkToBitcast = true;
        chain.bitcastNode = bottomNode;
        bottomNode = GetInputNode(*bottomNode, 0);
    }
    if (!IsType(bottomNode, OP_TYPE_RESHAPE)) {
        return false;
    }
    auto transposeNode = GetInputNode(*bottomNode, 0);
    if (!IsTransposeType(transposeNode)) {
        return false;
    }
    auto topReshapeNode = GetInputNode(*transposeNode, 0);
    if (!IsType(topReshapeNode, OP_TYPE_RESHAPE)) {
        return false;
    }
    int64_t srcOutputPort = 0;
    auto sourceNode = GetInputNode(*topReshapeNode, 0, &srcOutputPort);
    if (sourceNode == nullptr) {
        return false;
    }

    TensorDesc sourceDesc;
    TensorDesc targetDesc;
    if (sourceNode->GetOutputDesc(srcOutputPort, sourceDesc) != GRAPH_SUCCESS ||
        bottomNode->GetOutputDesc(0, targetDesc) != GRAPH_SUCCESS ||
        !IsExpectedSwap(sourceDesc.GetShape().GetDims(), targetDesc.GetShape().GetDims(), inputIndex, false) ||
        !CheckTransposeNode(transposeNode, inputIndex, true) || GetOutputConsumerNum(topReshapeNode) != 1 ||
        GetOutputConsumerNum(bottomNode) != 1 ||
        (chain.bitcastNode != nullptr && GetOutputConsumerNum(chain.bitcastNode) != 1) ||
        (chain.tailReshapeNode != nullptr && GetOutputConsumerNum(chain.tailReshapeNode) != 1)) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.transLikeNode = bottomNode;
    chain.sourceNode = sourceNode;
    chain.sourceOutputPort = srcOutputPort;
    chain.nodesToRemove = {bottomNode, transposeNode, topReshapeNode};
    return true;
}

bool TryResolveDirectChain(const GNode& nodeGmmSwiglu, int64_t inputIndex, TransposeChain& chain)
{
    auto transLikeNode = GetInputNode(nodeGmmSwiglu, inputIndex);
    if (transLikeNode == nullptr) {
        return false;
    }
    // A logical HIFLOAT8/FP4 unit-axis view can itself be the transpose.
    // Do not mistake it for a retained packing tail merely because of its dtype.
    if (IsSpecialFp4Path(transLikeNode, inputIndex) && !CheckDirectReshape(transLikeNode, inputIndex)) {
        chain.specialFp4Path = true;
        chain.tailReshapeNode = transLikeNode;
        transLikeNode = GetInputNode(*transLikeNode, 0);
    }
    if (IsType(transLikeNode, OP_TYPE_BITCAST)) {
        chain.relinkToBitcast = true;
        chain.bitcastNode = transLikeNode;
        transLikeNode = GetInputNode(*transLikeNode, 0);
    }
    if (!IsTransposeLikeType(transLikeNode) || !FillChainFromSource(chain, transLikeNode)) {
        return false;
    }
    if ((IsTransposeType(transLikeNode) && !CheckTransposeNode(transLikeNode, inputIndex, false)) ||
        (IsType(transLikeNode, OP_TYPE_RESHAPE) && !CheckDirectReshape(transLikeNode, inputIndex)) ||
        (chain.bitcastNode != nullptr && GetOutputConsumerNum(chain.bitcastNode) != 1) ||
        (chain.tailReshapeNode != nullptr && GetOutputConsumerNum(chain.tailReshapeNode) != 1)) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.nodesToRemove = {transLikeNode};
    return true;
}

bool ResolveTransposeChain(const GNode& nodeGmmSwiglu, int64_t inputIndex, TransposeChain& chain)
{
    if (TryResolveTwoReshapeChain(nodeGmmSwiglu, inputIndex, chain)) {
        return true;
    }
    chain = TransposeChain();
    return TryResolveDirectChain(nodeGmmSwiglu, inputIndex, chain);
}

bool CheckChainForCommit(const TransposeChain& chain)
{
    GNodePtr tailShapeNode = nullptr;
    if (chain.specialFp4Path) {
        TensorDesc sourceDesc;
        TensorDesc tailDesc;
        if (chain.tailReshapeNode == nullptr ||
            chain.sourceNode->GetOutputDesc(chain.sourceOutputPort, sourceDesc) != GRAPH_SUCCESS ||
            chain.tailReshapeNode->GetOutputDesc(0, tailDesc) != GRAPH_SUCCESS ||
            sourceDesc.GetShape().GetDimNum() != 3) {
            return false;
        }
        // Without a Bitcast the source already has the logical element type;
        // a Reshape alone cannot reinterpret byte carriers as FP4 values.
        if (chain.bitcastNode == nullptr) {
            if (sourceDesc.GetDataType() != tailDesc.GetDataType() ||
                !IsExpectedSwap(sourceDesc.GetShape().GetDims(), tailDesc.GetShape().GetDims(), chain.inputIndex,
                                false)) {
                return false;
            }
        }
        auto shapeProducer = GetInputNode(*chain.tailReshapeNode, 1);
        if (IsType(shapeProducer, OP_TYPE_SHAPE)) {
            tailShapeNode = shapeProducer;
        } else if (shapeProducer != nullptr) {
            tailShapeNode = GetInputNode(*shapeProducer, 0);
        }
    }
    // Only a Shape that builds the retained tail Reshape can follow the data
    // bypass. An unrelated shape user must keep its original layout.
    for (const auto& consumer : chain.transLikeNode->GetOutDataNodesAndPortIndexs(0)) {
        if (!IsType(consumer.first, OP_TYPE_SHAPE)) {
            continue;
        }
        AscendString actualName;
        AscendString expectedName;
        if (!IsType(tailShapeNode, OP_TYPE_SHAPE) || consumer.first->GetName(actualName) != GRAPH_SUCCESS ||
            tailShapeNode->GetName(expectedName) != GRAPH_SUCCESS || actualName != expectedName) {
            return false;
        }
    }
    return true;
}

bool RemoveNode(const GraphPtr& graph, const GNodePtr& dstNode)
{
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid when removing node.");
        return false;
    }
    if (dstNode == nullptr) {
        return true;
    }
    if (!dstNode->GetOutDataNodesAndPortIndexs(0).empty()) {
        OP_LOGW(PASS_NAME, "Skip removing transpose-like node: output consumers still exist; keep the node in graph.");
        return true;
    }

    const auto inputSize = dstNode->GetInputsSize();
    for (size_t dstInputPort = 0; dstInputPort < inputSize; ++dstInputPort) {
        int64_t srcOutputPort = 0;
        auto srcNode = GetInputNode(*dstNode, static_cast<int64_t>(dstInputPort), &srcOutputPort);
        if (srcNode == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*srcNode, srcOutputPort, *dstNode, dstInputPort) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to remove input edge before removing transpose-like node.");
            return false;
        }
    }
    if (graph->RemoveNode(*dstNode) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove unused transpose-like node.");
        return false;
    }
    return true;
}

bool UpdateSpecialFp4Desc(GNode& nodeGmmSwiglu, const TransposeChain& chain, int64_t inputIndex,
                          const TensorDesc& sourceDesc)
{
    if (chain.tailReshapeNode == nullptr) {
        OP_LOGW(PASS_NAME, "Special FP4 path misses tail Reshape node.");
        return false;
    }
    auto dims = sourceDesc.GetShape().GetDims();
    if (dims.size() <= INDEX_THREE) {
        OP_LOGW(PASS_NAME, "Special FP4 path source shape dims are too small.");
        return false;
    }

    auto tailInputDesc = sourceDesc;
    if (chain.bitcastNode != nullptr) {
        if (chain.bitcastNode->UpdateInputDesc(0, sourceDesc) != GRAPH_SUCCESS ||
            chain.bitcastNode->GetOutputDesc(0, tailInputDesc) != GRAPH_SUCCESS) {
            return false;
        }
        const bool unpackFp4 = tailInputDesc.GetDataType() == DT_FLOAT4_E2M1;
        const Shape bitcastShape =
            unpackFp4 ? Shape({dims[0], dims[1], dims[INDEX_THREE], DOUBLE_SHAPE}) : sourceDesc.GetShape();
        tailInputDesc.SetShape(bitcastShape);
        tailInputDesc.SetOriginShape(bitcastShape);
        if (chain.bitcastNode->UpdateOutputDesc(0, tailInputDesc) != GRAPH_SUCCESS) {
            return false;
        }
        if (unpackFp4 && !IsUnknownDim(dims[INDEX_THREE])) {
            dims[INDEX_THREE] *= DOUBLE_SHAPE;
        }
    }
    if (chain.tailReshapeNode->UpdateInputDesc(0, tailInputDesc) != GRAPH_SUCCESS) {
        return false;
    }

    TensorDesc tailReshapeOutputDesc;
    if (chain.tailReshapeNode->GetOutputDesc(0, tailReshapeOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get tail Reshape output desc.");
        return false;
    }
    Shape reshapeShape(dims);
    tailReshapeOutputDesc.SetShape(reshapeShape);
    tailReshapeOutputDesc.SetOriginShape(reshapeShape);
    if (chain.tailReshapeNode->UpdateOutputDesc(0, tailReshapeOutputDesc) != GRAPH_SUCCESS ||
        nodeGmmSwiglu.UpdateInputDesc(inputIndex, tailReshapeOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulSwigluQuantV2 desc for special FP4 path.");
        return false;
    }
    return true;
}

bool UpdateInputDesc(GNode& nodeGmmSwiglu, const TransposeChain& chain, int64_t inputIndex,
                     const TensorDesc& sourceDesc)
{
    if (chain.specialFp4Path) {
        return UpdateSpecialFp4Desc(nodeGmmSwiglu, chain, inputIndex, sourceDesc);
    }
    if (!chain.relinkToBitcast) {
        if (nodeGmmSwiglu.UpdateInputDesc(inputIndex, sourceDesc) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulSwigluQuantV2 input desc.");
            return false;
        }
        return true;
    }
    if (chain.bitcastNode == nullptr) {
        OP_LOGW(PASS_NAME, "Bitcast node is null when updating desc.");
        return false;
    }
    TensorDesc gmmInputDesc;
    if (nodeGmmSwiglu.GetInputDesc(inputIndex, gmmInputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get GroupedMatmulSwigluQuantV2 input desc.");
        return false;
    }
    if (chain.bitcastNode->UpdateInputDesc(0, sourceDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast input desc.");
        return false;
    }

    TensorDesc bitcastOutputDesc;
    if (chain.bitcastNode->GetOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get Bitcast output desc.");
        return false;
    }
    bitcastOutputDesc.SetShape(sourceDesc.GetShape());
    bitcastOutputDesc.SetOriginShape(sourceDesc.GetOriginShape());
    if (chain.bitcastNode->UpdateOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast output desc.");
        return false;
    }
    bitcastOutputDesc.SetDataType(gmmInputDesc.GetDataType());
    if (nodeGmmSwiglu.UpdateInputDesc(inputIndex, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulSwigluQuantV2 input desc from Bitcast.");
        return false;
    }
    return true;
}

bool RelinkNode(const GraphPtr& graph, GNode& nodeGmmSwiglu, const TransposeChain& chain, int64_t inputIndex)
{
    if (chain.transLikeNode == nullptr || chain.sourceNode == nullptr) {
        OP_LOGW(PASS_NAME, "Transpose-like chain is invalid.");
        return false;
    }
    TensorDesc sourceDesc;
    if (chain.sourceNode->GetOutputDesc(chain.sourceOutputPort, sourceDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get source output desc.");
        return false;
    }

    auto& dstNode = chain.bitcastNode != nullptr ?
                        *chain.bitcastNode :
                        (chain.tailReshapeNode != nullptr ? *chain.tailReshapeNode : nodeGmmSwiglu);
    const int64_t dstInputPort = chain.bitcastNode != nullptr || chain.tailReshapeNode != nullptr ? 0 : inputIndex;
    if (graph->RemoveEdge(*chain.transLikeNode, 0, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove edge from transpose-like node.");
        return false;
    }
    if (graph->AddDataEdge(*chain.sourceNode, chain.sourceOutputPort, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to add edge from source node.");
        return false;
    }
    if (chain.specialFp4Path) {
        // The tail Reshape can derive its runtime shape from the view being
        // bypassed. Update that dependency together with the data edge.
        for (const auto& consumer : chain.transLikeNode->GetOutDataNodesAndPortIndexs(0)) {
            if (!IsType(consumer.first, OP_TYPE_SHAPE)) {
                continue;
            }
            if (graph->RemoveEdge(*chain.transLikeNode, 0, *consumer.first, consumer.second) != GRAPH_SUCCESS ||
                graph->AddDataEdge(*chain.sourceNode, chain.sourceOutputPort, *consumer.first, consumer.second) !=
                    GRAPH_SUCCESS ||
                consumer.first->UpdateInputDesc(consumer.second, sourceDesc) != GRAPH_SUCCESS) {
                return false;
            }
        }
    }
    return UpdateInputDesc(nodeGmmSwiglu, chain, inputIndex, sourceDesc);
}

void AddNodesToReport(const GNode& nodeGmmSwiglu, const std::vector<TransposeChain>& chains,
                      std::vector<GNode>& nodesBeforeFuse, std::vector<GNodePtr>& nodesToRemove)
{
    nodesBeforeFuse = {nodeGmmSwiglu};
    nodesToRemove.clear();
    for (const auto& chain : chains) {
        for (const auto& node : chain.nodesToRemove) {
            if (node == nullptr || std::find(nodesToRemove.begin(), nodesToRemove.end(), node) != nodesToRemove.end()) {
                continue;
            }
            nodesBeforeFuse.emplace_back(*node);
            nodesToRemove.emplace_back(node);
        }
    }
}

void ReportTransposeFusion(const std::vector<GNode>& nodesBeforeFuse, const GNode& nodeGmmSwiglu,
                           CustomPassContext& passContext)
{
    auto reportFuse = GraphFuseInspectorUtils::ReportFuse;
    if (reportFuse == nullptr) {
        return;
    }
    if (reportFuse(nodesBeforeFuse, {nodeGmmSwiglu}, passContext) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to report fusion result.");
    }
}

bool RemoveFusionNodes(const GraphPtr& graph, const std::vector<GNodePtr>& nodesToRemove)
{
    for (const auto& nodeToRemove : nodesToRemove) {
        if (!RemoveNode(graph, nodeToRemove)) {
            return false;
        }
    }
    return true;
}

Status PrepareFusion(const GNode& nodeGmmSwiglu, std::vector<TransposeChain>& chains)
{
    if (!CheckGroupedMatmulSwigluQuantV2(nodeGmmSwiglu)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckNodeDtype(nodeGmmSwiglu) || !CheckNodeShape(nodeGmmSwiglu)) {
        return GRAPH_NOT_CHANGED;
    }
    bool transposeWeight = false;
    if (nodeGmmSwiglu.GetAttr(ATTR_TRANSPOSE_WEIGHT, transposeWeight) != GRAPH_SUCCESS || transposeWeight) {
        return GRAPH_NOT_CHANGED;
    }

    TransposeChain weightChain;
    if (!ResolveTransposeChain(nodeGmmSwiglu, WEIGHT_INDEX, weightChain) || !CheckChainForCommit(weightChain)) {
        return GRAPH_NOT_CHANGED;
    }
    chains.emplace_back(weightChain);

    if (IsMxWeightScale(nodeGmmSwiglu)) {
        TransposeChain weightScaleChain;
        if (!ResolveTransposeChain(nodeGmmSwiglu, WEIGHT_SCALE_INDEX, weightScaleChain) ||
            !CheckChainForCommit(weightScaleChain)) {
            return GRAPH_NOT_CHANGED;
        }
        chains.emplace_back(weightScaleChain);
    }
    return SUCCESS;
}

Status CommitFusion(const GraphPtr& graph, GNode& nodeGmmSwiglu, const std::vector<TransposeChain>& chains,
                    CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse;
    std::vector<GNodePtr> nodesToRemove;
    AddNodesToReport(nodeGmmSwiglu, chains, nodesBeforeFuse, nodesToRemove);
    const auto reportMatch = GraphFuseInspectorUtils::ReportMatch;
    if (reportMatch != nullptr && reportMatch(nodesBeforeFuse, passContext) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to report fusion match.");
    }

    for (const auto& chain : chains) {
        if (!RelinkNode(graph, nodeGmmSwiglu, chain, chain.inputIndex)) {
            OP_LOGE(PASS_NAME,
                    "Failed to relink input %ld; graph may be partially modified. Abort fusion without rollback.",
                    chain.inputIndex);
            return GRAPH_FAILED;
        }
    }
    bool transposeWeight = true;
    if (nodeGmmSwiglu.SetAttr(ATTR_TRANSPOSE_WEIGHT, transposeWeight) != GRAPH_SUCCESS) {
        OP_LOGE(
            PASS_NAME,
            "Failed to set transpose_weight after relinking inputs; graph is modified. Abort fusion without rollback.");
        return GRAPH_FAILED;
    }
    ReportTransposeFusion(nodesBeforeFuse, nodeGmmSwiglu, passContext);
    if (!RemoveFusionNodes(graph, nodesToRemove)) {
        OP_LOGE(
            PASS_NAME,
            "Failed to remove fusion nodes after relinking inputs; graph is modified. Abort fusion without rollback.");
        return GRAPH_FAILED;
    }
    return SUCCESS;
}

Status Fusion(const GraphPtr& graph, GNode& nodeGmmSwiglu, CustomPassContext& passContext)
{
    std::vector<TransposeChain> chains;
    const auto status = PrepareFusion(nodeGmmSwiglu, chains);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, nodeGmmSwiglu, chains, passContext);
}

Status RunGroupedMatmulSwigluQuantTransposeFusion(const GraphPtr& graph, CustomPassContext& passContext)
{
    OP_LOGD(PASS_NAME, "Enter GroupedMatmulSwigluQuantV2 transpose fusion pass.");
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> nodeGmmSwigluList;
    for (auto& node : graph->GetDirectNode()) {
        if (CheckGroupedMatmulSwigluQuantV2(node)) {
            nodeGmmSwigluList.emplace_back(node);
        }
    }
    if (nodeGmmSwigluList.empty()) {
        OP_LOGD(PASS_NAME, "No GroupedMatmulSwigluQuantV2 node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    bool changed = false;
    for (auto& nodeGmmSwiglu : nodeGmmSwigluList) {
        auto status = Fusion(graph, nodeGmmSwiglu, passContext);
        if (status == SUCCESS) {
            OP_LOGD(PASS_NAME, "GroupedMatmulSwigluQuantV2 transpose fusion succeeded.");
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status GroupedMatmulSwigluQuantTransposeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }
    return RunGroupedMatmulSwigluQuantTransposeFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(GroupedMatmulSwigluQuantTransposeFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif
} // namespace ops
#endif // CANN_VERSION_NUM

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
 * \file grouped_matmul_finalize_routing_transpose_fusion_pass.cpp
 * \brief Delete transpose-like nodes before GroupedMatmulFinalizeRouting weight inputs and set transpose_w.
 */

#include "grouped_matmul_finalize_routing_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "acl/acl_rt.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "../../../../common/include/op_host/op_const_def_transformer.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "securec.h"
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

constexpr char PASS_NAME[] = "GroupedMatmulFinalizeRoutingTransposeFusionPass";
constexpr char OP_TYPE_GMMFR[] = "GroupedMatmulFinalizeRouting";
constexpr char OP_TYPE_TRANSPOSE[] = "Transpose";
constexpr char OP_TYPE_TRANSPOSED[] = "TransposeD";
constexpr char OP_TYPE_RESHAPE[] = "Reshape";
constexpr char OP_TYPE_BITCAST[] = "Bitcast";
constexpr char OP_TYPE_SHAPE[] = "Shape";
constexpr char OP_TYPE_GATHER[] = "Gather";
constexpr char OP_TYPE_PACK[] = "Pack";
constexpr char ATTR_TRANSPOSE_WEIGHT[] = "transpose_w";
constexpr char ASCEND_950_SOC_VERSION[] = "Ascend950";
constexpr char ASCEND_350_SOC_VERSION[] = "Ascend350";
constexpr char ASCEND_960_DT_SOC_VERSION[] = "Ascend960DT";

constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_INDEX = 1;
constexpr int64_t WEIGHT_SCALE_INDEX = 2;
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
    bool twoReshapePattern = false;
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

std::size_t GetDataConsumerNum(const GNodePtr& node)
{
    if (node == nullptr) {
        return 0;
    }
    std::size_t count = 0;
    for (const auto& consumer : node->GetOutDataNodesAndPortIndexs(0)) {
        // Shape-only users are generated by dynamic view lowering to build
        // the following Reshape descriptor.  They do not consume the data
        // value and must not prevent removing the transpose-like view.
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
    return GetDataConsumerNum(transposeNode) == 1 && GetTransposePerm(transposeNode, perm) &&
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

bool GetConstScalar(const GNode& node, int64_t inputIndex, int64_t& value)
{
    Tensor tensor;
    TensorDesc desc;
    if (node.GetInputConstData(inputIndex, tensor) != GRAPH_SUCCESS ||
        node.GetInputDesc(inputIndex, desc) != GRAPH_SUCCESS || tensor.GetData() == nullptr) {
        return false;
    }
    if (desc.GetDataType() == DT_INT32 && tensor.GetSize() >= sizeof(int32_t)) {
        int32_t int32Value = 0;
        if (memcpy_s(&int32Value, sizeof(int32Value), tensor.GetData(), sizeof(int32_t)) != EOK) {
            return false;
        }
        value = static_cast<int64_t>(int32Value);
        return true;
    }
    if (desc.GetDataType() == DT_INT64 && tensor.GetSize() >= sizeof(int64_t)) {
        if (memcpy_s(&value, sizeof(value), tensor.GetData(), sizeof(int64_t)) != EOK) {
            return false;
        }
        return true;
    }
    return false;
}

bool IsSameNode(const GNodePtr& lhs, const GNodePtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    AscendString lhsName;
    AscendString rhsName;
    return lhs->GetName(lhsName) == GRAPH_SUCCESS && rhs->GetName(rhsName) == GRAPH_SUCCESS &&
           lhsName.GetString() != nullptr && rhsName.GetString() != nullptr &&
           std::string(lhsName.GetString()) == rhsName.GetString();
}

bool IsGatherOfSourceDim(const GNodePtr& gatherNode, const GNodePtr& sourceNode, int64_t dimIndex)
{
    if (!IsType(gatherNode, OP_TYPE_GATHER)) {
        return false;
    }
    auto shapeNode = GetInputNode(*gatherNode, 0);
    auto shapeSourceNode = shapeNode == nullptr ? nullptr : GetInputNode(*shapeNode, 0);
    int64_t actualDimIndex = 0;
    return IsType(shapeNode, OP_TYPE_SHAPE) && IsSameNode(shapeSourceNode, sourceNode) &&
           GetConstScalar(*gatherNode, 1, actualDimIndex) && actualDimIndex == dimIndex;
}

GNodePtr GetBaseViewSource(const GNodePtr& node)
{
    auto current = node;
    // Dynamic view lowering can add a small chain of Reshape/Transpose/
    // Bitcast nodes between the data input and the node used by Shape/Gather.
    // Follow only these view operations; stopping at any other op keeps the
    // equivalence proof tied to the same producer tensor.
    for (int32_t depth = 0; current != nullptr && depth < 8; ++depth) {
        if (!IsType(current, OP_TYPE_RESHAPE) && !IsTransposeType(current) && !IsType(current, OP_TYPE_BITCAST)) {
            break;
        }
        current = GetInputNode(*current, 0);
    }
    return current;
}

GNodePtr GetShapePackSource(const GNodePtr& packNode)
{
    if (!IsType(packNode, OP_TYPE_PACK)) {
        return nullptr;
    }
    for (int64_t inputIndex = 0; inputIndex < static_cast<int64_t>(packNode->GetInputsSize()); ++inputIndex) {
        auto shapeInput = GetInputNode(*packNode, inputIndex);
        if (!IsType(shapeInput, OP_TYPE_GATHER)) {
            continue;
        }
        auto shapeNode = GetInputNode(*shapeInput, 0);
        if (IsType(shapeNode, OP_TYPE_SHAPE)) {
            return GetInputNode(*shapeNode, 0);
        }
    }
    return nullptr;
}

bool IsPackAxisSourceDim(const GNodePtr& packNode, std::size_t outputAxis, const GNodePtr& sourceNode,
                         int64_t sourceAxis, const std::vector<int64_t>& sourceDims)
{
    auto shapeInput = GetInputNode(*packNode, static_cast<int64_t>(outputAxis));
    if (IsGatherOfSourceDim(shapeInput, sourceNode, sourceAxis)) {
        return true;
    }
    // TorchAir emits a Const directly into Pack for dimensions that are
    // statically known, while dynamic dimensions are represented by
    // Gather(Shape(source), axis).  Accept the former only when it exactly
    // matches the corresponding source dimension; this keeps the dynamic
    // proof structural and prevents an unrelated constant reshape from
    // being treated as an axis swap.
    if (sourceAxis < 0 || static_cast<std::size_t>(sourceAxis) >= sourceDims.size() ||
        IsUnknownDim(sourceDims[static_cast<std::size_t>(sourceAxis)])) {
        return false;
    }
    int64_t constantDim = 0;
    return GetConstScalar(*packNode, static_cast<int64_t>(outputAxis), constantDim) &&
           constantDim == sourceDims[static_cast<std::size_t>(sourceAxis)];
}

bool IsShapePackDerivedFromSource(const GNodePtr& reshapeNode, const GNodePtr& sourceNode, int64_t sourceOutputPort)
{
    if (!IsType(reshapeNode, OP_TYPE_RESHAPE) || sourceNode == nullptr || GetDataConsumerNum(reshapeNode) != 1) {
        return false;
    }
    auto packNode = GetInputNode(*reshapeNode, 1);
    if (!IsType(packNode, OP_TYPE_PACK) || packNode->GetInputsSize() == 0) {
        return false;
    }

    TensorDesc sourceDesc;
    if (sourceNode->GetOutputDesc(sourceOutputPort, sourceDesc) != GRAPH_SUCCESS) {
        return false;
    }
    const auto sourceDims = sourceDesc.GetShape().GetDims();
    bool hasGather = false;
    for (std::size_t outputAxis = 0; outputAxis < packNode->GetInputsSize(); ++outputAxis) {
        auto shapeInput = GetInputNode(*packNode, static_cast<int64_t>(outputAxis));
        bool isSourceDim = false;
        if (IsType(shapeInput, OP_TYPE_GATHER)) {
            int64_t sourceAxis = 0;
            if (GetConstScalar(*shapeInput, 1, sourceAxis) && sourceAxis >= 0 &&
                static_cast<std::size_t>(sourceAxis) < sourceDims.size() &&
                IsGatherOfSourceDim(shapeInput, sourceNode, sourceAxis)) {
                isSourceDim = true;
                hasGather = true;
            }
        }
        if (isSourceDim) {
            continue;
        }

        // A unit constant is used by TorchAir when a dynamic view drops or
        // restores a statically known leading axis.  Other constants are
        // accepted only when they match a known source dimension.
        int64_t constantDim = 0;
        if (!GetConstScalar(*packNode, static_cast<int64_t>(outputAxis), constantDim)) {
            return false;
        }
        const bool matchesSourceDim = std::any_of(sourceDims.begin(), sourceDims.end(), [constantDim](int64_t dim) {
            return !IsUnknownDim(dim) && dim == constantDim;
        });
        if (constantDim != 1 && !matchesSourceDim) {
            return false;
        }
    }
    return hasGather;
}

bool IsDynamicAxisSwapReshape(const GNodePtr& reshapeNode, int64_t inputIndex)
{
    if (!IsType(reshapeNode, OP_TYPE_RESHAPE)) {
        return false;
    }
    auto dataSourceNode = GetInputNode(*reshapeNode, 0);
    auto packNode = GetInputNode(*reshapeNode, 1);
    if (dataSourceNode == nullptr || !IsType(packNode, OP_TYPE_PACK)) {
        return false;
    }
    auto shapeSourceNode = GetShapePackSource(packNode);
    auto sourceNode = GetBaseViewSource(dataSourceNode);
    if (shapeSourceNode == nullptr || sourceNode == nullptr || !IsSameNode(sourceNode, shapeSourceNode)) {
        return false;
    }
    TensorDesc inputDesc;
    if (shapeSourceNode->GetOutputDesc(0, inputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    const auto sourceDims = inputDesc.GetShape().GetDims();
    const auto rank = sourceDims.size();
    if (rank < 3) {
        return false;
    }
    std::size_t firstAxis = rank - 2;
    std::size_t secondAxis = rank - 1;
    if (inputIndex == WEIGHT_SCALE_INDEX) {
        if (rank != 4) {
            return false;
        }
        firstAxis = 1;
        secondAxis = 2;
    }
    if (packNode->GetInputsSize() != rank) {
        return false;
    }
    for (std::size_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
        const auto sourceAxis =
            outputAxis == firstAxis ? secondAxis : (outputAxis == secondAxis ? firstAxis : outputAxis);
        if (!IsPackAxisSourceDim(packNode, outputAxis, sourceNode, static_cast<int64_t>(sourceAxis), sourceDims)) {
            return false;
        }
    }
    return true;
}

bool IsDynamicTwoReshapeChain(const GNodePtr& topReshapeNode, const GNodePtr& bottomReshapeNode,
                              const GNodePtr& sourceNode, int64_t sourceOutputPort)
{
    // The dynamic view optimizer may remove a unit leading axis.  Therefore
    // the two Pack nodes need not have the same rank as their Reshape inputs;
    // the Shape/Gather/Pack construction is the proof that no arbitrary
    // data-layout reshape was introduced.
    return IsShapePackDerivedFromSource(topReshapeNode, sourceNode, sourceOutputPort) &&
           IsShapePackDerivedFromSource(bottomReshapeNode, sourceNode, sourceOutputPort);
}

bool IsExpectedSwap(const std::vector<int64_t>& inputDims, const std::vector<int64_t>& outputDims, int64_t inputIndex,
                    bool requireUnitAxis, bool nonMxScale = false)
{
    if (inputDims.size() < (nonMxScale ? 2U : 3U)) {
        return false;
    }
    std::size_t firstAxis = inputDims.size() - 2;
    std::size_t secondAxis = inputDims.size() - 1;
    if (inputIndex == WEIGHT_SCALE_INDEX && !nonMxScale) {
        if (inputDims.size() != 4) {
            return false;
        }
        firstAxis = 1;
        secondAxis = 2;
    }
    if (requireUnitAxis && inputDims[firstAxis] != 1 && inputDims[secondAxis] != 1) {
        return false;
    }
    if (outputDims.empty()) {
        return false;
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

bool IsStaticFp4PackedWeightTranspose(const TensorDesc& sourceDesc, const TensorDesc& targetDesc, int64_t inputIndex,
                                      bool specialFp4Path)
{
    // A static FP4 weight can be represented by an int8/uint8 carrier before
    // Bitcast.  With a single group, the view optimizer first transposes the
    // carrier as a 2-D tensor and the FP4 packing doubles the final axis:
    //   source [1, N, K/2] -> target [1, K/2, 2*N].
    // Keep this exception deliberately narrow; ordinary Reshape nodes must
    // still satisfy the normal axis-swap proof below.
    if (inputIndex != WEIGHT_INDEX || !specialFp4Path) {
        return false;
    }
    if (sourceDesc.GetDataType() != DT_INT8 && sourceDesc.GetDataType() != DT_UINT8) {
        return false;
    }
    if (targetDesc.GetDataType() != DT_FLOAT4_E2M1) {
        return false;
    }
    const auto sourceDims = sourceDesc.GetShape().GetDims();
    const auto targetDims = targetDesc.GetShape().GetDims();
    if (sourceDims.size() != 3 || targetDims.size() != 3 || sourceDims[0] != 1 || targetDims[0] != 1) {
        return false;
    }
    if (std::any_of(sourceDims.begin(), sourceDims.end(), [](int64_t dim) { return dim <= 0; }) ||
        std::any_of(targetDims.begin(), targetDims.end(), [](int64_t dim) { return dim <= 0; })) {
        return false;
    }
    return targetDims[1] == sourceDims[2] && targetDims[2] == sourceDims[1] * DOUBLE_SHAPE;
}

bool CheckDirectReshape(const GNodePtr& reshapeNode, int64_t inputIndex)
{
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
    if (GetDataConsumerNum(reshapeNode) != 1 || !GetShapeDims(reshapeNode, true, inputDims) ||
        !GetShapeDims(reshapeNode, false, outputDims)) {
        return false;
    }
    // For a static graph, exact shape equality after the expected axis swap
    // is sufficient even when neither swapped axis is one.  This is the
    // ordinary transpose-as-reshape form emitted by view optimization.
    const bool hasUnknownOutputDim =
        std::any_of(outputDims.begin(), outputDims.end(), [](int64_t dim) { return IsUnknownDim(dim); });
    if (!outputDims.empty() && !hasUnknownOutputDim) {
        return IsExpectedSwap(inputDims, outputDims, inputIndex, false);
    }
    // Dynamic graphs do not always carry the Reshape output descriptor.  In
    // that case accept only the explicit Shape/Gather/Pack construction that
    // proves the last two (or MX scale N/K) axes were exchanged.
    return IsDynamicAxisSwapReshape(reshapeNode, inputIndex);
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
    return dtype == DT_FLOAT4_E2M1;
}

bool CheckPlatformSupportsTransposeFusion()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get platform information.");
        return false;
    }
    return platformInfo.str_info.short_soc_version == ASCEND_950_SOC_VERSION ||
           platformInfo.str_info.short_soc_version == ASCEND_350_SOC_VERSION ||
           platformInfo.str_info.short_soc_version == ASCEND_960_DT_SOC_VERSION;
}

bool CheckGroupedMatmulFinalizeRouting(const GNode& nodeGmmFr)
{
    return IsType(nodeGmmFr, OP_TYPE_GMMFR);
}

bool CheckNodeDtype(const GNode& nodeGmmFr)
{
    static const std::vector<DataType> legalInputDtypes = {DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_INT8,
                                                           DT_HIFLOAT8};
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (nodeGmmFr.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.GetInputDesc(WEIGHT_INDEX, weightDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get GroupedMatmulFinalizeRouting input desc.");
        return false;
    }
    if (!IsLegalDataType(xDesc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "X dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(xDesc.GetDataType()).c_str());
        return false;
    }
    if (!IsLegalDataType(weightDesc.GetDataType(), legalInputDtypes)) {
        OP_LOGW(PASS_NAME, "Weight dtype is %s, which does not support transpose fusion.",
                ge::TypeUtils::DataTypeToSerialString(weightDesc.GetDataType()).c_str());
        return false;
    }
    return true;
}

bool CheckNodeShape(const GNode& nodeGmmFr)
{
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (nodeGmmFr.GetInputDesc(X_INDEX, xDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.GetInputDesc(WEIGHT_INDEX, weightDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get shape desc before fusion.");
        return false;
    }
    const auto xRank = xDesc.GetOriginShape().GetDimNum();
    const auto weightRank = weightDesc.GetOriginShape().GetDimNum();
    // Dynamic GE graphs may leave the GMM input descriptor rank empty until
    // runtime shape inference.  In that case the view-chain proof below is
    // the source of truth; reject only ranks that are known and invalid.
    if ((xRank != 0 && xRank < MINI_SHAPE_LEN) || (weightRank != 0 && weightRank < WEIGHT_MINI_SHAPE_LEN)) {
        OP_LOGW(PASS_NAME, "X dim num or weight dim num does not support transpose fusion.");
        return false;
    }
    return true;
}

bool IsMxWeightScale(const GNode& nodeGmmFr)
{
    TensorDesc weightScaleDesc;
    return nodeGmmFr.GetInputDesc(WEIGHT_SCALE_INDEX, weightScaleDesc) == GRAPH_SUCCESS &&
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

bool TryResolveTwoReshapeChain(const GNode& nodeGmmFr, int64_t inputIndex, TransposeChain& chain)
{
    auto bottomNode = GetInputNode(nodeGmmFr, inputIndex);
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
    const bool nonMxScale = inputIndex == WEIGHT_SCALE_INDEX && !IsMxWeightScale(nodeGmmFr);
    // Non-MX scales use the ordinary last-two-axis layout, unlike the
    // four-dimensional MX scale whose trailing axis stores the two exponents.
    const int64_t layoutIndex = nonMxScale ? WEIGHT_INDEX : inputIndex;
    TensorDesc transposeInputDesc;
    if (nonMxScale && transposeNode->GetInputDesc(0, transposeInputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    const bool transposeReshapePattern = !nonMxScale || transposeInputDesc.GetShape().GetDimNum() == 2;
    if (sourceNode->GetOutputDesc(srcOutputPort, sourceDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.GetInputDesc(inputIndex, targetDesc) != GRAPH_SUCCESS ||
        !CheckTransposeNode(transposeNode, layoutIndex, transposeReshapePattern) ||
        GetDataConsumerNum(topReshapeNode) != 1 || GetDataConsumerNum(bottomNode) != 1 ||
        (chain.bitcastNode != nullptr && GetDataConsumerNum(chain.bitcastNode) != 1) ||
        (chain.tailReshapeNode != nullptr && GetDataConsumerNum(chain.tailReshapeNode) != 1)) {
        return false;
    }

    const auto sourceDims = sourceDesc.GetShape().GetDims();
    const auto targetDims = targetDesc.GetShape().GetDims();
    const bool exactShapeMatch = IsExpectedSwap(sourceDims, targetDims, inputIndex, false, nonMxScale);
    const bool hasUnknownShape = sourceDims.empty() || targetDims.empty() ||
                                 std::any_of(sourceDims.begin(), sourceDims.end(), IsUnknownDim) ||
                                 std::any_of(targetDims.begin(), targetDims.end(), IsUnknownDim);
    const bool staticFp4PackedMatch =
        IsStaticFp4PackedWeightTranspose(sourceDesc, targetDesc, inputIndex, chain.specialFp4Path);
    if (!exactShapeMatch && !staticFp4PackedMatch &&
        (!hasUnknownShape || !IsDynamicTwoReshapeChain(topReshapeNode, bottomNode, sourceNode, srcOutputPort))) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.transLikeNode = bottomNode;
    chain.sourceNode = sourceNode;
    chain.sourceOutputPort = srcOutputPort;
    chain.nodesToRemove = {bottomNode, transposeNode, topReshapeNode};
    chain.twoReshapePattern = true;
    return true;
}

bool TryResolveDirectChain(const GNode& nodeGmmFr, int64_t inputIndex, TransposeChain& chain)
{
    auto transLikeNode = GetInputNode(nodeGmmFr, inputIndex);
    if (transLikeNode == nullptr) {
        return false;
    }
    if (IsSpecialFp4Path(transLikeNode, inputIndex)) {
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
        (chain.bitcastNode != nullptr && GetDataConsumerNum(chain.bitcastNode) != 1) ||
        (chain.tailReshapeNode != nullptr && GetDataConsumerNum(chain.tailReshapeNode) != 1)) {
        return false;
    }
    chain.inputIndex = inputIndex;
    chain.nodesToRemove = {transLikeNode};
    return true;
}

bool ResolveTransposeChain(const GNode& nodeGmmFr, int64_t inputIndex, TransposeChain& chain)
{
    if (TryResolveTwoReshapeChain(nodeGmmFr, inputIndex, chain)) {
        return true;
    }
    chain = TransposeChain();
    return TryResolveDirectChain(nodeGmmFr, inputIndex, chain);
}

bool RemoveNode(const GraphPtr& graph, const GNodePtr& dstNode)
{
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid when removing node.");
        return false;
    }
    if (dstNode == nullptr || !dstNode->GetOutDataNodesAndPortIndexs(0).empty()) {
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

bool UpdateSpecialFp4Desc(GNode& nodeGmmFr, const TransposeChain& chain, int64_t inputIndex,
                          const TensorDesc& sourceDesc)
{
    if (chain.bitcastNode == nullptr || chain.tailReshapeNode == nullptr) {
        OP_LOGW(PASS_NAME, "Special FP4 path misses Bitcast or tail Reshape node.");
        return false;
    }
    auto dims = sourceDesc.GetShape().GetDims();
    if (dims.size() <= INDEX_THREE) {
        OP_LOGW(PASS_NAME, "Special FP4 path source shape dims are too small.");
        return false;
    }

    TensorDesc bitcastOutputDesc;
    if (chain.bitcastNode->GetOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get Bitcast output desc.");
        return false;
    }
    Shape bitcastShape({dims[0], dims[1], dims[INDEX_THREE], DOUBLE_SHAPE});
    bitcastOutputDesc.SetShape(bitcastShape);
    bitcastOutputDesc.SetOriginShape(bitcastShape);
    if (chain.bitcastNode->UpdateOutputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS ||
        chain.tailReshapeNode->UpdateInputDesc(0, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast or tail Reshape desc for special FP4 path.");
        return false;
    }

    TensorDesc tailReshapeOutputDesc;
    if (chain.tailReshapeNode->GetOutputDesc(0, tailReshapeOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get tail Reshape output desc.");
        return false;
    }
    Shape reshapeShape({dims[0], dims[1], dims[INDEX_THREE] * DOUBLE_SHAPE});
    tailReshapeOutputDesc.SetShape(reshapeShape);
    tailReshapeOutputDesc.SetOriginShape(reshapeShape);
    if (chain.tailReshapeNode->UpdateOutputDesc(0, tailReshapeOutputDesc) != GRAPH_SUCCESS ||
        nodeGmmFr.UpdateInputDesc(inputIndex, tailReshapeOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulFinalizeRouting desc for special FP4 path.");
        return false;
    }
    return true;
}

bool UpdateInputDesc(GNode& nodeGmmFr, const TransposeChain& chain, int64_t inputIndex, const TensorDesc& sourceDesc)
{
    if (!chain.relinkToBitcast) {
        if (nodeGmmFr.UpdateInputDesc(inputIndex, sourceDesc) != GRAPH_SUCCESS) {
            OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulFinalizeRouting input desc.");
            return false;
        }
        return true;
    }
    if (chain.bitcastNode == nullptr) {
        OP_LOGW(PASS_NAME, "Bitcast node is null when updating desc.");
        return false;
    }
    TensorDesc gmmInputDesc;
    if (nodeGmmFr.GetInputDesc(inputIndex, gmmInputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to get GroupedMatmulFinalizeRouting input desc.");
        return false;
    }
    if (chain.bitcastNode->UpdateInputDesc(0, sourceDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update Bitcast input desc.");
        return false;
    }
    if (chain.specialFp4Path) {
        return UpdateSpecialFp4Desc(nodeGmmFr, chain, inputIndex, sourceDesc);
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
    if (nodeGmmFr.UpdateInputDesc(inputIndex, bitcastOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to update GroupedMatmulFinalizeRouting input desc from Bitcast.");
        return false;
    }
    return true;
}

bool RelinkNode(const GraphPtr& graph, GNode& nodeGmmFr, const TransposeChain& chain, int64_t inputIndex)
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

    auto& dstNode = chain.relinkToBitcast ? *chain.bitcastNode : nodeGmmFr;
    const int64_t dstInputPort = chain.relinkToBitcast ? 0 : inputIndex;
    if (graph->RemoveEdge(*chain.transLikeNode, 0, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to remove edge from transpose-like node.");
        return false;
    }
    if (graph->AddDataEdge(*chain.sourceNode, chain.sourceOutputPort, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to add edge from source node.");
        return false;
    }
    if (chain.specialFp4Path && chain.tailReshapeNode != nullptr) {
        // The final FP4 Reshape has a dynamic shape subgraph (typically
        // Mul(Shape(bottom_reshape), 2)).  Once the data edge bypasses the
        // transpose chain, its Shape input must bypass it as well; otherwise
        // the runtime Reshape still materializes [G, K/2, N*2] even though the
        // descriptor was changed to [G, N, K].
        auto tailShapeProducer = GetInputNode(*chain.tailReshapeNode, 1);
        auto tailShapeNode = tailShapeProducer == nullptr ? nullptr : GetInputNode(*tailShapeProducer, 0);
        auto shapeInput = tailShapeNode == nullptr ? nullptr : GetInputNode(*tailShapeNode, 0);
        if (IsType(tailShapeNode, OP_TYPE_SHAPE) && IsSameNode(shapeInput, chain.transLikeNode)) {
            if (graph->RemoveEdge(*chain.transLikeNode, 0, *tailShapeNode, 0) != GRAPH_SUCCESS ||
                graph->AddDataEdge(*chain.sourceNode, chain.sourceOutputPort, *tailShapeNode, 0) != GRAPH_SUCCESS) {
                OP_LOGW(PASS_NAME, "Failed to relink dynamic FP4 shape input.");
                return false;
            }
        }
    }
    return UpdateInputDesc(nodeGmmFr, chain, inputIndex, sourceDesc);
}

void AddNodesToReport(const GNode& nodeGmmFr, const std::vector<TransposeChain>& chains,
                      std::vector<GNode>& nodesBeforeFuse, std::vector<GNodePtr>& nodesToRemove)
{
    nodesBeforeFuse = {nodeGmmFr};
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

void ReportTransposeFusion(const std::vector<GNode>& nodesBeforeFuse, const GNode& nodeGmmFr,
                           CustomPassContext& passContext)
{
    auto reportFuse = GraphFuseInspectorUtils::ReportFuse;
    if (reportFuse == nullptr) {
        return;
    }
    if (reportFuse(nodesBeforeFuse, {nodeGmmFr}, passContext) != SUCCESS) {
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

Status PrepareFusion(const GNode& nodeGmmFr, std::vector<TransposeChain>& chains)
{
    if (!CheckGroupedMatmulFinalizeRouting(nodeGmmFr)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckNodeDtype(nodeGmmFr) || !CheckNodeShape(nodeGmmFr)) {
        return GRAPH_NOT_CHANGED;
    }
    bool transposeWeight = false;
    if (nodeGmmFr.GetAttr(ATTR_TRANSPOSE_WEIGHT, transposeWeight) != GRAPH_SUCCESS || transposeWeight) {
        return GRAPH_NOT_CHANGED;
    }

    TransposeChain weightChain;
    if (!ResolveTransposeChain(nodeGmmFr, WEIGHT_INDEX, weightChain)) {
        return GRAPH_NOT_CHANGED;
    }
    chains.emplace_back(weightChain);

    TransposeChain weightScaleChain;
    if (!ResolveTransposeChain(nodeGmmFr, WEIGHT_SCALE_INDEX, weightScaleChain)) {
        return GRAPH_NOT_CHANGED;
    }
    // Match canndev: the direct scale path requires E8M0, but the
    // Reshape -> Transpose -> Reshape path has no MX dtype restriction.
    if (!IsMxWeightScale(nodeGmmFr) && !weightScaleChain.twoReshapePattern) {
        return GRAPH_NOT_CHANGED;
    }
    chains.emplace_back(weightScaleChain);
    return SUCCESS;
}

Status CommitFusion(const GraphPtr& graph, GNode& nodeGmmFr, const std::vector<TransposeChain>& chains,
                    CustomPassContext& passContext)
{
    std::vector<GNode> nodesBeforeFuse;
    std::vector<GNodePtr> nodesToRemove;
    AddNodesToReport(nodeGmmFr, chains, nodesBeforeFuse, nodesToRemove);
    const auto reportMatch = GraphFuseInspectorUtils::ReportMatch;
    if (reportMatch != nullptr && reportMatch(nodesBeforeFuse, passContext) != SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to report fusion match.");
    }

    bool transposeWeight = true;
    if (nodeGmmFr.SetAttr(ATTR_TRANSPOSE_WEIGHT, transposeWeight) != GRAPH_SUCCESS) {
        OP_LOGW(PASS_NAME, "Failed to set transpose_w attr.");
        return GRAPH_NOT_CHANGED;
    }
    for (const auto& chain : chains) {
        if (!RelinkNode(graph, nodeGmmFr, chain, chain.inputIndex)) {
            return GRAPH_FAILED;
        }
    }
    ReportTransposeFusion(nodesBeforeFuse, nodeGmmFr, passContext);
    if (!RemoveFusionNodes(graph, nodesToRemove)) {
        return GRAPH_FAILED;
    }
    return SUCCESS;
}

Status Fusion(const GraphPtr& graph, GNode& nodeGmmFr, CustomPassContext& passContext)
{
    std::vector<TransposeChain> chains;
    const auto status = PrepareFusion(nodeGmmFr, chains);
    if (status != SUCCESS) {
        return status;
    }
    return CommitFusion(graph, nodeGmmFr, chains, passContext);
}

Status RunGroupedMatmulFinalizeRoutingTransposeFusion(const GraphPtr& graph, CustomPassContext& passContext)
{
    OP_LOGD(PASS_NAME, "Enter GroupedMatmulFinalizeRouting transpose fusion pass.");
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(PASS_NAME, "Graph is null or invalid, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> nodeGmmFrList;
    for (auto& node : graph->GetDirectNode()) {
        if (CheckGroupedMatmulFinalizeRouting(node)) {
            nodeGmmFrList.emplace_back(node);
        }
    }
    if (nodeGmmFrList.empty()) {
        OP_LOGD(PASS_NAME, "No GroupedMatmulFinalizeRouting node, skip fusion pass.");
        return GRAPH_NOT_CHANGED;
    }

    bool changed = false;
    // Query once per run rather than once for each candidate node.
    if (!CheckPlatformSupportsTransposeFusion()) {
        return GRAPH_NOT_CHANGED;
    }
    for (auto& nodeGmmFr : nodeGmmFrList) {
        auto status = Fusion(graph, nodeGmmFr, passContext);
        if (status == SUCCESS) {
            OP_LOGD(PASS_NAME, "GroupedMatmulFinalizeRouting transpose fusion succeeded.");
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

Status GroupedMatmulFinalizeRoutingTransposeFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (!IsTargetVersion()) {
        return GRAPH_NOT_CHANGED;
    }
    return RunGroupedMatmulFinalizeRoutingTransposeFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(GroupedMatmulFinalizeRoutingTransposeFusionPass)
    .Stage(IsTargetVersion() ? CustomPassStage::kCompatibleInherited : CustomPassStage::kAfterInferShape);
#endif
} // namespace ops
#endif // CANN_VERSION_NUM

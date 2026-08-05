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
 * \file grouped_matmul_transpose_fusion_pass.cpp
 * \brief Remove transpose-like nodes before GroupedMatmul and preserve their semantics in transpose attrs.
 */

#include "grouped_matmul_transpose_fusion_pass.h"

#if CANN_VERSION_NUM >= GROUPED_MATMUL_GRAPH_FUSION_SUPPORT_VERSION
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "acl/acl_rt.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace ge::fusion;

// GE 9.1 does not expose the inspector declaration in public headers. Keep the weak compatibility shim local to this
// translation unit to avoid leaking a private GE type through the pass public header.
namespace ge {
namespace fusion {
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse,
        const std::vector<GNode>& nodesAfterFuse, CustomPassContext& ctx) __attribute__((weak));
};
} // namespace fusion
} // namespace ge

namespace ops {
namespace {
constexpr char kPassName[] = "GroupedMatmulTransFusionPass";
constexpr char kOpTypeGroupedMatmul[] = "GroupedMatmul";
constexpr char kOpTypeTranspose[] = "Transpose";
constexpr char kOpTypeTransposeD[] = "TransposeD";
constexpr char kOpTypeReshape[] = "Reshape";
constexpr char kOpTypeBitcast[] = "Bitcast";
constexpr char kOpTypeConst[] = "Const";
constexpr char kOpTypeShape[] = "Shape";
constexpr char kOpTypeGather[] = "Gather";
constexpr char kOpTypePack[] = "Pack";
constexpr char kTransposeXAttr[] = "transpose_x";
constexpr char kTransposeWeightAttr[] = "transpose_weight";
constexpr char kXInputName[] = "x";
constexpr char kWeightInputName[] = "weight";
constexpr char kFixPipeIntrinsic[] = "Intrinsic_fix_pipe_l0c2out";
constexpr char kDav3510ShortSocVersion[] = "Ascend950";

constexpr int32_t kXIndex = 0;
constexpr int32_t kWeightIndex = 1;
constexpr int32_t kScaleIndex = 3;
constexpr int32_t kAntiquantScaleIndex = 5;
constexpr int32_t kPertokenScaleIndex = 8;
constexpr int32_t kPermInputIndex = 1;
constexpr int32_t kMiniShapeLen = 2;
constexpr int32_t kCompatibleInheritedSupportVersion = 90100000;
const std::vector<int64_t> kPermWeight = {0, 2, 1};
const std::vector<int64_t> kPermWeightReshapePattern = {1, 0};
const std::vector<int64_t> kPermScaleReshapePattern = {1, 0, 2};
const std::vector<int64_t> kPermAntiquantScaleReshapePattern = {1, 0};
const std::vector<int64_t> kPermAntiquantScaleReshapePatternMx = {1, 0, 2};
const std::vector<int64_t> kPermX = {1, 0};
const std::vector<int64_t> kPermPertoken = {1, 0, 2};
const std::vector<int64_t> kPermAntiquantScale = {0, 2, 1};
const std::vector<int64_t> kPermAntiquantScaleMx = {0, 2, 1, 3};
const std::vector<int64_t> kPermScale = {0, 2, 1, 3};

struct GmmReshapePatternMatchInfo {
    bool isReshapePattern = false;
    bool isBitcastReshapePattern = false;
};

struct GmmInputPattern {
    GNodePtr immediateInput = nullptr;
    GNodePtr bitcastNode = nullptr;
    GNodePtr reshapeNode1 = nullptr;
    GNodePtr transposeNode = nullptr;
    GNodePtr reshapeNode2 = nullptr;
    GNodePtr sourceNode = nullptr;
    GNodePtr oldSrcNode = nullptr;
    int32_t sourceOutputPort = 0;
    bool isReshapePattern = false;
    bool isBitcastReshapePattern = false;
};

bool IsGraphFusionRuntimeSupported()
{
    int32_t version = 0;
    if (aclsysGetVersionNum("ge_compiler", &version) != ACL_SUCCESS) {
        OP_LOGW(kPassName, "Failed to get ge compiler version.");
        return false;
    }
    return version >= kCompatibleInheritedSupportVersion;
}

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
    return IsType(node, kOpTypeTranspose) || IsType(node, kOpTypeTransposeD);
}

GNodePtr GetInputNode(const GNode& dstNode, int32_t dstInputPort, int32_t* srcOutputPort = nullptr)
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
    return node->GetOutDataNodesAndPortIndexs(0).size();
}

bool IsUnknownDim(int64_t dim)
{
    return dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM || dim < 0;
}

bool IsSameOrBothUnknown(int64_t lhs, int64_t rhs)
{
    return lhs == rhs || (IsUnknownDim(lhs) && IsUnknownDim(rhs));
}

bool GetConstScalar(const GNode& node, int32_t inputIndex, int64_t& value)
{
    Tensor tensor;
    TensorDesc desc;
    if (node.GetInputConstData(inputIndex, tensor) != GRAPH_SUCCESS ||
        node.GetInputDesc(inputIndex, desc) != GRAPH_SUCCESS || tensor.GetData() == nullptr) {
        return false;
    }
    if (desc.GetDataType() == DT_INT32 && tensor.GetSize() >= sizeof(int32_t)) {
        value = static_cast<int64_t>(*reinterpret_cast<const int32_t*>(tensor.GetData()));
        return true;
    }
    if (desc.GetDataType() == DT_INT64 && tensor.GetSize() >= sizeof(int64_t)) {
        value = *reinterpret_cast<const int64_t*>(tensor.GetData());
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
    if (!IsType(gatherNode, kOpTypeGather)) {
        return false;
    }
    auto shapeNode = GetInputNode(*gatherNode, 0);
    auto shapeSourceNode = shapeNode == nullptr ? nullptr : GetInputNode(*shapeNode, 0);
    int64_t actualDimIndex = 0;
    return IsType(shapeNode, kOpTypeShape) && IsSameNode(shapeSourceNode, sourceNode) &&
           GetConstScalar(*gatherNode, 1, actualDimIndex) && actualDimIndex == dimIndex;
}

bool IsDynamicLastTwoAxisSwapReshape(const GNodePtr& reshapeNode)
{
    if (!IsType(reshapeNode, kOpTypeReshape)) {
        return false;
    }
    auto sourceNode = GetInputNode(*reshapeNode, 0);
    auto packNode = GetInputNode(*reshapeNode, 1);
    if (sourceNode == nullptr || !IsType(packNode, kOpTypePack)) {
        return false;
    }
    TensorDesc inputDesc;
    if (reshapeNode->GetInputDesc(0, inputDesc) != GRAPH_SUCCESS ||
        inputDesc.GetShape().GetDimNum() != kMiniShapeLen) {
        return false;
    }

    auto firstShapeInput = GetInputNode(*packNode, 0);
    auto secondShapeInput = GetInputNode(*packNode, 1);
    if (firstShapeInput == nullptr || secondShapeInput == nullptr) {
        return false;
    }
    int64_t firstConstValue = 0;
    int64_t secondConstValue = 0;
    const bool firstIsUnit = GetConstScalar(*packNode, 0, firstConstValue) && firstConstValue == 1;
    const bool secondIsUnit = GetConstScalar(*packNode, 1, secondConstValue) && secondConstValue == 1;
    return (IsGatherOfSourceDim(firstShapeInput, sourceNode, 1) && secondIsUnit) ||
           (firstIsUnit && IsGatherOfSourceDim(secondShapeInput, sourceNode, 0));
}

bool IsFloat4DataType(DataType dataType)
{
    return dataType == DT_FLOAT4_E1M2 || dataType == DT_FLOAT4_E2M1;
}

bool IsMxFp4QuantMode(const GNode& groupedMatmulNode)
{
    TensorDesc xDesc;
    TensorDesc weightDesc;
    return groupedMatmulNode.GetInputDesc(kXIndex, xDesc) == GRAPH_SUCCESS &&
           groupedMatmulNode.GetInputDesc(kWeightIndex, weightDesc) == GRAPH_SUCCESS &&
           IsFloat4DataType(xDesc.GetDataType()) && IsFloat4DataType(weightDesc.GetDataType());
}

bool CheckPlatformIsDav3510ForGmm()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGW(kPassName, "Get platform information failed.");
        return false;
    }
    return platformInfo.str_info.short_soc_version == kDav3510ShortSocVersion;
}

bool CheckPlatformSupportReshapePattern()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGW(kPassName, "Get platform information failed.");
        return false;
    }
    return platformInfo.ai_core_intrinsic_dtype_map.find(kFixPipeIntrinsic) !=
           platformInfo.ai_core_intrinsic_dtype_map.end();
}

bool IsWeightQuant(const GNode& groupedMatmulNode)
{
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (groupedMatmulNode.GetInputDesc(kXIndex, xDesc) != GRAPH_SUCCESS ||
        groupedMatmulNode.GetInputDesc(kWeightIndex, weightDesc) != GRAPH_SUCCESS) {
        return false;
    }
    return ge::GetSizeByDataType(xDesc.GetDataType()) != ge::GetSizeByDataType(weightDesc.GetDataType());
}

bool IsMxQuantMode(const GNode& groupedMatmulNode)
{
    TensorDesc scaleDesc;
    TensorDesc pertokenScaleDesc;
    if (groupedMatmulNode.GetInputDesc(kScaleIndex, scaleDesc) != GRAPH_SUCCESS ||
        groupedMatmulNode.GetInputDesc(kPertokenScaleIndex, pertokenScaleDesc) != GRAPH_SUCCESS) {
        return false;
    }
    return scaleDesc.GetDataType() == DT_FLOAT8_E8M0;
}

bool IsMxWeightQuantMode(const GNode& groupedMatmulNode)
{
    TensorDesc pertokenScaleDesc;
    if (groupedMatmulNode.GetInputDesc(kPertokenScaleIndex, pertokenScaleDesc) != GRAPH_SUCCESS) {
        return false;
    }
    TensorDesc antiquantScaleDesc;
    return groupedMatmulNode.GetInputDesc(kAntiquantScaleIndex, antiquantScaleDesc) == GRAPH_SUCCESS &&
           antiquantScaleDesc.GetDataType() == DT_FLOAT8_E8M0;
}

bool IsReshapeTransForScale(
    int32_t index, const GNodePtr& nodePerInput, bool isMxQuantMode, bool allowFullyDynamicMxFp4Scale)
{
    if (!IsType(nodePerInput, kOpTypeReshape) || index == kXIndex || index == kWeightIndex ||
        index == kAntiquantScaleIndex) {
        return false;
    }

    TensorDesc inputDesc;
    TensorDesc outputDesc;
    if (nodePerInput->GetInputDesc(0, inputDesc) != GRAPH_SUCCESS ||
        nodePerInput->GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    auto expectedOutputDims = inputDesc.GetShape().GetDims();
    const auto outputDims = outputDesc.GetShape().GetDims();
    if (expectedOutputDims.size() < kMiniShapeLen || expectedOutputDims.size() != outputDims.size()) {
        return false;
    }

    std::size_t firstTransposeAxis = expectedOutputDims.size() - kMiniShapeLen;
    std::size_t secondTransposeAxis = expectedOutputDims.size() - 1;
    const bool isMxScale =
        index == kScaleIndex && (isMxQuantMode || inputDesc.GetDataType() == DT_FLOAT8_E8M0);
    const bool isMxPertokenScale = index == kPertokenScaleIndex && inputDesc.GetDataType() == DT_FLOAT8_E8M0;
    if (isMxScale) {
        if (expectedOutputDims.size() < kPermScale.size()) {
            return false;
        }
        firstTransposeAxis = expectedOutputDims.size() - 3;
        secondTransposeAxis = expectedOutputDims.size() - 2;
    } else if (isMxPertokenScale) {
        firstTransposeAxis = 0;
        secondTransposeAxis = 1;
    }

    if (expectedOutputDims[firstTransposeAxis] != 1 && expectedOutputDims[secondTransposeAxis] != 1) {
        return false;
    }
    if (index == kPertokenScaleIndex && IsDynamicLastTwoAxisSwapReshape(nodePerInput)) {
        return true;
    }
    std::swap(expectedOutputDims[firstTransposeAxis], expectedOutputDims[secondTransposeAxis]);
    const bool isFullyDynamicOutput =
        std::all_of(outputDims.begin(), outputDims.end(), [](int64_t dim) { return IsUnknownDim(dim); });
    // A dynamic MXFP4 graph loses all Reshape output dimensions before this pass. This exception is only enabled
    // when the caller has already matched a valid Weight Transpose and the Scale source still proves a unit axis.
    if (allowFullyDynamicMxFp4Scale && isMxScale && expectedOutputDims.size() == kPermScale.size() &&
        isFullyDynamicOutput) {
        return true;
    }
    for (std::size_t i = 0; i < expectedOutputDims.size(); ++i) {
        if (!IsSameOrBothUnknown(expectedOutputDims[i], outputDims[i])) {
            return false;
        }
    }
    return true;
}

bool GetTransposePerm(const GNodePtr& transposeNode, std::vector<int64_t>& perm)
{
    if (transposeNode == nullptr) {
        return false;
    }
    if (IsType(transposeNode, kOpTypeTranspose)) {
        Tensor permTensor;
        if (transposeNode->GetInputConstData(kPermInputIndex, permTensor) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Get transpose perm const data failed.");
            return false;
        }

        TensorDesc permDesc;
        if (transposeNode->GetInputDesc(kPermInputIndex, permDesc) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Get transpose perm desc failed.");
            return false;
        }
        const uint8_t* constDataPtr = permTensor.GetData();
        if (constDataPtr == nullptr) {
            OP_LOGW(kPassName, "Get transpose perm data failed.");
            return false;
        }

        auto permDtype = permDesc.GetDataType();
        std::size_t size = 0;
        if (permDtype == DT_INT32) {
            size = permTensor.GetSize() / sizeof(int32_t);
            for (std::size_t i = 0; i < size; ++i) {
                perm.emplace_back(static_cast<int64_t>(*(reinterpret_cast<const int32_t*>(constDataPtr) + i)));
            }
        } else if (permDtype == DT_INT64) {
            size = permTensor.GetSize() / sizeof(int64_t);
            for (std::size_t i = 0; i < size; ++i) {
                perm.emplace_back(*(reinterpret_cast<const int64_t*>(constDataPtr) + i));
            }
        } else {
            OP_LOGW(kPassName, "Transpose perm dtype must be int32 or int64.");
            return false;
        }
        return true;
    }

    if (IsType(transposeNode, kOpTypeTransposeD)) {
        if (transposeNode->GetAttr("perm", perm) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Get TransposeD perm attr failed.");
            return false;
        }
        return true;
    }
    return false;
}

std::vector<int64_t> GetExpectedPerm(
    int32_t index, const std::vector<GmmReshapePatternMatchInfo>& reshapePattern, bool isMxQuantMode)
{
    std::vector<int64_t> inputPerm = kPermX;
    if (reshapePattern[index].isReshapePattern) {
        if (index == kWeightIndex || index == kScaleIndex) {
            inputPerm = kPermWeightReshapePattern;
        }
        if (index == kAntiquantScaleIndex) {
            inputPerm = isMxQuantMode ? kPermAntiquantScaleReshapePatternMx : kPermAntiquantScaleReshapePattern;
        }
        if (index == kScaleIndex && isMxQuantMode) {
            inputPerm = kPermScaleReshapePattern;
        }
        if (index == kPertokenScaleIndex && isMxQuantMode) {
            inputPerm = kPermPertoken;
        }
        return inputPerm;
    }

    if (index == kAntiquantScaleIndex) {
        inputPerm = isMxQuantMode ? kPermAntiquantScaleMx : kPermAntiquantScale;
    }
    if (index == kWeightIndex || index == kScaleIndex) {
        inputPerm = kPermWeight;
    }
    if (index == kScaleIndex && isMxQuantMode) {
        inputPerm = kPermScale;
    }
    if (index == kPertokenScaleIndex && isMxQuantMode) {
        inputPerm = kPermPertoken;
    }
    return inputPerm;
}

bool CheckTransposePara(
    const GNodePtr& transposeNode, int32_t index, const std::vector<GmmReshapePatternMatchInfo>& reshapePattern,
    bool isMxQuantMode)
{
    std::vector<int64_t> perm;
    if (!GetTransposePerm(transposeNode, perm)) {
        OP_LOGW(kPassName, "Get transpose perm failed for input index %d.", index);
        return false;
    }
    auto inputPerm = GetExpectedPerm(index, reshapePattern, isMxQuantMode);
    if (perm != inputPerm) {
        OP_LOGW(kPassName, "Transpose perm is invalid for input index %d.", index);
        return false;
    }
    if (GetOutputConsumerNum(transposeNode) != 1) {
        OP_LOGW(kPassName, "Transpose output must be used by only one node.");
        return false;
    }
    return true;
}

bool CheckFusionNodePara(const GNodePtr& transposeNode, GNode& groupedMatmulNode, int32_t index)
{
    if (!IsTransposeType(transposeNode)) {
        OP_LOGW(kPassName, "GroupedMatmul input index %d can only fuse transpose-like node.", index);
        return false;
    }

    bool transposeValue = true;
    if (index == kXIndex) {
        if (groupedMatmulNode.GetAttr(kTransposeXAttr, transposeValue) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Failed to get grouped matmul attr transpose_x.");
            return false;
        }
    } else if (index == kWeightIndex) {
        if (groupedMatmulNode.GetAttr(kTransposeWeightAttr, transposeValue) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Failed to get grouped matmul attr transpose_weight.");
            return false;
        }
    }

    if ((index == kXIndex || index == kWeightIndex) && transposeValue) {
        OP_LOGW(kPassName, "GroupedMatmul transpose attr for x/weight must be false before fusion.");
        return false;
    }
    return true;
}

bool IsTransposeWithTwoReshape(const GNodePtr& node)
{
    if (!IsType(node, kOpTypeReshape)) {
        return false;
    }
    auto transNode = GetInputNode(*node, 0);
    auto reshapeNode1 = transNode == nullptr ? nullptr : GetInputNode(*transNode, 0);
    return IsType(reshapeNode1, kOpTypeReshape) && IsTransposeType(transNode);
}

bool IsScaleNodeWithBitcastAndTwoReshape(const GNodePtr& node)
{
    if (!IsType(node, kOpTypeBitcast)) {
        return false;
    }
    auto reshapeNode2 = GetInputNode(*node, 0);
    auto transNode = reshapeNode2 == nullptr ? nullptr : GetInputNode(*reshapeNode2, 0);
    auto reshapeNode1 = transNode == nullptr ? nullptr : GetInputNode(*transNode, 0);
    return IsType(reshapeNode1, kOpTypeReshape) && IsTransposeType(transNode) &&
           IsType(reshapeNode2, kOpTypeReshape);
}

GmmReshapePatternMatchInfo AnalyzeNodePattern(const GNodePtr& node)
{
    GmmReshapePatternMatchInfo reshapePatternInfo;
    reshapePatternInfo.isReshapePattern = IsTransposeWithTwoReshape(node);
    reshapePatternInfo.isBitcastReshapePattern = IsScaleNodeWithBitcastAndTwoReshape(node);
    if (reshapePatternInfo.isBitcastReshapePattern) {
        reshapePatternInfo.isReshapePattern = true;
    }
    return reshapePatternInfo;
}

Status CheckGmmNode(GNode& groupedMatmulNode)
{
    if (!IsType(groupedMatmulNode, kOpTypeGroupedMatmul)) {
        return GRAPH_NOT_CHANGED;
    }
    std::vector<int32_t> xIndexes;
    std::vector<int32_t> weightIndexes;
    if (groupedMatmulNode.GetDynamicInputIndexesByName(kXInputName, xIndexes) != GRAPH_SUCCESS ||
        groupedMatmulNode.GetDynamicInputIndexesByName(kWeightInputName, weightIndexes) != GRAPH_SUCCESS ||
        xIndexes.size() != 1 || weightIndexes.size() != 1 || xIndexes[0] != kXIndex ||
        weightIndexes[0] != kWeightIndex) {
        OP_LOGD(kPassName, "Only one x and one weight input are supported by this fusion pass.");
        return GRAPH_NOT_CHANGED;
    }
    if (GetInputNode(groupedMatmulNode, kXIndex) == nullptr ||
        GetInputNode(groupedMatmulNode, kWeightIndex) == nullptr) {
        OP_LOGW(kPassName, "GroupedMatmul x/weight input node is null.");
        return GRAPH_NOT_CHANGED;
    }

    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (groupedMatmulNode.GetInputDesc(kXIndex, xDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "GroupedMatmul x input desc is invalid.");
        return GRAPH_FAILED;
    }
    if (groupedMatmulNode.GetInputDesc(kWeightIndex, weightDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "GroupedMatmul weight input desc is invalid.");
        return GRAPH_FAILED;
    }
    return SUCCESS;
}

void DetectReshapePattern(GNode& groupedMatmulNode, std::vector<GmmReshapePatternMatchInfo>& reshapePattern)
{
    reshapePattern.assign(kPertokenScaleIndex + 1, GmmReshapePatternMatchInfo());
    reshapePattern[kXIndex] = AnalyzeNodePattern(GetInputNode(groupedMatmulNode, kXIndex));
    reshapePattern[kWeightIndex] = AnalyzeNodePattern(GetInputNode(groupedMatmulNode, kWeightIndex));
    reshapePattern[kScaleIndex] = AnalyzeNodePattern(GetInputNode(groupedMatmulNode, kScaleIndex));
    reshapePattern[kAntiquantScaleIndex] = AnalyzeNodePattern(GetInputNode(groupedMatmulNode, kAntiquantScaleIndex));
    reshapePattern[kPertokenScaleIndex] = AnalyzeNodePattern(GetInputNode(groupedMatmulNode, kPertokenScaleIndex));

    const bool isDav3510 = CheckPlatformIsDav3510ForGmm();
    const bool isWeightQuant = IsWeightQuant(groupedMatmulNode);
    if (isDav3510 && !isWeightQuant) {
        return;
    }

    const bool hasReshapePattern = std::any_of(
        reshapePattern.begin(), reshapePattern.end(), [](const GmmReshapePatternMatchInfo& nodePattern) {
            return nodePattern.isReshapePattern;
        });
    if (!hasReshapePattern) {
        return;
    }

    if (!CheckPlatformSupportReshapePattern()) {
        reshapePattern.assign(reshapePattern.size(), GmmReshapePatternMatchInfo());
        return;
    }

    const bool weightReshapeFlag = reshapePattern[kWeightIndex].isReshapePattern;
    const bool antiquantScaleBitcastReshapeFlag = reshapePattern[kAntiquantScaleIndex].isBitcastReshapePattern;
    reshapePattern.assign(reshapePattern.size(), GmmReshapePatternMatchInfo());
    if (weightReshapeFlag) {
        reshapePattern[kWeightIndex].isReshapePattern = true;
    }
    if (antiquantScaleBitcastReshapeFlag) {
        reshapePattern[kAntiquantScaleIndex].isReshapePattern = true;
        reshapePattern[kAntiquantScaleIndex].isBitcastReshapePattern = true;
    }
}

bool BuildInputPattern(GNode& groupedMatmulNode, int32_t index,
    const std::vector<GmmReshapePatternMatchInfo>& reshapePattern, GmmInputPattern& pattern)
{
    pattern.immediateInput = GetInputNode(groupedMatmulNode, index);
    if (pattern.immediateInput == nullptr) {
        return false;
    }
    pattern.isReshapePattern = reshapePattern[index].isReshapePattern;
    pattern.isBitcastReshapePattern = reshapePattern[index].isBitcastReshapePattern;

    if (pattern.isBitcastReshapePattern) {
        pattern.bitcastNode = pattern.immediateInput;
        pattern.reshapeNode2 = GetInputNode(*pattern.bitcastNode, 0);
        pattern.transposeNode = pattern.reshapeNode2 == nullptr ? nullptr : GetInputNode(*pattern.reshapeNode2, 0);
        pattern.reshapeNode1 = pattern.transposeNode == nullptr ? nullptr : GetInputNode(*pattern.transposeNode, 0);
        if (pattern.reshapeNode1 == nullptr) {
            return false;
        }
        pattern.sourceNode = GetInputNode(*pattern.reshapeNode1, 0, &pattern.sourceOutputPort);
        pattern.oldSrcNode = pattern.reshapeNode2;
        return pattern.sourceNode != nullptr && pattern.transposeNode != nullptr;
    }

    if (pattern.isReshapePattern) {
        pattern.reshapeNode2 = pattern.immediateInput;
        pattern.transposeNode = GetInputNode(*pattern.reshapeNode2, 0);
        pattern.reshapeNode1 = pattern.transposeNode == nullptr ? nullptr : GetInputNode(*pattern.transposeNode, 0);
        if (pattern.reshapeNode1 == nullptr) {
            return false;
        }
        pattern.sourceNode = GetInputNode(*pattern.reshapeNode1, 0, &pattern.sourceOutputPort);
        pattern.oldSrcNode = pattern.reshapeNode2;
        return pattern.sourceNode != nullptr && pattern.transposeNode != nullptr;
    }

    if (IsType(pattern.immediateInput, kOpTypeBitcast)) {
        pattern.bitcastNode = pattern.immediateInput;
        pattern.transposeNode = GetInputNode(*pattern.bitcastNode, 0);
    } else {
        pattern.transposeNode = pattern.immediateInput;
    }

    if (pattern.transposeNode == nullptr) {
        return false;
    }
    pattern.sourceNode = GetInputNode(*pattern.transposeNode, 0, &pattern.sourceOutputPort);
    pattern.oldSrcNode = pattern.transposeNode;
    return pattern.sourceNode != nullptr;
}

bool CheckEquivalentTransposeNode(GNode& groupedMatmulNode, int32_t index,
    const std::vector<GmmReshapePatternMatchInfo>& reshapePattern, const GmmInputPattern& pattern)
{
    if (pattern.bitcastNode != nullptr && GetOutputConsumerNum(pattern.bitcastNode) != 1) {
        OP_LOGD(kPassName, "Bitcast output must be used by only one node.");
        return false;
    }
    TensorDesc xDesc;
    TensorDesc weightDesc;
    if (groupedMatmulNode.GetInputDesc(kXIndex, xDesc) != GRAPH_SUCCESS ||
        groupedMatmulNode.GetInputDesc(kWeightIndex, weightDesc) != GRAPH_SUCCESS) {
        return false;
    }
    const bool isWeightQuant =
        ge::GetSizeByDataType(xDesc.GetDataType()) != ge::GetSizeByDataType(weightDesc.GetDataType());
    const bool isMxFp4QuantMode = IsMxFp4QuantMode(groupedMatmulNode);
    const bool isMxQuantMode = isMxFp4QuantMode ||
        (isWeightQuant ? IsMxWeightQuantMode(groupedMatmulNode) : IsMxQuantMode(groupedMatmulNode));
    if (IsReshapeTransForScale(index, pattern.transposeNode, isMxQuantMode, isMxFp4QuantMode)) {
        return GetOutputConsumerNum(pattern.transposeNode) == 1;
    }
    return CheckFusionNodePara(pattern.transposeNode, groupedMatmulNode, index) &&
           CheckTransposePara(pattern.transposeNode, index, reshapePattern, isMxQuantMode);
}

bool UpdateBitcastDesc(GNode& bitcastNode, const TensorDesc& sourceOutputDesc, TensorDesc& newOutputDesc)
{
    if (bitcastNode.GetOutputDesc(0, newOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to get bitcast output desc.");
        return false;
    }
    if (bitcastNode.UpdateInputDesc(0, sourceOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to update bitcast input desc.");
        return false;
    }

    newOutputDesc.SetShape(sourceOutputDesc.GetShape());
    newOutputDesc.SetOriginShape(sourceOutputDesc.GetOriginShape());
    newOutputDesc.SetFormat(sourceOutputDesc.GetFormat());
    newOutputDesc.SetOriginFormat(sourceOutputDesc.GetOriginFormat());
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    if (sourceOutputDesc.GetShapeRange(shapeRange) == GRAPH_SUCCESS) {
        (void)newOutputDesc.SetShapeRange(shapeRange);
    }
    if (bitcastNode.UpdateOutputDesc(0, newOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to update bitcast output desc.");
        return false;
    }
    return true;
}

bool RelinkInput(GraphPtr& graph, GNode& groupedMatmulNode, int32_t index, const GmmInputPattern& pattern)
{
    if (graph == nullptr || !graph->IsValid() || pattern.sourceNode == nullptr || pattern.oldSrcNode == nullptr) {
        return false;
    }

    GNode& dstNode = pattern.bitcastNode == nullptr ? groupedMatmulNode : *pattern.bitcastNode;
    const int32_t dstInputPort = pattern.bitcastNode == nullptr ? index : 0;
    if (graph->RemoveEdge(*pattern.oldSrcNode, 0, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to remove edge before relinking GroupedMatmul input.");
        return false;
    }
    if (graph->AddDataEdge(*pattern.sourceNode, pattern.sourceOutputPort, dstNode, dstInputPort) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to add edge after relinking GroupedMatmul input.");
        return false;
    }

    TensorDesc sourceOutputDesc;
    if (pattern.sourceNode->GetOutputDesc(pattern.sourceOutputPort, sourceOutputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to get source output desc.");
        return false;
    }
    TensorDesc groupedMatmulInputDesc = sourceOutputDesc;
    if (pattern.bitcastNode != nullptr) {
        if (!UpdateBitcastDesc(*pattern.bitcastNode, sourceOutputDesc, groupedMatmulInputDesc)) {
            return false;
        }
    }
    if (groupedMatmulNode.UpdateInputDesc(index, groupedMatmulInputDesc) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to update GroupedMatmul input desc.");
        return false;
    }
    return true;
}

void ReportTransposeFusion(const std::vector<GNode>& nodesBeforeFuse, const GNode& groupedMatmulNode,
    CustomPassContext& passContext)
{
    const auto reportFuse = GraphFuseInspectorUtils::ReportFuse;
    if (reportFuse == nullptr) {
        OP_LOGD(kPassName, "Skip reporting fusion result because GraphFuseInspector is unavailable.");
        return;
    }
    if (reportFuse(nodesBeforeFuse, {groupedMatmulNode}, passContext) != SUCCESS) {
        OP_LOGW(kPassName, "Failed to report fusion result.");
    }
}

void CollectFusionNodes(const GmmInputPattern& pattern, std::vector<GNode>& nodesBeforeFuse)
{
    if (pattern.isReshapePattern) {
        nodesBeforeFuse.emplace_back(*pattern.reshapeNode1);
        nodesBeforeFuse.emplace_back(*pattern.transposeNode);
        nodesBeforeFuse.emplace_back(*pattern.reshapeNode2);
        return;
    }
    nodesBeforeFuse.emplace_back(*pattern.transposeNode);
}

bool SetTransposeAttr(GNode& groupedMatmulNode, int32_t index)
{
    const char* attrName = nullptr;
    if (index == kXIndex) {
        attrName = kTransposeXAttr;
    } else if (index == kWeightIndex) {
        attrName = kTransposeWeightAttr;
    } else {
        return true;
    }
    bool transposeValue = true;
    if (groupedMatmulNode.SetAttr(attrName, transposeValue) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to set GroupedMatmul transpose attr.");
        return false;
    }
    return true;
}

bool RemoveNodeIfUnused(GraphPtr& graph, const GNodePtr& node)
{
    if (graph == nullptr || !graph->IsValid() || node == nullptr) {
        return true;
    }
    if (!node->GetOutDataNodesAndPortIndexs(0).empty()) {
        return true;
    }

    std::vector<GNodePtr> inputConstNodes;
    const auto inputSize = node->GetInputsSize();
    for (std::size_t inputIndex = 0; inputIndex < inputSize; ++inputIndex) {
        int32_t srcOutputPort = 0;
        auto srcNode = GetInputNode(*node, static_cast<int32_t>(inputIndex), &srcOutputPort);
        if (srcNode == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(*srcNode, srcOutputPort, *node, static_cast<int32_t>(inputIndex)) != GRAPH_SUCCESS) {
            OP_LOGW(kPassName, "Failed to remove input edge before removing node.");
            return false;
        }
        if (IsType(srcNode, kOpTypeConst)) {
            inputConstNodes.emplace_back(srcNode);
        }
    }

    if (graph->RemoveNode(*node) != GRAPH_SUCCESS) {
        OP_LOGW(kPassName, "Failed to remove unused transpose-like node.");
        return false;
    }

    for (const auto& constNode : inputConstNodes) {
        if (!RemoveNodeIfUnused(graph, constNode)) {
            return false;
        }
    }
    return true;
}

bool RemoveFusionNodes(GraphPtr& graph, const GmmInputPattern& pattern)
{
    if (pattern.isReshapePattern) {
        if (!RemoveNodeIfUnused(graph, pattern.reshapeNode2)) {
            return false;
        }
        if (!RemoveNodeIfUnused(graph, pattern.transposeNode)) {
            return false;
        }
        return RemoveNodeIfUnused(graph, pattern.reshapeNode1);
    }
    return RemoveNodeIfUnused(graph, pattern.transposeNode);
}

Status FusionTransposeNode(GraphPtr& graph, GNode& groupedMatmulNode, int32_t index,
    const std::vector<GmmReshapePatternMatchInfo>& reshapePattern, std::vector<GmmInputPattern>& fusedPatterns)
{
    GmmInputPattern pattern;
    if (!BuildInputPattern(groupedMatmulNode, index, reshapePattern, pattern)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!CheckEquivalentTransposeNode(groupedMatmulNode, index, reshapePattern, pattern)) {
        return GRAPH_NOT_CHANGED;
    }
    if (!RelinkInput(graph, groupedMatmulNode, index, pattern)) {
        OP_LOGE(kPassName, "Failed to relink GroupedMatmul input index %d.", index);
        return GRAPH_FAILED;
    }
    if (!SetTransposeAttr(groupedMatmulNode, index)) {
        OP_LOGE(kPassName, "Failed to update GroupedMatmul transpose attr for input index %d.", index);
        return GRAPH_FAILED;
    }
    fusedPatterns.emplace_back(pattern);
    return SUCCESS;
}

Status FusionNodeX(GraphPtr& graph, GNode& groupedMatmulNode,
    const std::vector<GmmReshapePatternMatchInfo>& reshapePattern, std::vector<GmmInputPattern>& fusedPatterns)
{
    if (!CheckPlatformIsDav3510ForGmm() || IsWeightQuant(groupedMatmulNode)) {
        return GRAPH_NOT_CHANGED;
    }

    GmmInputPattern xPattern;
    if (!BuildInputPattern(groupedMatmulNode, kXIndex, reshapePattern, xPattern) ||
        !CheckEquivalentTransposeNode(groupedMatmulNode, kXIndex, reshapePattern, xPattern)) {
        return GRAPH_NOT_CHANGED;
    }
    auto pertokenScaleInput = GetInputNode(groupedMatmulNode, kPertokenScaleIndex);
    if (IsType(pertokenScaleInput, kOpTypeReshape)) {
        GmmInputPattern pertokenScalePattern;
        if (!BuildInputPattern(groupedMatmulNode, kPertokenScaleIndex, reshapePattern, pertokenScalePattern) ||
            !CheckEquivalentTransposeNode(
                groupedMatmulNode, kPertokenScaleIndex, reshapePattern, pertokenScalePattern)) {
            OP_LOGD(kPassName,
                "Skip x fusion because its per-token scale reshape cannot be proven transpose-equivalent.");
            return GRAPH_NOT_CHANGED;
        }
    }

    auto fusionNodeXResult = FusionTransposeNode(graph, groupedMatmulNode, kXIndex, reshapePattern, fusedPatterns);
    if (fusionNodeXResult != SUCCESS) {
        return fusionNodeXResult;
    }

    if (GetInputNode(groupedMatmulNode, kPertokenScaleIndex) != nullptr) {
        auto pertokenScaleRet =
            FusionTransposeNode(graph, groupedMatmulNode, kPertokenScaleIndex, reshapePattern, fusedPatterns);
        if (pertokenScaleRet == GRAPH_FAILED) {
            return GRAPH_FAILED;
        }
    }
    return SUCCESS;
}

Status FusionNodeWeight(GraphPtr& graph, GNode& groupedMatmulNode,
    const std::vector<GmmReshapePatternMatchInfo>& reshapePattern, std::vector<GmmInputPattern>& fusedPatterns)
{
    GmmInputPattern weightPattern;
    if (!BuildInputPattern(groupedMatmulNode, kWeightIndex, reshapePattern, weightPattern) ||
        !CheckEquivalentTransposeNode(groupedMatmulNode, kWeightIndex, reshapePattern, weightPattern)) {
        return GRAPH_NOT_CHANGED;
    }

    const bool isDav3510 = CheckPlatformIsDav3510ForGmm();
    const bool isWeightQuant = IsWeightQuant(groupedMatmulNode);
    if (isDav3510 && !isWeightQuant && GetInputNode(groupedMatmulNode, kScaleIndex) != nullptr) {
        GmmInputPattern scalePattern;
        if (BuildInputPattern(groupedMatmulNode, kScaleIndex, reshapePattern, scalePattern) &&
            IsType(scalePattern.transposeNode, kOpTypeReshape) &&
            !CheckEquivalentTransposeNode(groupedMatmulNode, kScaleIndex, reshapePattern, scalePattern)) {
            OP_LOGD(kPassName, "Skip weight fusion because its scale reshape cannot be proven transpose-equivalent.");
            return GRAPH_NOT_CHANGED;
        }
    }

    auto fusionNodeWeightResult =
        FusionTransposeNode(graph, groupedMatmulNode, kWeightIndex, reshapePattern, fusedPatterns);
    if (fusionNodeWeightResult != SUCCESS) {
        return fusionNodeWeightResult;
    }

    if (!isDav3510) {
        return SUCCESS;
    }

    if (!isWeightQuant && GetInputNode(groupedMatmulNode, kScaleIndex) != nullptr) {
        auto scaleRet = FusionTransposeNode(graph, groupedMatmulNode, kScaleIndex, reshapePattern, fusedPatterns);
        if (scaleRet == GRAPH_FAILED) {
            return GRAPH_FAILED;
        }
    } else if (isWeightQuant && GetInputNode(groupedMatmulNode, kAntiquantScaleIndex) != nullptr) {
        auto antiquantScaleRet =
            FusionTransposeNode(graph, groupedMatmulNode, kAntiquantScaleIndex, reshapePattern, fusedPatterns);
        if (antiquantScaleRet == GRAPH_FAILED) {
            return GRAPH_FAILED;
        }
    }
    return SUCCESS;
}

Status Fusion(GraphPtr& graph, GNode& groupedMatmulNode, CustomPassContext& passContext)
{
    auto checkResult = CheckGmmNode(groupedMatmulNode);
    if (checkResult != SUCCESS) {
        return checkResult;
    }

    std::vector<GmmReshapePatternMatchInfo> reshapePattern;
    DetectReshapePattern(groupedMatmulNode, reshapePattern);
    std::vector<GmmInputPattern> fusedPatterns;

    auto fusionNodeXResult = FusionNodeX(graph, groupedMatmulNode, reshapePattern, fusedPatterns);
    if (fusionNodeXResult == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    auto fusionNodeWeightResult = FusionNodeWeight(graph, groupedMatmulNode, reshapePattern, fusedPatterns);
    if (fusionNodeWeightResult == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    if (fusionNodeXResult == GRAPH_NOT_CHANGED && fusionNodeWeightResult == GRAPH_NOT_CHANGED) {
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> nodesBeforeFuse = {groupedMatmulNode};
    for (const auto& pattern : fusedPatterns) {
        CollectFusionNodes(pattern, nodesBeforeFuse);
    }
    ReportTransposeFusion(nodesBeforeFuse, groupedMatmulNode, passContext);
    for (const auto& pattern : fusedPatterns) {
        if (!RemoveFusionNodes(graph, pattern)) {
            return GRAPH_FAILED;
        }
    }
    return SUCCESS;
}

Status RunGroupedMatmulTransposeFusion(GraphPtr& graph, CustomPassContext& passContext)
{
    OP_LOGD(kPassName, "Enter GroupedMatmul transpose fusion pass.");
    if (!IsGraphFusionRuntimeSupported()) {
        OP_LOGD(kPassName, "Skip GroupedMatmul transpose fusion pass because graph fusion runtime is unsupported.");
        return GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kPassName, "Graph is null or invalid.");
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> groupedMatmulNodes;
    for (auto& node : graph->GetDirectNode()) {
        if (IsType(node, kOpTypeGroupedMatmul)) {
            groupedMatmulNodes.emplace_back(node);
        }
    }
    if (groupedMatmulNodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(kPassName);
    bool changed = false;
    for (auto& groupedMatmulNode : groupedMatmulNodes) {
        auto status = Fusion(graph, groupedMatmulNode, passContext);
        if (status == SUCCESS) {
            changed = true;
            continue;
        }
        if (status != GRAPH_NOT_CHANGED) {
            OP_LOGE(kPassName, "GroupedMatmul transpose fusion failed, status is %u.", status);
            return status;
        }
    }
    return changed ? SUCCESS : GRAPH_NOT_CHANGED;
}
} // namespace

Status GroupedMatmulTransFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    return RunGroupedMatmulTransposeFusion(graph, passContext);
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(GroupedMatmulTransFusionPass)
    .Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
#endif // CANN_VERSION_NUM >= GROUPED_MATMUL_GRAPH_FUSION_SUPPORT_VERSION

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_all_reduce_transpose_fusion_pass.h"

#if GE_COMPILER_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MatmulAllReduce.h"
#include "es_MatmulAllReduceAddRmsNorm.h"
#include "es_InplaceMatmulAllReduceAddRmsNorm.h"
#include "es_math_ops.h"
#include "common/utils/op_mc2.h"
#include "mc2_platform_info.h"
#include "mc2_fusion_pass_utils.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>
#include "acl/acl_rt.h"

namespace ops {
namespace {
const std::string PASS_NAME = "MatmulAllReduceTransposeFusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_SCALE = "HasScale";
const std::string PATTERN_OFFSET = "HasOffset";
const std::string PATTERN_DEQUANT = "HasDequant";
const std::string PATTERN_SCALE_TRANSPOSE = "ScaleTranspose";
const std::string PATTERN_OFFSET_TRANSPOSE = "OffsetTranspose";
const std::string PATTERN_ARN = "Arn";
const std::string PATTERN_INPLACE_ARN = "InplaceArn";

const char* const OP_TYPE_ARN = "MatmulAllReduceAddRmsNorm";
const char* const OP_TYPE_INPLACE_ARN = "InplaceMatmulAllReduceAddRmsNorm";

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;

enum class Mc2OpKind {
    Mar,
    Arn,
    InplaceArn
};

struct OriginalGraphInfo {
    Mc2OpKind opKind = Mc2OpKind::Mar;
    bool hasBias = false;
    bool hasScale = false;
    bool hasOffset = false;
    bool hasDequant = false;
    bool scaleTranspose = false;
    bool offsetTranspose = false;
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rResidual;
    ge::es::EsTensorHolder rGamma;
    ge::es::EsTensorHolder rAntiquantScale;
    ge::es::EsTensorHolder rAntiquantOffset;
    ge::es::EsTensorHolder rDequantScale;
};

typedef EsCTensorHolder* (*EsTransposeFunc)(EsCTensorHolder*, EsCTensorHolder*);

EsTransposeFunc GetEsTransposeFunc()
{
    void* handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E(PASS_NAME.c_str(), "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E(PASS_NAME.c_str(), "dlsym EsTranspose failed");
        return nullptr;
    }
    return func;
}

ge::es::EsTensorHolder TransposeDL(const ge::es::EsTensorLike& x, const ge::es::EsTensorLike& perm)
{
    static EsTransposeFunc func = GetEsTransposeFunc();
    auto* builder = ge::es::ResolveBuilder(x, perm);
    return func(x.ToTensorHolder(builder).GetCTensorHolder(), perm.ToTensorHolder(builder).GetCTensorHolder());
}

ge::CustomPassStage GetPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}

bool IsArnKind(Mc2OpKind kind)
{
    return kind == Mc2OpKind::Arn || kind == Mc2OpKind::InplaceArn;
}

Mc2OpKind ParseOpKind(const std::string& patternNameStr)
{
    if (patternNameStr.find(PATTERN_INPLACE_ARN) != std::string::npos) {
        return Mc2OpKind::InplaceArn;
    }
    if (patternNameStr.find(PATTERN_ARN) != std::string::npos) {
        return Mc2OpKind::Arn;
    }
    return Mc2OpKind::Mar;
}

size_t GetPatternInputNum(const OriginalGraphInfo& info)
{
    size_t n = 2; // x1 / x2
    if (info.hasBias) {
        ++n;
    }
    if (IsArnKind(info.opKind)) {
        n += 2; // residual / gamma 必选
    }
    if (info.hasScale) {
        ++n;
    }
    if (info.hasOffset) {
        ++n;
    }
    if (info.hasDequant) {
        ++n;
    }
    return n;
}

ge::es::EsTensorHolder MakeTransposeNode(const ge::es::EsTensorHolder& x)
{
    return TransposeDL(x, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
}

ge::fusion::PatternUniqPtr MakePattern(const std::string& passName, const OriginalGraphInfo& info)
{
    std::string patternName = passName;
    if (info.opKind == Mc2OpKind::Arn) {
        patternName += PATTERN_ARN;
    } else if (info.opKind == Mc2OpKind::InplaceArn) {
        patternName += PATTERN_INPLACE_ARN;
    }
    patternName += PATTERN_TRANSPOSE;
    if (info.hasBias) {
        patternName += PATTERN_BIAS;
    }
    if (info.hasScale) {
        patternName += PATTERN_SCALE;
    }
    if (info.hasOffset) {
        patternName += PATTERN_OFFSET;
    }
    if (info.hasDequant) {
        patternName += PATTERN_DEQUANT;
    }
    if (info.scaleTranspose) {
        patternName += PATTERN_SCALE_TRANSPOSE;
    }
    if (info.offsetTranspose) {
        patternName += PATTERN_OFFSET_TRANSPOSE;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto inputs = graphBuilder.CreateInputs(GetPatternInputNum(info));
    size_t idx = 0;
    auto x1 = inputs[idx++];
    auto x2 = inputs[idx++];
    ge::es::EsTensorHolder bias = nullptr;
    if (info.hasBias) {
        bias = inputs[idx++];
    }
    ge::es::EsTensorHolder residual = nullptr;
    ge::es::EsTensorHolder gamma = nullptr;
    if (IsArnKind(info.opKind)) {
        residual = inputs[idx++];
        gamma = inputs[idx++];
    }
    ge::es::EsTensorHolder antiquantScale = nullptr;
    if (info.hasScale) {
        antiquantScale = inputs[idx++];
    }
    ge::es::EsTensorHolder antiquantOffset = nullptr;
    if (info.hasOffset) {
        antiquantOffset = inputs[idx++];
    }
    // 非 A5：dequant 仅直连保留，不吸收其 Transpose
    ge::es::EsTensorHolder dequantScale = nullptr;
    if (info.hasDequant) {
        dequantScale = inputs[idx++];
    }

    auto transposeX2 = MakeTransposeNode(x2);
    ge::es::EsTensorHolder transposeScale = nullptr;
    ge::es::EsTensorHolder scaleIn = antiquantScale;
    if (info.scaleTranspose) {
        transposeScale = MakeTransposeNode(antiquantScale);
        scaleIn = transposeScale;
    }
    ge::es::EsTensorHolder transposeOffset = nullptr;
    ge::es::EsTensorHolder offsetIn = antiquantOffset;
    if (info.offsetTranspose) {
        transposeOffset = MakeTransposeNode(antiquantOffset);
        offsetIn = transposeOffset;
    }

    const char* group = "";
    std::unique_ptr<ge::fusion::Pattern> pattern;
    // x3/pertoken/comm_quant 固定 nullptr；dequant 按是否存在透传，避免量化路径丢失
    if (info.opKind == Mc2OpKind::Mar) {
        auto y = ge::es::MatmulAllReduce(x1, transposeX2, bias, nullptr, scaleIn, offsetIn, dequantScale, nullptr,
                                         nullptr, nullptr, group);
        auto graph = graphBuilder.BuildAndReset({y});
        pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
        pattern->CaptureTensor({*y.GetProducer(), 0}).CaptureTensor({*transposeX2.GetProducer(), 0});
    } else if (info.opKind == Mc2OpKind::Arn) {
        auto out = ge::es::MatmulAllReduceAddRmsNorm(x1, transposeX2, bias, residual, gamma, scaleIn, offsetIn,
                                                     dequantScale, group);
        auto graph = graphBuilder.BuildAndReset({out.y, out.norm_out});
        pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
        pattern->CaptureTensor({*out.y.GetProducer(), 0}).CaptureTensor({*transposeX2.GetProducer(), 0});
    } else {
        auto out = ge::es::InplaceMatmulAllReduceAddRmsNorm(x1, transposeX2, bias, residual, gamma, scaleIn, offsetIn,
                                                            dequantScale, group);
        auto graph = graphBuilder.BuildAndReset({out.ref_residual, out.norm_out});
        pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
        pattern->CaptureTensor({*out.ref_residual.GetProducer(), 0}).CaptureTensor({*transposeX2.GetProducer(), 0});
    }
    if (info.scaleTranspose) {
        pattern->CaptureTensor({*transposeScale.GetProducer(), 0});
    }
    if (info.offsetTranspose) {
        pattern->CaptureTensor({*transposeOffset.GetProducer(), 0});
    }
    return pattern;
}

std::vector<ge::fusion::PatternUniqPtr> BuildAllPatterns(const std::string& passName)
{
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (auto opKind : {Mc2OpKind::Mar, Mc2OpKind::Arn, Mc2OpKind::InplaceArn}) {
        for (bool hasBias : {false, true}) {
            for (bool hasScale : {false, true}) {
                for (bool hasOffset : {false, true}) {
                    for (bool hasDequant : {false, true}) {
                        for (bool scaleTranspose : {false, true}) {
                            if (scaleTranspose && !hasScale) {
                                continue;
                            }
                            for (bool offsetTranspose : {false, true}) {
                                if (offsetTranspose && !hasOffset) {
                                    continue;
                                }
                                // bias/scale/offset 可吸 T；dequant 仅直连透传（不吸 T）
                                OriginalGraphInfo info{opKind,     hasBias,        hasScale,       hasOffset,
                                                       hasDequant, scaleTranspose, offsetTranspose};
                                patternGraphs.emplace_back(MakePattern(passName, info));
                            }
                        }
                    }
                }
            }
        }
    }
    return patternGraphs;
}
size_t GetExpectedSubgraphInputNum(const std::string& patternNameStr)
{
    OriginalGraphInfo info;
    info.opKind = ParseOpKind(patternNameStr);
    info.hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    info.hasScale = patternNameStr.find(PATTERN_SCALE) != std::string::npos;
    info.hasOffset = patternNameStr.find(PATTERN_OFFSET) != std::string::npos;
    info.hasDequant = patternNameStr.find(PATTERN_DEQUANT) != std::string::npos;
    return GetPatternInputNum(info);
}

bool GetTransposePerm(const ge::GNode& transposeNode, std::vector<int64_t>& permValue)
{
    ge::TensorDesc permDesc;
    transposeNode.GetInputDesc(TRANSPOSE_PERM_IDX, permDesc);
    ge::Shape permShape = permDesc.GetShape();
    if (permShape.GetDimNum() != ONE_DIM_SIZE) {
        OPS_LOG_D(PASS_NAME.c_str(), "Transpose permutation dim size must be 1, but got %zu.", permShape.GetDimNum());
        return false;
    }
    ge::Tensor permTensor;
    if (transposeNode.GetInputConstData(TRANSPOSE_PERM_IDX, permTensor) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Failed to get transpose permutation const data.");
        return false;
    }
    ge::DataType permDType = permDesc.GetDataType();
    uint8_t* permData = permTensor.GetData();
    if (permData == nullptr) {
        OPS_LOG_D(PASS_NAME.c_str(), "Transpose permutation data is nullptr.");
        return false;
    }
    if (permDType == ge::DT_INT32) {
        size_t size = permTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(static_cast<int64_t>(*(reinterpret_cast<int32_t*>(permData) + i)));
        }
    } else if (permDType == ge::DT_INT64) {
        size_t size = permTensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(*(reinterpret_cast<int64_t*>(permData) + i));
        }
    } else {
        OPS_LOG_D(PASS_NAME.c_str(), "Transpose permutation dtype must be int32 or int64.");
        return false;
    }
    return !permValue.empty();
}

bool IsLastTwoDimsSwapPermValid(const std::vector<int64_t>& permValue)
{
    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(PASS_NAME.c_str(), "Transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }
    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(PASS_NAME.c_str(), "Transpose permutation last 2 dims must be swapped.");
        return false;
    }
    return true;
}
bool ValidateCapturedTransposePerm(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, int64_t captureIdx)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, transposeOutput) != ge::SUCCESS, false, PASS_NAME.c_str(),
               "Get captured Transpose failed, idx %ld.", captureIdx);
    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeOutput.node, permValue)) {
        OPS_LOG_D(PASS_NAME.c_str(), "Failed to get transpose permutation, capture idx %ld.", captureIdx);
        return false;
    }
    return IsLastTwoDimsSwapPermValid(permValue);
}

bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                          const std::string& patternNameStr)
{
    int64_t captureIdx = MC2_CAPTURE_IDX + 1;
    if (!ValidateCapturedTransposePerm(matchResult, captureIdx)) {
        return false;
    }
    ++captureIdx;
    if (patternNameStr.find(PATTERN_SCALE_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_OFFSET_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx)) {
            return false;
        }
    }
    return true;
}

bool IsAntiquantDirectConnectAllowed(const ge::GNode& mc2Node, const std::string& patternNameStr)
{
    const bool hasScale = patternNameStr.find(PATTERN_SCALE) != std::string::npos;
    const bool hasOffset = patternNameStr.find(PATTERN_OFFSET) != std::string::npos;
    if (!hasScale && !hasOffset) {
        return true;
    }

    ge::AscendString nodeType("");
    mc2Node.GetType(nodeType);
    const std::string typeStr = nodeType.GetString() != nullptr ? nodeType.GetString() : "";
    const bool isArn = (typeStr == OP_TYPE_ARN || typeStr == OP_TYPE_INPLACE_ARN);
    const int32_t scaleIdx = isArn ? static_cast<int32_t>(ops::MC2AddRmsNormInputIdx::K_SCALE) :
                                     static_cast<int32_t>(ops::MC2InputIdx::K_SCALE);

    ge::TensorDesc scaleDesc;
    if (mc2Node.GetInputDesc(scaleIdx, scaleDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(PASS_NAME.c_str(), "Get antiquant scale input desc failed.");
        return false;
    }
    const size_t antiquantDimNum = scaleDesc.GetShape().GetDimNum();
    if (antiquantDimNum <= ONE_DIM_SIZE) {
        return true;
    }
    if (hasScale && patternNameStr.find(PATTERN_SCALE_TRANSPOSE) == std::string::npos) {
        OPS_LOG_D(PASS_NAME.c_str(), "antiquant_scale dim num is %zu (>1) without scale Transpose, fusion is skipped.",
                  antiquantDimNum);
        return false;
    }
    if (hasOffset && patternNameStr.find(PATTERN_OFFSET_TRANSPOSE) == std::string::npos) {
        OPS_LOG_D(PASS_NAME.c_str(),
                  "antiquant_offset present with scale dim %zu (>1) without offset Transpose, fusion is skipped.",
                  antiquantDimNum);
        return false;
    }
    return true;
}

bool Mc2HasConnectedInput(const ge::GNode& mc2Node, int32_t inputIdx)
{
    ge::TensorDesc desc;
    return mc2Node.GetInputDesc(inputIdx, desc) == ge::GRAPH_SUCCESS;
}

int32_t GetDequantInputIdx(const std::string& patternNameStr)
{
    const Mc2OpKind opKind = ParseOpKind(patternNameStr);
    if (IsArnKind(opKind)) {
        return static_cast<int32_t>(ops::MC2AddRmsNormInputIdx::K_DEQUANT);
    }
    return static_cast<int32_t>(ops::MC2InputIdx::K_DEQUANT);
}

// ES pattern 对 optional 口可能误匹配；以原 MC2 节点是否真有 dequant 边为准
bool IsDequantPresenceConsistent(const ge::GNode& mc2Node, const std::string& patternNameStr)
{
    const bool patternHasDequant = patternNameStr.find(PATTERN_DEQUANT) != std::string::npos;
    const bool nodeHasDequant = Mc2HasConnectedInput(mc2Node, GetDequantInputIdx(patternNameStr));
    if (patternHasDequant != nodeHasDequant) {
        OPS_LOG_D(PASS_NAME.c_str(), "Skip pattern=%s: patternHasDequant=%d nodeHasDequant=%d (must be consistent).",
                  patternNameStr.c_str(), static_cast<int>(patternHasDequant), static_cast<int>(nodeHasDequant));
        return false;
    }
    return true;
}

bool MeetCore(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, const char* passName)
{
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(passName, "Skip when cann version not compatible.");
        return false;
    }

    std::string patternNameStr;
    if (!ops::GetPatternNameStr(matchResult, patternNameStr, PASS_NAME.c_str())) {
        return false;
    }

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(passName, "Get subgraph inputs failed.");
        return false;
    }
    if (subgraphInputs.size() != GetExpectedSubgraphInputNum(patternNameStr)) {
        OPS_LOG_D(passName, "Skip pattern=%s: subgraph size mismatch.", patternNameStr.c_str());
        return false;
    }
    if (!IsTransposePermValid(matchResult, patternNameStr)) {
        return false;
    }

    ge::GNode mc2Node;
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, MC2_CAPTURE_IDX, PASS_NAME.c_str(),
                                 "Capture MC2 node failed.")) {
        return false;
    }
    if (!IsAntiquantDirectConnectAllowed(mc2Node, patternNameStr)) {
        return false;
    }
    if (!IsDequantPresenceConsistent(mc2Node, patternNameStr)) {
        return false;
    }
    OPS_LOG_I(passName, "Found One pattern that meets requirements, pattern=%s", patternNameStr.c_str());
    return true;
}

bool CollectSubgraphInputsInfo(const std::vector<ge::fusion::SubgraphInput>& subGraphInputs,
                               std::vector<ge::Shape>& inputShapes, std::vector<ge::DataType>& inputDTypes,
                               std::vector<ge::Format>& inputFormats)
{
    inputShapes.clear();
    inputDTypes.clear();
    inputFormats.clear();
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_E(PASS_NAME.c_str(), "CollectSubgraphInputsInfo: matchNodes is empty.");
            return false;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "CollectSubgraphInputsInfo: GetInputDesc failed.");
            return false;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    return true;
}

bool CreateReplaceGraphInputs(ReplaceGraphInputs& inputs, ge::es::EsGraphBuilder& replaceGraphBuilder,
                              const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                              const std::string& patternNameStr)
{
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        return false;
    }
    if (inputShapes.size() < 2) {
        return false;
    }

    const Mc2OpKind opKind = ParseOpKind(patternNameStr);
    int64_t inputIdx = 0;
    inputs.rX1 = replaceGraphBuilder.CreateInput(inputIdx, "x1", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rX2 = replaceGraphBuilder.CreateInput(inputIdx, "x2", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;

    inputs.rBias = nullptr;
    if (patternNameStr.find(PATTERN_BIAS) != std::string::npos) {
        inputs.rBias = replaceGraphBuilder.CreateInput(inputIdx, "bias", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }

    inputs.rResidual = nullptr;
    inputs.rGamma = nullptr;
    if (IsArnKind(opKind)) {
        inputs.rResidual = replaceGraphBuilder.CreateInput(inputIdx, "residual", inputDTypes[inputIdx],
                                                           inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
        inputs.rGamma = replaceGraphBuilder.CreateInput(inputIdx, "gamma", inputDTypes[inputIdx],
                                                        inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }

    inputs.rAntiquantScale = nullptr;
    if (patternNameStr.find(PATTERN_SCALE) != std::string::npos) {
        inputs.rAntiquantScale =
            replaceGraphBuilder.CreateInput(inputIdx, "antiquant_scale", inputDTypes[inputIdx], inputFormats[inputIdx],
                                            inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rAntiquantOffset = nullptr;
    if (patternNameStr.find(PATTERN_OFFSET) != std::string::npos) {
        inputs.rAntiquantOffset =
            replaceGraphBuilder.CreateInput(inputIdx, "antiquant_offset", inputDTypes[inputIdx], inputFormats[inputIdx],
                                            inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rDequantScale = nullptr;
    if (patternNameStr.find(PATTERN_DEQUANT) != std::string::npos) {
        inputs.rDequantScale = replaceGraphBuilder.CreateInput(inputIdx, "dequant_scale", inputDTypes[inputIdx],
                                                               inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    return true;
}

bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr& replaceGraph,
                            const std::vector<ge::fusion::SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_D(PASS_NAME.c_str(), "InferShapeReplaceGraph: matchNodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(PASS_NAME.c_str(), "InferShapeReplaceGraph: GetInputDesc failed.");
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    if (ge::GeUtils::InferShape(*replaceGraph, inputShapes) != ge::SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "GeUtils::InferShape failed for replace graph.");
        return false;
    }
    return true;
}

ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                           const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                           const char* passName)
{
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    std::string patternNameStr;
    if (!ops::GetPatternNameStr(matchResult, patternNameStr, PASS_NAME.c_str())) {
        OPS_LOG_E(passName, "Get pattern graph name failed in BuildReplaceGraph.");
        return nullptr;
    }
    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, patternNameStr)) {
        OPS_LOG_E(passName, "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::GNode mc2Node;
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, MC2_CAPTURE_IDX, PASS_NAME.c_str(),
                                 "Capture MC2 node failed.")) {
        return nullptr;
    }

    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, passName, "Get Attr group failed.");
    ge::AscendString reduceOp;
    OP_LOGE_IF(mc2Node.GetAttr("reduce_op", reduceOp) != ge::GRAPH_SUCCESS, nullptr, passName,
               "Get Attr reduce_op failed.");
    bool isTransA = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_a", isTransA) != ge::GRAPH_SUCCESS, nullptr, passName,
               "Get Attr is_trans_a failed.");
    bool isTransB = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_b", isTransB) != ge::GRAPH_SUCCESS, nullptr, passName,
               "Get Attr is_trans_b failed.");
    // 吸收 x2 Transpose 后置反 is_trans_b
    isTransB = !isTransB;

    int64_t commTurn = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_turn", commTurn) != ge::GRAPH_SUCCESS, nullptr, passName,
               "Get Attr comm_turn failed.");
    int64_t antiquantGroupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("antiquant_group_size", antiquantGroupSize) != ge::GRAPH_SUCCESS, nullptr, passName,
               "Get Attr antiquant_group_size failed.");

    const Mc2OpKind opKind = ParseOpKind(patternNameStr);
    if (opKind == Mc2OpKind::Mar) {
        int64_t groupSize = 0;
        OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, passName,
                   "Get Attr group_size failed.");
        // y_dtype 常为 UNDEFINED：非量化跟随 x1，量化等继承原输出，避免重建后落到 DT_FLOAT
        int64_t yDtype = static_cast<int64_t>(ge::DT_UNDEFINED);
        OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, passName,
                   "Get Attr y_dtype failed.");
        if (yDtype == static_cast<int64_t>(ge::DT_UNDEFINED)) {
            ge::TensorDesc x1Desc;
            if (mc2Node.GetInputDesc(static_cast<int>(ops::MC2InputIdx::K_X1), x1Desc) == ge::GRAPH_SUCCESS &&
                (x1Desc.GetDataType() == ge::DT_FLOAT16 || x1Desc.GetDataType() == ge::DT_BF16)) {
                yDtype = static_cast<int64_t>(x1Desc.GetDataType());
            } else {
                ge::TensorDesc yDesc;
                if (mc2Node.GetOutputDesc(0, yDesc) == ge::GRAPH_SUCCESS && yDesc.GetDataType() != ge::DT_UNDEFINED) {
                    yDtype = static_cast<int64_t>(yDesc.GetDataType());
                }
            }
        }
        int64_t commQuantMode = 0;
        OP_LOGE_IF(mc2Node.GetAttr("comm_quant_mode", commQuantMode) != ge::GRAPH_SUCCESS, nullptr, passName,
                   "Get Attr comm_quant_mode failed.");
        ge::AscendString commMode;
        OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, passName,
                   "Get Attr comm_mode failed.");
        auto y = ge::es::MatmulAllReduce(inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, nullptr,
                                         inputTensors.rAntiquantScale, inputTensors.rAntiquantOffset,
                                         inputTensors.rDequantScale, nullptr, nullptr, nullptr, group.GetString(),
                                         reduceOp.GetString(), isTransA, isTransB, commTurn, antiquantGroupSize,
                                         groupSize, yDtype, commQuantMode, commMode.GetString());
        return replaceGraphBuilder.BuildAndReset({y});
    }

    float epsilon = 1e-6f;
    (void)mc2Node.GetAttr("epsilon", epsilon);

    if (opKind == Mc2OpKind::Arn) {
        auto out = ge::es::MatmulAllReduceAddRmsNorm(
            inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rResidual, inputTensors.rGamma,
            inputTensors.rAntiquantScale, inputTensors.rAntiquantOffset, inputTensors.rDequantScale, group.GetString(),
            reduceOp.GetString(), isTransA, isTransB, commTurn, antiquantGroupSize, epsilon);
        return replaceGraphBuilder.BuildAndReset({out.y, out.norm_out});
    }

    auto out = ge::es::InplaceMatmulAllReduceAddRmsNorm(
        inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rResidual, inputTensors.rGamma,
        inputTensors.rAntiquantScale, inputTensors.rAntiquantOffset, inputTensors.rDequantScale, group.GetString(),
        reduceOp.GetString(), isTransA, isTransB, commTurn, antiquantGroupSize, epsilon);
    return replaceGraphBuilder.BuildAndReset({out.ref_residual, out.norm_out});
}

ge::fusion::GraphUniqPtr DoReplacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                       const char* passName)
{
    OPS_LOG_D(passName, "Enter Replacement");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(passName, "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult, passName);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(passName, "BuildReplaceGraph failed.");
        return nullptr;
    }
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(passName, "InferShapeReplaceGraph failed.");
        return nullptr;
    }
    return replaceGraph;
}
} // namespace

ge::Status MatmulAllReduceTransposeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context)
{
    if (graph == nullptr || !GraphHasAnyOpType(*graph, {"MatmulAllReduce", OP_TYPE_ARN, OP_TYPE_INPLACE_ARN})) {
        OPS_LOG_D(PASS_NAME.c_str(), "Skip Patterns: no MatmulAllReduce/ARN anchor in graph.");
        return ge::SUCCESS;
    }
    return PatternFusionPass::Run(graph, pass_context);
}

std::vector<ge::fusion::PatternUniqPtr> MatmulAllReduceTransposeFusionPass::Patterns()
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter Patterns for MatmulAllReduceTransposeFusionPass");
    return BuildAllPatterns(PASS_NAME);
}

bool MatmulAllReduceTransposeFusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(PASS_NAME.c_str(), "Enter MeetRequirements for MatmulAllReduceTransposeFusionPass");
    // 不支持 310P
    if (IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_310P)) {
        OPS_LOG_D(PASS_NAME.c_str(), "Currently not supports NPUARCH 310P.");
        return false;
    }
    return MeetCore(matchResult, PASS_NAME.c_str());
}

ge::fusion::GraphUniqPtr MatmulAllReduceTransposeFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    return DoReplacement(matchResult, PASS_NAME.c_str());
}

REG_FUSION_PASS(MatmulAllReduceTransposeFusionPass).Stage(GetPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION

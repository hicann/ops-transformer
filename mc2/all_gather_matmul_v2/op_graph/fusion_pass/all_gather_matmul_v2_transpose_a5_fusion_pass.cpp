/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_matmul_v2_transpose_a5_fusion_pass.h"

#if GE_COMPILER_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_AllGatherMatmulV2.h" // es autogen header
#include "es_math_ops.h"          // math ops stub
#include "common/utils/op_mc2.h"
#include "mc2_platform_info.h"
#include "mc2_fusion_pass_utils.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "AllGatherMatmulV2TransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_SCALE = "HasScale";
const std::string PATTERN_X2SCALE_TRANSPOSE = "X2ScaleTranspose";
const std::string PATTERN_X2_BITCAST = "X2Bitcast";
const std::string PATTERN_X2SCALE_BITCAST = "X2ScaleBitcast";

const int64_t AGMMV2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_X2_CAPTURE_IDX = 1l;
const int64_t TRANSPOSE_X2SCALE_CAPTURE_IDX = 2l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t INPUT_NUM_NO_BIAS = 2;
const size_t INPUT_NUM_HAS_BIAS = 3;
const size_t INPUT_NUM_HAS_SCALE = 4; // x1_scale/x2_scale 成对出现；无 bias 至少 4 路
const size_t INPUT_NUM_HAS_BIAS_AND_SCALE = 5;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
// Pattern 中 Bitcast 的 type 仅占位；默认 matcher 不校验 IR attr，Replacement 用真实 MC2 输入 dtype
const ge::DataType BITCAST_PATTERN_DTYPE = ge::DT_HIFLOAT8;

// ES matcher 需覆盖会改变入边个数/拓扑的组合：bias、scale、x2_scale Transpose、以及 Transpose→Bitcast
struct OriginalGraphInfo {
    bool hasBias = false;
    bool hasScale = false;
    bool x2ScaleTranspose = false; // true 时必须 hasScale，且 Capture x2 与 x2_scale 两个 Transpose
    bool x2Bitcast = false;
    bool x2ScaleBitcast = false; // 仅 x2ScaleTranspose=true 时可开
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rX1Scale;
    ge::es::EsTensorHolder rX2Scale;
    ge::es::EsTensorHolder rQuantScale;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder* (*EsTransposeFunc)(EsCTensorHolder*, EsCTensorHolder*);

EsTransposeFunc GetEsTransposeFunc()
{
    void* handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("AllGatherMatmulV2TransposeA5FusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("AllGatherMatmulV2TransposeA5FusionPass", "dlsym EsTranspose failed");
        return nullptr;
    }
    return func;
}

ge::es::EsTensorHolder TransposeDL(const ge::es::EsTensorLike& x, const ge::es::EsTensorLike& perm)
{
    static EsTransposeFunc func = GetEsTransposeFunc();
    auto* builder = ge::es::ResolveBuilder(x, perm);
    auto result = func(x.ToTensorHolder(builder).GetCTensorHolder(), perm.ToTensorHolder(builder).GetCTensorHolder());
    return result;
}

ge::CustomPassStage GetAllGatherMatmulV2TransposeA5FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
} // namespace

static size_t GetPatternInputNum(const OriginalGraphInfo& info)
{
    if (info.hasScale) {
        return info.hasBias ? INPUT_NUM_HAS_BIAS_AND_SCALE : INPUT_NUM_HAS_SCALE;
    }
    return info.hasBias ? INPUT_NUM_HAS_BIAS : INPUT_NUM_NO_BIAS;
}

static ge::es::EsTensorHolder MakeTransposeNode(const ge::es::EsTensorHolder& x)
{
    return TransposeDL(x, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
}

static ge::es::EsTensorHolder MaybeBitcastAfterTranspose(bool needBitcast, const ge::es::EsTensorHolder& transposed)
{
    if (!needBitcast) {
        return transposed;
    }
    // Transpose → Bitcast → MC2；type 占位，matcher 默认只比 op type
    return ge::es::Bitcast(transposed, BITCAST_PATTERN_DTYPE);
}

static ge::fusion::PatternUniqPtr MakePattern(const OriginalGraphInfo& info)
{
    std::string patternName = FUSION_PASS_NAME;
    patternName += PATTERN_TRANSPOSE;
    if (info.x2ScaleTranspose) {
        patternName += PATTERN_X2SCALE_TRANSPOSE;
    }
    if (info.hasBias) {
        patternName += PATTERN_BIAS;
    }
    if (info.hasScale) {
        patternName += PATTERN_SCALE;
    }
    if (info.x2Bitcast) {
        patternName += PATTERN_X2_BITCAST;
    }
    if (info.x2ScaleBitcast) {
        patternName += PATTERN_X2SCALE_BITCAST;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    const size_t inputNum = GetPatternInputNum(info);
    auto inputs = graphBuilder.CreateInputs(inputNum);
    size_t idx = 0;
    auto x1 = inputs[idx++];
    auto x2 = inputs[idx++];
    ge::es::EsTensorHolder bias = nullptr;
    if (info.hasBias) {
        bias = inputs[idx++];
    }
    ge::es::EsTensorHolder x1Scale = nullptr;
    ge::es::EsTensorHolder x2Scale = nullptr;
    if (info.hasScale) {
        x1Scale = inputs[idx++];
        x2Scale = inputs[idx++];
    }

    auto transposeX2 = MakeTransposeNode(x2);
    auto x2In = MaybeBitcastAfterTranspose(info.x2Bitcast, transposeX2);
    ge::es::EsTensorHolder x2ScaleIn = x2Scale;
    ge::es::EsTensorHolder transposeX2Scale = nullptr;
    if (info.x2ScaleTranspose) {
        transposeX2Scale = MakeTransposeNode(x2Scale);
        x2ScaleIn = MaybeBitcastAfterTranspose(info.x2ScaleBitcast, transposeX2Scale);
    }

    const char* group = "";
    auto mc2Out = ge::es::AllGatherMatmulV2(x1, x2In, bias, x1Scale, x2ScaleIn, nullptr, group);
    auto graph = graphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2Out.y.GetProducer(), 0}).CaptureTensor({*transposeX2.GetProducer(), 0});
    if (info.x2ScaleTranspose) {
        pattern->CaptureTensor({*transposeX2Scale.GetProducer(), 0});
    }
    return pattern;
}
static bool PatternHasScaleInputs(const std::string& patternNameStr)
{
    return patternNameStr.find(PATTERN_SCALE) != std::string::npos;
}

static bool PatternHasX2ScaleTranspose(const std::string& patternNameStr)
{
    return patternNameStr.find(PATTERN_X2SCALE_TRANSPOSE) != std::string::npos;
}

static size_t GetMinRequiredSubgraphInputNum(const std::string& patternNameStr)
{
    const bool hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    if (PatternHasScaleInputs(patternNameStr)) {
        return hasBias ? INPUT_NUM_HAS_BIAS_AND_SCALE : INPUT_NUM_HAS_SCALE;
    }
    return hasBias ? INPUT_NUM_HAS_BIAS : INPUT_NUM_NO_BIAS;
}

static bool GetTransposePerm(const ge::GNode& transposeNode, std::vector<int64_t>& permValue)
{
    ge::TensorDesc permDesc;
    transposeNode.GetInputDesc(TRANSPOSE_PERM_IDX, permDesc);
    ge::Shape permShape = permDesc.GetShape();
    if (permShape.GetDimNum() != ONE_DIM_SIZE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation dim size must be 1, but got %zu.",
                  permShape.GetDimNum());
        return false;
    }

    ge::Tensor permTensor;
    if (transposeNode.GetInputConstData(TRANSPOSE_PERM_IDX, permTensor) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation const data.");
        return false;
    }

    ge::DataType permDType = permDesc.GetDataType();
    uint8_t* permData = permTensor.GetData();
    if (permData == nullptr) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation data is nullptr.");
        return false;
    }

    size_t size = 0;
    if (permDType == ge::DT_INT32) {
        size = permTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(static_cast<int64_t>(*(reinterpret_cast<int32_t*>(permData) + i)));
        }
    } else if (permDType == ge::DT_INT64) {
        size = permTensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(*(reinterpret_cast<int64_t*>(permData) + i));
        }
    } else {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation dtype must be int32 or int64.");
        return false;
    }

    return !permValue.empty();
}

static bool IsX2TransposePermValid(const std::vector<int64_t>& permValue)
{
    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2 transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }

    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2 transpose permutation last 2 dims must be swapped.");
        return false;
    }
    return true;
}

static bool IsX2ScaleTransposePermValid(const std::vector<int64_t>& permValue)
{
    if (permValue.size() < PERM_SIZE_TWO) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2_scale transpose permutation size must be at least 2, but got %zu.",
                  permValue.size());
        return false;
    }

    if (permValue[0] != 1 || permValue[1] != 0) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "x2_scale transpose permutation first 2 dims must be [1, 0].");
        return false;
    }
    return true;
}
// 吸掉 Transpose，保留 Bitcast，dtype 取自原 MC2 该口输入
static ge::es::EsTensorHolder MaybeKeepBitcast(const ge::GNode& mc2Node, int64_t inputIdx,
                                               const ge::es::EsTensorHolder& x, bool keepBitcast)
{
    if (!keepBitcast) {
        return x;
    }
    ge::TensorDesc inDesc;
    if (mc2Node.GetInputDesc(static_cast<int>(inputIdx), inDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "GetInputDesc failed for bitcast dtype, idx=%ld, fallback HIFLOAT8.",
                  inputIdx);
        return ge::es::Bitcast(x, BITCAST_PATTERN_DTYPE);
    }
    return ge::es::Bitcast(x, inDesc.GetDataType());
}

static bool IsX2ScaleDirectConnectAllowed(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                          const std::string& patternNameStr)
{
    if (!PatternHasScaleInputs(patternNameStr) || PatternHasX2ScaleTranspose(patternNameStr)) {
        return true;
    }

    ge::GNode mc2Node;
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, AGMMV2_CAPTURE_IDX, FUSION_PASS_NAME.c_str(),
                                 "Capture AllGatherMatmulV2 node failed.")) {
        return false;
    }

    const int32_t x2ScaleInputIdx = static_cast<int32_t>(ops::MC2V2InputIdx::K_X2SCALE);
    ge::TensorDesc x2ScaleDesc;
    if (mc2Node.GetInputDesc(x2ScaleInputIdx, x2ScaleDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Get x2_scale input desc failed.");
        return false;
    }

    const size_t x2ScaleDimNum = x2ScaleDesc.GetShape().GetDimNum();
    if (x2ScaleDimNum > ONE_DIM_SIZE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "x2_scale dim num is %zu (>1) without x2_scale Transpose, fusion is skipped.", x2ScaleDimNum);
        return false;
    }
    return true;
}

static bool ValidateCapturedTransposePerm(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                          int64_t captureIdx, bool isX2ScaleTranspose)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get captured Transpose node failed, capture idx %ld.", captureIdx);

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeOutput.node, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation, capture idx %ld.", captureIdx);
        return false;
    }

    return isX2ScaleTranspose ? IsX2ScaleTransposePermValid(permValue) : IsX2TransposePermValid(permValue);
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                 const std::string& patternNameStr)
{
    if (!ValidateCapturedTransposePerm(matchResult, TRANSPOSE_X2_CAPTURE_IDX, false)) {
        return false;
    }

    if (PatternHasX2ScaleTranspose(patternNameStr)) {
        return ValidateCapturedTransposePerm(matchResult, TRANSPOSE_X2SCALE_CAPTURE_IDX, true);
    }
    return true;
}

static bool CollectSubgraphInputsInfo(const std::vector<ge::fusion::SubgraphInput>& subGraphInputs,
                                      std::vector<ge::Shape>& inputShapes, std::vector<ge::DataType>& inputDTypes,
                                      std::vector<ge::Format>& inputFormats)
{
    inputShapes.clear();
    inputDTypes.clear();
    inputFormats.clear();
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo: matchNodes is empty.");
            return false;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo: GetInputDesc failed.");
            return false;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    return true;
}

static bool CreateReplaceGraphInputs(ReplaceGraphInputs& inputs, ge::es::EsGraphBuilder& replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                     const std::string& patternNameStr)
{
    // 按 MakePattern 的 CreateInputs 顺序映射 subgraph 边界
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo failed in CreateReplaceGraphInputs.");
        return false;
    }
    if (inputShapes.size() < INPUT_NUM_NO_BIAS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement without bias, size=%zu.",
                  inputShapes.size());
        return false;
    }

    const bool hasScaleInputs = PatternHasScaleInputs(patternNameStr);

    int64_t inputIdx = 0;
    inputs.rX1 = replaceGraphBuilder.CreateInput(inputIdx, "x1", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rX2 = replaceGraphBuilder.CreateInput(inputIdx, "x2", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;

    inputs.rBias = nullptr;
    if (patternNameStr.find(PATTERN_BIAS) != std::string::npos) {
        const size_t expectWithBias = hasScaleInputs ? INPUT_NUM_HAS_BIAS_AND_SCALE : INPUT_NUM_HAS_BIAS;
        if (inputShapes.size() < expectWithBias) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement with bias, size=%zu.",
                      inputShapes.size());
            return false;
        }
        inputs.rBias = replaceGraphBuilder.CreateInput(inputIdx, "bias", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }

    inputs.rX1Scale = nullptr;
    inputs.rX2Scale = nullptr;
    if (hasScaleInputs) {
        const size_t expectWithScale =
            patternNameStr.find(PATTERN_BIAS) != std::string::npos ? INPUT_NUM_HAS_BIAS_AND_SCALE : INPUT_NUM_HAS_SCALE;
        if (inputShapes.size() < expectWithScale) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement with scale, size=%zu.",
                      inputShapes.size());
            return false;
        }
        inputs.rX1Scale = replaceGraphBuilder.CreateInput(inputIdx, "x1_scale", inputDTypes[inputIdx],
                                                          inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
        inputs.rX2Scale = replaceGraphBuilder.CreateInput(inputIdx, "x2_scale", inputDTypes[inputIdx],
                                                          inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    }
    inputs.rQuantScale = nullptr;
    return true;
}

static ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to build replaceGraph EsGraphBuilder.");
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    std::string patternNameStr;
    if (!ops::GetPatternNameStr(matchResult, patternNameStr, FUSION_PASS_NAME.c_str())) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get pattern graph name failed in BuildReplaceGraph.");
        return nullptr;
    }

    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, patternNameStr)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::GNode mc2Node;
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, AGMMV2_CAPTURE_IDX, FUSION_PASS_NAME.c_str(),
                                 "Capture AllGatherMatmulV2 node failed.")) {
        return nullptr;
    }

    const bool keepX2Bitcast = patternNameStr.find(PATTERN_X2_BITCAST) != std::string::npos;
    const bool keepX2ScaleBitcast = patternNameStr.find(PATTERN_X2SCALE_BITCAST) != std::string::npos;
    auto x2In =
        MaybeKeepBitcast(mc2Node, static_cast<int64_t>(ops::MC2V2InputIdx::K_X2), inputTensors.rX2, keepX2Bitcast);
    ge::es::EsTensorHolder x2ScaleIn = MaybeKeepBitcast(mc2Node, static_cast<int64_t>(ops::MC2V2InputIdx::K_X2SCALE),
                                                        inputTensors.rX2Scale, keepX2ScaleBitcast);

    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    bool isTransA = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_a", isTransA) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr is_trans_a failed.");

    bool isTransB = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_b", isTransB) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr is_trans_b failed.");
    // 移除 Transpose 节点后，is_trans_b 置反
    isTransB = !isTransB;

    int64_t gatherIndex = 0;
    OP_LOGE_IF(mc2Node.GetAttr("gather_index", gatherIndex) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr gather_index failed.");

    int64_t commTurn = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_turn", commTurn) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_turn failed.");

    int64_t rankSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("rank_size", rankSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr rank_size failed.");

    int64_t blockSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("block_size", blockSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr block_size failed.");

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    bool isGatherOut = true;
    OP_LOGE_IF(mc2Node.GetAttr("is_gather_out", isGatherOut) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr is_gather_out failed.");

    bool isAmaxOut = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_amax_out", isAmaxOut) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr is_amax_out failed.");

    int64_t yDtype = 0;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");

    ge::AscendString commMode;
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");

    auto mc2Out = ge::es::AllGatherMatmulV2(inputTensors.rX1, x2In, inputTensors.rBias, inputTensors.rX1Scale,
                                            x2ScaleIn, inputTensors.rQuantScale, group.GetString(), isTransA, isTransB,
                                            gatherIndex, commTurn, rankSize, blockSize, groupSize, isGatherOut,
                                            isAmaxOut, yDtype, commMode.GetString());

    return replaceGraphBuilder.BuildAndReset({mc2Out.y, mc2Out.gather_out, mc2Out.amax_out});
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr& replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph: matchNodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputDesc failed.");
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
    }
    if (ge::GeUtils::InferShape(*replaceGraph, inputShapes) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "GeUtils::InferShape failed for replace graph.");
        return false;
    }
    return true;
}

ge::Status AllGatherMatmulV2TransposeA5FusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context)
{
    if (graph == nullptr || !GraphHasAnyOpType(*graph, {"AllGatherMatmulV2"})) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip Patterns: no AllGatherMatmulV2 in graph.");
        return ge::SUCCESS;
    }
    return PatternFusionPass::Run(graph, pass_context);
}

std::vector<ge::fusion::PatternUniqPtr> AllGatherMatmulV2TransposeA5FusionPass::Patterns()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Patterns for AllGatherMatmulV2TransposeA5FusionPass");
    // 仅 x2 Transpose：hasBias × hasScale × x2Bitcast → 8 种
    // x2 + x2_scale 双 Transpose：hasBias × x2Bitcast × x2ScaleBitcast → 8 种；共 16 种
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (bool hasBias : {false, true}) {
        for (bool hasScale : {false, true}) {
            for (bool x2Bitcast : {false, true}) {
                patternGraphs.emplace_back(MakePattern({hasBias, hasScale, false, x2Bitcast, false}));
            }
        }
        for (bool x2Bitcast : {false, true}) {
            for (bool x2ScaleBitcast : {false, true}) {
                patternGraphs.emplace_back(MakePattern({hasBias, true, true, x2Bitcast, x2ScaleBitcast}));
            }
        }
    }
    return patternGraphs;
}

bool AllGatherMatmulV2TransposeA5FusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AllGatherMatmulV2TransposeA5FusionPass");

    // 9.0.0 版本前运行降级stage 空跑
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip in MeetRequirements when cann version not compatible.");
        return false;
    }

    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    std::string patternNameStr;
    if (!ops::GetPatternNameStr(matchResult, patternNameStr, FUSION_PASS_NAME.c_str())) {
        return false;
    }

    if (!IsTransposePermValid(matchResult, patternNameStr)) {
        return false;
    }

    // AGMMV2 特有：x2_scale 维数 > 1 时必须有 x2_scale Transpose
    if (!IsX2ScaleDirectConnectAllowed(matchResult, patternNameStr)) {
        return false;
    }

    // 按 pattern 期望的 subgraph 输入个数校验（bias/scale 组合）
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in MeetRequirements.");
        return false;
    }
    const size_t expectNum = GetMinRequiredSubgraphInputNum(patternNameStr);
    if (subgraphInputs.size() != expectNum) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip pattern=%s: subgraph size=%zu, expect=%zu.", patternNameStr.c_str(),
                  subgraphInputs.size(), expectNum);
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr AllGatherMatmulV2TransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AllGatherMatmulV2TransposeA5FusionPass");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }

    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "BuildReplaceGraph failed in Replacement.");
        return nullptr;
    }
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(AllGatherMatmulV2TransposeA5FusionPass).Stage(GetAllGatherMatmulV2TransposeA5FusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION

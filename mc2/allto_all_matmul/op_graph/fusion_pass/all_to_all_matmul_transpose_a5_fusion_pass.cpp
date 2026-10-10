/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_all_matmul_transpose_a5_fusion_pass.h"

#if GE_COMPILER_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_AlltoAllMatmul.h" // es autogen header
#include "es_math_ops.h"       // math ops stub
#include "mc2_platform_info.h"
#include "mc2_fusion_pass_utils.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "AllToAllMatmulTransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_SCALE = "HasScale"; // MX：scale 固定进 pattern，仅写入名字便于区分
const std::string PATTERN_BITCAST = "X2Bitcast";

const int64_t MX_QUANT_MODE = 6;
const int64_t ALL2ALLMM_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_CAPTURE_IDX = 1l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const int64_t X2_INPUT_IDX = 1l; // IR: x1 / x2 / ...
const size_t INPUT_NUM_NO_BIAS = 4;
const size_t INPUT_NUM_HAS_BIAS = 5;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
// Pattern 中 Bitcast 的 type 仅占位；默认 matcher 不校验 IR attr，Replacement 用真实 MC2 输入 dtype
const ge::DataType BITCAST_PATTERN_DTYPE = ge::DT_HIFLOAT8;

struct OriginalGraphInfo {
    bool hasBias = false;
    bool hasBitcast = false; // true：Transpose → Bitcast → x2
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rX1Scale;
    ge::es::EsTensorHolder rX2Scale;
    ge::es::EsTensorHolder rCommScale;
    ge::es::EsTensorHolder rX1Offset;
    ge::es::EsTensorHolder rX2Offset;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder* (*EsTransposeFunc)(EsCTensorHolder*, EsCTensorHolder*);

EsTransposeFunc GetEsTransposeFunc()
{
    void* handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("AllToAllMatmulTransposeA5FusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("AllToAllMatmulTransposeA5FusionPass", "dlsym EsTranspose failed");
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

ge::CustomPassStage GetAllToAllMatmulTransposeA5FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
} // namespace

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
    if (info.hasBias) {
        patternName += PATTERN_BIAS;
    }
    patternName += PATTERN_SCALE;
    if (info.hasBitcast) {
        patternName += PATTERN_BITCAST;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    // MX：x1/x2/scale 常驻，bias 可选；comm/offset 不进 pattern
    const size_t inputNum = info.hasBias ? INPUT_NUM_HAS_BIAS : INPUT_NUM_NO_BIAS;
    auto inputs = graphBuilder.CreateInputs(inputNum);
    size_t idx = 0;
    auto x1 = inputs[idx++];
    auto x2 = inputs[idx++];
    ge::es::EsTensorHolder bias = nullptr;
    if (info.hasBias) {
        bias = inputs[idx++];
    }
    auto x1Scale = inputs[idx++];
    auto x2Scale = inputs[idx++];

    auto transpose = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    auto x2In = MaybeBitcastAfterTranspose(info.hasBitcast, transpose);

    const char* group = "";
    int64_t worldSize = 0;
    auto mc2 = ge::es::AlltoAllMatmul(x1, x2In, bias, x1Scale, x2Scale, nullptr, nullptr, nullptr, group, worldSize);
    auto graph = graphBuilder.BuildAndReset({mc2.y, mc2.all2all_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.y.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

static bool IsMXQuantMode(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(ALL2ALLMM_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get AlltoAllMatmul node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    if (mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS ||
        mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get quant mode Attr failed.");
        return false;
    }
    if (x1QuantMode != MX_QUANT_MODE || x2QuantMode != MX_QUANT_MODE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Quant mode is not MX QUANT (x1=%ld, x2=%ld), fusion skipped.", x1QuantMode,
                  x2QuantMode);
        return false;
    }
    return true;
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

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(TRANSPOSE_CAPTURE_IDX, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get Transpose node failed.");
    auto transposeNode = transposeOutput.node;

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeNode, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation.");
        return false;
    }

    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }

    // x2 路径：最后两维必须互换，即 2维 [1,0] / 3维 [0,2,1]
    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation last 2 dims must be swapped.");
        return false;
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

    int64_t inputIdx = 0;
    inputs.rX1 = replaceGraphBuilder.CreateInput(inputIdx, "x1", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rX2 = replaceGraphBuilder.CreateInput(inputIdx, "x2", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                 inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rBias = nullptr;
    if (patternNameStr.find(PATTERN_BIAS) != std::string::npos) {
        if (inputShapes.size() < INPUT_NUM_HAS_BIAS) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement with bias, size=%zu.",
                      inputShapes.size());
            return false;
        }
        inputs.rBias = replaceGraphBuilder.CreateInput(inputIdx, "bias", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
        ++inputIdx;
    }
    inputs.rX1Scale = replaceGraphBuilder.CreateInput(inputIdx, "x1_scale", inputDTypes[inputIdx],
                                                      inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rX2Scale = replaceGraphBuilder.CreateInput(inputIdx, "x2_scale", inputDTypes[inputIdx],
                                                      inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    inputs.rCommScale = nullptr;
    inputs.rX1Offset = nullptr;
    inputs.rX2Offset = nullptr;
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

static ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                                                  const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
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
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, ALL2ALLMM_CAPTURE_IDX, FUSION_PASS_NAME.c_str(),
                                 "Capture AlltoAllMatmul node failed.")) {
        return nullptr;
    }

    const bool keepX2Bitcast = patternNameStr.find(PATTERN_BITCAST) != std::string::npos;
    auto x2In = MaybeKeepBitcast(mc2Node, X2_INPUT_IDX, inputTensors.rX2, keepX2Bitcast);

    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    int64_t worldSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("world_size", worldSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr world_size failed.");

    std::vector<int64_t> all2allAxes;
    OP_LOGE_IF(mc2Node.GetAttr("all2all_axes", all2allAxes) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr all2all_axes failed.");

    int64_t yDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");

    int64_t x1QuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_mode failed.");

    int64_t x2QuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x2_quant_mode failed.");

    int64_t commQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_mode", commQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr comm_quant_mode failed.");

    int64_t x1QuantDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("x1_quant_dtype", x1QuantDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr x1_quant_dtype failed.");

    int64_t commQuantDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_dtype", commQuantDtype) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr comm_quant_dtype failed.");

    bool transposeX1 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x1", transposeX1) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x1 failed.");

    bool transposeX2 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x2", transposeX2) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x2 failed.");
    // MX tiling 要求吸收 x2 Transpose 后 transpose_x2 必须为 true
    transposeX2 = true;

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    ge::AscendString commMode;
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");

    bool alltoallOutFlag = true;
    OP_LOGE_IF(mc2Node.GetAttr("alltoall_out_flag", alltoallOutFlag) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr alltoall_out_flag failed.");

    auto mc2 = ge::es::AlltoAllMatmul(inputTensors.rX1, x2In, inputTensors.rBias, inputTensors.rX1Scale,
                                      inputTensors.rX2Scale, inputTensors.rCommScale, inputTensors.rX1Offset,
                                      inputTensors.rX2Offset, group.GetString(), worldSize, all2allAxes, yDtype,
                                      x1QuantMode, x2QuantMode, commQuantMode, x1QuantDtype, commQuantDtype,
                                      transposeX1, transposeX2, groupSize, commMode.GetString(), alltoallOutFlag);
    return replaceGraphBuilder.BuildAndReset({mc2.y, mc2.all2all_out});
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

ge::Status AllToAllMatmulTransposeA5FusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context)
{
    if (graph == nullptr || !GraphHasAnyOpType(*graph, {"AlltoAllMatmul"})) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip Patterns: no AlltoAllMatmul in graph.");
        return ge::SUCCESS;
    }
    return PatternFusionPass::Run(graph, pass_context);
}

std::vector<ge::fusion::PatternUniqPtr> AllToAllMatmulTransposeA5FusionPass::Patterns()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Patterns for AllToAllMatmulTransposeA5FusionPass");
    // ES：排列 hasBias × hasBitcast；MX scale 固定 → 共 4 种
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (bool hasBias : {false, true}) {
        for (bool hasBitcast : {false, true}) {
            patternGraphs.emplace_back(MakePattern({hasBias, hasBitcast}));
        }
    }
    return patternGraphs;
}

bool AllToAllMatmulTransposeA5FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AllToAllMatmulTransposeA5FusionPass");

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

    if (!IsMXQuantMode(matchResult)) {
        return false;
    }

    if (!IsTransposePermValid(matchResult)) {
        return false;
    }

    // 按 subgraph 输入个数区分有无 bias
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in MeetRequirements.");
        return false;
    }
    std::string patternNameStr;
    if (!ops::GetPatternNameStr(matchResult, patternNameStr, FUSION_PASS_NAME.c_str())) {
        return false;
    }
    const bool hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    const size_t expectNum = hasBias ? INPUT_NUM_HAS_BIAS : INPUT_NUM_NO_BIAS;
    if (subgraphInputs.size() != expectNum) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip pattern=%s: subgraph size=%zu, expect=%zu.", patternNameStr.c_str(),
                  subgraphInputs.size(), expectNum);
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr AllToAllMatmulTransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AllToAllMatmulTransposeA5FusionPass");
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

REG_FUSION_PASS(AllToAllMatmulTransposeA5FusionPass).Stage(GetAllToAllMatmulTransposeA5FusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION

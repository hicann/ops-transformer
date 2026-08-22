/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_all_to_all_transpose_a5_fusion_pass.h"

#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MatmulAlltoAll.h" // es autogen header
#include "es_TransposeD.h"     // es autogen header for stub op
#include "es_math_ops.h"       // math ops stub
#include "mc2_platform_info.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "MatmulAllToAllTransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_TRANSPOSE_D = "TransposeD";
const std::string PATTERN_BIAS = "HasBias";
const int64_t MMALL2ALL_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_CAPTURE_IDX = 1l;
const int64_t MX_QUANT_MODE = 6;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t INPUT_NUM_NO_BIAS = 4;
const size_t INPUT_NUM_HAS_BIAS = 5;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;

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
typedef EsCTensorHolder *(*EsTransposeFunc)(EsCTensorHolder *, EsCTensorHolder *);

EsTransposeFunc GetEsTransposeFunc()
{
    void *handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("MatmulAllToAllTransposeA5FusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("MatmulAllToAllTransposeA5FusionPass", "dlsym EsTranspose failed");
        return nullptr;
    }
    return func;
}

ge::es::EsTensorHolder TransposeDL(const ge::es::EsTensorLike &x, const ge::es::EsTensorLike &perm)
{
    static EsTransposeFunc func = GetEsTransposeFunc();
    auto *builder = ge::es::ResolveBuilder(x, perm);
    auto result = func(x.ToTensorHolder(builder).GetCTensorHolder(), perm.ToTensorHolder(builder).GetCTensorHolder());
    return result;
}

ge::CustomPassStage GetMatmulAlltoAllTransposeA5FusionPassStage()
{
    int32_t version = 0;
    aclsysGetVersionNum("ge_compiler", &version);
    if (version >= GRAPH_FUSION_SUPPORT_VERSION) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
} // namespace

static ge::fusion::PatternUniqPtr MakePatternForTranspose(bool hasBias)
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transpose = TransposeDL(x2, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
    // 必选attr 设置默认值
    std::string group = std::string("");
    int64_t worldSize = 0;
    // 有输入的data节点传入默认, 无输入的可选节点传入nullptr, 若后续场景扩充, 需仿照bias 扩充子pattern
    auto mc2 = ge::es::MatmulAlltoAll(x1, transpose, hasBias ? bias : nullptr, x1Scale, x2Scale, nullptr, nullptr,
                                      nullptr, group.c_str(), worldSize);
    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.GetProducer(), 0}).CaptureTensor({*transpose.GetProducer(), 0});
    return pattern;
}

static ge::fusion::PatternUniqPtr MakePatternForTransposeD(bool hasBias)
{
    std::string patternName = FUSION_PASS_NAME + PATTERN_TRANSPOSE_D;
    if (hasBias) {
        patternName += PATTERN_BIAS;
    }
    auto graphBuilder = ge::es::EsGraphBuilder(FUSION_PASS_NAME.c_str());
    auto [x1, x2, bias, x1Scale, x2Scale] = graphBuilder.CreateInputs<5>();
    auto transposeD = ge::es::TransposeD(x2, std::vector<int64_t>{1, 0});
    std::string group = std::string("");
    int64_t worldSize = 0;
    auto mc2 = ge::es::MatmulAlltoAll(x1, transposeD, hasBias ? bias : nullptr, x1Scale, x2Scale, nullptr, nullptr,
                                      nullptr, group.c_str(), worldSize);
    auto graph = graphBuilder.BuildAndReset({mc2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mc2.GetProducer(), 0}).CaptureTensor({*transposeD.GetProducer(), 0});
    return pattern;
}

static bool IsMXQuantMode(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(MMALL2ALL_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get MatmulAlltoAll node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    if (mc2Node.GetAttr("x1_quant_mode", x1QuantMode) != ge::GRAPH_SUCCESS ||
        mc2Node.GetAttr("x2_quant_mode", x2QuantMode) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get quant mode Attr failed.");
        return false;
    }
    return (x1QuantMode == MX_QUANT_MODE && x2QuantMode == MX_QUANT_MODE);
}

static bool GetTransposePerm(const ge::GNode &transposeNode, std::vector<int64_t> &permValue)
{
    ge::AscendString nodeType("");
    transposeNode.GetType(nodeType);
    std::string nodeTypeStr = nodeType.GetString();
    if (nodeTypeStr == PATTERN_TRANSPOSE_D) { // return when node is TransposeD
        OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Now in TransposeD node for getting perm");
        if (transposeNode.GetAttr("perm", permValue) != ge::GRAPH_SUCCESS) {
            OPS_LOG_W(FUSION_PASS_NAME.c_str(), "Get perm Attr from TransposeD node failed.");
            return false;
        }
        return !permValue.empty();
    }

    // Transpose 的场景
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
    uint8_t *permData = permTensor.GetData();

    if (permData == nullptr) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation data is nullptr.");
        return false;
    }

    size_t size = 0;
    if (permDType == ge::DT_INT32) {
        size = permTensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(static_cast<int64_t>(*(reinterpret_cast<int32_t *>(permData) + i)));
        }
    } else if (permDType == ge::DT_INT64) {
        size = permTensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
            permValue.emplace_back(*(reinterpret_cast<int64_t *>(permData) + i));
        }
    } else {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation dtype must be int32 or int64.");
        return false;
    }

    return !permValue.empty();
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
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

    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation last 2 dims must be [1, 0].");
        return false;
    }
    return true;
}

static void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput> &subGraphInputs,
                          std::vector<ge::Shape> &inputShapes, std::vector<ge::DataType> &inputDTypes,
                          std::vector<ge::Format> &inputFormats)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter GetInputsInfo for MatmulAllToAllTransposeA5FusionPass");
    for (const auto &subGraphInput : subGraphInputs) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputsInfo: collecting input datas.");
        auto matchNodes = subGraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputsInfo: matchNodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0); // NodeIo, input node 被多个当前pattern的节点消费时, 序号递增
        ge::TensorDesc tmpDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tmpDesc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(FUSION_PASS_NAME.c_str(), "GetInputDesc failed.");
            continue;
        }
        inputShapes.emplace_back(tmpDesc.GetShape());
        inputDTypes.emplace_back(tmpDesc.GetDataType());
        inputFormats.emplace_back(tmpDesc.GetOriginFormat());
    }
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Finish GetInputsInfo for MatmulAllToAllTransposeA5FusionPass");
}

static bool InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr &replaceGraph,
                                   const std::vector<ge::fusion::SubgraphInput> &subgraphInputs)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter InferShapeReplaceGraph for MatmulAllToAllTransposeA5FusionPass");
    std::vector<ge::Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
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
    return (ge::GeUtils::InferShape(*replaceGraph, inputShapes) == ge::SUCCESS);
}

static bool CreateReplaceGraphInputs(ReplaceGraphInputs &inputs, ge::es::EsGraphBuilder &replaceGraphBuilder,
                                     const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                     const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to create inputs in Replacement.");
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats);
    if (inputShapes.size() < INPUT_NUM_NO_BIAS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement without bias.");
        return false;
    }
    ge::AscendString patternName = "";
    matchResult->GetPatternGraph().GetName(patternName);
    std::string patternNameStr = patternName.GetString();

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
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement with bias.");
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
    // 当前支持场景下, 下面三个optional input 都为空, 后续扩展时可按上述inputs 仿写
    inputs.rCommScale = nullptr;
    inputs.rX1Offset = nullptr;
    inputs.rX2Offset = nullptr;
    return true;
}

static ge::fusion::GraphUniqPtr BuildReplaceGraph(const std::vector<ge::fusion::SubgraphInput> &subgraphInputs,
                                                  const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Start to build replaceGraph EsGraphBuilder.");
    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");

    ReplaceGraphInputs inputTensors;
    if (!CreateReplaceGraphInputs(inputTensors, replaceGraphBuilder, subgraphInputs, matchResult)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CreateReplaceGraphInputs failed.");
        return nullptr;
    }

    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(MMALL2ALL_CAPTURE_IDX, mc2NodeIo) != ge::SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Capture MatmulAlltoAll node failed.");
    ge::GNode mc2Node = mc2NodeIo.node;

    ge::AscendString group = "";
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    int64_t worldSize = -1;
    OP_LOGE_IF(mc2Node.GetAttr("world_size", worldSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr world_size failed.");

    std::vector<int64_t> all2allAxes = {-1, -2};
    OP_LOGE_IF(mc2Node.GetAttr("all2all_axes", all2allAxes) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr all2all_axes failed.");

    int64_t yDType = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDType) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
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

    int64_t commQuantDType = 28;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_dtype", commQuantDType) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr comm_quant_dtype failed.");

    bool transposeX1 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x1", transposeX1) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x1 failed.");

    bool transposeX2 = false;
    OP_LOGE_IF(mc2Node.GetAttr("transpose_x2", transposeX2) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr transpose_x2 failed.");
    // 移除Transpose节点后，transpose x2 attr 置反
    transposeX2 = !transposeX2;

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    ge::AscendString commMode = "";
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");

    auto mc2 = ge::es::MatmulAlltoAll(inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, inputTensors.rX1Scale,
                                      inputTensors.rX2Scale, inputTensors.rCommScale, inputTensors.rX1Offset,
                                      inputTensors.rX2Offset, group.GetString(), worldSize, all2allAxes, yDType,
                                      x1QuantMode, x2QuantMode, commQuantMode, commQuantDType, transposeX1, transposeX2,
                                      groupSize, commMode.GetString());

    return replaceGraphBuilder.BuildAndReset({mc2});
}

std::vector<ge::fusion::PatternUniqPtr> MatmulAllToAllTransposeA5FusionPass::Patterns()
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter Patterns for MatmulAllToAllTransposeA5FusionPass");
    bool hasBias = true;
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePatternForTranspose(hasBias));
    patternGraphs.emplace_back(MakePatternForTranspose(!hasBias));
    patternGraphs.emplace_back(MakePatternForTransposeD(hasBias));
    patternGraphs.emplace_back(MakePatternForTransposeD(!hasBias));
    return patternGraphs;
}

bool MatmulAllToAllTransposeA5FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for MatmulAllToAllTransposeA5FusionPass");

    // 9.0.0 版本前运行降级stage 空跑
    int32_t geCompilerVersion = 0;
    aclsysGetVersionNum("ge_compiler", &geCompilerVersion);
    if (geCompilerVersion < GRAPH_FUSION_SUPPORT_VERSION) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip in MeetRequirements when cann version not compatible.");
        return false;
    }

    // 是否ascend950
    if (!IsTargetPlatformNpuArch(FUSION_PASS_NAME.c_str(), NPUARCH_A5)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Check target platform fail!");
        return false;
    }

    // 是否mx quant
    if (!IsMXQuantMode(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Quant mode is not MX QUANT, fusion will be skipped.");
        return false;
    }

    // 是否transpose 条件吻合
    if (!IsTransposePermValid(matchResult)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Transpose permutation is not valid.");
        return false;
    }
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
    return true;
}

ge::fusion::GraphUniqPtr MatmulAllToAllTransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult> &matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for MatmulAllToAllTransposeA5FusionPass");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    size_t subgraphInputSize = subgraphInputs.size();
    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Subgraph input size is %zu.", subgraphInputSize);

    // build replace graph with transpose x2 reversed
    ge::fusion::GraphUniqPtr replaceGraph = BuildReplaceGraph(subgraphInputs, matchResult);
    if (replaceGraph == nullptr) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "BuildReplaceGraph failed in Replacement.");
        return nullptr;
    }
    // infershape for node x2
    if (!InferShapeReplaceGraph(replaceGraph, subgraphInputs)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "InferShapeReplaceGraph failed in Replacement.");
        return nullptr;
    }
    return std::move(replaceGraph);
}

REG_FUSION_PASS(MatmulAllToAllTransposeA5FusionPass).Stage(GetMatmulAlltoAllTransposeA5FusionPassStage());
} // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION

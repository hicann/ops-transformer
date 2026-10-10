/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_all_reduce_transpose_a5_fusion_pass.h"

#if GE_COMPILER_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_MatmulAllReduce.h" // es autogen header
#include "es_math_ops.h"        // math ops stub
#include "common/utils/op_mc2.h"
#include "mc2_platform_info.h"
#include "mc2_fusion_pass_utils.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "MatmulAllReduceTransposeA5FusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_BIAS = "HasBias";
const std::string PATTERN_SCALE = "HasScale";
const std::string PATTERN_OFFSET = "HasOffset";
const std::string PATTERN_DEQUANT = "HasDequant";
const std::string PATTERN_SCALE_TRANSPOSE = "ScaleTranspose";
const std::string PATTERN_OFFSET_TRANSPOSE = "OffsetTranspose";
const std::string PATTERN_DEQUANT_TRANSPOSE = "DequantTranspose";
const std::string PATTERN_DEQUANT_BITCAST = "DequantBitcast";

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
// Pattern 中 Bitcast 的 type 仅占位；默认 matcher 不校验 IR attr，Replacement 用真实 MC2 输入 dtype
const ge::DataType BITCAST_PATTERN_DTYPE = ge::DT_HIFLOAT8;

enum class TransposePort {
    X2,
    Scale,
    Offset,
    Dequant
};

// ES：仅排列影响匹配拓扑的组合；x3/pertoken/comm_quant 固定 nullptr
struct OriginalGraphInfo {
    bool hasBias = false;
    bool hasScale = false;
    bool hasOffset = false;
    bool hasDequant = false;
    bool scaleTranspose = false;   // 需 hasScale
    bool offsetTranspose = false;  // 需 hasOffset
    bool dequantTranspose = false; // 需 hasDequant
    bool dequantBitcast = false;   // 需 dequantTranspose
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rX1;
    ge::es::EsTensorHolder rX2;
    ge::es::EsTensorHolder rBias;
    ge::es::EsTensorHolder rAntiquantScale;
    ge::es::EsTensorHolder rAntiquantOffset;
    ge::es::EsTensorHolder rDequantScale;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder* (*EsTransposeFunc)(EsCTensorHolder*, EsCTensorHolder*);

EsTransposeFunc GetEsTransposeFunc()
{
    void* handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("MatmulAllReduceTransposeA5FusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("MatmulAllReduceTransposeA5FusionPass", "dlsym EsTranspose failed");
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

ge::CustomPassStage GetMatmulAllReduceTransposeA5FusionPassStage()
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
    size_t n = 2; // x1 / x2
    if (info.hasBias) {
        ++n;
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

static ge::es::EsTensorHolder MakeTransposeNode(const ge::es::EsTensorHolder& x)
{
    return TransposeDL(x, ge::es::EsTensorLike(std::vector<int64_t>{1, 0}));
}

static ge::es::EsTensorHolder MaybeBitcastAfterTranspose(bool needBitcast, const ge::es::EsTensorHolder& transposed)
{
    if (!needBitcast) {
        return transposed;
    }
    // 仅 dequant 支持 Transpose → Bitcast → MC2
    return ge::es::Bitcast(transposed, BITCAST_PATTERN_DTYPE);
}

static ge::fusion::PatternUniqPtr MakePattern(const OriginalGraphInfo& info)
{
    std::string patternName = FUSION_PASS_NAME;
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
    if (info.dequantTranspose) {
        patternName += PATTERN_DEQUANT_TRANSPOSE;
    }
    if (info.dequantBitcast) {
        patternName += PATTERN_DEQUANT_BITCAST;
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
    ge::es::EsTensorHolder antiquantScale = nullptr;
    if (info.hasScale) {
        antiquantScale = inputs[idx++];
    }
    ge::es::EsTensorHolder antiquantOffset = nullptr;
    if (info.hasOffset) {
        antiquantOffset = inputs[idx++];
    }
    ge::es::EsTensorHolder dequantScale = nullptr;
    if (info.hasDequant) {
        dequantScale = inputs[idx++];
    }

    // x2 必挂 Transpose（无 Bitcast）
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

    ge::es::EsTensorHolder transposeDequant = nullptr;
    ge::es::EsTensorHolder dequantIn = dequantScale;
    if (info.dequantTranspose) {
        transposeDequant = MakeTransposeNode(dequantScale);
        dequantIn = MaybeBitcastAfterTranspose(info.dequantBitcast, transposeDequant);
    }

    const char* group = "";
    // x3 / pertoken / comm_quant 固定 nullptr（不参与融合效果）
    auto y = ge::es::MatmulAllReduce(x1, transposeX2, bias, nullptr, scaleIn, offsetIn, dequantIn, nullptr, nullptr,
                                     nullptr, group);
    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    // Capture：0=mc2；其后按 x2 → scale → offset → dequant 顺序追加 Transpose
    pattern->CaptureTensor({*y.GetProducer(), 0}).CaptureTensor({*transposeX2.GetProducer(), 0});
    if (info.scaleTranspose) {
        pattern->CaptureTensor({*transposeScale.GetProducer(), 0});
    }
    if (info.offsetTranspose) {
        pattern->CaptureTensor({*transposeOffset.GetProducer(), 0});
    }
    if (info.dequantTranspose) {
        pattern->CaptureTensor({*transposeDequant.GetProducer(), 0});
    }
    return pattern;
}
static size_t GetExpectedSubgraphInputNum(const std::string& patternNameStr)
{
    OriginalGraphInfo info;
    info.hasBias = patternNameStr.find(PATTERN_BIAS) != std::string::npos;
    info.hasScale = patternNameStr.find(PATTERN_SCALE) != std::string::npos;
    info.hasOffset = patternNameStr.find(PATTERN_OFFSET) != std::string::npos;
    info.hasDequant = patternNameStr.find(PATTERN_DEQUANT) != std::string::npos;
    return GetPatternInputNum(info);
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

static bool IsLastTwoDimsSwapPermValid(const std::vector<int64_t>& permValue)
{
    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "transpose permutation size must be 2 or 3, but got %zu.", permSize);
        return false;
    }
    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "transpose permutation last 2 dims must be swapped.");
        return false;
    }
    return true;
}

static bool IsDequantTransposePermValid(const std::vector<int64_t>& permValue)
{
    const size_t permSize = permValue.size();
    // 先要求 size∈{2,3}，再校验前两维 [1,0]
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "dequant transpose permutation size must be 2 or 3, but got %zu.",
                  permSize);
        return false;
    }
    if (permValue[0] != static_cast<int64_t>(1) || permValue[1] != static_cast<int64_t>(0)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "dequant transpose first 2 dims must be [1, 0].");
        return false;
    }
    return true;
}
static bool ValidateCapturedTransposePerm(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                          int64_t captureIdx, TransposePort port)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get captured Transpose node failed, capture idx %ld.", captureIdx);

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeOutput.node, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation, capture idx %ld.", captureIdx);
        return false;
    }

    if (port == TransposePort::Dequant) {
        return IsDequantTransposePermValid(permValue);
    }
    return IsLastTwoDimsSwapPermValid(permValue);
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                 const std::string& patternNameStr)
{
    // Capture：mc2=0，其后 x2 → scale → offset → dequant
    int64_t captureIdx = MC2_CAPTURE_IDX + 1;
    if (!ValidateCapturedTransposePerm(matchResult, captureIdx, TransposePort::X2)) {
        return false;
    }
    ++captureIdx;
    if (patternNameStr.find(PATTERN_SCALE_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, TransposePort::Scale)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_OFFSET_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, TransposePort::Offset)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_DEQUANT_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, TransposePort::Dequant)) {
            return false;
        }
    }
    return true;
}

// antiquant dim>1 且无合法 Transpose → 整次不融；offset 门控用 scale 的 dim
static bool IsAntiquantDirectConnectAllowed(const ge::GNode& mc2Node, const std::string& patternNameStr)
{
    const bool hasScale = patternNameStr.find(PATTERN_SCALE) != std::string::npos;
    const bool hasOffset = patternNameStr.find(PATTERN_OFFSET) != std::string::npos;
    if (!hasScale && !hasOffset) {
        return true;
    }

    const int32_t scaleIdx = static_cast<int32_t>(ops::MC2InputIdx::K_SCALE);
    ge::TensorDesc scaleDesc;
    if (mc2Node.GetInputDesc(scaleIdx, scaleDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Get antiquant_scale input desc failed.");
        return false;
    }
    const size_t antiquantDimNum = scaleDesc.GetShape().GetDimNum();
    if (antiquantDimNum <= ONE_DIM_SIZE) {
        return true;
    }

    if (hasScale && patternNameStr.find(PATTERN_SCALE_TRANSPOSE) == std::string::npos) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "antiquant_scale dim num is %zu (>1) without scale Transpose, fusion is skipped.", antiquantDimNum);
        return false;
    }
    if (hasOffset && patternNameStr.find(PATTERN_OFFSET_TRANSPOSE) == std::string::npos) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "antiquant_offset required Transpose when scale dim num is %zu (>1), fusion is skipped.",
                  antiquantDimNum);
        return false;
    }
    return true;
}

static bool IsDequantDirectConnectAllowed(const ge::GNode& mc2Node, const std::string& patternNameStr)
{
    if (patternNameStr.find(PATTERN_DEQUANT) == std::string::npos) {
        return true;
    }
    if (patternNameStr.find(PATTERN_DEQUANT_TRANSPOSE) != std::string::npos) {
        return true;
    }

    const int32_t dequantIdx = static_cast<int32_t>(ops::MC2InputIdx::K_DEQUANT);
    ge::TensorDesc dequantDesc;
    if (mc2Node.GetInputDesc(dequantIdx, dequantDesc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Get dequant_scale input desc failed.");
        return false;
    }
    const size_t dequantDimNum = dequantDesc.GetShape().GetDimNum();
    if (dequantDimNum > ONE_DIM_SIZE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "dequant_scale dim num is %zu (>1) without dequant Transpose, fusion is skipped.", dequantDimNum);
        return false;
    }
    return true;
}

ge::Status MatmulAllReduceTransposeA5FusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context)
{
    if (graph == nullptr || !GraphHasAnyOpType(*graph, {"MatmulAllReduce"})) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip Patterns: no MatmulAllReduce in graph.");
        return ge::SUCCESS;
    }
    return PatternFusionPass::Run(graph, pass_context);
}

std::vector<ge::fusion::PatternUniqPtr> MatmulAllReduceTransposeA5FusionPass::Patterns()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Patterns for MatmulAllReduceTransposeA5FusionPass");
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
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
                            for (bool dequantTranspose : {false, true}) {
                                if (dequantTranspose && !hasDequant) {
                                    continue;
                                }
                                for (bool dequantBitcast : {false, true}) {
                                    if (dequantBitcast && !dequantTranspose) {
                                        continue;
                                    }
                                    OriginalGraphInfo info{hasBias,          hasScale,       hasOffset,
                                                           hasDequant,       scaleTranspose, offsetTranspose,
                                                           dequantTranspose, dequantBitcast};
                                    patternGraphs.emplace_back(MakePattern(info));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return patternGraphs;
}

bool MatmulAllReduceTransposeA5FusionPass::MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for MatmulAllReduceTransposeA5FusionPass");

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

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in MeetRequirements.");
        return false;
    }
    const size_t expectNum = GetExpectedSubgraphInputNum(patternNameStr);
    if (subgraphInputs.size() != expectNum) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip pattern=%s: subgraph size=%zu, expect=%zu.", patternNameStr.c_str(),
                  subgraphInputs.size(), expectNum);
        return false;
    }

    if (!IsTransposePermValid(matchResult, patternNameStr)) {
        return false;
    }

    ge::GNode mc2Node;
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, MC2_CAPTURE_IDX, FUSION_PASS_NAME.c_str(),
                                 "Capture MatmulAllReduce node failed.")) {
        return false;
    }

    if (!IsAntiquantDirectConnectAllowed(mc2Node, patternNameStr)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "antiquant direct connect check failed.");
        return false;
    }

    if (!IsDequantDirectConnectAllowed(mc2Node, patternNameStr)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "dequant direct connect check failed.");
        return false;
    }

    OPS_LOG_I(FUSION_PASS_NAME.c_str(), "Found One pattern that meets requirements");
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
    if (inputShapes.size() < 2) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement, size=%zu.", inputShapes.size());
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
        inputs.rBias = replaceGraphBuilder.CreateInput(inputIdx, "bias", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                       inputShapes[inputIdx].GetDims());
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
    if (!ops::GetCapturedMc2Node(matchResult, mc2Node, MC2_CAPTURE_IDX, FUSION_PASS_NAME.c_str(),
                                 "Capture MatmulAllReduce node failed.")) {
        return nullptr;
    }

    const bool keepDequantBitcast = patternNameStr.find(PATTERN_DEQUANT_BITCAST) != std::string::npos;
    auto dequantIn = MaybeKeepBitcast(mc2Node, static_cast<int64_t>(ops::MC2InputIdx::K_DEQUANT),
                                      inputTensors.rDequantScale, keepDequantBitcast);

    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    ge::AscendString reduceOp;
    OP_LOGE_IF(mc2Node.GetAttr("reduce_op", reduceOp) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr reduce_op failed.");

    bool isTransA = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_a", isTransA) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr is_trans_a failed.");

    bool isTransB = false;
    OP_LOGE_IF(mc2Node.GetAttr("is_trans_b", isTransB) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr is_trans_b failed.");
    // 吸收 x2 Transpose 后置反
    isTransB = !isTransB;

    int64_t commTurn = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_turn", commTurn) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_turn failed.");

    int64_t antiquantGroupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("antiquant_group_size", antiquantGroupSize) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr antiquant_group_size failed.");

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    int64_t yDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");

    int64_t commQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("comm_quant_mode", commQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr comm_quant_mode failed.");

    ge::AscendString commMode;
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");

    auto y = ge::es::MatmulAllReduce(
        inputTensors.rX1, inputTensors.rX2, inputTensors.rBias, nullptr, inputTensors.rAntiquantScale,
        inputTensors.rAntiquantOffset, dequantIn, nullptr, nullptr, nullptr, group.GetString(), reduceOp.GetString(),
        isTransA, isTransB, commTurn, antiquantGroupSize, groupSize, yDtype, commQuantMode, commMode.GetString());
    return replaceGraphBuilder.BuildAndReset({y});
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

ge::fusion::GraphUniqPtr MatmulAllReduceTransposeA5FusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for MatmulAllReduceTransposeA5FusionPass");
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

REG_FUSION_PASS(MatmulAllReduceTransposeA5FusionPass).Stage(GetMatmulAllReduceTransposeA5FusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION

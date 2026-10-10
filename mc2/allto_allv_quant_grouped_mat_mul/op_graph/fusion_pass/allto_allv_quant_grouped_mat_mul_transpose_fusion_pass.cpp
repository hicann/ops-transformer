/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allto_allv_quant_grouped_mat_mul_transpose_fusion_pass.h"

#if GE_COMPILER_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "es_AlltoAllvQuantGroupedMatMul.h" // es autogen header
#include "es_math_ops.h"                    // math ops stub
#include "mc2_platform_info.h"
#include "mc2_fusion_pass_utils.h"
#include "mc2_common_log.h"
#include "ge/ge_utils.h"
#include <dlfcn.h>      // dlopen 动态加载
#include "acl/acl_rt.h" // 运行时判断cann ver

namespace ops {
const std::string FUSION_PASS_NAME = "AlltoAllvQuantGroupedMatMulTransposeFusionPass";
const std::string PATTERN_TRANSPOSE = "Transpose";
const std::string PATTERN_HAS_MM = "HasMm";
const std::string PATTERN_GMM_WEIGHT_TRANSPOSE = "GmmWeightTranspose";
const std::string PATTERN_GMM_WEIGHT_SCALE_TRANSPOSE = "GmmWeightScaleTranspose";
const std::string PATTERN_MM_WEIGHT_TRANSPOSE = "MmWeightTranspose";
const std::string PATTERN_MM_WEIGHT_SCALE_TRANSPOSE = "MmWeightScaleTranspose";
const std::string PATTERN_GMM_WEIGHT_BITCAST = "GmmWeightBitcast";
const std::string PATTERN_GMM_WEIGHT_SCALE_BITCAST = "GmmWeightScaleBitcast";
const std::string PATTERN_MM_WEIGHT_BITCAST = "MmWeightBitcast";
const std::string PATTERN_MM_WEIGHT_SCALE_BITCAST = "MmWeightScaleBitcast";

// 仅排列影响融合效果的维度；send/recv_counts_tensor 不进 pattern（同 all2all 的 comm/offset）
const size_t INPUT_NUM_NO_MM = 4;  // gmm_x / gmm_weight / gmm_x_scale / gmm_weight_scale
const size_t INPUT_NUM_HAS_MM = 8; // + mm_x / mm_weight / mm_x_scale / mm_weight_scale

const int64_t MC2_CAPTURE_IDX = 0l;
const int64_t TRANSPOSE_PERM_IDX = 1l;
const size_t ONE_DIM_SIZE = 1;
const size_t PERM_SIZE_ONE = 1;
const size_t PERM_SIZE_TWO = 2;
const size_t PERM_SIZE_THREE = 3;
const size_t PERM_SIZE_FOUR = 4;
// Pattern 中 Bitcast 的 type 仅占位；默认 matcher 不校验 IR attr，Replacement 用真实 MC2 输入 dtype
const ge::DataType BITCAST_PATTERN_DTYPE = ge::DT_HIFLOAT8;

enum class WeightTransposePort {
    GmmWeight,
    GmmWeightScale,
    MmWeight,
    MmWeightScale
};

// 仅覆盖影响融合拓扑/结果的组合
// - hasMm：可选 mm 路径
// - 四路 weight/scale 是否挂 Transpose（至少一路）
// - 对应口是否为 Transpose→Bitcast→MC2（仅 Transpose=true 时可开）
struct OriginalGraphInfo {
    bool hasMm = false;
    bool gmmWeightTranspose = false;
    bool gmmWeightScaleTranspose = false;
    bool mmWeightTranspose = false;
    bool mmWeightScaleTranspose = false;
    bool gmmWeightBitcast = false;
    bool gmmWeightScaleBitcast = false;
    bool mmWeightBitcast = false;
    bool mmWeightScaleBitcast = false;
};

struct ReplaceGraphInputs {
    ge::es::EsTensorHolder rGmmX;
    ge::es::EsTensorHolder rGmmWeight;
    ge::es::EsTensorHolder rGmmXScale;
    ge::es::EsTensorHolder rGmmWeightScale;
    ge::es::EsTensorHolder rMmX;
    ge::es::EsTensorHolder rMmWeight;
    ge::es::EsTensorHolder rMmXScale;
    ge::es::EsTensorHolder rMmWeightScale;
};

// 加载 EsTranspose 符号
namespace {
typedef EsCTensorHolder* (*EsTransposeFunc)(EsCTensorHolder*, EsCTensorHolder*);

EsTransposeFunc GetEsTransposeFunc()
{
    void* handle = dlopen("libes_math.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        OPS_LOG_E("AlltoAllvQuantGroupedMatMulTransposeFusionPass", "dlopen failed: %s", dlerror());
        return nullptr;
    }
    dlerror();
    auto func = reinterpret_cast<EsTransposeFunc>(dlsym(handle, "EsTranspose"));
    if (dlerror() != nullptr) {
        OPS_LOG_E("AlltoAllvQuantGroupedMatMulTransposeFusionPass", "dlsym EsTranspose failed");
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

ge::CustomPassStage GetAlltoAllvQuantGroupedMatMulTransposeFusionPassStage()
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
    return info.hasMm ? INPUT_NUM_HAS_MM : INPUT_NUM_NO_MM;
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
    if (info.hasMm) {
        patternName += PATTERN_HAS_MM;
    }
    if (info.gmmWeightTranspose) {
        patternName += PATTERN_GMM_WEIGHT_TRANSPOSE;
    }
    if (info.gmmWeightScaleTranspose) {
        patternName += PATTERN_GMM_WEIGHT_SCALE_TRANSPOSE;
    }
    if (info.mmWeightTranspose) {
        patternName += PATTERN_MM_WEIGHT_TRANSPOSE;
    }
    if (info.mmWeightScaleTranspose) {
        patternName += PATTERN_MM_WEIGHT_SCALE_TRANSPOSE;
    }
    if (info.gmmWeightBitcast) {
        patternName += PATTERN_GMM_WEIGHT_BITCAST;
    }
    if (info.gmmWeightScaleBitcast) {
        patternName += PATTERN_GMM_WEIGHT_SCALE_BITCAST;
    }
    if (info.mmWeightBitcast) {
        patternName += PATTERN_MM_WEIGHT_BITCAST;
    }
    if (info.mmWeightScaleBitcast) {
        patternName += PATTERN_MM_WEIGHT_SCALE_BITCAST;
    }

    auto graphBuilder = ge::es::EsGraphBuilder(patternName.c_str());
    // gmm_* 必选；mm_* 按 hasMm；send/recv 固定 nullptr（不参与融合效果，不排列）
    const size_t inputNum = GetPatternInputNum(info);
    auto inputs = graphBuilder.CreateInputs(inputNum);
    size_t idx = 0;
    auto gmmX = inputs[idx++];
    auto gmmWeight = inputs[idx++];
    auto gmmXScale = inputs[idx++];
    auto gmmWeightScale = inputs[idx++];

    ge::es::EsTensorHolder mmX = nullptr;
    ge::es::EsTensorHolder mmWeight = nullptr;
    ge::es::EsTensorHolder mmXScale = nullptr;
    ge::es::EsTensorHolder mmWeightScale = nullptr;
    if (info.hasMm) {
        mmX = inputs[idx++];
        mmWeight = inputs[idx++];
        mmXScale = inputs[idx++];
        mmWeightScale = inputs[idx++];
    }

    ge::es::EsTensorHolder transposeGmmWeight = nullptr;
    ge::es::EsTensorHolder gmmWeightIn = gmmWeight;
    if (info.gmmWeightTranspose) {
        transposeGmmWeight = MakeTransposeNode(gmmWeight);
        gmmWeightIn = MaybeBitcastAfterTranspose(info.gmmWeightBitcast, transposeGmmWeight);
    }

    ge::es::EsTensorHolder transposeGmmWeightScale = nullptr;
    ge::es::EsTensorHolder gmmWeightScaleIn = gmmWeightScale;
    if (info.gmmWeightScaleTranspose) {
        transposeGmmWeightScale = MakeTransposeNode(gmmWeightScale);
        gmmWeightScaleIn = MaybeBitcastAfterTranspose(info.gmmWeightScaleBitcast, transposeGmmWeightScale);
    }

    ge::es::EsTensorHolder transposeMmWeight = nullptr;
    ge::es::EsTensorHolder mmWeightIn = mmWeight;
    if (info.mmWeightTranspose) {
        transposeMmWeight = MakeTransposeNode(mmWeight);
        mmWeightIn = MaybeBitcastAfterTranspose(info.mmWeightBitcast, transposeMmWeight);
    }

    ge::es::EsTensorHolder transposeMmWeightScale = nullptr;
    ge::es::EsTensorHolder mmWeightScaleIn = mmWeightScale;
    if (info.mmWeightScaleTranspose) {
        transposeMmWeightScale = MakeTransposeNode(mmWeightScale);
        mmWeightScaleIn = MaybeBitcastAfterTranspose(info.mmWeightScaleBitcast, transposeMmWeightScale);
    }

    const char* group = "";
    const int64_t epWorldSize = 0;
    const std::vector<int64_t> sendCounts;
    const std::vector<int64_t> recvCounts;
    const int64_t gmmXQuantMode = 0;
    const int64_t gmmWeightQuantMode = 0;
    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(gmmX, gmmWeightIn, gmmXScale, gmmWeightScaleIn, nullptr, nullptr,
                                                   mmX, mmWeightIn, mmXScale, mmWeightScaleIn, group, epWorldSize,
                                                   sendCounts, recvCounts, gmmXQuantMode, gmmWeightQuantMode);

    auto graph = graphBuilder.BuildAndReset({mc2.gmm_y, mc2.mm_y, mc2.permute_out});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    // Capture：0=mc2；其后按固定顺序追加本 pattern 中存在的 Transpose，便于 MeetRequirements 校验 perm
    pattern->CaptureTensor({*mc2.gmm_y.GetProducer(), 0});
    if (info.gmmWeightTranspose) {
        pattern->CaptureTensor({*transposeGmmWeight.GetProducer(), 0});
    }
    if (info.gmmWeightScaleTranspose) {
        pattern->CaptureTensor({*transposeGmmWeightScale.GetProducer(), 0});
    }
    if (info.mmWeightTranspose) {
        pattern->CaptureTensor({*transposeMmWeight.GetProducer(), 0});
    }
    if (info.mmWeightScaleTranspose) {
        pattern->CaptureTensor({*transposeMmWeightScale.GetProducer(), 0});
    }
    return pattern;
}

ge::Status AlltoAllvQuantGroupedMatMulTransposeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context)
{
    if (graph == nullptr || !GraphHasAnyOpType(*graph, {"AlltoAllvQuantGroupedMatMul"})) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip Patterns: no AlltoAllvQuantGroupedMatMul in graph.");
        return ge::SUCCESS;
    }
    return PatternFusionPass::Run(graph, pass_context);
}

std::vector<ge::fusion::PatternUniqPtr> AlltoAllvQuantGroupedMatMulTransposeFusionPass::Patterns()
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Patterns for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
    // hasMm × 四路 Transpose（至少一路）× 对应 Bitcast（仅 Transpose 口）
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    for (bool hasMm : {false, true}) {
        for (bool gmmWeightTranspose : {false, true}) {
            for (bool gmmWeightScaleTranspose : {false, true}) {
                for (bool mmWeightTranspose : {false, true}) {
                    for (bool mmWeightScaleTranspose : {false, true}) {
                        if (!hasMm && (mmWeightTranspose || mmWeightScaleTranspose)) {
                            continue;
                        }
                        if (!gmmWeightTranspose && !gmmWeightScaleTranspose && !mmWeightTranspose &&
                            !mmWeightScaleTranspose) {
                            continue;
                        }
                        for (bool gmmWeightBitcast : {false, true}) {
                            if (gmmWeightBitcast && !gmmWeightTranspose) {
                                continue;
                            }
                            for (bool gmmWeightScaleBitcast : {false, true}) {
                                if (gmmWeightScaleBitcast && !gmmWeightScaleTranspose) {
                                    continue;
                                }
                                for (bool mmWeightBitcast : {false, true}) {
                                    if (mmWeightBitcast && !mmWeightTranspose) {
                                        continue;
                                    }
                                    for (bool mmWeightScaleBitcast : {false, true}) {
                                        if (mmWeightScaleBitcast && !mmWeightScaleTranspose) {
                                            continue;
                                        }
                                        OriginalGraphInfo info{hasMm,
                                                               gmmWeightTranspose,
                                                               gmmWeightScaleTranspose,
                                                               mmWeightTranspose,
                                                               mmWeightScaleTranspose,
                                                               gmmWeightBitcast,
                                                               gmmWeightScaleBitcast,
                                                               mmWeightBitcast,
                                                               mmWeightScaleBitcast};
                                        patternGraphs.emplace_back(MakePattern(info));
                                    }
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

static bool IsWeightTransposePermValid(const std::vector<int64_t>& permValue)
{
    const size_t permSize = permValue.size();
    if (permSize != PERM_SIZE_TWO && permSize != PERM_SIZE_THREE && permSize != PERM_SIZE_FOUR) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "weight transpose permutation size must be 2/3/4, but got %zu.", permSize);
        return false;
    }
    // 最后两维必须互换
    if ((permValue[permSize - PERM_SIZE_ONE] != static_cast<int64_t>(permSize - PERM_SIZE_TWO)) ||
        (permValue[permSize - PERM_SIZE_TWO] != static_cast<int64_t>(permSize - PERM_SIZE_ONE))) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "weight transpose permutation last 2 dims must be swapped.");
        return false;
    }
    return true;
}

static bool IsGmmWeightScaleTransposePermValid(const std::vector<int64_t>& permValue)
{
    if (permValue.size() < PERM_SIZE_THREE) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "gmm_weight_scale transpose permutation size must be at least 3, but got %zu.", permValue.size());
        return false;
    }
    // perm[1]==2 && perm[2]==1
    if (permValue[1] != static_cast<int64_t>(2) || permValue[2] != static_cast<int64_t>(1)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "gmm_weight_scale transpose perm[1]/perm[2] must be 2/1.");
        return false;
    }
    return true;
}

static bool IsMmWeightScaleTransposePermValid(const std::vector<int64_t>& permValue)
{
    if (permValue.size() < PERM_SIZE_TWO) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(),
                  "mm_weight_scale transpose permutation size must be at least 2, but got %zu.", permValue.size());
        return false;
    }
    // perm[0]==1 && perm[1]==0
    if (permValue[0] != static_cast<int64_t>(1) || permValue[1] != static_cast<int64_t>(0)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "mm_weight_scale transpose first 2 dims must be [1, 0].");
        return false;
    }
    return true;
}

static bool ValidateCapturedTransposePerm(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                          int64_t captureIdx, WeightTransposePort port)
{
    ge::fusion::NodeIo transposeOutput;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, transposeOutput) != ge::SUCCESS, false,
               FUSION_PASS_NAME.c_str(), "Get captured Transpose node failed, capture idx %ld.", captureIdx);

    std::vector<int64_t> permValue;
    if (!GetTransposePerm(transposeOutput.node, permValue)) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Failed to get transpose permutation, capture idx %ld.", captureIdx);
        return false;
    }

    switch (port) {
        case WeightTransposePort::GmmWeight:
        case WeightTransposePort::MmWeight:
            return IsWeightTransposePermValid(permValue);
        case WeightTransposePort::GmmWeightScale:
            return IsGmmWeightScaleTransposePermValid(permValue);
        case WeightTransposePort::MmWeightScale:
            return IsMmWeightScaleTransposePermValid(permValue);
        default:
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Unexpected WeightTransposePort.");
            return false;
    }
}

static bool IsTransposePermValid(const std::unique_ptr<ge::fusion::MatchResult>& matchResult,
                                 const std::string& patternNameStr)
{
    // Capture 顺序与 MakePattern 一致：mc2=0，其后按 gmm_w → gmm_ws → mm_w → mm_ws
    int64_t captureIdx = MC2_CAPTURE_IDX + 1;
    if (patternNameStr.find(PATTERN_GMM_WEIGHT_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::GmmWeight)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_GMM_WEIGHT_SCALE_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::GmmWeightScale)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_MM_WEIGHT_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::MmWeight)) {
            return false;
        }
        ++captureIdx;
    }
    if (patternNameStr.find(PATTERN_MM_WEIGHT_SCALE_TRANSPOSE) != std::string::npos) {
        if (!ValidateCapturedTransposePerm(matchResult, captureIdx, WeightTransposePort::MmWeightScale)) {
            return false;
        }
    }
    return true;
}
bool AlltoAllvQuantGroupedMatMulTransposeFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter MeetRequirements for AlltoAllvQuantGroupedMatMulTransposeFusionPass");

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
    const bool hasMm = patternNameStr.find(PATTERN_HAS_MM) != std::string::npos;
    const size_t expectNum = hasMm ? INPUT_NUM_HAS_MM : INPUT_NUM_NO_MM;

    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Get subgraph inputs failed in MeetRequirements.");
        return false;
    }
    if (subgraphInputs.size() != expectNum) {
        OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Skip pattern=%s: subgraph size=%zu, expect=%zu.", patternNameStr.c_str(),
                  subgraphInputs.size(), expectNum);
        return false;
    }

    if (!IsTransposePermValid(matchResult, patternNameStr)) {
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
    // 按 MakePattern 的 CreateInputs 顺序映射 subgraph 边界（已是 Transpose 前的 tensor）
    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDTypes;
    std::vector<ge::Format> inputFormats;
    if (!CollectSubgraphInputsInfo(subgraphInputs, inputShapes, inputDTypes, inputFormats)) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "CollectSubgraphInputsInfo failed in CreateReplaceGraphInputs.");
        return false;
    }
    if (inputShapes.size() < INPUT_NUM_NO_MM) {
        OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement without mm, size=%zu.",
                  inputShapes.size());
        return false;
    }

    const bool hasMm = patternNameStr.find(PATTERN_HAS_MM) != std::string::npos;

    int64_t inputIdx = 0;
    inputs.rGmmX = replaceGraphBuilder.CreateInput(inputIdx, "gmm_x", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                   inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rGmmWeight = replaceGraphBuilder.CreateInput(inputIdx, "gmm_weight", inputDTypes[inputIdx],
                                                        inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rGmmXScale = replaceGraphBuilder.CreateInput(inputIdx, "gmm_x_scale", inputDTypes[inputIdx],
                                                        inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    ++inputIdx;
    inputs.rGmmWeightScale = replaceGraphBuilder.CreateInput(inputIdx, "gmm_weight_scale", inputDTypes[inputIdx],
                                                             inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
    ++inputIdx;

    inputs.rMmX = nullptr;
    inputs.rMmWeight = nullptr;
    inputs.rMmXScale = nullptr;
    inputs.rMmWeightScale = nullptr;
    if (hasMm) {
        if (inputShapes.size() < INPUT_NUM_HAS_MM) {
            OPS_LOG_E(FUSION_PASS_NAME.c_str(), "Input Nums do not meet requirement with mm, size=%zu.",
                      inputShapes.size());
            return false;
        }
        inputs.rMmX = replaceGraphBuilder.CreateInput(inputIdx, "mm_x", inputDTypes[inputIdx], inputFormats[inputIdx],
                                                      inputShapes[inputIdx].GetDims());
        ++inputIdx;
        inputs.rMmWeight = replaceGraphBuilder.CreateInput(inputIdx, "mm_weight", inputDTypes[inputIdx],
                                                           inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
        inputs.rMmXScale = replaceGraphBuilder.CreateInput(inputIdx, "mm_x_scale", inputDTypes[inputIdx],
                                                           inputFormats[inputIdx], inputShapes[inputIdx].GetDims());
        ++inputIdx;
        inputs.rMmWeightScale =
            replaceGraphBuilder.CreateInput(inputIdx, "mm_weight_scale", inputDTypes[inputIdx], inputFormats[inputIdx],
                                            inputShapes[inputIdx].GetDims());
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
                                 "Capture AlltoAllvQuantGroupedMatMul node failed.")) {
        return nullptr;
    }

    ge::AscendString group;
    OP_LOGE_IF(mc2Node.GetAttr("group", group) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group failed.");

    int64_t epWorldSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("ep_world_size", epWorldSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr ep_world_size failed.");

    std::vector<int64_t> sendCounts;
    OP_LOGE_IF(mc2Node.GetAttr("send_counts", sendCounts) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr send_counts failed.");

    std::vector<int64_t> recvCounts;
    OP_LOGE_IF(mc2Node.GetAttr("recv_counts", recvCounts) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr recv_counts failed.");

    int64_t gmmXQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("gmm_x_quant_mode", gmmXQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr gmm_x_quant_mode failed.");

    int64_t gmmWeightQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("gmm_weight_quant_mode", gmmWeightQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr gmm_weight_quant_mode failed.");

    bool transGmmWeight = false;
    OP_LOGE_IF(mc2Node.GetAttr("trans_gmm_weight", transGmmWeight) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr trans_gmm_weight failed.");
    // 吸收 gmm_weight Transpose 后取反
    if (patternNameStr.find(PATTERN_GMM_WEIGHT_TRANSPOSE) != std::string::npos) {
        transGmmWeight = !transGmmWeight;
    }

    bool transMmWeight = false;
    OP_LOGE_IF(mc2Node.GetAttr("trans_mm_weight", transMmWeight) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr trans_mm_weight failed.");
    if (patternNameStr.find(PATTERN_MM_WEIGHT_TRANSPOSE) != std::string::npos) {
        transMmWeight = !transMmWeight;
    }

    bool permuteOutFlag = false;
    OP_LOGE_IF(mc2Node.GetAttr("permute_out_flag", permuteOutFlag) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr permute_out_flag failed.");

    int64_t mmXQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("mm_x_quant_mode", mmXQuantMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr mm_x_quant_mode failed.");

    int64_t mmWeightQuantMode = 0;
    OP_LOGE_IF(mc2Node.GetAttr("mm_weight_quant_mode", mmWeightQuantMode) != ge::GRAPH_SUCCESS, nullptr,
               FUSION_PASS_NAME.c_str(), "Get Attr mm_weight_quant_mode failed.");

    int64_t groupSize = 0;
    OP_LOGE_IF(mc2Node.GetAttr("group_size", groupSize) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr group_size failed.");

    int64_t yDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("y_dtype", yDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr y_dtype failed.");

    int64_t mmDtype = 28;
    OP_LOGE_IF(mc2Node.GetAttr("mm_dtype", mmDtype) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr mm_dtype failed.");

    ge::AscendString commMode;
    OP_LOGE_IF(mc2Node.GetAttr("comm_mode", commMode) != ge::GRAPH_SUCCESS, nullptr, FUSION_PASS_NAME.c_str(),
               "Get Attr comm_mode failed.");

    // IR：0 gmm_x / 1 gmm_weight / 2 gmm_x_scale / 3 gmm_weight_scale / 6 mm_x / 7 mm_weight / 8 mm_x_scale / 9
    // mm_weight_scale
    const bool gmmWeightBitcast = patternNameStr.find(PATTERN_GMM_WEIGHT_BITCAST) != std::string::npos;
    const bool gmmWeightScaleBitcast = patternNameStr.find(PATTERN_GMM_WEIGHT_SCALE_BITCAST) != std::string::npos;
    const bool mmWeightBitcast = patternNameStr.find(PATTERN_MM_WEIGHT_BITCAST) != std::string::npos;
    const bool mmWeightScaleBitcast = patternNameStr.find(PATTERN_MM_WEIGHT_SCALE_BITCAST) != std::string::npos;

    auto gmmWeightIn = MaybeKeepBitcast(mc2Node, 1, inputTensors.rGmmWeight, gmmWeightBitcast);
    auto gmmWeightScaleIn = MaybeKeepBitcast(mc2Node, 3, inputTensors.rGmmWeightScale, gmmWeightScaleBitcast);
    auto mmWeightIn = MaybeKeepBitcast(mc2Node, 7, inputTensors.rMmWeight, mmWeightBitcast);
    auto mmWeightScaleIn = MaybeKeepBitcast(mc2Node, 9, inputTensors.rMmWeightScale, mmWeightScaleBitcast);

    auto mc2 = ge::es::AlltoAllvQuantGroupedMatMul(
        inputTensors.rGmmX, gmmWeightIn, inputTensors.rGmmXScale, gmmWeightScaleIn, nullptr, nullptr, inputTensors.rMmX,
        mmWeightIn, inputTensors.rMmXScale, mmWeightScaleIn, group.GetString(), epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag, mmXQuantMode,
        mmWeightQuantMode, groupSize, yDtype, mmDtype, commMode.GetString());
    return replaceGraphBuilder.BuildAndReset({mc2.gmm_y, mc2.mm_y, mc2.permute_out});
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

ge::fusion::GraphUniqPtr AlltoAllvQuantGroupedMatMulTransposeFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OPS_LOG_D(FUSION_PASS_NAME.c_str(), "Enter Replacement for AlltoAllvQuantGroupedMatMulTransposeFusionPass");
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

REG_FUSION_PASS(AlltoAllvQuantGroupedMatMulTransposeFusionPass)
    .Stage(GetAlltoAllvQuantGroupedMatMulTransposeFusionPassStage());
} // namespace ops

#endif // GRAPH_FUSION_SUPPORT_VERSION

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
 * \file kda_input_proj_tiling.cpp
 * \brief Orchestrates per-module tiling aligned with kernel components.
 */

#include "kda_input_proj_tiling.h"

#include <algorithm>
#include <cstring>

#include "../op_kernel/kda_input_proj_template_tiling_key.h"
#include "../op_kernel/kda_input_proj_workspace.h"
#include "kda_input_proj_tiling_mm_bgg.h"
#include "kda_input_proj_tiling_mx_quant.h"
#include "kda_input_proj_tiling_qmm_qkv.h"
#include "kda_input_proj_tiling_sigmoid.h"

namespace optiling {
namespace {
constexpr uint32_t DIM_IDX_ZERO = 0U;
constexpr uint32_t DIM_IDX_ONE = 1U;
constexpr uint32_t DIM_IDX_TWO = 2U;
constexpr uint32_t DIM_NUM_TWO = 2U;
constexpr uint32_t DIM_NUM_THREE = 3U;
constexpr int64_t MX_BLOCK_SIZE = 64L;
constexpr int64_t WEIGHT_QKV_SCALE_PACK_NUM = 2L;

static bool GetAttrOrDefault(const bool *ptr, bool defaultValue)
{
    return (ptr != nullptr) ? *ptr : defaultValue;
}

static const gert::Shape &GetWeightLogicShape(const gert::StorageShape *shape)
{
    const auto &origin = shape->GetOriginShape();
    if (origin.GetDimNum() == DIM_NUM_TWO) {
        return origin;
    }
    return shape->GetStorageShape();
}

// trans=true：存储 [N, K]，N=dim0；trans=false：逻辑 [K, N]，N=dim1。
static uint32_t InferOutFeatures(const gert::Shape &weightShape, bool transWeight)
{
    return static_cast<uint32_t>(transWeight ? weightShape.GetDim(DIM_IDX_ZERO) : weightShape.GetDim(DIM_IDX_ONE));
}

ge::graphStatus CheckExpectDataType(ge::DataType actual, ge::DataType expect, const char *tensorName,
                                    const char *expectName, const char *opName)
{
    OP_CHECK_IF(actual != expect, OP_LOGE(opName, "%s dtype must be %s.", tensorName, expectName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus KdaInputProjInfoParser::GetOpName()
{
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaInputProj", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    const char *opName = context_->GetNodeName();
    OP_CHECK_IF(opName == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaInputProj", "Node name is null."),
                return ge::GRAPH_FAILED);
    opName_ = opName;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0U || aivNum == 0U, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);
    aicNum_ = aicNum;
    aivNum_ = aivNum;

    const auto socVersion = ascendcPlatform.GetSocVersion();
    OP_CHECK_IF(socVersion != platform_ascendc::SocVersion::ASCEND950,
                OP_LOGE(opName_, "SOC Version[%d] is not supported, only ASCEND950 is supported.",
                        static_cast<int32_t>(socVersion)),
                return ge::GRAPH_FAILED);

    l1Size_ = 0UL;
    l0cSize_ = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    OP_CHECK_IF(l1Size_ == 0UL || l0cSize_ == 0UL,
                OP_LOGE(opName_, "l1Size or l0cSize is 0, platform mem info is invalid."), return ge::GRAPH_FAILED);

    ubSize_ = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0UL, OP_LOGE(opName_, "ubSize is 0, platform mem info is invalid."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void KdaInputProjInfoParser::GetInputParaInfo()
{
    opParamInfo_.x.desc = context_->GetInputDesc(X_INDEX);
    opParamInfo_.x.shape = context_->GetInputShape(X_INDEX);
    opParamInfo_.weightQkv.desc = context_->GetInputDesc(WEIGHT_QKV_INDEX);
    opParamInfo_.weightQkv.shape = context_->GetInputShape(WEIGHT_QKV_INDEX);
    opParamInfo_.weightBeta.desc = context_->GetInputDesc(WEIGHT_BETA_INDEX);
    opParamInfo_.weightBeta.shape = context_->GetInputShape(WEIGHT_BETA_INDEX);
    opParamInfo_.weightGate.desc = context_->GetInputDesc(WEIGHT_GATE_INDEX);
    opParamInfo_.weightGate.shape = context_->GetInputShape(WEIGHT_GATE_INDEX);
    opParamInfo_.weightG.desc = context_->GetInputDesc(WEIGHT_G_INDEX);
    opParamInfo_.weightG.shape = context_->GetInputShape(WEIGHT_G_INDEX);
    opParamInfo_.weightQkvScale.desc = context_->GetInputDesc(WEIGHT_QKV_SCALE_INDEX);
    opParamInfo_.weightQkvScale.shape = context_->GetInputShape(WEIGHT_QKV_SCALE_INDEX);
}

void KdaInputProjInfoParser::GetOutputParaInfo()
{
    opParamInfo_.qkvOutDesc = context_->GetOutputDesc(QKV_INDEX);
    opParamInfo_.betaOutDesc = context_->GetOutputDesc(BETA_INDEX);
    opParamInfo_.gateOutDesc = context_->GetOutputDesc(GATE_INDEX);
    opParamInfo_.gOutDesc = context_->GetOutputDesc(G_INDEX);
}

ge::graphStatus KdaInputProjInfoParser::GetAndCheckAttrParaInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        opParamInfo_.transWeightQkv = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_QKV_INDEX);
        opParamInfo_.transWeightBeta = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_BETA_INDEX);
        opParamInfo_.transWeightGate = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_GATE_INDEX);
        opParamInfo_.transWeightG = attrs->GetAttrPointer<bool>(ATTR_TRANS_WEIGHT_G_INDEX);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.x.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_input_proj::X_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.x.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::X_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkv.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_input_proj::WEIGHT_QKV_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkv.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::WEIGHT_QKV_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightBeta.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_input_proj::WEIGHT_BETA_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightBeta.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::WEIGHT_BETA_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightGate.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_input_proj::WEIGHT_GATE_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightGate.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::WEIGHT_GATE_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightG.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_input_proj::WEIGHT_G_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightG.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::WEIGHT_G_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkvScale.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor %s is nullptr.", kda_input_proj::WEIGHT_QKV_SCALE_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkvScale.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::WEIGHT_QKV_SCALE_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qkvOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::QKV_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.betaOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::BETA_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.gateOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::GATE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.gOutDesc == nullptr,
                OP_LOGE(opName_, "Desc of tensor %s is nullptr.", kda_input_proj::G_NAME), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::GetAndCheckInOutDataType()
{
    const ge::DataType xType = opParamInfo_.x.desc->GetDataType();
    const ge::DataType weightQkvType = opParamInfo_.weightQkv.desc->GetDataType();
    const ge::DataType weightBetaType = opParamInfo_.weightBeta.desc->GetDataType();
    const ge::DataType weightGateType = opParamInfo_.weightGate.desc->GetDataType();
    const ge::DataType weightGType = opParamInfo_.weightG.desc->GetDataType();
    const ge::DataType weightQkvScaleType = opParamInfo_.weightQkvScale.desc->GetDataType();
    const ge::DataType qkvOutType = opParamInfo_.qkvOutDesc->GetDataType();
    const ge::DataType betaOutType = opParamInfo_.betaOutDesc->GetDataType();
    const ge::DataType gateOutType = opParamInfo_.gateOutDesc->GetDataType();
    const ge::DataType gOutType = opParamInfo_.gOutDesc->GetDataType();

    if (CheckExpectDataType(xType, ge::DT_BF16, kda_input_proj::X_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightQkvType, ge::DT_FLOAT8_E4M3FN, kda_input_proj::WEIGHT_QKV_NAME, "FLOAT8_E4M3FN",
                            opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightBetaType, ge::DT_BF16, kda_input_proj::WEIGHT_BETA_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightGateType, ge::DT_BF16, kda_input_proj::WEIGHT_GATE_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightGType, ge::DT_BF16, kda_input_proj::WEIGHT_G_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightQkvScaleType, ge::DT_FLOAT8_E8M0, kda_input_proj::WEIGHT_QKV_SCALE_NAME,
                            "FLOAT8_E8M0", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(qkvOutType, ge::DT_BF16, kda_input_proj::QKV_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(betaOutType, ge::DT_FLOAT, kda_input_proj::BETA_NAME, "FLOAT", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(gateOutType, ge::DT_BF16, kda_input_proj::GATE_NAME, "BF16", opName_) !=
            ge::GRAPH_SUCCESS ||
        CheckExpectDataType(gOutType, ge::DT_BF16, kda_input_proj::G_NAME, "BF16", opName_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::CheckShapeDim()
{
    OP_CHECK_IF(opParamInfo_.x.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_input_proj::X_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkv.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_input_proj::WEIGHT_QKV_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightBeta.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_input_proj::WEIGHT_BETA_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightGate.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_input_proj::WEIGHT_GATE_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightG.shape->GetStorageShape().GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(opName_, "%s must be 2D.", kda_input_proj::WEIGHT_G_NAME), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weightQkvScale.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(opName_, "%s must be 3D.", kda_input_proj::WEIGHT_QKV_SCALE_NAME), return ge::GRAPH_FAILED);

    const int64_t tSize = opParamInfo_.x.shape->GetStorageShape().GetDim(0);
    const int64_t hiddenSize = opParamInfo_.x.shape->GetStorageShape().GetDim(1);
    OP_CHECK_IF(tSize <= 0L || hiddenSize <= 0L,
                OP_LOGE(opName_, "%s shape [%ld, %ld] is invalid.", kda_input_proj::X_NAME, tSize, hiddenSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::GetBaseShapeInfo()
{
    const auto &xShape = opParamInfo_.x.shape->GetStorageShape();
    baseParams_.tSize = static_cast<uint32_t>(xShape.GetDim(DIM_IDX_ZERO));
    baseParams_.hiddenSize = static_cast<uint32_t>(xShape.GetDim(DIM_IDX_ONE));
    OP_CHECK_IF(baseParams_.tSize == 0 || baseParams_.hiddenSize == 0,
                OP_LOGE(opName_, "x shape [%u, %u] is invalid.", baseParams_.tSize, baseParams_.hiddenSize),
                return ge::GRAPH_FAILED);

    const bool transQkv = GetAttrOrDefault(opParamInfo_.transWeightQkv, true);
    const bool transBeta = GetAttrOrDefault(opParamInfo_.transWeightBeta, true);
    const bool transGate = GetAttrOrDefault(opParamInfo_.transWeightGate, true);
    const bool transG = GetAttrOrDefault(opParamInfo_.transWeightG, true);

    baseParams_.qkvSize = InferOutFeatures(GetWeightLogicShape(opParamInfo_.weightQkv.shape), transQkv);
    baseParams_.betaSize = InferOutFeatures(GetWeightLogicShape(opParamInfo_.weightBeta.shape), transBeta);
    baseParams_.gateSize = InferOutFeatures(GetWeightLogicShape(opParamInfo_.weightGate.shape), transGate);
    baseParams_.gSize = InferOutFeatures(GetWeightLogicShape(opParamInfo_.weightG.shape), transG);
    OP_CHECK_IF(
        baseParams_.qkvSize == 0 || baseParams_.betaSize == 0 || baseParams_.gateSize == 0 || baseParams_.gSize == 0,
        OP_LOGE(opName_, "output features qkv/beta/gate/g must be >0, got %u/%u/%u/%u.", baseParams_.qkvSize,
                baseParams_.betaSize, baseParams_.gateSize, baseParams_.gSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjInfoParser::ValidateInputShapesMatch()
{
    const bool transWeightQkv = opParamInfo_.transWeightQkv != nullptr ? *opParamInfo_.transWeightQkv : true;
    const int64_t hiddenSize = static_cast<int64_t>(baseParams_.hiddenSize);
    const int64_t qkvSize = static_cast<int64_t>(baseParams_.qkvSize);
    const int64_t mxHiddenSize = (hiddenSize + MX_BLOCK_SIZE - 1L) / MX_BLOCK_SIZE;
    const auto &weightQkvScaleStorage = opParamInfo_.weightQkvScale.shape->GetStorageShape();
    const int64_t scaleDim0 = weightQkvScaleStorage.GetDim(0);
    const int64_t scaleDim1 = weightQkvScaleStorage.GetDim(1);
    const int64_t scaleDim2 = weightQkvScaleStorage.GetDim(2);
    const int64_t expectScaleDim0 = transWeightQkv ? qkvSize : mxHiddenSize;
    const int64_t expectScaleDim1 = transWeightQkv ? mxHiddenSize : qkvSize;
    OP_CHECK_IF(
        scaleDim0 != expectScaleDim0 || scaleDim1 != expectScaleDim1 || scaleDim2 != WEIGHT_QKV_SCALE_PACK_NUM,
        OP_LOGE(opName_, "%s shape [%ld, %ld, %ld] does not match expected [%ld, %ld, %ld] (trans_weight_qkv=%d).",
                kda_input_proj::WEIGHT_QKV_SCALE_NAME, scaleDim0, scaleDim1, scaleDim2, expectScaleDim0,
                expectScaleDim1, WEIGHT_QKV_SCALE_PACK_NUM, static_cast<int32_t>(transWeightQkv)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void KdaInputProjInfoParser::GenerateInfo(KdaInputProjTilingInfo &tilingInfo)
{
    tilingInfo.opName = opName_;
    tilingInfo.platformInfo = platformInfo_;
    tilingInfo.baseParams = baseParams_;
    tilingInfo.transWeightQkv = opParamInfo_.transWeightQkv != nullptr ? *opParamInfo_.transWeightQkv : true;
    tilingInfo.transWeightBeta = opParamInfo_.transWeightBeta != nullptr ? *opParamInfo_.transWeightBeta : true;
    tilingInfo.transWeightGate = opParamInfo_.transWeightGate != nullptr ? *opParamInfo_.transWeightGate : true;
    tilingInfo.transWeightG = opParamInfo_.transWeightG != nullptr ? *opParamInfo_.transWeightG : true;
    tilingInfo.aicNum = aicNum_;
    tilingInfo.aivNum = aivNum_;
    tilingInfo.l1Size = l1Size_;
    tilingInfo.l0cSize = l0cSize_;
    tilingInfo.ubSize = ubSize_;

    OP_LOGI(opName_,
            "KdaInputProj ParseAndCheck: T=%u K=%u qkv=%u beta=%u gate=%u g=%u aic=%u aiv=%u l1=%lu l0c=%lu ub=%lu.",
            tilingInfo.baseParams.tSize, tilingInfo.baseParams.hiddenSize, tilingInfo.baseParams.qkvSize,
            tilingInfo.baseParams.betaSize, tilingInfo.baseParams.gateSize, tilingInfo.baseParams.gSize,
            tilingInfo.aicNum, tilingInfo.aivNum, tilingInfo.l1Size, tilingInfo.l0cSize, tilingInfo.ubSize);
}

ge::graphStatus KdaInputProjInfoParser::ParseAndCheck(KdaInputProjTilingInfo &tilingInfo)
{
    if (GetOpName() != ge::GRAPH_SUCCESS || GetNpuInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetInputParaInfo();
    GetOutputParaInfo();
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || GetAndCheckAttrParaInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckInOutDataType() != ge::GRAPH_SUCCESS || CheckShapeDim() != ge::GRAPH_SUCCESS ||
        GetBaseShapeInfo() != ge::GRAPH_SUCCESS || ValidateInputShapesMatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(tilingInfo);
    OP_LOGI(opName_, "KdaInputProj parse: T=%u K=%u qkv=%u beta=%u gate=%u g=%u trans(qkv/beta/gate/g)=%d/%d/%d/%d.",
            baseParams_.tSize, baseParams_.hiddenSize, baseParams_.qkvSize, baseParams_.betaSize, baseParams_.gateSize,
            baseParams_.gSize, static_cast<int32_t>(tilingInfo.transWeightQkv),
            static_cast<int32_t>(tilingInfo.transWeightBeta), static_cast<int32_t>(tilingInfo.transWeightGate),
            static_cast<int32_t>(tilingInfo.transWeightG));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForKdaInputProj(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::FillBaseParams(const KdaInputProjTilingInfo &tilingInfo)
{
    tilingData_.baseParams = tilingInfo.baseParams;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::CalcModuleTilings(const KdaInputProjTilingInfo &tilingInfo)
{
    // 与 kernel 四组件一一对应；跨模块 workspace / 核数协调在编排层汇总
    if (KdaInputProjMmBggTiling(tilingInfo, context_).CalcTiling(tilingData_.mmBggParams) != ge::GRAPH_SUCCESS ||
        KdaInputProjMxQuantTiling(tilingInfo).CalcTiling(tilingData_.mxQuantParams) != ge::GRAPH_SUCCESS ||
        KdaInputProjQmmQkvTiling(tilingInfo).CalcTiling(tilingData_.qmmQkvParams) != ge::GRAPH_SUCCESS ||
        KdaInputProjSigmoidTiling(tilingInfo).CalcTiling(tilingData_.sigmoidParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::CalcWorkspaceSize(const KdaInputProjTilingInfo &tilingInfo)
{
    const auto &base = tilingInfo.baseParams;
    workspaceSize_ = KdaInputProj::KdaInputProjWorkspace::TotalBytes(base.tSize, base.hiddenSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::SetTilingKey(const KdaInputProjTilingInfo &tilingInfo)
{
    // 与 ASCENDC_TPL_ARGS_DECL 顺序一致：TRANS_WEIGHT_{QKV,BETA,GATE,G}
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tilingInfo.transWeightQkv, tilingInfo.transWeightBeta,
                                                  tilingInfo.transWeightGate, tilingInfo.transWeightG);
    context_->SetTilingKey(tilingKey);
    OP_LOGI(tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaInputProj",
            "KdaInputProj tilingKey=%lu (trans_qkv=%d, trans_beta=%d, trans_gate=%d, trans_g=%d).", tilingKey,
            static_cast<int32_t>(tilingInfo.transWeightQkv), static_cast<int32_t>(tilingInfo.transWeightBeta),
            static_cast<int32_t>(tilingInfo.transWeightGate), static_cast<int32_t>(tilingInfo.transWeightG));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::WriteTilingResult(const KdaInputProjTilingInfo &tilingInfo)
{
    const char *opName = tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaInputProj";
    OP_CHECK_IF(tilingInfo.platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "platformInfo is null."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0U, OP_LOGE(opName, "num of aic obtained is 0."), return ge::GRAPH_FAILED);

    const uint32_t tileNum =
        tilingData_.mmBggParams.numBetaTile + tilingData_.mmBggParams.numGateTile + tilingData_.mmBggParams.numGTile;
    constexpr uint32_t kAivPerAic = 2U;
    // MxQuant 把行按核号分给 0..usedCoreNum-1，编号超出实际启动核数的那部分行不会有核去处理，
    // quantX 里对应行保持脏数据，Stage2 的 QMM 会照着算出错误的 qkv。所以 blockDim 不能只看
    // mm_bgg 的块数，必须同时够 MxQuant 用。
    const uint32_t mxQuantAic =
        static_cast<uint32_t>((tilingData_.mxQuantParams.usedCoreNum + static_cast<int64_t>(kAivPerAic) - 1) /
                              static_cast<int64_t>(kAivPerAic));
    // 与 MatMulV3 usedCoreNum = min(mCnt*nCnt, aicNum) 一致：块数不够时允许不满核。
    const uint32_t needAic = std::max(tileNum, mxQuantAic);
    const uint32_t blockDim = (needAic == 0U) ? 1U : std::min(aicNum, needAic);

    // 这类不一致是静默的（输出脏数据而非报错），宁可在 tiling 阶段就拦下来。
    OP_CHECK_IF(tilingData_.mxQuantParams.usedCoreNum > static_cast<int64_t>(blockDim * kAivPerAic),
                OP_LOGE(opName, "MxQuant usedCoreNum=%ld exceeds launched aiv=%u (blockDim=%u).",
                        tilingData_.mxQuantParams.usedCoreNum, blockDim * kAivPerAic, blockDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KdaInputProjSigmoidTiling(tilingInfo).FillAivSplit(tilingData_.sigmoidParams, blockDim * kAivPerAic) !=
                    ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Sigmoid FillAivSplit failed."), return ge::GRAPH_FAILED);

    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(1);

    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "GetWorkspaceSizes returned nullptr."),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize_) + static_cast<size_t>(sysWorkspaceSize);

    KdaInputProjTilingData *outTiling = context_->GetTilingData<KdaInputProjTilingData>();
    OP_CHECK_IF(outTiling == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "GetTilingData returned nullptr."),
                return ge::GRAPH_FAILED);
    *outTiling = tilingData_;

    OP_LOGI(opName, "KdaInputProj WriteTilingResult: blockDim=%u tileNum=%u workspace=%zu.", blockDim, tileNum,
            workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KdaInputProjTiling::DoTiling(const KdaInputProjTilingInfo *tilingInfo)
{
    OP_CHECK_IF(tilingInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaInputProj", "Tiling info is null."),
                return ge::GRAPH_FAILED);

    if (FillBaseParams(*tilingInfo) != ge::GRAPH_SUCCESS || CalcModuleTilings(*tilingInfo) != ge::GRAPH_SUCCESS ||
        CalcWorkspaceSize(*tilingInfo) != ge::GRAPH_SUCCESS || SetTilingKey(*tilingInfo) != ge::GRAPH_SUCCESS ||
        WriteTilingResult(*tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForKdaInputProj(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KdaInputProj", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    KdaInputProjTilingInfo tilingInfo;
    KdaInputProjInfoParser infoParser(context);
    if (infoParser.ParseAndCheck(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    KdaInputProjTiling tiling(context);
    return tiling.DoTiling(&tilingInfo);
}

IMPL_OP_OPTILING(KdaInputProj)
    .Tiling(TilingForKdaInputProj)
    .TilingParse<KdaInputProjCompileInfo>(TilingPrepareForKdaInputProj);
} // namespace optiling

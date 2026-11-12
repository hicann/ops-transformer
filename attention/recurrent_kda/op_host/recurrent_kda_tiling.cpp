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
 * \file recurrent_kda_tiling.cpp
 * \brief
 */
#include "recurrent_kda_tiling.h"

#include "err/ops_err.h"
#include "log/log.h"
#include "platform/soc_spec.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"
#include <cstring>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RecurrentKda, RecurrentKdaTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t GATE_INDEX = 3;
const size_t BETA_INDEX = 4;
const size_t STATE_INDEX = 5;
const size_t CU_SEQLENS_INDEX = 6;
const size_t SSM_STATE_INDICES_INDEX = 7;
const size_t A_LOG_INDEX = 8;
const size_t DT_BIAS_INDEX = 9;
const size_t ACC_TOKEN_INDEX = 10;

const size_t INITIAL_STATE_OUTPUT_INDEX = 1;
const size_t FINAL_STATE_OUTPUT_INDEX = 2;

const size_t ATTR_LAYOUT_INDEX = 0;
const size_t ATTR_SCALE_INDEX = 1;
const size_t ATTR_OUTPUT_FINAL_STATE_INDEX = 2;
const size_t ATTR_INPLACE_FINAL_STATE_INDEX = 3;
const size_t ATTR_USE_QK_L2NORM_INDEX = 4;
const size_t ATTR_USE_GATE_INDEX = 5;
const size_t ATTR_USE_BETA_SIGMOID_INDEX = 6;
const size_t ATTR_ALLOW_NEG_EIGVAL_INDEX = 7;
const size_t ATTR_SAFE_GATE_INDEX = 8;
const size_t ATTR_LOWER_BOUND_INDEX = 9;
const size_t ATTR_STATE_V_FIRST_INDEX = 10;

void RecurrentKdaTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.isA5 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.vectorCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

namespace {
void CopyOptionalOriginShape(gert::TilingContext* context, size_t index, gert::Shape& dst)
{
    auto shape = context->GetOptionalInputShape(index);
    if (shape != nullptr) {
        dst = shape->GetOriginShape();
    }
}

template <typename StrideType>
void CopyStateStrides(const StrideType* src, std::array<int64_t, RKDA_STATE_DIM_NUM>& dst, bool& hasStrides)
{
    if (src == nullptr || src->GetDimNum() != RKDA_STATE_DIM_NUM) {
        return;
    }
    for (size_t i = 0; i < RKDA_STATE_DIM_NUM; ++i) {
        dst[i] = src->GetStride(i);
    }
    hasStrides = true;
}

template <typename StrideType>
void CopyQkvStrides(const StrideType* src, size_t expectedDimNum, std::array<int64_t, RKDA_RANK4_QKV_DIM_NUM>& dst,
                    bool& hasStrides)
{
    if (src == nullptr || src->GetDimNum() != expectedDimNum || expectedDimNum > RKDA_RANK4_QKV_DIM_NUM) {
        return;
    }
    for (size_t i = 0; i < expectedDimNum; ++i) {
        dst[i] = src->GetStride(i);
    }
    hasStrides = true;
}
} // namespace

RecurrentKdaTilingContext RecurrentKdaTiling::BuildProcessorContext() const
{
    RecurrentKdaTilingContext ctx;
    ctx.nodeName = context_->GetNodeName();
    ctx.queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
    ctx.keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
    ctx.valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
    CopyQkvStrides(context_->GetInputStride(QUERY_INDEX), ctx.queryShape.GetDimNum(), ctx.queryStrides,
                   ctx.hasQueryStrides);
    CopyQkvStrides(context_->GetInputStride(KEY_INDEX), ctx.keyShape.GetDimNum(), ctx.keyStrides, ctx.hasKeyStrides);
    CopyQkvStrides(context_->GetInputStride(VALUE_INDEX), ctx.valueShape.GetDimNum(), ctx.valueStrides,
                   ctx.hasValueStrides);
    ctx.gateShape = context_->GetInputShape(GATE_INDEX)->GetOriginShape();
    ctx.betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
    ctx.stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
    CopyStateStrides(context_->GetInputStride(STATE_INDEX), ctx.stateInStrides, ctx.hasStateInStrides);
    const size_t stateOutputIndex =
        tilingData_.inplaceFinalState == 1 ? INITIAL_STATE_OUTPUT_INDEX : FINAL_STATE_OUTPUT_INDEX;
    CopyStateStrides(context_->GetOutputStride(stateOutputIndex), ctx.stateOutStrides, ctx.hasStateOutStrides);
    if (!ctx.hasStateOutStrides && tilingData_.inplaceFinalState == 1 && ctx.hasStateInStrides) {
        ctx.stateOutStrides = ctx.stateInStrides;
        ctx.hasStateOutStrides = true;
    }
    CopyOptionalOriginShape(context_, CU_SEQLENS_INDEX, ctx.cuSeqlensShape);
    CopyOptionalOriginShape(context_, SSM_STATE_INDICES_INDEX, ctx.ssmStateShape);
    CopyOptionalOriginShape(context_, A_LOG_INDEX, ctx.aLogShape);
    CopyOptionalOriginShape(context_, DT_BIAS_INDEX, ctx.dtBiasShape);
    CopyOptionalOriginShape(context_, ACC_TOKEN_INDEX, ctx.acceptedTokensShape);
    ctx.aivNum = compileInfo_.aivNum;
    ctx.ubSize = compileInfo_.ubSize;
    ctx.isA5 = compileInfo_.isA5 ? 1 : 0;
    ctx.stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    ctx.gateDtype = context_->GetInputDesc(GATE_INDEX)->GetDataType();
    ctx.betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    if (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr) {
        ctx.cuSeqlensDtype = context_->GetOptionalInputDesc(CU_SEQLENS_INDEX)->GetDataType();
    }
    if (context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX) != nullptr) {
        ctx.ssmStateIndicesDtype = context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    }
    if (context_->GetOptionalInputDesc(ACC_TOKEN_INDEX) != nullptr) {
        ctx.acceptedTokensDtype = context_->GetOptionalInputDesc(ACC_TOKEN_INDEX)->GetDataType();
    }
    ctx.scale = tilingData_.scale;
    ctx.lowerBound = tilingData_.lowerBound;
    ctx.layout = tilingData_.layout;
    ctx.hasCuSeqlens = tilingData_.hasCuSeqlens;
    ctx.hasSsmStateIndices = tilingData_.hasSsmStateIndices;
    ctx.hasALog = tilingData_.hasALog;
    ctx.hasDtBias = tilingData_.hasDtBias;
    ctx.hasAcceptedTokens = tilingData_.hasAcceptedTokens;
    ctx.useQkL2norm = tilingData_.useQkL2norm;
    ctx.useGateInKernel = tilingData_.useGateInKernel;
    ctx.useBetaSigmoid = tilingData_.useBetaSigmoid;
    ctx.allowNegEigval = tilingData_.allowNegEigval;
    ctx.safeGate = tilingData_.safeGate;
    ctx.stateVFirst = tilingData_.stateVFirst;
    ctx.outputFinalState = tilingData_.outputFinalState;
    ctx.inplaceFinalState = tilingData_.inplaceFinalState;
    return ctx;
}

ge::graphStatus RecurrentKdaTiling::GetPlatformInfo()
{
    if (context_->GetPlatformInfo() == nullptr || compileInfo_.aivNum <= 0 || compileInfo_.ubSize == 0) {
        const std::string reason = "GetPlatformInfo: platformInfo is " +
                                   std::string(context_->GetPlatformInfo() == nullptr ? "null" : "available") +
                                   ", aivNum=" + std::to_string(compileInfo_.aivNum) +
                                   ", ubSize=" + std::to_string(compileInfo_.ubSize) +
                                   "; check the SoC-specific OPP package and platform information loading.";
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(), reason.c_str()}));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentKdaTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid optional input."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid attrs."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "Invalid format."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::DoOpTiling()
{
    OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(inputParams_.opName, "CalUbSize failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
};

uint64_t RecurrentKdaTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus RecurrentKdaTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(RKDA_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentKdaTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RecurrentKdaTilingData);
    auto rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
        rawTilingData->GetCapacity() < tilingDataSize) {
        const std::string reason =
            "PostTiling: raw tiling buffer is missing or too small, requiredBytes=" + std::to_string(tilingDataSize) +
            ", actualBytes=" + std::to_string(rawTilingData == nullptr ? 0 : rawTilingData->GetCapacity()) +
            "; check TilingContext buffer construction.";
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(), reason.c_str()}));
        return ge::GRAPH_FAILED;
    }
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      ("PostTiling: memcpy_s failed, ret=" + std::to_string(ret) +
                                       ", requiredBytes=" + std::to_string(tilingDataSize) +
                                       ", actualBytes=" + std::to_string(rawTilingData->GetCapacity()))
                                          .c_str()}));
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(),
                                                            "PostTiling: GetWorkspaceSizes(1) returned null; required "
                                                            "output slots=1. Check TilingContext construction."})),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::CheckContext()
{
    OP_CHECK_IF(context_->GetInputShape(QUERY_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "query"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(QUERY_INDEX) == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      "CheckContext: query input descriptor is null although its tensor is present; "
                                      "check TilingContext metadata construction."})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(KEY_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "key"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(KEY_INDEX) == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      "CheckContext: key input descriptor is null although its tensor is present; "
                                      "check TilingContext metadata construction."})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(VALUE_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "value"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(VALUE_INDEX) == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      "CheckContext: value input descriptor is null although its tensor is present; "
                                      "check TilingContext metadata construction."})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(GATE_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "gate"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(GATE_INDEX) == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      "CheckContext: gate input descriptor is null although its tensor is present; "
                                      "check TilingContext metadata construction."})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(BETA_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "beta"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(BETA_INDEX) == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context_->GetNodeName(),
                                      "CheckContext: beta input descriptor is null although its tensor is present; "
                                      "check TilingContext metadata construction."})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(STATE_INDEX) == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "initial_state"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputDesc(STATE_INDEX) == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        context_->GetNodeName(),
                        "CheckContext: initial_state input descriptor is null although its tensor is present; "
                        "check TilingContext metadata construction."})),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(
        queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "query, key, value",
                                               Ops::Base::ToString(queryDtype) + ", " + Ops::Base::ToString(keyDtype) +
                                                   ", " + Ops::Base::ToString(valueDtype),
                                               "query, key and value must be bfloat16"),
        return ge::GRAPH_FAILED);

    auto gateDtype = context_->GetInputDesc(GATE_INDEX)->GetDataType();
    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(
        (gateDtype != ge::DT_FLOAT && gateDtype != ge::DT_BF16 && gateDtype != ge::DT_FLOAT16) ||
            (betaDtype != ge::DT_FLOAT && betaDtype != ge::DT_BF16 && betaDtype != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "gate, beta",
                                               Ops::Base::ToString(gateDtype) + ", " + Ops::Base::ToString(betaDtype),
                                               "gate and beta must be float32, bfloat16 or float16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != ge::DT_FLOAT && stateDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "initial_state", Ops::Base::ToString(stateDtype),
                                          "bfloat16 or float32"),
                return ge::GRAPH_FAILED);
    if (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr) {
        auto dtype = context_->GetOptionalInputDesc(CU_SEQLENS_INDEX)->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "cu_seqlens", Ops::Base::ToString(dtype),
                                              "int32 or int64"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX) != nullptr) {
        auto dtype = context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "ssm_state_indices", Ops::Base::ToString(dtype),
                                              "int32 or int64"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(A_LOG_INDEX) != nullptr) {
        OP_CHECK_IF(context_->GetOptionalInputDesc(A_LOG_INDEX)->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        context_->GetNodeName(), "A_log",
                        Ops::Base::ToString(context_->GetOptionalInputDesc(A_LOG_INDEX)->GetDataType()), "float32"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(DT_BIAS_INDEX) != nullptr) {
        OP_CHECK_IF(context_->GetOptionalInputDesc(DT_BIAS_INDEX)->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        context_->GetNodeName(), "dt_bias",
                        Ops::Base::ToString(context_->GetOptionalInputDesc(DT_BIAS_INDEX)->GetDataType()), "float32"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(ACC_TOKEN_INDEX) != nullptr) {
        auto dtype = context_->GetOptionalInputDesc(ACC_TOKEN_INDEX)->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "num_accepted_tokens",
                                              Ops::Base::ToString(dtype), "int32 or int64"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::AnalyzeShapes()
{
    RecurrentKdaTilingProcessor processor(BuildProcessorContext());
    return processor.ProcessShapes(tilingData_);
}

bool RecurrentKdaTiling::CheckFormat(ge::Format format, const std::string& desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), desc, Ops::Base::ToString(format),
                                               "FRACTAL_NZ is not supported; use an ND tensor");
        return false;
    }
    return true;
}

ge::graphStatus RecurrentKdaTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
        !CheckFormat(context_->GetInputDesc(GATE_INDEX)->GetStorageFormat(), "gate") ||
        !CheckFormat(context_->GetInputDesc(BETA_INDEX)->GetStorageFormat(), "beta") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "initial_state")) {
        return ge::GRAPH_FAILED;
    }

    const std::array<std::pair<size_t, const char*>, 5> optionalInputs = {{
        {CU_SEQLENS_INDEX, "cu_seqlens"},
        {SSM_STATE_INDICES_INDEX, "ssm_state_indices"},
        {A_LOG_INDEX, "A_log"},
        {DT_BIAS_INDEX, "dt_bias"},
        {ACC_TOKEN_INDEX, "num_accepted_tokens"},
    }};
    for (const auto& item : optionalInputs) {
        auto desc = context_->GetOptionalInputDesc(item.first);
        if (desc != nullptr && !CheckFormat(desc->GetStorageFormat(), item.second)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::GetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                context_->GetNodeName(),
                "GetAttrsInfo: runtime attrs are null; check registered attributes and TilingContext construction."})),
        return ge::GRAPH_FAILED);
    const char* layout = attrs->GetAttrPointer<char>(ATTR_LAYOUT_INDEX);
    if (layout == nullptr || std::strcmp(layout, "BSND") == 0) {
        tilingData_.layout = RKDA_LAYOUT_BSND;
    } else if (std::strcmp(layout, "TND") == 0) {
        tilingData_.layout = RKDA_LAYOUT_TND;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout", layout, "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    tilingData_.scale = *attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    tilingData_.outputFinalState = *attrs->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE_INDEX) ? 1 : 0;
    tilingData_.inplaceFinalState = *attrs->GetAttrPointer<bool>(ATTR_INPLACE_FINAL_STATE_INDEX) ? 1 : 0;
    tilingData_.useQkL2norm = *attrs->GetAttrPointer<bool>(ATTR_USE_QK_L2NORM_INDEX) ? 1 : 0;
    tilingData_.useGateInKernel = *attrs->GetAttrPointer<bool>(ATTR_USE_GATE_INDEX) ? 1 : 0;
    tilingData_.useBetaSigmoid = *attrs->GetAttrPointer<bool>(ATTR_USE_BETA_SIGMOID_INDEX) ? 1 : 0;
    tilingData_.allowNegEigval = *attrs->GetAttrPointer<bool>(ATTR_ALLOW_NEG_EIGVAL_INDEX) ? 1 : 0;
    tilingData_.safeGate = *attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE_INDEX) ? 1 : 0;
    tilingData_.lowerBound = *attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_INDEX);
    tilingData_.stateVFirst = *attrs->GetAttrPointer<bool>(ATTR_STATE_V_FIRST_INDEX) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentKdaTiling::GetOptionalInput()
{
    tilingData_.hasCuSeqlens = (context_->GetOptionalInputDesc(CU_SEQLENS_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasSsmStateIndices = (context_->GetOptionalInputDesc(SSM_STATE_INDICES_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasALog = (context_->GetOptionalInputDesc(A_LOG_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasDtBias = (context_->GetOptionalInputDesc(DT_BIAS_INDEX) == nullptr) ? 0 : 1;
    tilingData_.hasAcceptedTokens = (context_->GetOptionalInputDesc(ACC_TOKEN_INDEX) == nullptr) ? 0 : 1;
    return ge::GRAPH_SUCCESS;
}

void RecurrentKdaTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "seqLen: [%u]", tilingData_.seqLen);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "ssmStateStride: [%u]", tilingData_.ssmStateStride);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "stateOutBufferNum: [%u]", tilingData_.stateOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "attnOutBufferNum: [%u]", tilingData_.attnOutBufferNum);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "lowerBound: [%f]", tilingData_.lowerBound);
    OP_LOGD(context_->GetNodeName(), "layout: [%u]", tilingData_.layout);
    OP_LOGD(context_->GetNodeName(), "hasCuSeqlens: [%u]", tilingData_.hasCuSeqlens);
    OP_LOGD(context_->GetNodeName(), "hasSsmStateIndices: [%u]", tilingData_.hasSsmStateIndices);
    OP_LOGD(context_->GetNodeName(), "hasALog: [%u]", tilingData_.hasALog);
    OP_LOGD(context_->GetNodeName(), "hasDtBias: [%u]", tilingData_.hasDtBias);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
    OP_LOGD(context_->GetNodeName(), "useQkL2norm: [%u]", tilingData_.useQkL2norm);
    OP_LOGD(context_->GetNodeName(), "useGateInKernel: [%u]", tilingData_.useGateInKernel);
    OP_LOGD(context_->GetNodeName(), "useBetaSigmoid: [%u]", tilingData_.useBetaSigmoid);
    OP_LOGD(context_->GetNodeName(), "allowNegEigval: [%u]", tilingData_.allowNegEigval);
    OP_LOGD(context_->GetNodeName(), "safeGate: [%u]", tilingData_.safeGate);
    OP_LOGD(context_->GetNodeName(), "stateVFirst: [%u]", tilingData_.stateVFirst);
    OP_LOGD(context_->GetNodeName(), "outputFinalState: [%u]", tilingData_.outputFinalState);
    OP_LOGD(context_->GetNodeName(), "inplaceFinalState: [%u]", tilingData_.inplaceFinalState);
    OP_LOGD(context_->GetNodeName(), "gateDtype: [%u]", tilingData_.gateDtype);
    OP_LOGD(context_->GetNodeName(), "betaDtype: [%u]", tilingData_.betaDtype);
    OP_LOGD(context_->GetNodeName(), "cuSeqlensDtype: [%u]", tilingData_.cuSeqlensDtype);
    OP_LOGD(context_->GetNodeName(), "ssmStateIndicesDtype: [%u]", tilingData_.ssmStateIndicesDtype);
    OP_LOGD(context_->GetNodeName(), "acceptedTokensDtype: [%u]", tilingData_.acceptedTokensDtype);
    OP_LOGD(context_->GetNodeName(), "queryTokenStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.queryTokenStride));
    OP_LOGD(context_->GetNodeName(), "queryHeadStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.queryHeadStride));
    OP_LOGD(context_->GetNodeName(), "keyTokenStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.keyTokenStride));
    OP_LOGD(context_->GetNodeName(), "keyHeadStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.keyHeadStride));
    OP_LOGD(context_->GetNodeName(), "valueTokenStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.valueTokenStride));
    OP_LOGD(context_->GetNodeName(), "valueHeadStride: [%llu]",
            static_cast<unsigned long long>(tilingData_.valueHeadStride));
}

ge::graphStatus RecurrentKdaTiling::CalUbSize()
{
    RecurrentKdaTilingProcessor processor(BuildProcessorContext());
    return processor.ProcessUb(tilingData_);
}

static ge::graphStatus RecurrentKdaTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"RecurrentKda", "Tiling: context is null; check TilingContext construction."})),
        return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentKda(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"RecurrentKda", "Tiling: context is null; check TilingContext construction."})),
        return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                context->GetNodeName(),
                "TilingPrepare: platformInfo is null; check SoC-specific OPP and platform information loading."})),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentKdaCompileInfo>();
    OP_CHECK_IF(
        compileInfoPtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                context->GetNodeName(),
                "TilingPrepare: compileInfo is null; check compiled metadata and TilingParseContext construction."})),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentKda)
    .Tiling(RecurrentKdaTilingFunc)
    .TilingParse<RecurrentKdaCompileInfo>(TilingPrepareForRecurrentKda);
} // namespace optiling

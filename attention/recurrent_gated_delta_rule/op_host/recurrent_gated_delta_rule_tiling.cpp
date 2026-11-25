/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling.cpp
 * \brief
 */
#include "recurrent_gated_delta_rule_tiling.h"

#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRule, RecurrentGatedDeltaRuleTiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t ACTUAL_SEQ_LENGTHS_INDEX = 5;
const size_t SSM_STATE_INDICES_INDEX = 6;
const size_t G_INDEX = 7;
const size_t GK_INDEX = 8;
const size_t NUM_ACCEPTED_TOKENS_INDEX = 9;
const size_t OUT_INDEX = 0;

const size_t QKV_DIM_NUM = 3;
const size_t OUT_DIM_NUM = 3;
const size_t BETA_DIM_NUM = 2;
const size_t STATE_DIM_NUM = 4;
const size_t ACTUAL_SEQ_LENGTHS_DIM_NUM = 1;
const size_t SSM_STATE_INDICES_DIM_NUM = 1;
const size_t G_DIM_NUM = 2;
const size_t GK_DIM_NUM = 3;
const size_t NUM_ACCEPTED_TOKENS_DIM_NUM = 1;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

const size_t MAX_MTP = 8;

void RecurrentGatedDeltaRuleTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.vectorCoreNum = compileInfo_.aivNum;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetScale() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetScale."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetStrides() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid strides."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoOpTiling()
{
    OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::DoLibApiTiling()
{
    tilingKey_ = (stateDtype_ == ge::DT_FLOAT) ? 1UL : 0UL;
    return ge::GRAPH_SUCCESS;
};

uint64_t RecurrentGatedDeltaRuleTiling::GetTilingKey() const
{
    return tilingKey_;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetWorkspaceSize()
{
    // system workspace size is 16 * 1024 * 1024 = 16M;
    constexpr int64_t sysWorkspaceSize = 16777216;
    workspaceSize_ = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
};

ge::graphStatus RecurrentGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RecurrentGatedDeltaRuleTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null."),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CheckOptionalInputContext(const size_t optionalIndex,
                                                                         const std::string& optionalName)
{
    OP_CHECK_IF(context_->GetOptionalInputDesc(optionalIndex) == nullptr &&
                    context_->GetOptionalInputTensor(optionalIndex) != nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "The desc of %s is nullptr, but the tensor of %s is not nullptr. They must be either both null "
                        "or both non-null.",
                        optionalName.c_str(), optionalName.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputDesc(optionalIndex) != nullptr &&
                    context_->GetOptionalInputTensor(optionalIndex) == nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "The tensor of %s is nullptr, but the desc of %s is not nullptr. They must be either both null "
                        "or both non-null.",
                        optionalName.c_str(), optionalName.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(BETA_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(BETA_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(STATE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(STATE_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(SSM_STATE_INDICES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(SSM_STATE_INDICES_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(OUT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUT_INDEX));

    // 可选参数校验
    if (CheckOptionalInputContext(G_INDEX, "g") != ge::GRAPH_SUCCESS ||
        CheckOptionalInputContext(GK_INDEX, "gk") != ge::GRAPH_SUCCESS ||
        CheckOptionalInputContext(NUM_ACCEPTED_TOKENS_INDEX, "num_accepted_tokens") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeDtype()
{
    auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(queryDtype != ge::DT_BF16 || keyDtype != ge::DT_BF16 || valueDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "query/key/value",
                                                      (std::string(ge::TypeUtils::DataTypeToSerialString(queryDtype)) +
                                                       "/" + ge::TypeUtils::DataTypeToSerialString(keyDtype) + "/" +
                                                       ge::TypeUtils::DataTypeToSerialString(valueDtype))
                                                          .c_str(),
                                                      "the dtypes of query, key and value must be bfloat16"),
                return ge::GRAPH_FAILED);

    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(betaDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "beta",
                                                      ge::TypeUtils::DataTypeToSerialString(betaDtype).c_str(),
                                                      "the dtype of beta must be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != ge::DT_BF16 && stateDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "state",
                                                      ge::TypeUtils::DataTypeToSerialString(stateDtype).c_str(),
                                                      "the dtype of state must be bfloat16 or float32"),
                return ge::GRAPH_FAILED);
    stateDtype_ = stateDtype;

    auto actSeqlensDtype = context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetDataType();
    auto ssmStateIndicesDtype = context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    OP_CHECK_IF(
        actSeqlensDtype != ge::DT_INT32 || ssmStateIndicesDtype != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "actual_seq_lengths/ssm_state_indices",
                                              (std::string(ge::TypeUtils::DataTypeToSerialString(actSeqlensDtype)) +
                                               "/" + ge::TypeUtils::DataTypeToSerialString(ssmStateIndicesDtype))
                                                  .c_str(),
                                              "the dtypes of actual_seq_lengths and ssm_state_indices must be int32"),
        return ge::GRAPH_FAILED);

    if (tilingData_.hasGama == 1) {
        auto gamaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gamaDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "g",
                                                          ge::TypeUtils::DataTypeToSerialString(gamaDtype).c_str(),
                                                          "the dtype of g must be float32"),
                    return ge::GRAPH_FAILED);
    }

    if (tilingData_.hasGamaK == 1) {
        auto gamaKDtype = context_->GetOptionalInputDesc(GK_INDEX)->GetDataType();
        OP_CHECK_IF(gamaKDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gk",
                                                          ge::TypeUtils::DataTypeToSerialString(gamaKDtype).c_str(),
                                                          "the dtype of gk must be float32"),
                    return ge::GRAPH_FAILED);
    }

    if (context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX) != nullptr) {
        auto numAcceptedTokensDtype = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX)->GetDataType();
        OP_CHECK_IF(
            numAcceptedTokensDtype != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "num_accepted_tokens",
                                                  ge::TypeUtils::DataTypeToSerialString(numAcceptedTokensDtype).c_str(),
                                                  "the dtype of num_accepted_tokens must be int32"),
            return ge::GRAPH_FAILED);
    }
    auto outDtype = context_->GetOutputDesc(OUT_INDEX)->GetDataType();
    OP_CHECK_IF(outDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "out",
                                                      ge::TypeUtils::DataTypeToSerialString(outDtype).c_str(),
                                                      "the dtype of out must be bfloat16"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查维度数量是否符合预期
bool RecurrentGatedDeltaRuleTiling::CheckDim(const gert::Shape shape, const size_t dim, const std::string& dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), dimDesc.c_str(), (std::to_string(shape.GetDimNum()) + "D").c_str(),
            ("the dim num of " + dimDesc + " must be " + std::to_string(dim)).c_str());
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapesParser()
{
    const auto& queryShape = context_->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto& valueShape = context_->GetInputShape(VALUE_INDEX)->GetStorageShape();
    const auto& stateShape = context_->GetInputShape(STATE_INDEX)->GetStorageShape();
    const auto& actSeqlensShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageShape();

    int64_t t = queryShape.GetDim(DIM_0);
    int64_t nk = queryShape.GetDim(DIM_1);
    int64_t dk = queryShape.GetDim(DIM_2);
    int64_t nv = valueShape.GetDim(DIM_1);
    int64_t dv = valueShape.GetDim(DIM_2);
    int64_t sBlockNum = stateShape.GetDim(DIM_0);
    int64_t b = actSeqlensShape.GetDim(DIM_0);

    // T>0
    OP_CHECK_IF(t <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "T", std::to_string(t).c_str(),
                                                      "T must be greater than 0"),
                return ge::GRAPH_FAILED);
    // // B>=0
    OP_CHECK_IF(b <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "B", std::to_string(b).c_str(),
                                                      "B must be greater than 0"),
                return ge::GRAPH_FAILED);
    // nk>0 nk<=256
    OP_CHECK_IF(nk <= 0 || nk > 256,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "Nk", std::to_string(nk).c_str(),
                                                      "Nk must be greater than 0 and less than or equal to 256"),
                return ge::GRAPH_FAILED);
    // nv>0 nv<=256
    OP_CHECK_IF(nv <= 0 || nv > 256,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "Nv", std::to_string(nv).c_str(),
                                                      "Nv must be greater than 0 and less than or equal to 256"),
                return ge::GRAPH_FAILED);
    // dk>0 dk<=512
    OP_CHECK_IF(dk <= 0 || dk > 512,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "Dk", std::to_string(dk).c_str(),
                                                      "Dk must be greater than 0 and less than or equal to 512"),
                return ge::GRAPH_FAILED);
    // dv>0 dv<=512
    OP_CHECK_IF(dv <= 0 || dv > 512,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "Dv", std::to_string(dv).c_str(),
                                                      "Dv must be greater than 0 and less than or equal to 512"),
                return ge::GRAPH_FAILED);
    // nv>=nk
    OP_CHECK_IF(nv % nk != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "Nv/Nk",
                                                      (std::to_string(nv) + "/" + std::to_string(nk)).c_str(),
                                                      "Nv must be an integer multiple of Nk"),
                return ge::GRAPH_FAILED);
    // blockNum >= T
    OP_CHECK_IF(sBlockNum < t,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "sBlockNum",
                                                      (std::to_string(sBlockNum) + "/" + std::to_string(t)).c_str(),
                                                      "BlockNum must be greater than or equal to T"),
                return ge::GRAPH_FAILED);

    tilingData_.t = static_cast<uint32_t>(t);
    tilingData_.nk = static_cast<uint32_t>(nk);
    tilingData_.dk = static_cast<uint32_t>(dk);
    tilingData_.nv = static_cast<uint32_t>(nv);
    tilingData_.dv = static_cast<uint32_t>(dv);
    tilingData_.sBlockNum = static_cast<uint32_t>(sBlockNum);
    tilingData_.b = static_cast<uint32_t>(b);

    uint64_t batchHeadTaskNum = static_cast<uint64_t>(b) * static_cast<uint64_t>(nv);
    if (batchHeadTaskNum > 0 && batchHeadTaskNum < tilingData_.vectorCoreNum) {
        tilingData_.vectorCoreNum = static_cast<uint32_t>(batchHeadTaskNum);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeEmptyTensor()
{
    OP_CHECK_IF(
        context_->GetInputShape(QUERY_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "query", "0", "query can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputShape(KEY_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "key", "0", "key can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputShape(VALUE_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "value", "0", "value can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputShape(BETA_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "beta", "0", "beta can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputShape(STATE_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "state", "0", "state can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "actual_seq_lengths", "0",
                                                      "actual_seq_lengths can not be an empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "ssm_state_indices", "0",
                                                      "ssm_state_indices can not be an empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetOutputShape(OUT_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "out", "0", "out can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingData_.hasGama == 1 && context_->GetOptionalInputShape(G_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "g", "0", "g can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingData_.hasGamaK == 1 && context_->GetOptionalInputShape(GK_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "gk", "0", "gk can not be an empty tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.hasAcceptedTokens == 1 &&
                    context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "num_accepted_tokens", "0",
                                                      "num_accepted_tokens can not be an empty tensor"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeOptionalShapes()
{
    // g (T, NV)
    gert::Shape expectGShape = gert::Shape({tilingData_.t, tilingData_.nv});
    // gk (T, NV, DK)
    gert::Shape expectGkShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dk});
    // num_accepted_tokens (B)
    gert::Shape expectNumAcceptedShape = gert::Shape({tilingData_.b});
    if (tilingData_.hasGama == 1) {
        const auto& gShape = context_->GetOptionalInputShape(G_INDEX)->GetStorageShape();
        if (!CheckDim(gShape, G_DIM_NUM, "g")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(gShape != expectGShape,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "g", Ops::Base::ToString(gShape).c_str(),
                        ("the shape of g must be [" + Ops::Base::ToString(expectGShape) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (tilingData_.hasGamaK == 1) {
        const auto& gkShape = context_->GetOptionalInputShape(GK_INDEX)->GetStorageShape();
        if (!CheckDim(gkShape, GK_DIM_NUM, "gk")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(gkShape != expectGkShape,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "gk", Ops::Base::ToString(gkShape).c_str(),
                        ("the shape of gk must be [" + Ops::Base::ToString(expectGkShape) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (tilingData_.hasAcceptedTokens == 1) {
        const auto& numAcceptedShape = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX)->GetStorageShape();
        if (!CheckDim(numAcceptedShape, NUM_ACCEPTED_TOKENS_DIM_NUM, "num_accepted_tokens")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            numAcceptedShape != expectNumAcceptedShape,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "num_accepted_tokens", Ops::Base::ToString(numAcceptedShape).c_str(),
                ("the shape of num_accepted_tokens must be [" + Ops::Base::ToString(expectNumAcceptedShape) + "]")
                    .c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    const auto& queryShape = context_->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto& keyShape = context_->GetInputShape(KEY_INDEX)->GetStorageShape();
    const auto& valueShape = context_->GetInputShape(VALUE_INDEX)->GetStorageShape();
    const auto& betaShape = context_->GetInputShape(BETA_INDEX)->GetStorageShape();
    const auto& stateShape = context_->GetInputShape(STATE_INDEX)->GetStorageShape();
    const auto& actSeqlensShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageShape();
    const auto& ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetStorageShape();
    const auto& outShape = context_->GetOutputShape(OUT_INDEX)->GetStorageShape();

    if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(keyShape, QKV_DIM_NUM, "key") ||
        !CheckDim(valueShape, QKV_DIM_NUM, "value") || !CheckDim(betaShape, BETA_DIM_NUM, "beta") ||
        !CheckDim(stateShape, STATE_DIM_NUM, "state") ||
        !CheckDim(actSeqlensShape, ACTUAL_SEQ_LENGTHS_DIM_NUM, "actual_seq_lengths") ||
        !CheckDim(ssmStateShape, SSM_STATE_INDICES_DIM_NUM, "ssm_state_indices") ||
        !CheckDim(outShape, OUT_DIM_NUM, "out")) {
        return ge::GRAPH_FAILED;
    }

    if (AnalyzeShapesParser() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // qk (T, NK, DK)
    gert::Shape expectQKShape = gert::Shape({tilingData_.t, tilingData_.nk, tilingData_.dk});
    // value (T, NV, DV)
    gert::Shape expectValueShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dv});
    // state (BlockNum, NV, DV, Dk)
    gert::Shape expectStateShape = gert::Shape({tilingData_.sBlockNum, tilingData_.nv, tilingData_.dv, tilingData_.dk});
    // beta (T, NV)
    gert::Shape expectBetaShape = gert::Shape({tilingData_.t, tilingData_.nv});
    // out (T, NV, DV)
    gert::Shape expectOutShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dv});
    // actSeqlensShape (B)
    gert::Shape expectActSeqlensShape = gert::Shape({tilingData_.b});
    // ssmStateShape (T)
    gert::Shape expectSsmStateShape = gert::Shape({tilingData_.t});

    OP_CHECK_IF(queryShape != expectQKShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "query", Ops::Base::ToString(queryShape).c_str(),
                    ("the shape of query must be [" + Ops::Base::ToString(expectQKShape) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape != expectQKShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "key", Ops::Base::ToString(keyShape).c_str(),
                    ("the shape of key must be [" + Ops::Base::ToString(expectQKShape) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueShape != expectValueShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "value", Ops::Base::ToString(valueShape).c_str(),
                    ("the shape of value must be [" + Ops::Base::ToString(expectValueShape) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateShape != expectStateShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "state", Ops::Base::ToString(stateShape).c_str(),
                    ("the shape of state must be [" + Ops::Base::ToString(expectStateShape) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaShape != expectBetaShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "beta", Ops::Base::ToString(betaShape).c_str(),
                    ("the shape of beta must be [" + Ops::Base::ToString(expectBetaShape) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape != expectOutShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "out", Ops::Base::ToString(outShape).c_str(),
                    ("the shape of out must be [" + Ops::Base::ToString(expectOutShape) + "]").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        actSeqlensShape != expectActSeqlensShape,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "actual_seq_lengths", Ops::Base::ToString(actSeqlensShape).c_str(),
            ("the shape of actual_seq_lengths must be [" + Ops::Base::ToString(expectActSeqlensShape) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ssmStateShape != expectSsmStateShape,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "ssm_state_indices", Ops::Base::ToString(ssmStateShape).c_str(),
            ("the shape of ssm_state_indices must be [" + Ops::Base::ToString(expectSsmStateShape) + "]").c_str()),
        return ge::GRAPH_FAILED);

    if (AnalyzeOptionalShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (AnalyzeEmptyTensor() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 检查是否为FormatND格式
bool RecurrentGatedDeltaRuleTiling::CheckFormat(ge::Format format, const std::string& Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), Desc.c_str(),
                                               ge::TypeUtils::FormatToSerialString(format).c_str(),
                                               ("the format of " + Desc + " can not be FRACTAL_NZ").c_str());
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
        !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
        !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
        !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "state") ||
        !CheckFormat(context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageFormat(), "actual_seq_lengths") ||
        !CheckFormat(context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetStorageFormat(), "ssm_state_indices")) {
        return ge::GRAPH_FAILED;
    }

    if (tilingData_.hasGama == 1) {
        auto gamaFormat = context_->GetOptionalInputDesc(G_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "g",
                                                           ge::TypeUtils::FormatToSerialString(gamaFormat).c_str(),
                                                           "the format of g can not be FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);
    }
    if (tilingData_.hasGamaK == 1) {
        auto gamaKFormat = context_->GetOptionalInputDesc(GK_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaKFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "gk",
                                                           ge::TypeUtils::FormatToSerialString(gamaKFormat).c_str(),
                                                           "the format of gk can not be FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX) != nullptr) {
        auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX)->GetStorageFormat();
        OP_CHECK_IF(
            numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "num_accepted_tokens",
                                                   ge::TypeUtils::FormatToSerialString(numAcceptedTokensFormat).c_str(),
                                                   "the format of num_accepted_tokens can not be FRACTAL_NZ"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetScale()
{
    auto attrs = context_->GetAttrs();
    float scaleValue = *attrs->GetAttrPointer<float>(0);
    tilingData_.scale = scaleValue;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetOptionalInput()
{
    if (context_->GetOptionalInputDesc(G_INDEX) != nullptr && context_->GetOptionalInputTensor(G_INDEX) != nullptr) {
        tilingData_.hasGama = 1;
    } else {
        tilingData_.hasGama = 0;
    }
    if (context_->GetOptionalInputDesc(GK_INDEX) != nullptr && context_->GetOptionalInputTensor(GK_INDEX) != nullptr) {
        tilingData_.hasGamaK = 1;
    } else {
        tilingData_.hasGamaK = 0;
    }
    if (context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX) != nullptr &&
        context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX) != nullptr) {
        tilingData_.hasAcceptedTokens = 1;
    } else {
        tilingData_.hasAcceptedTokens = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::GetStrides()
{
    auto strideState = context_->GetInputStride(STATE_INDEX);
    if (strideState != nullptr && strideState->GetDimNum() == STATE_DIM_NUM) {
        int64_t stride0 = strideState->GetStride(0);
        int64_t stride1 = strideState->GetStride(1);
        int64_t stride2 = strideState->GetStride(2);
        int64_t stride3 = strideState->GetStride(3);
        uint64_t expectStride2 = static_cast<uint64_t>(tilingData_.dk);
        OP_CHECK_IF(stride2 != static_cast<int64_t>(expectStride2) || stride3 != 1,
                    OP_LOGE_FOR_INVALID_STRIDE(inputParams_.opName, "state",
                                               (std::to_string(stride0) + ", " + std::to_string(stride1) + ", " +
                                                std::to_string(stride2) + ", " + std::to_string(stride3))
                                                   .c_str(),
                                               ("stride2=" + std::to_string(expectStride2) +
                                                ", stride3=1 (only dim 0 and dim 1 can be non-contiguous)")
                                                   .c_str()),
                    return ge::GRAPH_FAILED);
        tilingData_.stateStride0 = static_cast<uint64_t>(stride0);
        tilingData_.stateStride1 = static_cast<uint64_t>(stride1);
    } else {
        tilingData_.stateStride1 = static_cast<uint64_t>(tilingData_.dk) * static_cast<uint64_t>(tilingData_.dv);
        tilingData_.stateStride0 = static_cast<uint64_t>(tilingData_.nv) * tilingData_.stateStride1;
    }
    return ge::GRAPH_SUCCESS;
}

void RecurrentGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
    OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
    OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
    OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
    OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
    OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
    OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
    OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
    OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
    OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
    OP_LOGD(context_->GetNodeName(), "hasGama: [%u]", tilingData_.hasGama);
    OP_LOGD(context_->GetNodeName(), "hasGamaK: [%u]", tilingData_.hasGamaK);
    OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
    OP_LOGD(context_->GetNodeName(), "stateStride0: [%lu]", tilingData_.stateStride0);
    OP_LOGD(context_->GetNodeName(), "stateStride1: [%lu]", tilingData_.stateStride1);
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CalUbSize()
{
    int64_t ubSize = compileInfo_.ubSize;
    int64_t aNv = Ops::Base::CeilAlign(tilingData_.nv, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    int64_t aDv = Ops::Base::CeilAlign(tilingData_.dv, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    int64_t aDk = Ops::Base::CeilAlign(tilingData_.dk, static_cast<uint32_t>(16)); // 16 * 2 = 32B
    int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv); // 4 for qInQueue_ & kInQueue_, 2 for vInQueue_
    usedUbBytes += 128;                                  // reserve 128 Bytes
    if (tilingData_.hasGamaK) {
        usedUbBytes += MAX_MTP * 4 * aDk; // 4 for gk gamaInQueue_
    }
    if (tilingData_.hasGama) {
        usedUbBytes += MAX_MTP * 4 * aNv; // 4 for g gamaInQueue_
    }
    usedUbBytes += MAX_MTP * 2 * aNv; // 2 for betaInQueue_
    tilingData_.ubRestBytes = ubSize - usedUbBytes;
    usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv); // 8 for qk in ub, 4 for v in ub, 4 for beta in ub
    int64_t stateInElemBytes = (stateDtype_ == ge::DT_FLOAT) ? sizeof(float) : sizeof(uint16_t);
    int64_t stateOutElemBytes = stateInElemBytes;                     // stateOut has the same dtype as stateIn
    int64_t coeff = (stateInElemBytes + stateOutElemBytes) * aDk + 4; // stateInQueue_, stateOutQueue_, attnOutQueue_
    coeff += (4 + 4) * aDk + 4 + 4;                                   // 4 for qInUb, kInUb, vInUb, deltaInUb, attnInUb
    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;           // 8 * sizeof(float) = 32
    if (vStep < 8) {                                                  // vStep不小于8
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "query/value",
            (Ops::Base::ToString(context_->GetInputShape(QUERY_INDEX)->GetOriginShape()) + "/" +
             Ops::Base::ToString(context_->GetInputShape(VALUE_INDEX)->GetOriginShape()))
                .c_str(),
            "the shape is too large for the current platform, reduce T, Nv, Dv or Dk");
        return ge::GRAPH_FAILED;
    }
    int64_t rptime = Ops::Base::CeilDiv(tilingData_.dv, static_cast<uint32_t>(vStep));
    vStep = Ops::Base::CeilAlign(Ops::Base::CeilDiv(tilingData_.dv, static_cast<uint32_t>(rptime)),
                                 static_cast<uint32_t>(8)); // 8 * sizeof(float) = 32
    tilingData_.ubCalSize = compileInfo_.ubSize;
    tilingData_.vStep = vStep;
    tilingData_.ubRestBytes -= ((stateInElemBytes + stateOutElemBytes) * aDk + 4) * vStep;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RecurrentGatedDeltaRuleTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null."),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentGatedDeltaRule(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null."),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null."),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RecurrentGatedDeltaRuleCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRule)
    .Tiling(RecurrentGatedDeltaRuleTilingFunc)
    .TilingParse<RecurrentGatedDeltaRuleCompileInfo>(TilingPrepareForRecurrentGatedDeltaRule);
} // namespace optiling

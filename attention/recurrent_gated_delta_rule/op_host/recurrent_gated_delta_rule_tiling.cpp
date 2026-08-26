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
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
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
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null."),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::CheckOptionalInputContext(const size_t optionalIndex,
                                                                         const std::string &optionalName)
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
                OP_LOGE(context_->GetNodeName(), "query dtype, key dtype and value dtype should be bfloat16."),
                return ge::GRAPH_FAILED);

    auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
    auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
    OP_CHECK_IF(betaDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "beta dtype should be bfloat16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateDtype != ge::DT_BF16 && stateDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "state dtype should be bfloat16 or float32."),
                return ge::GRAPH_FAILED);
    stateDtype_ = stateDtype;

    auto actSeqlensDtype = context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetDataType();
    auto ssmStateIndicesDtype = context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
    OP_CHECK_IF(actSeqlensDtype != ge::DT_INT32 || ssmStateIndicesDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "actual_seq_lengths dtype and ssmStateIndices dtype should be int32."),
                return ge::GRAPH_FAILED);

    if (tilingData_.hasGama == 1) {
        auto gamaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
        OP_CHECK_IF(gamaDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "g dtype should be float32."),
                    return ge::GRAPH_FAILED);
    }

    if (tilingData_.hasGamaK == 1) {
        auto gamaKDtype = context_->GetOptionalInputDesc(GK_INDEX)->GetDataType();
        OP_CHECK_IF(gamaKDtype != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "gk dtype should be float32."),
                    return ge::GRAPH_FAILED);
    }

    if (context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX) != nullptr) {
        auto numAcceptedTokensDtype = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX)->GetDataType();
        OP_CHECK_IF(numAcceptedTokensDtype != ge::DT_INT32,
                    OP_LOGE(context_->GetNodeName(), "num_accepted_tokens dtype should be int32."),
                    return ge::GRAPH_FAILED);
    }
    auto outDtype = context_->GetOutputDesc(OUT_INDEX)->GetDataType();
    OP_CHECK_IF(outDtype != ge::DT_BF16, OP_LOGE(context_->GetNodeName(), "out dtype should be bfloat16."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查维度数量是否符合预期
bool RecurrentGatedDeltaRuleTiling::CheckDim(const gert::Shape shape, const size_t dim, const std::string &dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu.",
                dimDesc.c_str(), dim, shape.GetDimNum());
        return false;
    }
    return true;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapesParser()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetStorageShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetStorageShape();
    const auto &actSeqlensShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageShape();

    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
    tilingData_.b = actSeqlensShape.GetDim(DIM_0);

    // T>0
    OP_CHECK_IF(tilingData_.t <= 0,
                OP_LOGE(inputParams_.opName, "T should be greater than 0, but T is %u.", tilingData_.t),
                return ge::GRAPH_FAILED);
    // // B>=0
    OP_CHECK_IF((tilingData_.b <= 0),
                OP_LOGE(inputParams_.opName, "B should be greater than 0, but B is %u.", tilingData_.b),
                return ge::GRAPH_FAILED);
    // nk>0 nk<=256
    OP_CHECK_IF(tilingData_.nk <= 0 || tilingData_.nk > 256,
                OP_LOGE(inputParams_.opName, "Nk should be greater than 0 and less than or equal to 256, but Nk is %u.",
                        tilingData_.nk),
                return ge::GRAPH_FAILED);
    // nv>0 nv<=256
    OP_CHECK_IF(tilingData_.nv <= 0 || tilingData_.nv > 256,
                OP_LOGE(inputParams_.opName, "Nv should be greater than 0 and less than or equal to 256, but Nv is %u.",
                        tilingData_.nv),
                return ge::GRAPH_FAILED);
    // dk>0 dk<=512
    OP_CHECK_IF(tilingData_.dk <= 0 || tilingData_.dk > 512,
                OP_LOGE(inputParams_.opName, "Dk should be greater than 0 and less than or equal to 512, but Dk is %u.",
                        tilingData_.dk),
                return ge::GRAPH_FAILED);
    // dv>0 dv<=512
    OP_CHECK_IF(tilingData_.dv <= 0 || tilingData_.dv > 512,
                OP_LOGE(inputParams_.opName, "Dv should be greater than 0 and less than or equal to 512, but Dv is %u.",
                        tilingData_.dv),
                return ge::GRAPH_FAILED);
    // nv>=nk
    OP_CHECK_IF(tilingData_.nv % tilingData_.nk != 0,
                OP_LOGE(inputParams_.opName, "Nv should be an integer multiple of Nk, but Nv is %u, Nk is %u.",
                        tilingData_.nv, tilingData_.nk),
                return ge::GRAPH_FAILED);
    // blockNum >= T
    OP_CHECK_IF(tilingData_.sBlockNum < tilingData_.t,
                OP_LOGE(inputParams_.opName,
                        "BlockNum should be greater than or equal to T, Current values: BlockNum=%u, T=%u.",
                        tilingData_.sBlockNum, tilingData_.t),
                return ge::GRAPH_FAILED);

    uint64_t batchHeadTaskNum = static_cast<uint64_t>(tilingData_.b) * static_cast<uint64_t>(tilingData_.nv);
    if (batchHeadTaskNum > 0 && batchHeadTaskNum < tilingData_.vectorCoreNum) {
        tilingData_.vectorCoreNum = static_cast<uint32_t>(batchHeadTaskNum);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeEmptyTensor()
{
    OP_CHECK_IF(context_->GetInputShape(QUERY_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "query not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(KEY_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "key not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(VALUE_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "value not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(BETA_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "beta not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(STATE_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "state not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "actual_seq_lengths not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "ssm_state_indices not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(OUT_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "out not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingData_.hasGama == 1 && context_->GetOptionalInputShape(G_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "g not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingData_.hasGamaK == 1 && context_->GetOptionalInputShape(GK_INDEX)->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(inputParams_.opName, "gk not support empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData_.hasAcceptedTokens == 1 &&
                    context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX)->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(inputParams_.opName, "num_accepted_tokens not support empty tensor."), return ge::GRAPH_FAILED);
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
        const auto &gShape = context_->GetOptionalInputShape(G_INDEX)->GetStorageShape();
        if (!CheckDim(gShape, G_DIM_NUM, "g")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            gShape != expectGShape,
            OP_LOGE(context_->GetNodeName(), "The shape of g parameter[%ld, %ld] is not expected, Expect [%ld, %ld].",
                    gShape.GetDim(DIM_0), gShape.GetDim(DIM_1), expectGShape.GetDim(DIM_0), expectGShape.GetDim(DIM_1)),
            return ge::GRAPH_FAILED);
    }
    if (tilingData_.hasGamaK == 1) {
        const auto &gkShape = context_->GetOptionalInputShape(GK_INDEX)->GetStorageShape();
        if (!CheckDim(gkShape, GK_DIM_NUM, "gk")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(gkShape != expectGkShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of gk parameter[%ld, %ld, %ld] is not expected, Expect [%ld, %ld %ld].",
                            gkShape.GetDim(DIM_0), gkShape.GetDim(DIM_1), gkShape.GetDim(DIM_2),
                            expectGkShape.GetDim(DIM_0), expectGkShape.GetDim(DIM_1), expectGkShape.GetDim(DIM_2)),
                    return ge::GRAPH_FAILED);
    }
    if (tilingData_.hasAcceptedTokens == 1) {
        const auto &numAcceptedShape = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX)->GetStorageShape();
        if (!CheckDim(numAcceptedShape, NUM_ACCEPTED_TOKENS_DIM_NUM, "num_accepted_tokens")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(numAcceptedShape != expectNumAcceptedShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of num_accepted_tokens parameter[%ld] is not expected, Expect [%ld].",
                            numAcceptedShape.GetDim(DIM_0), expectNumAcceptedShape.GetDim(DIM_0)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetStorageShape();
    const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetStorageShape();
    const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetStorageShape();
    const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetStorageShape();
    const auto &actSeqlensShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetStorageShape();
    const auto &ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetStorageShape();
    const auto &outShape = context_->GetOutputShape(OUT_INDEX)->GetStorageShape();

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
                OP_LOGE(context_->GetNodeName(),
                        "The shape of query parameter[%ld, %ld, %ld] is not expected, Expect [%ld, %ld, %ld].",
                        queryShape.GetDim(DIM_0), queryShape.GetDim(DIM_1), queryShape.GetDim(DIM_2),
                        expectQKShape.GetDim(DIM_0), expectQKShape.GetDim(DIM_1), expectQKShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape != expectQKShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of key parameter[%ld, %ld, %ld] is not expected, Expect [%ld, %ld, %ld].",
                        keyShape.GetDim(DIM_0), keyShape.GetDim(DIM_1), keyShape.GetDim(DIM_2),
                        expectQKShape.GetDim(DIM_0), expectQKShape.GetDim(DIM_1), expectQKShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueShape != expectValueShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of value parameter[%ld, %ld, %ld] is not expected, Expect [%ld, %ld, %ld].",
                        valueShape.GetDim(DIM_0), valueShape.GetDim(DIM_1), valueShape.GetDim(DIM_2),
                        expectValueShape.GetDim(DIM_0), expectValueShape.GetDim(DIM_1), expectValueShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        stateShape != expectStateShape,
        OP_LOGE(context_->GetNodeName(),
                "The shape of state parameter[%ld, %ld, %ld, %ld] is not expected, Expect [%ld, %ld, %ld, %ld].",
                stateShape.GetDim(DIM_0), stateShape.GetDim(DIM_1), stateShape.GetDim(DIM_2), stateShape.GetDim(DIM_3),
                expectStateShape.GetDim(DIM_0), expectStateShape.GetDim(DIM_1), expectStateShape.GetDim(DIM_2),
                expectStateShape.GetDim(DIM_3)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        betaShape != expectBetaShape,
        OP_LOGE(context_->GetNodeName(), "The shape of beta parameter[%ld, %ld] is not expected, Expect [%ld, %ld].",
                betaShape.GetDim(DIM_0), betaShape.GetDim(DIM_1), expectBetaShape.GetDim(DIM_0),
                expectBetaShape.GetDim(DIM_1)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape != expectOutShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of out parameter[%ld, %ld, %ld] is not expected, Expect [%ld, %ld, %ld].",
                        outShape.GetDim(DIM_0), outShape.GetDim(DIM_1), outShape.GetDim(DIM_2),
                        expectOutShape.GetDim(DIM_0), expectOutShape.GetDim(DIM_1), expectOutShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(actSeqlensShape != expectActSeqlensShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of actual_seq_lengths parameter[%ld] is not expected, Expect [%ld].",
                        actSeqlensShape.GetDim(DIM_0), expectActSeqlensShape.GetDim(DIM_0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ssmStateShape != expectSsmStateShape,
        OP_LOGE(context_->GetNodeName(), "The shape of ssm_state_indices parameter[%ld] is not expected, Expect [%ld].",
                ssmStateShape.GetDim(DIM_0), expectSsmStateShape.GetDim(DIM_0)),
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
bool RecurrentGatedDeltaRuleTiling::CheckFormat(ge::Format format, const std::string &Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
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
        OP_CHECK_IF(gamaFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "g format not support NZ."),
                    return ge::GRAPH_FAILED);
    }
    if (tilingData_.hasGamaK == 1) {
        auto gamaKFormat = context_->GetOptionalInputDesc(GK_INDEX)->GetStorageFormat();
        OP_CHECK_IF(gamaKFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context_->GetNodeName(), "gk format not support NZ."),
                    return ge::GRAPH_FAILED);
    }
    if (context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX) != nullptr) {
        auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX)->GetStorageFormat();
        OP_CHECK_IF(numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context_->GetNodeName(), "num_accepted_tokens format not support NZ."),
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
                    OP_LOGE(inputParams_.opName,
                            "State only supports non-contiguous on dim 0 and dim 1, the innermost two dims must be "
                            "contiguous, but got stride[%ld, %ld, %ld, %ld], expected stride2=%lu, stride3=1.",
                            stride0, stride1, stride2, stride3, expectStride2),
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
        OP_LOGE(context_->GetNodeName(), "vStep should be bigger than 8, shape is too big.");
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

static ge::graphStatus RecurrentGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null."),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRecurrentGatedDeltaRule(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("RecurrentGatedDeltaRule", "context is null."),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
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

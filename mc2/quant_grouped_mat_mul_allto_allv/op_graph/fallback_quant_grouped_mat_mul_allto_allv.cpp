/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fallback_quant_grouped_mat_mul_allto_allv.cpp
 * \brief fallback function of op QuantGroupedMatMulAlltoAllv
 */
#include "mc2_common_log.h"
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include <cstring>
namespace fallback {
constexpr size_t INPUT_K_GMM_X = 0;
constexpr size_t INPUT_K_GMM_WEIGHT = 1;
constexpr size_t INPUT_K_SEND_COUNTS_TENSOR = 4;
constexpr size_t INPUT_K_RECV_COUNTS_TENSOR = 5;
constexpr size_t INPUT_K_MM_X = 6;
constexpr size_t INPUT_K_MM_WEIGHT = 7;
constexpr size_t INPUT_K_GMM_X_SCALE = 2;
constexpr size_t INPUT_K_GMM_WEIGHT_SCALE = 3;
constexpr size_t INPUT_K_MM_X_SCALE = 8;
constexpr size_t INPUT_K_MM_WEIGHT_SCALE = 9;
constexpr size_t INPUT_K_COMM_QUANT_SCALE = 10;

constexpr size_t OUTPUT_K_Y = 0;
constexpr size_t OUTPUT_K_MM_Y = 1;

constexpr size_t ATTR_K_GROUP = 0;
constexpr size_t ATTR_K_EP_WORLD_SIZE = 1;
constexpr size_t ATTR_K_SEND_COUNTS = 2;
constexpr size_t ATTR_K_RECV_COUNTS = 3;
constexpr size_t ATTR_K_GMM_X_QUANT_MODE = 4;
constexpr size_t ATTR_K_GMM_WEIGHT_QUANT_MODE = 5;
constexpr size_t ATTR_K_TRANS_GMM_WEIGHT = 6;
constexpr size_t ATTR_K_TRANS_MM_WEIGHT = 7;
constexpr size_t ATTR_K_MM_X_QUANT_MODE = 8;
constexpr size_t ATTR_K_MM_WEIGHT_QUANT_MODE = 9;
constexpr size_t ATTR_K_COMM_QUANT_MODE = 10;
constexpr size_t ATTR_K_GROUP_SIZE = 11;
constexpr size_t ATTR_K_Y_DTYPE = 12;
constexpr size_t ATTR_K_MM_DTYPE = 13;
constexpr size_t ATTR_K_COMM_QUANT_DTYPE = 14;
constexpr size_t ATTR_K_COMM_MODE = 15;

struct QuantGroupedMatMulAlltoAllvInputs {
    const gert::Tensor *gmmX;
    const gert::Tensor *gmmWeight;
    const gert::Tensor *gmmXScale;
    const gert::Tensor *gmmWeightScale;
    const gert::Tensor *sendCountsTensor;
    const gert::Tensor *recvCountsTensor;
    const gert::Tensor *mmX;
    const gert::Tensor *mmWeight;
    const gert::Tensor *mmXScale;
    const gert::Tensor *mmWeightScale;
    const gert::Tensor *commQuantScale;
};

struct QuantGroupedMatMulAlltoAllvAttrs {
    const char *group;
    const int64_t *epWorldSize;
    const gert::TypedContinuousVector<int64_t> *sendCounts;
    const gert::TypedContinuousVector<int64_t> *recvCounts;
    const int64_t *gmmXQuantMode;
    const int64_t *gmmWeightQuantMode;
    const bool *transGmmWeight;
    const bool *transMmWeight;
    const int64_t *mmXQuantMode;
    const int64_t *mmWeightQuantMode;
    const int64_t *commQuantMode;
    const int64_t *groupSize;
    const int64_t *yDtype;
    const int64_t *mmDtype;
    const int64_t *commQuantDtype;
    const char *commMode;
};

static ge::graphStatus GetInputs(gert::OpExecuteContext *host_api_ctx, QuantGroupedMatMulAlltoAllvInputs &inputs)
{
    inputs.gmmX = host_api_ctx->GetInputTensor(INPUT_K_GMM_X);
    inputs.gmmWeight = host_api_ctx->GetInputTensor(INPUT_K_GMM_WEIGHT);
    inputs.gmmXScale = host_api_ctx->GetOptionalInputTensor(INPUT_K_GMM_X_SCALE);
    inputs.gmmWeightScale = host_api_ctx->GetOptionalInputTensor(INPUT_K_GMM_WEIGHT_SCALE);
    inputs.sendCountsTensor = host_api_ctx->GetOptionalInputTensor(INPUT_K_SEND_COUNTS_TENSOR);
    inputs.recvCountsTensor = host_api_ctx->GetOptionalInputTensor(INPUT_K_RECV_COUNTS_TENSOR);
    inputs.mmX = host_api_ctx->GetOptionalInputTensor(INPUT_K_MM_X);
    inputs.mmWeight = host_api_ctx->GetOptionalInputTensor(INPUT_K_MM_WEIGHT);
    inputs.mmXScale = host_api_ctx->GetOptionalInputTensor(INPUT_K_MM_X_SCALE);
    inputs.mmWeightScale = host_api_ctx->GetOptionalInputTensor(INPUT_K_MM_WEIGHT_SCALE);
    inputs.commQuantScale = host_api_ctx->GetOptionalInputTensor(INPUT_K_COMM_QUANT_SCALE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrs(const gert::RuntimeAttrs *attrs, QuantGroupedMatMulAlltoAllvAttrs &attrsData)
{
    attrsData.group = attrs->GetStr(ATTR_K_GROUP);
    attrsData.epWorldSize = attrs->GetInt(ATTR_K_EP_WORLD_SIZE);
    attrsData.sendCounts = attrs->GetListInt(ATTR_K_SEND_COUNTS);
    attrsData.recvCounts = attrs->GetListInt(ATTR_K_RECV_COUNTS);
    attrsData.gmmXQuantMode = attrs->GetInt(ATTR_K_GMM_X_QUANT_MODE);
    attrsData.gmmWeightQuantMode = attrs->GetInt(ATTR_K_GMM_WEIGHT_QUANT_MODE);
    attrsData.transGmmWeight = attrs->GetBool(ATTR_K_TRANS_GMM_WEIGHT);
    attrsData.transMmWeight = attrs->GetBool(ATTR_K_TRANS_MM_WEIGHT);
    attrsData.mmXQuantMode = attrs->GetInt(ATTR_K_MM_X_QUANT_MODE);
    attrsData.mmWeightQuantMode = attrs->GetInt(ATTR_K_MM_WEIGHT_QUANT_MODE);
    attrsData.commQuantMode = attrs->GetInt(ATTR_K_COMM_QUANT_MODE);
    attrsData.groupSize = attrs->GetInt(ATTR_K_GROUP_SIZE);
    attrsData.yDtype = attrs->GetInt(ATTR_K_Y_DTYPE);
    attrsData.mmDtype = attrs->GetInt(ATTR_K_MM_DTYPE);
    attrsData.commQuantDtype = attrs->GetInt(ATTR_K_COMM_QUANT_DTYPE);
    attrsData.commMode = attrs->GetStr(ATTR_K_COMM_MODE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputsAndAttrs(const gert::Tensor *gmmX, const gert::Tensor *gmmWeight, const char *group,
                                           const bool *transGmmWeight, const int64_t *epWorldSize, const char *commMode)
{
    OPS_ERR_IF(gmmX == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "gmmX"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmmWeight == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "gmmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(group == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "group"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(epWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "epWorldSize"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(transGmmWeight == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "transGmmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(commMode == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "commMode"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQuantAttrs(const QuantGroupedMatMulAlltoAllvAttrs &attrsData)
{
    OPS_ERR_IF(attrsData.transMmWeight == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "transMmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.gmmXQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "gmmXQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.gmmWeightQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "gmmWeightQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.mmXQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "mmXQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.mmWeightQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "mmWeightQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.commQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "commQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.groupSize == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "groupSize"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.yDtype == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "yDtype"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.mmDtype == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "mmDtype"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.commQuantDtype == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "commQuantDtype"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSendRecvCounts(const gert::TypedContinuousVector<int64_t> *sendCounts,
                                           const gert::TypedContinuousVector<int64_t> *recvCounts,
                                           std::vector<int64_t> &actSendCountsSeqArray,
                                           std::vector<int64_t> &actRecvCountsSeqArray)
{
    OPS_ERR_IF(sendCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "sendCounts"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(recvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "recvCounts"),
               return ge::GRAPH_FAILED);

    const size_t sendLen = static_cast<size_t>(sendCounts->GetSize());
    const int64_t *actSendSeqData = sendCounts->GetData();
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }

    const size_t recvLen = static_cast<size_t>(recvCounts->GetSize());
    const int64_t *actRecvSeqData = recvCounts->GetData();
    for (size_t i = 0UL; i < recvLen; i++) {
        actRecvCountsSeqArray.push_back(actRecvSeqData[i]);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensors(const gert::Tensor *y, const gert::Tensor *&mmY)
{
    OPS_ERR_IF(y == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "y"),
               return ge::GRAPH_FAILED);
    if ((mmY != nullptr) && (mmY->GetStorageShape().GetDimNum() == 0)) {
        mmY = nullptr;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ExecuteQuantGroupedMatMulAlltoAllv(gert::OpExecuteContext *host_api_ctx,
                                                          const QuantGroupedMatMulAlltoAllvInputs &inputs,
                                                          const QuantGroupedMatMulAlltoAllvAttrs &attrsData,
                                                          const std::vector<int64_t> &actSendCountsSeqArray,
                                                          const std::vector<int64_t> &actRecvCountsSeqArray,
                                                          const gert::Tensor *y, const gert::Tensor *mmY)
{
    OPS_LOG_D("QuantGroupedMatMulAlltoAllvFallback", "aclnnQuantGroupedMatMulAlltoAllvV2.");
    const auto apiRet = EXEC_OPAPI_CMD(
        aclnnQuantGroupedMatMulAlltoAllvV2, inputs.gmmX, inputs.gmmWeight, inputs.gmmXScale, inputs.gmmWeightScale,
        inputs.sendCountsTensor, inputs.recvCountsTensor, inputs.mmX, inputs.mmWeight, inputs.mmXScale,
        inputs.mmWeightScale, inputs.commQuantScale, *attrsData.gmmXQuantMode, *attrsData.gmmWeightQuantMode,
        *attrsData.mmXQuantMode, *attrsData.mmWeightQuantMode, *attrsData.commQuantMode, *attrsData.commQuantDtype,
        *attrsData.groupSize, attrsData.group, attrsData.commMode, *attrsData.epWorldSize, actSendCountsSeqArray,
        actRecvCountsSeqArray, *attrsData.transGmmWeight, *attrsData.transMmWeight, y, mmY);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS,
               OP_LOGE("QuantGroupedMatMulAlltoAllvFallback", "Aclnn api error code %u", apiRet),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantGroupedMatMulAlltoAllvExecuteFunc(gert::OpExecuteContext *host_api_ctx)
{
    OPS_LOG_D("QuantGroupedMatMulAlltoAllvFallback", "Start QuantGroupedMatMulAlltoAllvFallback.");
    OPS_ERR_IF(host_api_ctx == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "host_api_ctx"),
               return ge::GRAPH_FAILED);

    QuantGroupedMatMulAlltoAllvInputs inputs;
    ge::graphStatus ret = GetInputs(host_api_ctx, inputs);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::RuntimeAttrs *attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantGroupedMatMulAlltoAllvFallback", "attrs"),
               return ge::GRAPH_FAILED);

    QuantGroupedMatMulAlltoAllvAttrs attrsData;
    ret = GetAttrs(attrs, attrsData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputsAndAttrs(inputs.gmmX, inputs.gmmWeight, attrsData.group, attrsData.transGmmWeight,
                              attrsData.epWorldSize, attrsData.commMode);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckQuantAttrs(attrsData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    std::vector<int64_t> actRecvCountsSeqArray;
    std::vector<int64_t> actSendCountsSeqArray;
    ret = ParseSendRecvCounts(attrsData.sendCounts, attrsData.recvCounts, actSendCountsSeqArray, actRecvCountsSeqArray);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto y = host_api_ctx->GetOutputTensor(OUTPUT_K_Y);
    auto mmY = host_api_ctx->GetOutputTensor(OUTPUT_K_MM_Y);
    ret = CheckOutputTensors(y, mmY);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ExecuteQuantGroupedMatMulAlltoAllv(host_api_ctx, inputs, attrsData, actSendCountsSeqArray,
                                              actRecvCountsSeqArray, y, mmY);
}

IMPL_OP(QuantGroupedMatMulAlltoAllv).OpExecuteFunc(QuantGroupedMatMulAlltoAllvExecuteFunc);
} // namespace fallback

/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file fallback_allto_allv_quant_grouped_mat_mul.cpp
 * \brief fallback function of op AlltoAllvQuantGroupedMatMul
 */
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"
#include <cstring>

namespace fallback {
constexpr size_t ATTR_K_GROUP = 0;
constexpr size_t ATTR_K_EP_WORLD_SIZE = 1;
constexpr size_t ATTR_K_SEND_COUNTS = 2;
constexpr size_t ATTR_K_RECV_COUNTS = 3;
constexpr size_t ATTR_K_GMM_X_QUANT_MODE = 4;
constexpr size_t ATTR_K_GMM_WEIGHT_QUANT_MODE = 5;
constexpr size_t ATTR_K_TRANS_GMM_WEIGHT = 6;
constexpr size_t ATTR_K_TRANS_MM_WEIGHT = 7;
constexpr size_t ATTR_K_PERMUTE_OUT_FLAG = 8;
constexpr size_t ATTR_K_MM_X_QUANT_MODE = 9;
constexpr size_t ATTR_K_MM_WEIGHT_QUANT_MODE = 10;
constexpr size_t ATTR_K_GROUP_SIZE = 11;
constexpr size_t ATTR_K_Y_DTYPE = 12;
constexpr size_t ATTR_K_MM_DTYPE = 13;
constexpr size_t ATTR_COMM_MODE = 14;

constexpr size_t INPUT_K_GMM_X = 0;
constexpr size_t INPUT_K_GMM_WEIGHT = 1;
constexpr size_t INPUT_K_GMM_X_SCALE = 2;
constexpr size_t INPUT_K_GMM_WEIGHT_SCALE = 3;
constexpr size_t INPUT_K_SEND_COUNTS_TENSOR = 4;
constexpr size_t INPUT_K_RECV_COUNTS_TENSOR = 5;
constexpr size_t INPUT_K_MM_X = 6;
constexpr size_t INPUT_K_MM_WEIGHT = 7;
constexpr size_t INPUT_K_MM_X_SCALE = 8;
constexpr size_t INPUT_K_MM_WEIGHT_SCALE = 9;

constexpr size_t OUTPUT_K_GMM_Y = 0;
constexpr size_t OUTPUT_K_MM_Y = 1;
constexpr size_t OUTPUT_K_PERMUTE_OUT = 2;

struct AlltoAllvQuantGroupedMatMulInputs {
    const gert::Tensor* gmmX;
    const gert::Tensor* gmmWeight;
    const gert::Tensor* gmmXScale;
    const gert::Tensor* gmmWeightScale;
    const gert::Tensor* sendCountsTensor;
    const gert::Tensor* recvCountsTensor;
    const gert::Tensor* mmX;
    const gert::Tensor* mmWeight;
    const gert::Tensor* mmXScale;
    const gert::Tensor* mmWeightScale;
};

struct AlltoAllvQuantGroupedMatMulAttrs {
    const char* group;
    const int64_t* epWorldSize;
    const gert::TypedContinuousVector<int64_t>* sendCounts;
    const gert::TypedContinuousVector<int64_t>* recvCounts;
    const int64_t* gmmXQuantMode;
    const int64_t* gmmWeightQuantMode;
    const bool* transGmmWeight;
    const bool* transMmWeight;
    const bool* permuteOutFlag;
    const int64_t* mmXQuantMode;
    const int64_t* mmWeightQuantMode;
    const int64_t* groupSize;
    const int64_t* yDtype;
    const int64_t* mmDtype;
    const char* commMode;
};

static ge::graphStatus GetInputs(gert::OpExecuteContext* host_api_ctx, AlltoAllvQuantGroupedMatMulInputs& inputs)
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
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrs(const gert::RuntimeAttrs* attrs, AlltoAllvQuantGroupedMatMulAttrs& attrsData)
{
    attrsData.group = attrs->GetStr(ATTR_K_GROUP);
    attrsData.epWorldSize = attrs->GetInt(ATTR_K_EP_WORLD_SIZE);
    attrsData.sendCounts = attrs->GetListInt(ATTR_K_SEND_COUNTS);
    attrsData.recvCounts = attrs->GetListInt(ATTR_K_RECV_COUNTS);
    attrsData.gmmXQuantMode = attrs->GetInt(ATTR_K_GMM_X_QUANT_MODE);
    attrsData.gmmWeightQuantMode = attrs->GetInt(ATTR_K_GMM_WEIGHT_QUANT_MODE);
    attrsData.transGmmWeight = attrs->GetBool(ATTR_K_TRANS_GMM_WEIGHT);
    attrsData.transMmWeight = attrs->GetBool(ATTR_K_TRANS_MM_WEIGHT);
    attrsData.permuteOutFlag = attrs->GetBool(ATTR_K_PERMUTE_OUT_FLAG);
    attrsData.mmXQuantMode = attrs->GetInt(ATTR_K_MM_X_QUANT_MODE);
    attrsData.mmWeightQuantMode = attrs->GetInt(ATTR_K_MM_WEIGHT_QUANT_MODE);
    attrsData.groupSize = attrs->GetInt(ATTR_K_GROUP_SIZE);
    attrsData.yDtype = attrs->GetInt(ATTR_K_Y_DTYPE);
    attrsData.mmDtype = attrs->GetInt(ATTR_K_MM_DTYPE);
    attrsData.commMode = attrs->GetStr(ATTR_COMM_MODE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputsAndAttrs(const gert::Tensor* gmmX, const gert::Tensor* gmmWeight, const char* group,
                                           const int64_t* epWorldSize, const bool* transGmmWeight, const char* commMode)
{
    OPS_ERR_IF(gmmX == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "gmmX"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmmWeight == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "gmmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(group == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "group"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(epWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "epWorldSize"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(transGmmWeight == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "transGmmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(commMode == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "commMode is null"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQuantAttrs(const AlltoAllvQuantGroupedMatMulAttrs& attrsData)
{
    OPS_ERR_IF(attrsData.transMmWeight == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "transMmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.permuteOutFlag == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "permuteOutFlag"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.gmmXQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "gmmXQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.gmmWeightQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "gmmWeightQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.mmXQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "mmXQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.mmWeightQuantMode == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "mmWeightQuantMode"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.groupSize == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "groupSize"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.yDtype == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "yDtype"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(attrsData.mmDtype == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "mmDtype"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSendRecvCounts(const gert::TypedContinuousVector<int64_t>* sendCounts,
                                           const gert::TypedContinuousVector<int64_t>* recvCounts,
                                           std::vector<int64_t>& actSendCountsSeqArray,
                                           std::vector<int64_t>& actRecvCountsSeqArray)
{
    OPS_ERR_IF(sendCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "sendCounts"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(recvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "recvCounts"),
               return ge::GRAPH_FAILED);

    const int64_t* actSend = sendCounts->GetData();
    const size_t sendLen = sendCounts->GetSize();
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSend[i]);
    }

    const int64_t* actRecv = recvCounts->GetData();
    const size_t recvLen = recvCounts->GetSize();
    for (size_t i = 0UL; i < recvLen; i++) {
        actRecvCountsSeqArray.push_back(actRecv[i]);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensors(const gert::Tensor* gmmY, const gert::Tensor*& mmY,
                                          const gert::Tensor*& permuteOut)
{
    OPS_ERR_IF(gmmY == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvQuantGroupedMatMulFallback", "gmmY"),
               return ge::GRAPH_FAILED);
    if ((permuteOut != nullptr) && (permuteOut->GetStorageShape().GetDimNum() == 0)) {
        permuteOut = nullptr;
    }
    if ((mmY != nullptr) && (mmY->GetStorageShape().GetDimNum() == 0)) {
        mmY = nullptr;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ExecuteAlltoAllvQuantGroupedMatMul(gert::OpExecuteContext* host_api_ctx,
                                                          const AlltoAllvQuantGroupedMatMulInputs& inputs,
                                                          const AlltoAllvQuantGroupedMatMulAttrs& attrsData,
                                                          const std::vector<int64_t>& actSendCountsSeqArray,
                                                          const std::vector<int64_t>& actRecvCountsSeqArray,
                                                          const gert::Tensor* gmmY, const gert::Tensor* mmY,
                                                          const gert::Tensor* permuteOut)
{
    OPS_LOG_D("AlltoAllvQuantGroupedMatMulFallback", "aclnnAlltoAllvQuantGroupedMatMulV2.");
    const auto apiRet = EXEC_OPAPI_CMD(
        aclnnAlltoAllvQuantGroupedMatMulV2, inputs.gmmX, inputs.gmmWeight, inputs.gmmXScale, inputs.gmmWeightScale,
        inputs.sendCountsTensor, inputs.recvCountsTensor, inputs.mmX, inputs.mmWeight, inputs.mmXScale,
        inputs.mmWeightScale, *attrsData.gmmXQuantMode, *attrsData.gmmWeightQuantMode, *attrsData.mmXQuantMode,
        *attrsData.mmWeightQuantMode, attrsData.group, attrsData.commMode, *attrsData.epWorldSize,
        actSendCountsSeqArray, actRecvCountsSeqArray, *attrsData.transGmmWeight, *attrsData.transMmWeight,
        *attrsData.groupSize, *attrsData.permuteOutFlag, gmmY, mmY, permuteOut);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS,
               OP_LOGE("AlltoAllvGroupedMatMulFallback", "Aclnn api error code %u", apiRet), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AlltoAllvQuantGroupedMatMulExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D("AlltoAllvQuantGroupedMatMulFallback", "Start AlltoAllvQuantGroupedMatMulFallback.");
    OPS_ERR_IF(host_api_ctx == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "host_api_ctx is null"),
               return ge::GRAPH_FAILED);

    AlltoAllvQuantGroupedMatMulInputs inputs;
    ge::graphStatus ret = GetInputs(host_api_ctx, inputs);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::RuntimeAttrs* attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE("AlltoAllvQuantGroupedMatMulFallback", "attrs is null"),
               return ge::GRAPH_FAILED);

    AlltoAllvQuantGroupedMatMulAttrs attrsData;
    ret = GetAttrs(attrs, attrsData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputsAndAttrs(inputs.gmmX, inputs.gmmWeight, attrsData.group, attrsData.epWorldSize,
                              attrsData.transGmmWeight, attrsData.commMode);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckQuantAttrs(attrsData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    std::vector<int64_t> actSendCountsSeqArray;
    std::vector<int64_t> actRecvCountsSeqArray;
    ret = ParseSendRecvCounts(attrsData.sendCounts, attrsData.recvCounts, actSendCountsSeqArray, actRecvCountsSeqArray);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto gmmY = host_api_ctx->GetOutputTensor(OUTPUT_K_GMM_Y);
    auto mmY = host_api_ctx->GetOutputTensor(OUTPUT_K_MM_Y);
    auto permuteOut = host_api_ctx->GetOutputTensor(OUTPUT_K_PERMUTE_OUT);
    ret = CheckOutputTensors(gmmY, mmY, permuteOut);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ExecuteAlltoAllvQuantGroupedMatMul(host_api_ctx, inputs, attrsData, actSendCountsSeqArray,
                                              actRecvCountsSeqArray, gmmY, mmY, permuteOut);
}

IMPL_OP(AlltoAllvQuantGroupedMatMul).OpExecuteFunc(AlltoAllvQuantGroupedMatMulExecuteFunc);
} // namespace fallback

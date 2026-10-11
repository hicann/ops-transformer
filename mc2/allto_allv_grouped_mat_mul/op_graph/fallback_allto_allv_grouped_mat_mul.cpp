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
 * \file fallback_allto_allv_grouped_mat_mul.cpp
 * \brief fallback function of op AlltoAllvGroupedMatMul
 */
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"
#include <cstring>
namespace fallback {
constexpr size_t K_COMM_MODE = 7;
// 输入参数和属性的校验
static ge::graphStatus CheckInputsAndAttrs(const gert::Tensor* gmmX, const gert::Tensor* gmmWeight, const char* group,
                                           const int64_t* epWorldSize, const bool* transGmmWeight, const char* commMode)
{
    OPS_ERR_IF(gmmX == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "gmmX"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmmWeight == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "gmmWeight"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(group == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "group"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(epWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "epWorldSize"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(transGmmWeight == nullptr,
               OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "transGmmWeight"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(commMode == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "commMode"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 解析 sendCounts 和 recvCounts
static ge::graphStatus ParseSendRecvCounts(const gert::TypedContinuousVector<int64_t>* sendCounts,
                                           const gert::TypedContinuousVector<int64_t>* recvCounts,
                                           std::vector<int64_t>& actSendCountsSeqArray,
                                           std::vector<int64_t>& actRecvCountsSeqArray)
{
    OPS_ERR_IF(sendCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "sendCounts"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(recvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "recvCounts"),
               return ge::GRAPH_FAILED);

    const int64_t* actSendSeqData = sendCounts->GetData();
    const size_t sendLen = static_cast<size_t>(sendCounts->GetSize());
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }

    const int64_t* actRecvSeqData = recvCounts->GetData();
    const size_t recvLen = static_cast<size_t>(recvCounts->GetSize());
    for (size_t i = 0UL; i < recvLen; i++) {
        actRecvCountsSeqArray.push_back(actRecvSeqData[i]);
    }

    return ge::GRAPH_SUCCESS;
}

// 校验输出参数
static ge::graphStatus CheckOutputTensors(const gert::Tensor* gmmY, const gert::Tensor*& mmY,
                                          const gert::Tensor*& permuteOut)
{
    OPS_ERR_IF(gmmY == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "gmmY"),
               return ge::GRAPH_FAILED);
    if ((permuteOut != nullptr) && (permuteOut->GetStorageShape().GetDimNum() == 0)) {
        permuteOut = nullptr;
    }
    if ((mmY != nullptr) && (mmY->GetStorageShape().GetDimNum() == 0)) {
        mmY = nullptr;
    }

    return ge::GRAPH_SUCCESS;
}

// 主执行函数
static ge::graphStatus AlltoAllvGroupedMatMulExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D("AlltoAllvGroupedMatMulFallback", "Start AlltoAllvGroupedMatMulFallback.");
    OPS_ERR_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "host_api_ctx"),
               return ge::GRAPH_FAILED);

    // 获取输入参数
    const gert::Tensor* gmmX =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_GMM_X));
    const gert::Tensor* gmmWeight =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_GMM_WEIGHT));
    const gert::Tensor* sendCountsTensor = host_api_ctx->GetOptionalInputTensor(
        static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_SEND_COUNTS_TENSOR));
    const gert::Tensor* recvCountsTensor = host_api_ctx->GetOptionalInputTensor(
        static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_RECV_COUNTS_TENSOR));
    const gert::Tensor* mmX =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_MM_X));
    const gert::Tensor* mmWeight =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_MM_WEIGHT));

    // 获取属性
    const gert::RuntimeAttrs* attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT("AlltoAllvGroupedMatMulFallback", "attrs"),
               return ge::GRAPH_FAILED);
    const char* group = attrs->GetStr(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_GROUP));
    const int64_t* epWorldSize =
        attrs->GetInt(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_EP_WORLD_SIZE));
    const gert::TypedContinuousVector<int64_t>* sendCounts =
        attrs->GetListInt(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_SEND_COUNTS));
    const gert::TypedContinuousVector<int64_t>* recvCounts =
        attrs->GetListInt(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_RECV_COUNTS));
    const bool* transGmmWeight =
        attrs->GetBool(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_TRANS_GMM_WEIGHT));
    const bool* transMmWeight =
        attrs->GetBool(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_TRANS_MM_WEIGHT));
    const bool* permuteOutFlag =
        attrs->GetBool(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_PERMUTE_OUT_FLAG));
    const char* commMode = attrs->GetStr(static_cast<size_t>(K_COMM_MODE));

    // 输入参数和属性的校验
    ge::graphStatus ret = CheckInputsAndAttrs(gmmX, gmmWeight, group, epWorldSize, transGmmWeight, commMode);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 解析 sendCounts 和 recvCounts
    std::vector<int64_t> actSendCountsSeqArray;
    std::vector<int64_t> actRecvCountsSeqArray;
    ret = ParseSendRecvCounts(sendCounts, recvCounts, actSendCountsSeqArray, actRecvCountsSeqArray);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 获取并校验输出参数
    auto gmmY = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulOutputIdx::K_GMM_Y));
    auto mmY = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulOutputIdx::K_MM_Y));
    auto permuteOut =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulOutputIdx::K_PERMUTE_OUT));
    ret = CheckOutputTensors(gmmY, mmY, permuteOut);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    const auto apiRet =
        EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMulV2, gmmX, gmmWeight, sendCountsTensor, recvCountsTensor, mmX,
                       mmWeight, group, commMode, *epWorldSize, actSendCountsSeqArray, actRecvCountsSeqArray,
                       *transGmmWeight, *transMmWeight, *permuteOutFlag, gmmY, mmY, permuteOut);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS,
               OP_LOGE("AlltoAllvGroupedMatMulFallback", "Aclnn api error code %u", apiRet), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP(AlltoAllvGroupedMatMul).OpExecuteFunc(AlltoAllvGroupedMatMulExecuteFunc);
} // namespace fallback

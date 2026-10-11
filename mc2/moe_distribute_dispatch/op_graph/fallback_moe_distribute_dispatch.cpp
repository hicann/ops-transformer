/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

const char* moeDistributeDispatchInfo = "MoeDistributeDispatchFallback";

namespace {
struct OpInput {
    const gert::Tensor* x;
    const gert::Tensor* expertIds;
};

struct OpOptionalInput {
    const gert::Tensor* scales;
    const gert::Tensor* xActiveMask;
    const gert::Tensor* expertScales;
};

struct OpOutput {
    const gert::Tensor* expandX;
    const gert::Tensor* dynamicScales;
    const gert::Tensor* expandIdx;
    const gert::Tensor* expertTokenNums;
    const gert::Tensor* epRecvCounts;
    const gert::Tensor* tpRecvCounts;
    const gert::Tensor* expandScales;
};

struct OpAttrs {
    const char* groupEp;
    const int64_t* epWorldSize;
    const int64_t* epRankId;
    const int64_t* moeExpertNum;
    const char* groupTp;
    const int64_t* tpWorldSize;
    const int64_t* tpRankId;
    const int64_t* expertShardType;
    const int64_t* sharedExpertNum;
    const int64_t* sharedExpertRankNum;
    const int64_t* quantMode;
    const int64_t* globalBs;
    const int64_t* expertTokenNumsType;
};

// 获取必选输入
graphStatus MoeDistributeDispatchGetOpInput(OpExecuteContext* host_api_ctx, OpInput& opInput)
{
    opInput.x = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeDispatchInputIdx::K_X));
    OP_CHECK_IF(opInput.x == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "x"),
                return ge::GRAPH_FAILED);

    opInput.expertIds =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeDispatchInputIdx::K_EXPERT_IDS));
    OP_CHECK_IF(opInput.expertIds == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expertIds"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取可选输入
void MoeDistributeDispatchGetOpOptionalInput(OpExecuteContext* host_api_ctx, OpOptionalInput& opOptionalInput)
{
    opOptionalInput.scales =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeDispatchInputIdx::K_SCALES));
    opOptionalInput.xActiveMask =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeDispatchInputIdx::K_X_ACTIVE_MASK));
    opOptionalInput.expertScales =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeDispatchInputIdx::K_EXPERT_SCALES));
}

// 获取输出
graphStatus MoeDistributeDispatchGetOpOutput(OpExecuteContext* host_api_ctx, OpOutput& opOutput)
{
    opOutput.expandX =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_EXPAND_X));
    OP_CHECK_IF(opOutput.expandX == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expandX"),
                return ge::GRAPH_FAILED);

    opOutput.dynamicScales =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_DYNAMIC_SCALES));
    OP_CHECK_IF(opOutput.dynamicScales == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "dynamicScales"), return ge::GRAPH_FAILED);

    opOutput.expandIdx =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_EXPAND_IDX));
    OP_CHECK_IF(opOutput.expandIdx == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expandIdx"),
                return ge::GRAPH_FAILED);

    opOutput.expertTokenNums =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_EXPERT_TOKEN_NUMS));
    OP_CHECK_IF(opOutput.expertTokenNums == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expertTokenNums"), return ge::GRAPH_FAILED);

    opOutput.epRecvCounts =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_EP_RECV_COUNTS));
    OP_CHECK_IF(opOutput.epRecvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "epRecvCounts"),
                return ge::GRAPH_FAILED);

    opOutput.tpRecvCounts =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_TP_RECV_COUNTS));
    OP_CHECK_IF(opOutput.tpRecvCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "tpRecvCounts"),
                return ge::GRAPH_FAILED);

    opOutput.expandScales =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeDispatchOutputIdx::K_EXPAND_SCALES));
    OP_CHECK_IF(opOutput.expandScales == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expandScales"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性
graphStatus MoeDistributeDispatchGetOpAttrs(OpExecuteContext* host_api_ctx, OpAttrs& opAttrs)
{
    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "attrs"),
                return ge::GRAPH_FAILED);

    opAttrs.groupEp = attrs->GetStr(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_GROUP_EP));
    OP_CHECK_IF(opAttrs.groupEp == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "groupEp"),
                return ge::GRAPH_FAILED);

    opAttrs.epWorldSize = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_EP_WORLD_SIZE));
    OP_CHECK_IF(opAttrs.epWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "epWorldSize"),
                return ge::GRAPH_FAILED);

    opAttrs.epRankId = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_EP_RANK_ID));
    OP_CHECK_IF(opAttrs.epRankId == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "epRankId"),
                return ge::GRAPH_FAILED);

    opAttrs.moeExpertNum = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_MOE_EXPERT_NUM));
    OP_CHECK_IF(opAttrs.moeExpertNum == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "moeExpertNum"),
                return ge::GRAPH_FAILED);

    opAttrs.groupTp = attrs->GetStr(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_GROUP_TP));
    OP_CHECK_IF(opAttrs.groupTp == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "groupTp"),
                return ge::GRAPH_FAILED);

    opAttrs.tpWorldSize = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_TP_WORLD_SIZE));
    OP_CHECK_IF(opAttrs.tpWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "tpWorldSize"),
                return ge::GRAPH_FAILED);

    opAttrs.tpRankId = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_TP_RANK_ID));
    OP_CHECK_IF(opAttrs.tpRankId == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "tpRankId"),
                return ge::GRAPH_FAILED);

    opAttrs.expertShardType =
        attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_EXPERT_SHARD_TYPE));
    OP_CHECK_IF(opAttrs.expertShardType == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expertShardType"), return ge::GRAPH_FAILED);

    opAttrs.sharedExpertNum =
        attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_SHARED_EXPERT_NUM));
    OP_CHECK_IF(opAttrs.sharedExpertNum == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "sharedExpertNum"), return ge::GRAPH_FAILED);

    opAttrs.sharedExpertRankNum =
        attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_SHARED_EXPERT_RANK_NUM));
    OP_CHECK_IF(opAttrs.sharedExpertRankNum == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "sharedExpertRankNum"), return ge::GRAPH_FAILED);

    opAttrs.quantMode = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_QUANT_MODE));
    OP_CHECK_IF(opAttrs.quantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "quantMode"),
                return ge::GRAPH_FAILED);

    opAttrs.globalBs = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_GLOBAL_BS));
    OP_CHECK_IF(opAttrs.globalBs == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "globalBs"),
                return ge::GRAPH_FAILED);

    opAttrs.expertTokenNumsType =
        attrs->GetInt(static_cast<size_t>(ops::MoeDistributeDispatchAttrIdx::K_EXPERT_TOKEN_NUMS_TYPE));
    OP_CHECK_IF(opAttrs.expertTokenNumsType == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "expertTokenNumsType"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

static graphStatus MoeDistributeDispatchExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_LOGD(moeDistributeDispatchInfo, "start to fallback for moeDistributeDispatch");
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeDispatchInfo, "host_api_ctx"),
                return ge::GRAPH_FAILED);

    OpInput opInput;
    OpOptionalInput opOptionalInput;
    OpOutput opOutput;
    OpAttrs opAttrs;

    OP_CHECK_IF(MoeDistributeDispatchGetOpInput(host_api_ctx, opInput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeDispatchInfo, "get input failed"), return ge::GRAPH_FAILED);
    MoeDistributeDispatchGetOpOptionalInput(host_api_ctx, opOptionalInput);
    OP_CHECK_IF(MoeDistributeDispatchGetOpOutput(host_api_ctx, opOutput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeDispatchInfo, "get output failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeDispatchGetOpAttrs(host_api_ctx, opAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeDispatchInfo, "get attrs failed"), return ge::GRAPH_FAILED);

    const auto apiRet = EXEC_OPAPI_CMD(
        aclnnMoeDistributeDispatch, opInput.x, opInput.expertIds, opOptionalInput.scales, opOptionalInput.xActiveMask,
        opOptionalInput.expertScales, opAttrs.groupEp, *opAttrs.epWorldSize, *opAttrs.epRankId, *opAttrs.moeExpertNum,
        opAttrs.groupTp, *opAttrs.tpWorldSize, *opAttrs.tpRankId, *opAttrs.expertShardType, *opAttrs.sharedExpertNum,
        *opAttrs.sharedExpertRankNum, *opAttrs.quantMode, *opAttrs.globalBs, *opAttrs.expertTokenNumsType,
        opOutput.expandX, opOutput.dynamicScales, opOutput.expandIdx, opOutput.expertTokenNums, opOutput.epRecvCounts,
        opOutput.tpRecvCounts, opOutput.expandScales);
    OP_CHECK_IF(apiRet != ge::GRAPH_SUCCESS, OP_LOGE(moeDistributeDispatchInfo, "aclnn api error code %u", apiRet),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeDispatch).OpExecuteFunc(MoeDistributeDispatchExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif

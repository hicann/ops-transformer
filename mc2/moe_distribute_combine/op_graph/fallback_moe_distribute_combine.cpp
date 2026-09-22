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

namespace fallback {
using namespace ge;
using namespace gert;

const char *moeDistributeCombineInfo = "MoeDistributeCombineFallback";

namespace {
struct OpInput {
    const gert::Tensor *expandX;
    const gert::Tensor *expertIds;
    const gert::Tensor *expandIdx;
    const gert::Tensor *epSendCounts;
    const gert::Tensor *expertScales;
};

struct OpOptionalInput {
    const gert::Tensor *tpSendCounts;
    const gert::Tensor *xActiveMask;
    const gert::Tensor *activationScale;
    const gert::Tensor *weightScale;
    const gert::Tensor *groupList;
    const gert::Tensor *expandScales;
};

struct OpOutput {
    const gert::Tensor *y;
};

struct OpAttrs {
    const char *groupEp;
    const int64_t *epWorldSize;
    const int64_t *epRankId;
    const int64_t *moeExpertNum;
    const char *groupTp;
    const int64_t *tpWorldSize;
    const int64_t *tpRankId;
    const int64_t *expertShardType;
    const int64_t *sharedExpertNum;
    const int64_t *sharedExpertRankNum;
    const int64_t *globalBs;
    const int64_t *outDtype;
    const int64_t *commQuantMode;
    const int64_t *groupListType;
};

// 获取必选输入
graphStatus MoeDistributeCombineGetOpInput(OpExecuteContext *host_api_ctx, OpInput &opInput)
{
    opInput.expandX = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_EXPAND_X));
    OP_CHECK_IF(opInput.expandX == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "expandX"),
                return ge::GRAPH_FAILED);

    opInput.expertIds =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_EXPERT_IDS));
    OP_CHECK_IF(opInput.expertIds == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "expertIds"),
                return ge::GRAPH_FAILED);

    opInput.expandIdx =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_EXPAND_IDX));
    OP_CHECK_IF(opInput.expandIdx == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "expandIdx"),
                return ge::GRAPH_FAILED);

    opInput.epSendCounts =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_EP_SEND_COUNTS));
    OP_CHECK_IF(opInput.epSendCounts == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "epSendCounts"),
                return ge::GRAPH_FAILED);

    opInput.expertScales =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_EXPERT_SCALES));
    OP_CHECK_IF(opInput.expertScales == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "expertScales"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取可选输入
graphStatus MoeDistributeCombineGetOpOptionalInput(OpExecuteContext *host_api_ctx, OpOptionalInput &opOptionalInput)
{
    opOptionalInput.tpSendCounts =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_TP_SEND_COUNTS));
    OP_CHECK_IF(opOptionalInput.tpSendCounts == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "tpSendCounts"), return ge::GRAPH_FAILED);

    opOptionalInput.xActiveMask =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_X_ACTIVE_MASK));
    opOptionalInput.activationScale = host_api_ctx->GetOptionalInputTensor(
        static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_ACTIVATION_SCALE));
    opOptionalInput.weightScale =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_WEIGHT_SCALE));
    opOptionalInput.groupList =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_GROUP_LIST));
    opOptionalInput.expandScales =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineInputIdx::K_EXPAND_SCALES));
    return ge::GRAPH_SUCCESS;
}

// 获取输出
graphStatus MoeDistributeCombineGetOpOutput(OpExecuteContext *host_api_ctx, OpOutput &opOutput)
{
    opOutput.y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeCombineOutputIdx::K_Y));
    OP_CHECK_IF(opOutput.y == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "y"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取通信分组属性
graphStatus MoeDistributeCombineGetCommGroupAttrs(const gert::RuntimeAttrs *attrs, OpAttrs &opAttrs)
{
    opAttrs.groupEp = attrs->GetStr(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_GROUP_EP));
    OP_CHECK_IF(opAttrs.groupEp == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "groupEp"),
                return ge::GRAPH_FAILED);

    opAttrs.epWorldSize = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_EP_WORLD_SIZE));
    OP_CHECK_IF(opAttrs.epWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "epWorldSize"),
                return ge::GRAPH_FAILED);

    opAttrs.epRankId = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_EP_RANK_ID));
    OP_CHECK_IF(opAttrs.epRankId == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "epRankId"),
                return ge::GRAPH_FAILED);

    opAttrs.moeExpertNum = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_MOE_EXPERT_NUM));
    OP_CHECK_IF(opAttrs.moeExpertNum == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "moeExpertNum"),
                return ge::GRAPH_FAILED);

    opAttrs.groupTp = attrs->GetStr(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_GROUP_TP));
    OP_CHECK_IF(opAttrs.groupTp == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "groupTp"),
                return ge::GRAPH_FAILED);

    opAttrs.tpWorldSize = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_TP_WORLD_SIZE));
    OP_CHECK_IF(opAttrs.tpWorldSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "tpWorldSize"),
                return ge::GRAPH_FAILED);

    opAttrs.tpRankId = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_TP_RANK_ID));
    OP_CHECK_IF(opAttrs.tpRankId == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "tpRankId"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取专家与量化属性
graphStatus MoeDistributeCombineGetExpertQuantAttrs(const gert::RuntimeAttrs *attrs, OpAttrs &opAttrs)
{
    opAttrs.expertShardType = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_EXPERT_SHARD_TYPE));
    OP_CHECK_IF(opAttrs.expertShardType == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "expertShardType"), return ge::GRAPH_FAILED);

    opAttrs.sharedExpertNum = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_SHARED_EXPERT_NUM));
    OP_CHECK_IF(opAttrs.sharedExpertNum == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "sharedExpertNum"), return ge::GRAPH_FAILED);

    opAttrs.sharedExpertRankNum =
        attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_SHARED_EXPERT_RANK_NUM));
    OP_CHECK_IF(opAttrs.sharedExpertRankNum == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "sharedExpertRankNum"), return ge::GRAPH_FAILED);

    opAttrs.globalBs = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_GLOBAL_BS));
    OP_CHECK_IF(opAttrs.globalBs == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "globalBs"),
                return ge::GRAPH_FAILED);

    opAttrs.outDtype = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_OUT_DTYPE));
    OP_CHECK_IF(opAttrs.outDtype == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "out_dtype"),
                return ge::GRAPH_FAILED);

    opAttrs.commQuantMode = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_COMM_QUANT_MODE));
    OP_CHECK_IF(opAttrs.commQuantMode == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "comm_quant_mode"), return ge::GRAPH_FAILED);

    opAttrs.groupListType = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineAttrIdx::K_GROUP_LIST_TYPE));
    OP_CHECK_IF(opAttrs.groupListType == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "group_list_type"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取属性
graphStatus MoeDistributeCombineGetOpAttrs(OpExecuteContext *host_api_ctx, OpAttrs &opAttrs)
{
    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "attrs"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(MoeDistributeCombineGetCommGroupAttrs(attrs, opAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo, "get comm group attrs failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetExpertQuantAttrs(attrs, opAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo, "get expert quant attrs failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace

static graphStatus MoeDistributeCombineExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD(moeDistributeCombineInfo, "start to fallback for moeDistributeCombine");
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT(moeDistributeCombineInfo, "host_api_ctx"),
                return ge::GRAPH_FAILED);

    OpInput opInput;
    OpOptionalInput opOptionalInput;
    OpOutput opOutput;
    OpAttrs opAttrs;

    OP_CHECK_IF(MoeDistributeCombineGetOpInput(host_api_ctx, opInput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo, "get input failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetOpOptionalInput(host_api_ctx, opOptionalInput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo, "get optional input failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetOpOutput(host_api_ctx, opOutput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo, "get output failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetOpAttrs(host_api_ctx, opAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo, "get attrs failed"), return ge::GRAPH_FAILED);

    const auto apiRet = EXEC_OPAPI_CMD(
        aclnnMoeDistributeCombine, opInput.expandX, opInput.expertIds, opInput.expandIdx, opInput.epSendCounts,
        opInput.expertScales, opOptionalInput.tpSendCounts, opOptionalInput.xActiveMask,
        opOptionalInput.activationScale, opOptionalInput.weightScale, opOptionalInput.groupList,
        opOptionalInput.expandScales, opAttrs.groupEp, *opAttrs.epWorldSize, *opAttrs.epRankId, *opAttrs.moeExpertNum,
        opAttrs.groupTp, *opAttrs.tpWorldSize, *opAttrs.tpRankId, *opAttrs.expertShardType, *opAttrs.sharedExpertNum,
        *opAttrs.sharedExpertRankNum, *opAttrs.globalBs, *opAttrs.outDtype, *opAttrs.commQuantMode,
        *opAttrs.groupListType, opOutput.y);
    OP_CHECK_IF(apiRet != ge::GRAPH_SUCCESS, OP_LOGE(moeDistributeCombineInfo, "aclnn api error code %u", apiRet),
                return apiRet);
    return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombine).OpExecuteFunc(MoeDistributeCombineExecuteFunc);

} // namespace fallback

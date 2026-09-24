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

const char *MoeDistributeCombineV2Info = "MoeDistributeCombineV2Fallback";

namespace {
struct CombineV2Params {
    const gert::Tensor *expand_x = nullptr;
    const gert::Tensor *expert_ids = nullptr;
    const gert::Tensor *assist_info_for_combine = nullptr;
    const gert::Tensor *ep_send_counts = nullptr;
    const gert::Tensor *expert_scales = nullptr;
    const gert::Tensor *tp_send_counts = nullptr;
    const gert::Tensor *x_active_mask = nullptr;
    const gert::Tensor *activation_scale = nullptr;
    const gert::Tensor *weight_scale = nullptr;
    const gert::Tensor *group_list = nullptr;
    const gert::Tensor *shared_expert_x = nullptr;
    const gert::Tensor *elastic_info = nullptr;
    const gert::Tensor *ori_x = nullptr;
    const gert::Tensor *const_expert_alpha_1 = nullptr;
    const gert::Tensor *const_expert_alpha_2 = nullptr;
    const gert::Tensor *const_expert_v = nullptr;
    const gert::Tensor *x = nullptr;
    const char *group_ep = nullptr;
    const char *group_tp = nullptr;
    const int64_t *ep_word_size = nullptr;
    const int64_t *ep_rank_id = nullptr;
    const int64_t *moe_expert_num = nullptr;
    const int64_t *tp_word_size = nullptr;
    const int64_t *tp_rank_id = nullptr;
    const int64_t *expert_shard_type = nullptr;
    const int64_t *shared_expert_num = nullptr;
    const int64_t *shared_expert_rank_num = nullptr;
    const int64_t *global_bs_ptr = nullptr;
    const int64_t *out_dtype_ptr = nullptr;
    const int64_t *comm_quant_mode_ptr = nullptr;
    const int64_t *group_list_type_ptr = nullptr;
    const int64_t *comm_alg_ptr = nullptr;
    const int64_t *zero_expert_num = nullptr;
    const int64_t *copy_expert_num = nullptr;
    const int64_t *const_expert_num = nullptr;
};

static bool CheckParamNotNull(const void *param, const char *name)
{
    if (param == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(MoeDistributeCombineV2Info, name);
        return false;
    }
    return true;
}

#define MC2_CHECK_PARAM_NOT_NULL(param, name) \
    do { \
        if (!CheckParamNotNull((param), (name))) { \
            return ge::GRAPH_FAILED; \
        } \
    } while (0)

static ge::graphStatus FetchInputs(const OpExecuteContext *ctx, CombineV2Params &p)
{
    p.expand_x = ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_EXPAND_X));
    p.expert_ids = ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_EXPERT_IDS));
    p.assist_info_for_combine =
        ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_ASSIST_INFO_FOR_COMBINE));
    p.ep_send_counts = ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_EP_SEND_COUNTS));
    p.expert_scales = ctx->GetInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_EXPERT_SCALES));
    p.tp_send_counts =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_TP_SEND_COUNTS));
    p.x_active_mask =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_X_ACTIVE_MASK));
    p.activation_scale =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_ACTIVATION_SCALE));
    p.weight_scale =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_WEIGHT_SCALE));
    p.group_list = ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_GROUP_LIST));
    p.shared_expert_x =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_SHARED_EXPERT_X));
    p.elastic_info =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_ELASTIC_INFO));
    p.ori_x = ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_ORI_X));
    p.const_expert_alpha_1 =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_CONST_EXPERT_ALPHA_1));
    p.const_expert_alpha_2 =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_CONST_EXPERT_ALPHA_2));
    p.const_expert_v =
        ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2InputIdx::K_CONST_EXPERT_V));
    p.x = ctx->GetOutputTensor(static_cast<size_t>(ops::MoeDistributeCombineV2OutputIdx::K_X));

    MC2_CHECK_PARAM_NOT_NULL(p.expand_x, "expand_x");
    MC2_CHECK_PARAM_NOT_NULL(p.expert_ids, "expert_ids");
    MC2_CHECK_PARAM_NOT_NULL(p.assist_info_for_combine, "assist_info_for_combine");
    MC2_CHECK_PARAM_NOT_NULL(p.ep_send_counts, "ep_send_counts");
    MC2_CHECK_PARAM_NOT_NULL(p.expert_scales, "expert_scales");
    MC2_CHECK_PARAM_NOT_NULL(p.tp_send_counts, "tp_send_counts");
    MC2_CHECK_PARAM_NOT_NULL(p.x, "x");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FetchAttrs(const OpExecuteContext *ctx, CombineV2Params &p)
{
    const auto attrs = ctx->GetAttrs();
    MC2_CHECK_PARAM_NOT_NULL(attrs, "attrs");

    p.group_ep = attrs->GetStr(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_GROUP_EP));
    p.ep_word_size = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_EP_WORLD_SIZE));
    p.ep_rank_id = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_EP_RANK_ID));
    p.moe_expert_num = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_MOE_EXPERT_NUM));
    p.group_tp = attrs->GetStr(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_GROUP_TP));
    p.tp_word_size = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_TP_WORLD_SIZE));
    p.tp_rank_id = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_TP_RANK_ID));
    p.expert_shard_type = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_EXPERT_SHARD_TYPE));
    p.shared_expert_num = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_SHARED_EXPERT_NUM));
    p.shared_expert_rank_num =
        attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_SHARED_EXPERT_RANK_NUM));
    p.global_bs_ptr = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_GLOBAL_BS));
    p.out_dtype_ptr = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_OUT_DTYPE));
    p.comm_quant_mode_ptr = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_COMM_QUANT_MODE));
    p.group_list_type_ptr = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_GROUP_LIST_TYPE));
    p.comm_alg_ptr = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_COMM_ALG));
    p.zero_expert_num = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_ZERO_EXPERT_NUM));
    p.copy_expert_num = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_COPY_EXPERT_NUM));
    p.const_expert_num = attrs->GetInt(static_cast<size_t>(ops::MoeDistributeCombineV2AttrIdx::K_CONST_EXPERT_NUM));

    MC2_CHECK_PARAM_NOT_NULL(p.group_ep, "group_ep");
    MC2_CHECK_PARAM_NOT_NULL(p.ep_word_size, "ep_word_size");
    MC2_CHECK_PARAM_NOT_NULL(p.ep_rank_id, "ep_rank_id");
    MC2_CHECK_PARAM_NOT_NULL(p.moe_expert_num, "moe_expert_num");
    MC2_CHECK_PARAM_NOT_NULL(p.group_tp, "group_tp");
    MC2_CHECK_PARAM_NOT_NULL(p.tp_word_size, "tp_word_size");
    MC2_CHECK_PARAM_NOT_NULL(p.tp_rank_id, "tp_rank_id");
    MC2_CHECK_PARAM_NOT_NULL(p.expert_shard_type, "expert_shard_type");
    MC2_CHECK_PARAM_NOT_NULL(p.shared_expert_num, "shared_expert_num");
    MC2_CHECK_PARAM_NOT_NULL(p.shared_expert_rank_num, "shared_expert_rank_num");
    MC2_CHECK_PARAM_NOT_NULL(p.global_bs_ptr, "global_bs_ptr");
    MC2_CHECK_PARAM_NOT_NULL(p.out_dtype_ptr, "out_dtype");
    MC2_CHECK_PARAM_NOT_NULL(p.comm_quant_mode_ptr, "comm_quant_mode");
    MC2_CHECK_PARAM_NOT_NULL(p.group_list_type_ptr, "group_list_type");
    MC2_CHECK_PARAM_NOT_NULL(p.comm_alg_ptr, "comm_alg_ptr");
    MC2_CHECK_PARAM_NOT_NULL(p.zero_expert_num, "zero_expert_num");
    MC2_CHECK_PARAM_NOT_NULL(p.copy_expert_num, "copy_expert_num");
    MC2_CHECK_PARAM_NOT_NULL(p.const_expert_num, "const_expert_num");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchCombineV2(OpExecuteContext *host_api_ctx, const CombineV2Params &p)
{
    if (p.elastic_info != nullptr || p.ori_x != nullptr || p.const_expert_alpha_1 != nullptr ||
        p.const_expert_alpha_2 != nullptr || p.const_expert_v != nullptr || *p.zero_expert_num != 0 ||
        *p.copy_expert_num != 0 || *p.const_expert_num != 0) {
        const auto api_ret_newfeature = EXEC_OPAPI_CMD(
            aclnnMoeDistributeCombineV3, p.expand_x, p.expert_ids, p.assist_info_for_combine, p.ep_send_counts,
            p.expert_scales, p.tp_send_counts, p.x_active_mask, p.activation_scale, p.weight_scale, p.group_list,
            p.shared_expert_x, p.elastic_info, p.ori_x, p.const_expert_alpha_1, p.const_expert_alpha_2,
            p.const_expert_v, p.group_ep, *p.ep_word_size, *p.ep_rank_id, *p.moe_expert_num, p.group_tp,
            *p.tp_word_size, *p.tp_rank_id, *p.expert_shard_type, *p.shared_expert_num, *p.shared_expert_rank_num,
            *p.global_bs_ptr, *p.out_dtype_ptr, *p.comm_quant_mode_ptr, *p.group_list_type_ptr, *p.comm_alg_ptr,
            *p.zero_expert_num, *p.copy_expert_num, *p.const_expert_num, p.x);
        OP_CHECK_IF(api_ret_newfeature != ge::GRAPH_SUCCESS,
                    OP_LOGE(MoeDistributeCombineV2Info, "aclnn api error code %u", api_ret_newfeature),
                    return api_ret_newfeature);
    } else {
        const auto api_ret = EXEC_OPAPI_CMD(
            aclnnMoeDistributeCombineV2, p.expand_x, p.expert_ids, p.assist_info_for_combine, p.ep_send_counts,
            p.expert_scales, p.tp_send_counts, p.x_active_mask, p.activation_scale, p.weight_scale, p.group_list,
            p.shared_expert_x, p.group_ep, *p.ep_word_size, *p.ep_rank_id, *p.moe_expert_num, p.group_tp,
            *p.tp_word_size, *p.tp_rank_id, *p.expert_shard_type, *p.shared_expert_num, *p.shared_expert_rank_num,
            *p.global_bs_ptr, *p.out_dtype_ptr, *p.comm_quant_mode_ptr, *p.group_list_type_ptr, *p.comm_alg_ptr, p.x);
        OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS,
                    OP_LOGE(MoeDistributeCombineV2Info, "aclnn api error code %u", api_ret), return api_ret);
    }
    return GRAPH_SUCCESS;
}

#undef MC2_CHECK_PARAM_NOT_NULL

} // namespace

static graphStatus MoeDistributeCombineV2ExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD(MoeDistributeCombineV2Info, "start to fallback for moeDistributeCombineV2");
    if (host_api_ctx == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(MoeDistributeCombineV2Info, "host_api_ctx");
        return ge::GRAPH_FAILED;
    }
    CombineV2Params params;
    if (FetchInputs(host_api_ctx, params) != ge::GRAPH_SUCCESS ||
        FetchAttrs(host_api_ctx, params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return DispatchCombineV2(host_api_ctx, params);
}

IMPL_OP(MoeDistributeCombineV2).OpExecuteFunc(MoeDistributeCombineV2ExecuteFunc);

} // namespace fallback
